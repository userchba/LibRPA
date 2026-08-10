#ifndef DDLA_COMM_TRAITS_H
#define DDLA_COMM_TRAITS_H

#include <ddla/ddla_connector.h>
#include <ddla/ddla_comm.h>
#include "ddla_stream_impl.h"
#include "mpi_datatype.h"
#include <vector>
#include <cstddef>

// ---------------------------------------------------------------------------
// CommTraits<DdlaBackend Backend>: a DdlaBackend-templated abstraction over
// LibDDLA's communication layer, replacing the old per-file
// `#ifdef DDLA_USE_CCL ncclComm_t X = handle->nccl_X; #else MPI_Comm X = handle->X; #endif`
// aliasing idiom with real functions, mirroring RuntimeTraits<Backend> in
// ddla_connector.h. Lives directly in namespace ddla (not nested under
// ddla::detail) -- this header is already private (src/, not include/ddla/),
// so an additional detail nesting inside an already-internal header would be
// redundant.
// ---------------------------------------------------------------------------
namespace ddla {

// ---------------------------------------------------------------------------
// Host staging for the GPU_CPU_TUNNEL path.  These return a typed pointer
// into a handle-owned, grow-on-demand byte buffer (see DdlaStream), so the
// tunnel collectives below avoid allocating a fresh std::vector per call.
// Each comm* wrapper is synchronous (blocking MPI plus a stream sync where
// an async host<->device copy follows), so a buffer is never live across two
// wrappers and reuse is safe.
//
// Caveats:
//  - The buffers grow to the largest size seen and never shrink (a
//    high-water mark), which is the intended allocation-amortization tradeoff.
//  - The buffers are owned per DdlaHandle_t and are therefore NOT safe for
//    concurrent host-side use of the same handle from multiple threads;
//    LibDDLA handles are single-threaded per rank today.
// ---------------------------------------------------------------------------
namespace detail {

template <typename T>
inline T* tunnel_staging(DdlaHandle_t h, int which, std::size_t count)
{
    std::vector<std::byte>& buf = (which == 0) ? h->tunnel_host_staging_a
                                               : h->tunnel_host_staging_b;
    const std::size_t need = (count ? count : 1) * sizeof(T);
    if (buf.size() < need) {
        buf.resize(need);
    }
    return reinterpret_cast<T*>(buf.data());
}

} // namespace detail

enum class CommScope { Grid, Row, Col };

template <DdlaBackend Backend> struct CommTraits;   // primary: intentionally undefined

template <>
struct CommTraits<DdlaBackend::CPU> {
    using comm_t = MPI_Comm;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return h->row_comm;
            case CommScope::Col: return h->col_comm;
            default:             return h->comm;
        }
    }
    static void group_start(const DdlaHandle_t&) {}
    static void group_end  (const DdlaHandle_t&) {}
};

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
// RuntimeTraits<GPU>-style: only *defined* in a TU that actually compiled in
// a GPU vendor, exactly like RuntimeTraits<DdlaBackend::GPU> in ddla_connector.h.
template <>
struct CommTraits<DdlaBackend::GPU> {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
    // Tunnel wins if both DDLA_USE_CCL and DDLA_USE_GPU_CPU_TUNNEL are set
    // (matches this project's existing precedence elsewhere, e.g.
    // transport_block.cpp's `#if DDLA_USE_CCL && !DDLA_USE_GPU_CPU_TUNNEL`
    // group-batch guard).
    using comm_t = MPI_Comm;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return h->row_comm;
            case CommScope::Col: return h->col_comm;
            default:             return h->comm;
        }
    }
    static void group_start(const DdlaHandle_t&) {}
    static void group_end  (const DdlaHandle_t&) {}
#elif defined(DDLA_USE_CCL)
    using comm_t = ncclComm_t;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return h->nccl_row_comm;
            case CommScope::Col: return h->nccl_col_comm;
            default:             return h->nccl_comm;
        }
    }
    static void group_start(const DdlaHandle_t&) { CCL_CHECK(ncclGroupStart()); }
    static void group_end  (const DdlaHandle_t&) { CCL_CHECK(ncclGroupEnd());   }
#else
    // "Dead" configuration: GPU vendor compiled in, but neither DDLA_USE_CCL
    // nor DDLA_USE_GPU_CPU_TUNNEL set. Preserves today's direct-MPI-on-
    // device-pointer behavior verbatim (ddla_comm.h's plain `#else` cclSend/
    // cclRecv/cclBcast) -- unsupported/untested, but unchanged from before
    // this refactor, so no regression.
    using comm_t = MPI_Comm;
    static comm_t comm(const DdlaHandle_t& h, CommScope s) {
        switch (s) {
            case CommScope::Row: return h->row_comm;
            case CommScope::Col: return h->col_comm;
            default:             return h->comm;
        }
    }
    static void group_start(const DdlaHandle_t&) {}
    static void group_end  (const DdlaHandle_t&) {}
#endif
};
#endif // defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)

// ---------------------------------------------------------------------------
// comm* function templates. Backend defaults to detail::local_backend_v (the
// per-TU constant already used by the runtime* family in ddla_connector.h),
// so GPU-only callers can invoke these bracket-free, e.g.
// `commBcast(h, CommScope::Row, buf, count, root)`, exactly like they already
// call `runtimeMallocAsync(...)` bracket-free today.
//
// Non-tunnel GPU calls forward to ddla_comm.h's existing 5-arg cclBcast/
// cclSend/cclRecv/cclAllReduce -- which resolve, by the very same macro that
// selected CommTraits<GPU>::comm_t's type above, to either the real NCCL/RCCL
// implementation (DDLA_USE_CCL) or the "dead config" direct-MPI-on-device-
// pointer implementation (neither CCL nor tunnel) -- comm_traits.h does not
// need to distinguish those two cases itself.
//
// The tunnel case stages through handle-owned host scratch buffers (see
// detail::tunnel_staging below) -- callers never see or manage a host
// pointer, unlike ddla_comm.h's tunnel-overloaded cclBcast/cclSend/
// cclRecv/cclAllReduce (reused here as the underlying implementation,
// selected by argument count: the tunnel overloads take an extra leading
// host-buffer argument). Because the tunnel cclBcast/cclRecv/cclAllReduce
// helpers do NOT synchronize after their final, asynchronous host-to-device
// copy (by design -- their original callers keep the host buffer alive
// across a broader scope), an extra runtimeStreamSynchronize is added here
// after any such call so the async copy completes before the scratch buffer
// is reused by the next wrapper. cclSend needs no such extra sync:
// MPI_Send is a blocking call, so the host buffer is safe to reuse the
// instant it returns.
// ---------------------------------------------------------------------------

template <DdlaBackend Backend = detail::local_backend_v>
inline void commGroupStart(const DdlaHandle_t& h) { CommTraits<Backend>::group_start(h); }

template <DdlaBackend Backend = detail::local_backend_v>
inline void commGroupEnd(const DdlaHandle_t& h) { CommTraits<Backend>::group_end(h); }

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commBcast(const DdlaHandle_t& h, CommScope scope, T* buf, std::size_t count, int root)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Bcast(buf, (int)count, detail::mpi_datatype<T>(), root,
                            CommTraits<Backend>::comm(h, scope)));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* host = detail::tunnel_staging<T>(h, 0, count);
        MPI_CHECK(cclBcast(host, buf, count, root, CommTraits<Backend>::comm(h, scope), h->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(h->stream));
#else
        CCL_CHECK(cclBcast(buf, count, root, CommTraits<Backend>::comm(h, scope), h->stream));
#endif
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commSend(const DdlaHandle_t& h, CommScope scope, const T* buf, std::size_t count, int peer)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Send(buf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                           CommTraits<Backend>::comm(h, scope)));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* host = detail::tunnel_staging<T>(h, 0, count);
        MPI_CHECK(cclSend(host, buf, count, peer, CommTraits<Backend>::comm(h, scope), h->stream));
#else
        CCL_CHECK(cclSend(buf, count, peer, CommTraits<Backend>::comm(h, scope), h->stream));
#endif
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commRecv(const DdlaHandle_t& h, CommScope scope, T* buf, std::size_t count, int peer)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Recv(buf, (int)count, detail::mpi_datatype<T>(), peer, 0,
                           CommTraits<Backend>::comm(h, scope), MPI_STATUS_IGNORE));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* host = detail::tunnel_staging<T>(h, 0, count);
        MPI_CHECK(cclRecv(host, buf, count, peer, CommTraits<Backend>::comm(h, scope), h->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(h->stream));
#else
        CCL_CHECK(cclRecv(buf, count, peer, CommTraits<Backend>::comm(h, scope), h->stream));
#endif
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commAllReduce(const DdlaHandle_t& h, CommScope scope,
                          const T* sbuf, T* rbuf, int count, cclOp op)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Allreduce_ddla(sbuf, rbuf, count, op, CommTraits<Backend>::comm(h, scope)));
    } else {
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
        T* hs = detail::tunnel_staging<T>(h, 0, count);
        T* hr = detail::tunnel_staging<T>(h, 1, count);
        MPI_CHECK(cclAllReduce(hs, sbuf, hr, rbuf, count, op,
                               CommTraits<Backend>::comm(h, scope), h->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(h->stream));
#else
        CCL_CHECK(cclAllReduce(sbuf, rbuf, count, op, CommTraits<Backend>::comm(h, scope), h->stream));
#endif
    }
}

template <DdlaBackend Backend = detail::local_backend_v, typename T>
inline void commAlltoallv(const DdlaHandle_t& h, CommScope scope, int nprocs,
                          const T* sendbuf, const int* sendcounts, const int* sdispls,
                          T* recvbuf, const int* recvcounts, const int* rdispls)
{
    if constexpr (Backend == DdlaBackend::CPU) {
        MPI_CHECK(MPI_Alltoallv(sendbuf, sendcounts, sdispls, detail::mpi_datatype<T>(),
                                recvbuf, recvcounts, rdispls, detail::mpi_datatype<T>(),
                                CommTraits<Backend>::comm(h, scope)));
    } else {
#if defined(DDLA_USE_CCL) && !defined(DDLA_USE_GPU_CPU_TUNNEL)
        // One message per peer, aggregating every block destined for/arriving
        // from that peer -- correct because send_blocks/recv_blocks are laid
        // out contiguously per peer (matches the sort-by-peer-rank ordering
        // callers already use to build sendcounts/sdispls/recvcounts/rdispls).
        CommTraits<Backend>::group_start(h);
        for (int p = 0; p < nprocs; ++p) {
            if (sendcounts[p] > 0)
                CCL_CHECK(cclSend(sendbuf + sdispls[p], sendcounts[p], p,
                                  CommTraits<Backend>::comm(h, scope), h->stream));
            if (recvcounts[p] > 0)
                CCL_CHECK(cclRecv(recvbuf + rdispls[p], recvcounts[p], p,
                                  CommTraits<Backend>::comm(h, scope), h->stream));
        }
        CommTraits<Backend>::group_end(h);
#else
        // Tunnel (and "dead config" without CCL): host-staged single
        // MPI_Alltoallv collective -- today's non-CCL ptran.cpp logic,
        // hoisted out of its #ifdef unchanged.
        int send_total = nprocs > 0 ? sdispls[nprocs - 1] + sendcounts[nprocs - 1] : 0;
        int recv_total = nprocs > 0 ? rdispls[nprocs - 1] + recvcounts[nprocs - 1] : 0;
        T* h_send = detail::tunnel_staging<T>(h, 0, static_cast<std::size_t>(send_total));
        T* h_recv = detail::tunnel_staging<T>(h, 1, static_cast<std::size_t>(recv_total));
        if (send_total > 0) {
            RUNTIME_CHECK(runtimeMemcpyAsync(h_send, sendbuf, sizeof(T) * send_total,
                                             runtimeMemcpyDeviceToHost, h->stream));
            RUNTIME_CHECK(runtimeStreamSynchronize(h->stream));
        }
        MPI_CHECK(MPI_Alltoallv(h_send, sendcounts, sdispls, detail::mpi_datatype<T>(),
                                h_recv, recvcounts, rdispls, detail::mpi_datatype<T>(),
                                CommTraits<Backend>::comm(h, scope)));
        if (recv_total > 0) {
            RUNTIME_CHECK(runtimeMemcpyAsync(recvbuf, h_recv, sizeof(T) * recv_total,
                                             runtimeMemcpyHostToDevice, h->stream));
            RUNTIME_CHECK(runtimeStreamSynchronize(h->stream));
        }
#endif
    }
}

} // namespace ddla

#endif // DDLA_COMM_TRAITS_H
