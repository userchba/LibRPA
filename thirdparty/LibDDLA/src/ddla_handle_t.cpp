#include "ddla_stream_impl.h"
#include <stdexcept>
#include <cstring>

namespace ddla {

// ---------------------------------------------------------------------------
// Backend availability (compile-time capabilities from generated config)
// ---------------------------------------------------------------------------
bool ddla_backend_available(DdlaBackend backend)
{
    switch (backend) {
    case DdlaBackend::CPU:
#if DDLA_HAS_CPU
        return true;
#else
        return false;
#endif
    case DdlaBackend::GPU:
#if DDLA_HAS_GPU
        return true;
#else
        return false;
#endif
    }
    return false;
}

// ---------------------------------------------------------------------------
// ddla_init — allocate opaque handle, validate backend
// ---------------------------------------------------------------------------
void ddla_init(DdlaHandle_t& handle)
{
    ddla_init(handle, default_backend_v);
}

void ddla_init(DdlaHandle_t& handle, DdlaBackend requested_backend)
{
    // Validate availability
    if (!ddla_backend_available(requested_backend)) {
        throw std::runtime_error(
            "Requested backend is not available in this build of LibDDLA");
    }

    // Allocate fresh handle — do NOT read the incoming value (callers may
    // pass uninitialized pointer variables; reading them is UB).
    handle = new DdlaStream();
    handle->backend = requested_backend;
}

// ---------------------------------------------------------------------------
// ddla_get_backend
// ---------------------------------------------------------------------------
DdlaBackend ddla_get_backend(const DdlaHandle_t& handle)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddla_get_backend: handle is null");
    }
    return handle->backend;
}

// ---------------------------------------------------------------------------
// Shared by both ddla_set overloads: validate the handle backend is
// available in this build, and collectively verify every MPI rank picked
// the same backend. Packs {value, -value} into a single MPI_MAX Allreduce
// to recover both the max and the min in one collective call instead of two.
// ---------------------------------------------------------------------------
static void resolve_and_validate_backend(DdlaHandle_t handle, const MPI_Comm& comm)
{
    const DdlaBackend resolved = handle->backend;

    if (!ddla_backend_available(resolved)) {
        throw std::runtime_error(
            "Resolved backend is not available in this build of LibDDLA");
    }

    int send[2] = { static_cast<int>(resolved), -static_cast<int>(resolved) };
    int recv[2] = { 0, 0 };
    MPI_Allreduce(send, recv, 2, MPI_INT, MPI_MAX, comm);
    int max_backend = recv[0];
    int min_backend = -recv[1];
    if (max_backend != min_backend) {
        throw std::runtime_error(
            "Not all MPI ranks selected the same backend");
    }
}

// ---------------------------------------------------------------------------
// ddla_set — initialize process grid, verify backend consistency collectively
// ---------------------------------------------------------------------------
void ddla_set(DdlaHandle_t handle, const MPI_Comm& comm, const char& major)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddla_set: handle is null");
    }

    resolve_and_validate_backend(handle, comm);

    // CPU handles must not execute GPU device setup
    handle->init(comm, major);
    handle->initialized = true;
}

void ddla_set(DdlaHandle_t handle, const MPI_Comm& comm,
              const int& nprows, const int& npcols, const char& major)
{
    if (handle == nullptr) {
        throw std::runtime_error("ddla_set: handle is null");
    }

    resolve_and_validate_backend(handle, comm);

    handle->init(nprows, npcols, comm, major);
    handle->initialized = true;
}

// ---------------------------------------------------------------------------
// ddla_destroy — idempotent cleanup
// ---------------------------------------------------------------------------
void ddla_destroy(DdlaHandle_t& handle)
{
    if (handle == nullptr) return;
    if (handle->destroyed) {
        handle = nullptr;
        return;
    }
    handle->clean();
    handle->destroyed = true;
    delete handle;
    handle = nullptr;
}

// ---------------------------------------------------------------------------
// ddla_get_stream — CPU returns nullptr; GPU returns the stream handle
// ---------------------------------------------------------------------------
void* ddla_get_stream(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return nullptr;
    if (handle->backend == DdlaBackend::CPU) return nullptr;
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
    return reinterpret_cast<void*>(handle->stream);
#else
    // GPU backend requested but this is a CPU-only build — unreachable
    // if ddla_backend_available is checked correctly before init.
    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
// Memory helpers — dispatch by handle backend, with uniform validation
// ---------------------------------------------------------------------------
int ddla_malloc(void** ptr, std::size_t bytes, const DdlaHandle_t& handle)
{
    if (handle == nullptr || ptr == nullptr) return 1;

    // Zero-byte allocation: set *ptr = nullptr, return success
    if (bytes == 0) {
        *ptr = nullptr;
        return 0;
    }

    if (handle->backend == DdlaBackend::CPU) {
        *ptr = std::malloc(bytes);
        return (*ptr != nullptr) ? 0 : 1;
    }
    return static_cast<int>(runtimeMallocAsync(ptr, bytes, handle->stream));
}

int ddla_free(void* ptr, const DdlaHandle_t& handle)
{
    if (handle == nullptr) return 1;
    // Freeing nullptr is a valid no-op on both CPU and GPU
    if (ptr == nullptr) return 0;
    if (handle->backend == DdlaBackend::CPU) {
        std::free(ptr);
        return 0;
    }
    return static_cast<int>(runtimeFreeAsync(ptr, handle->stream));
}

int ddla_memcpy(void* dst, const void* src, std::size_t bytes,
                DdlaMemoryCopyKind kind, const DdlaHandle_t& handle)
{
    if (handle == nullptr) return 1;

    // Zero-byte copy: no-op, do not dereference src/dst
    if (bytes == 0) return 0;

    // Nonzero copy: reject null source or destination
    if (dst == nullptr || src == nullptr) return 1;

    // Reject invalid copy kind values
    switch (kind) {
    case DdlaMemoryCopyKind::HostToDevice:
    case DdlaMemoryCopyKind::DeviceToHost:
    case DdlaMemoryCopyKind::DeviceToDevice:
        break;
    default:
        return 1;
    }

    if (handle->backend == DdlaBackend::CPU) {
        std::memcpy(dst, src, bytes);
        return 0;
    }
    runtimeMemcpyKind dkind;
    switch (kind) {
    case DdlaMemoryCopyKind::HostToDevice:   dkind = runtimeMemcpyHostToDevice;   break;
    case DdlaMemoryCopyKind::DeviceToHost:   dkind = runtimeMemcpyDeviceToHost;   break;
    case DdlaMemoryCopyKind::DeviceToDevice: dkind = runtimeMemcpyDeviceToDevice; break;
    default: return 1; // unreachable (validated above)
    }
    return static_cast<int>(
        runtimeMemcpyAsync(dst, src, bytes, dkind, handle->stream));
}

int ddla_synchronize(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return 1;
    if (handle->backend == DdlaBackend::CPU) return 0;  // CPU is synchronous
    return static_cast<int>(runtimeStreamSynchronize(handle->stream));
}

// ---------------------------------------------------------------------------
// Public accessors
// ---------------------------------------------------------------------------
MPI_Comm ddla_get_communicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->comm;
}

int ddla_get_rank(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return -1;
    if (handle->comm == MPI_COMM_NULL) return -1;
    int rank;
    MPI_Comm_rank(handle->comm, &rank);
    return rank;
}

int ddla_get_size(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return 0;
    if (handle->comm == MPI_COMM_NULL) return 0;
    int size;
    MPI_Comm_size(handle->comm, &size);
    return size;
}

void ddla_get_grid_coords(const DdlaHandle_t& handle,
                          int& myprow, int& mypcol)
{
    if (handle == nullptr) {
        myprow = -1; mypcol = -1;
        return;
    }
    myprow = handle->myprow_;
    mypcol = handle->mypcol_;
}

void ddla_get_grid_dims(const DdlaHandle_t& handle,
                        int& nprows, int& npcols)
{
    if (handle == nullptr) {
        nprows = 0; npcols = 0;
        return;
    }
    nprows = handle->nprows_;
    npcols = handle->npcols_;
}

void ddla_rank_to_rc(const DdlaHandle_t& handle,
                     int rank, int& row, int& col)
{
    if (handle == nullptr) {
        row = -1; col = -1;
        return;
    }
    handle->rank_to_rc(rank, row, col);
}

int ddla_rc_to_rank(const DdlaHandle_t& handle, int row, int col)
{
    if (handle == nullptr) return -1;
    return handle->rc_to_rank(row, col);
}

MPI_Comm ddla_get_row_communicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->row_comm;
}

MPI_Comm ddla_get_col_communicator(const DdlaHandle_t& handle)
{
    if (handle == nullptr) return MPI_COMM_NULL;
    return handle->col_comm;
}

} // namespace ddla
