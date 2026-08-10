/**
 * @file test_backend_handle.cpp
 * @brief Test backend selection, handle lifecycle, and public accessors.
 *
 * Covers:
 *  - Default resolution to the compile-time default backend
 *  - Explicit available backend
 *  - Explicit unavailable backend (throws)
 *  - Backend query (ddla_get_backend)
 *  - CPU stream returns nullptr (when CPU)
 *  - Two handles created from the same original MPI communicator
 *  - Cleanup in both orders
 *  - All operations must be collective
 */

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>

using namespace ddla;

static int myid = -1;
static int nprocs = 0;

#define TEST(name, cond)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "FAIL [" << myid << "]: " << (name) << std::endl;    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
        if (myid == 0) std::cout << "  PASS: " << (name) << std::endl;        \
    } while (0)

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myid == 0) std::cout << "=== Backend & Handle Test ===" << std::endl;

    // ---- 1. Default backend resolution --------------------------------
    {
        DdlaHandle_t h1 = nullptr;
        ddla_init(h1);
        TEST("Default resolves to compile-time default backend",
             ddla_get_backend(h1) == default_backend_v);
        TEST("Resolved backend is available",
             ddla_backend_available(ddla_get_backend(h1)));
        ddla_destroy(h1);
    }

    // ---- 2. Explicit available backend ---------------------------------
    {
        // Determine which backends are available
        bool cpu_avail = ddla_backend_available(DdlaBackend::CPU);
        bool gpu_avail = ddla_backend_available(DdlaBackend::GPU);
        TEST("At least one backend available", cpu_avail || gpu_avail);

        DdlaBackend explicit_be = cpu_avail ? DdlaBackend::CPU : DdlaBackend::GPU;
        DdlaHandle_t h2 = nullptr;
        ddla_init(h2, explicit_be);
        TEST("Explicit available backend matches request",
             ddla_get_backend(h2) == explicit_be);
        ddla_destroy(h2);
    }

    // ---- 3. Explicit unavailable backend (must throw) ------------------
    {
        DdlaBackend unavailable = ddla_backend_available(DdlaBackend::CPU)
                                      ? DdlaBackend::GPU
                                      : DdlaBackend::CPU;
        if (!ddla_backend_available(unavailable)) {
            bool threw = false;
            DdlaHandle_t h3 = nullptr;
            try {
                ddla_init(h3, unavailable);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            TEST("Unavailable backend throws std::runtime_error", threw);
            // handle should remain nullptr after failed init
            TEST("Handle is nullptr after failed init", h3 == nullptr);
        } else {
            if (myid == 0)
                std::cout << "  SKIP: both backends available (unavailable test "
                             "not applicable)"
                          << std::endl;
        }
    }

    // ---- 4. Backend query on null handle -------------------------------
    {
        DdlaHandle_t null_h = nullptr;
        bool threw = false;
        try {
            ddla_get_backend(null_h);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        TEST("ddla_get_backend(nullptr) throws std::runtime_error", threw);
        TEST("ddla_get_stream(nullptr) returns nullptr",
             ddla_get_stream(null_h) == nullptr);
    }

    // ---- 5. CPU stream = nullptr (when CPU backend is available) -------
    if (ddla_backend_available(DdlaBackend::CPU)) {
        DdlaHandle_t h_cpu = nullptr;
        ddla_init(h_cpu, DdlaBackend::CPU);
        ddla_set(h_cpu, MPI_COMM_WORLD, 'R');
        void* s = ddla_get_stream(h_cpu);
#if DDLA_HAS_CPU
        TEST("CPU handle stream is nullptr", s == nullptr);
#else
        // In a GPU-only build, CPU backend shouldn't be available
        (void)s;
#endif
        ddla_destroy(h_cpu);
    } else {
        if (myid == 0)
            std::cout << "  SKIP: CPU backend not available" << std::endl;
    }

    // ---- 6. Public accessors after ddla_set ----------------------------
    {
        DdlaHandle_t h = nullptr;
        ddla_init(h);
        ddla_set(h, MPI_COMM_WORLD, 'R');

        int rank = ddla_get_rank(h);
        int size = ddla_get_size(h);
        TEST("ddla_get_rank in range", rank >= 0 && rank < size);
        TEST("ddla_get_size matches nprocs", size == nprocs);

        MPI_Comm comm = ddla_get_communicator(h);
        TEST("ddla_get_communicator != MPI_COMM_NULL", comm != MPI_COMM_NULL);
        int comm_rank, comm_size;
        MPI_Comm_rank(comm, &comm_rank);
        MPI_Comm_size(comm, &comm_size);
        TEST("communicator rank matches", comm_rank == rank);
        TEST("communicator size matches", comm_size == size);

        int myprow = -1, mypcol = -1;
        ddla_get_grid_coords(h, myprow, mypcol);
        TEST("grid coords in range",
             myprow >= 0 && mypcol >= 0);

        int nprows = 0, npcols = 0;
        ddla_get_grid_dims(h, nprows, npcols);
        TEST("grid dims product matches nprocs", nprows * npcols == nprocs);

        // rank_to_rc / rc_to_rank roundtrip
        for (int r = 0; r < nprocs; ++r) {
            int rr, rc;
            ddla_rank_to_rc(h, r, rr, rc);
            int back = ddla_rc_to_rank(h, rr, rc);
            TEST("rank_to_rc/rc_to_rank roundtrip", back == r);
        }

        ddla_destroy(h);
    }

    // ---- 7. Two handles from the same original communicator ------------
    {
        DdlaHandle_t ha = nullptr, hb = nullptr;
        ddla_init(ha);
        ddla_init(hb);
        ddla_set(ha, MPI_COMM_WORLD, 'R');
        ddla_set(hb, MPI_COMM_WORLD, 'R');

        TEST("Both handles initialized", ha != nullptr && hb != nullptr);
        TEST("Both handles have same backend",
             ddla_get_backend(ha) == ddla_get_backend(hb));

        MPI_Comm ca = ddla_get_communicator(ha);
        MPI_Comm cb = ddla_get_communicator(hb);
        int cmp_result = MPI_UNEQUAL;
        MPI_Comm_compare(ca, cb, &cmp_result);
        TEST("Duplicated communicators are congruent (not identical)",
             cmp_result == MPI_CONGRUENT);

        int ra, rb;
        MPI_Comm_rank(ca, &ra);
        MPI_Comm_rank(cb, &rb);
        TEST("Same rank in both communicators", ra == rb);

        // Clean up in reverse order
        ddla_destroy(hb);
        ddla_destroy(ha);
    }

    // ---- 8. Cleanup in both orders --------------------------------------
    {
        DdlaHandle_t hx = nullptr, hy = nullptr;
        ddla_init(hx);
        ddla_init(hy);
        ddla_set(hx, MPI_COMM_WORLD, 'R');
        ddla_set(hy, MPI_COMM_WORLD, 'R');

        // Destroy in same order as creation
        ddla_destroy(hx);
        ddla_destroy(hy);
        TEST("Same-order destroy: handles null", hx == nullptr && hy == nullptr);
    }

    // ---- 9. Idempotent double-destroy -----------------------------------
    {
        DdlaHandle_t h = nullptr;
        ddla_init(h);
        ddla_set(h, MPI_COMM_WORLD, 'R');
        ddla_destroy(h);
        ddla_destroy(h);  // should be safe
        TEST("Double destroy is safe", h == nullptr);
    }

    // ---- 9b. Repeated ddla_set (reinit lifecycle) -----------------------
    {
        DdlaHandle_t h = nullptr;
        ddla_init(h);
        ddla_set(h, MPI_COMM_WORLD, 'R');
        int rank1 = ddla_get_rank(h);
        int size1 = ddla_get_size(h);
        MPI_Comm comm1 = ddla_get_communicator(h);
        MPI_Comm comm1_copy = MPI_COMM_NULL;
        MPI_Comm_dup(comm1, &comm1_copy);

        // Second ddla_set on the same handle — must clean and reinitialize
        ddla_set(h, MPI_COMM_WORLD, 'R');
        int rank2 = ddla_get_rank(h);
        int size2 = ddla_get_size(h);
        MPI_Comm comm2 = ddla_get_communicator(h);

        TEST("Repeated ddla_set preserves rank", rank1 == rank2);
        TEST("Repeated ddla_set preserves size", size1 == size2);
        TEST("Repeated ddla_set creates new communicator", comm2 != MPI_COMM_NULL);

        int cmp_result = MPI_UNEQUAL;
        MPI_Comm_compare(comm1_copy, comm2, &cmp_result);
        TEST("Repeated ddla_set communicators are congruent",
             cmp_result == MPI_CONGRUENT);

        MPI_Comm_free(&comm1_copy);
        ddla_destroy(h);
    }

    // ---- 10. Memory helpers smoke test ----------------------------------
    {
        DdlaHandle_t h = nullptr;
        ddla_init(h);
        ddla_set(h, MPI_COMM_WORLD, 'R');

        void* ptr = nullptr;
        int rc = ddla_malloc(&ptr, 1024, h);
        TEST("ddla_malloc succeeds", rc == 0 && ptr != nullptr);

        // Create host pattern buffer and fill
        std::vector<char> host_src(1024, 0);
        for (size_t i = 0; i < 1024; ++i)
            host_src[i] = static_cast<char>(i & 0xFF);

        // Copy host -> device
        rc = ddla_memcpy(ptr, host_src.data(), 1024,
                         DdlaMemoryCopyKind::HostToDevice, h);
        TEST("ddla_memcpy H2D succeeds", rc == 0);
        rc = ddla_synchronize(h);
        TEST("ddla_synchronize after H2D succeeds", rc == 0);

        // Allocate host buffer for readback
        std::vector<char> host(1024, 0);

        // Copy device -> host
        rc = ddla_memcpy(host.data(), ptr, 1024,
                         DdlaMemoryCopyKind::DeviceToHost, h);
        TEST("ddla_memcpy D2H succeeds", rc == 0);

        rc = ddla_synchronize(h);
        TEST("ddla_synchronize after D2H succeeds", rc == 0);

        // Verify pattern
        bool pattern_ok = true;
        for (size_t i = 0; i < 1024; ++i) {
            if (static_cast<unsigned char>(host[i]) !=
                static_cast<unsigned char>(i & 0xFF)) {
                pattern_ok = false;
                break;
            }
        }
        TEST("Memory pattern preserved", pattern_ok);

        rc = ddla_free(ptr, h);
        TEST("ddla_free succeeds", rc == 0);

        ddla_destroy(h);
    }

    // ---- 11. Boundary tests for every compiled/available backend ---------
    // Test zero-byte allocation, zero-byte copy, invalid args, and enum validation.
    {
        bool cpu_avail = ddla_backend_available(DdlaBackend::CPU);
        bool gpu_avail = ddla_backend_available(DdlaBackend::GPU);
        DdlaBackend backends_to_test[2];
        int nb = 0;
        if (cpu_avail) backends_to_test[nb++] = DdlaBackend::CPU;
        if (gpu_avail) backends_to_test[nb++] = DdlaBackend::GPU;

        for (int bi = 0; bi < nb; ++bi) {
            DdlaBackend be = backends_to_test[bi];
            const char* be_name = (be == DdlaBackend::CPU) ? "CPU" : "GPU";

            DdlaHandle_t h = nullptr;
            ddla_init(h, be);
            ddla_set(h, MPI_COMM_WORLD, 'R');

            // 11a. Zero-byte allocation succeeds, returns nullptr
            {
                void* zp = reinterpret_cast<void*>(0x1); // non-null sentinel
                int rc = ddla_malloc(&zp, 0, h);
                TEST(std::string(be_name) + " zero-byte malloc succeeds", rc == 0);
                TEST(std::string(be_name) + " zero-byte malloc returns nullptr",
                     zp == nullptr);
            }

            // 11b. Zero-byte copy with null pointers succeeds
            {
                int rc = ddla_memcpy(nullptr, nullptr, 0,
                                     DdlaMemoryCopyKind::HostToDevice, h);
                TEST(std::string(be_name) + " zero-byte memcpy(null,null) succeeds",
                     rc == 0);
            }

            // 11c. Nonzero copy with null source fails
            {
                char buf[1] = {0};
                int rc = ddla_memcpy(buf, nullptr, 1,
                                     DdlaMemoryCopyKind::HostToDevice, h);
                TEST(std::string(be_name) + " nonzero memcpy null src fails",
                     rc != 0);
            }

            // 11d. Nonzero copy with null destination fails
            {
                char buf[1] = {0};
                int rc = ddla_memcpy(nullptr, buf, 1,
                                     DdlaMemoryCopyKind::HostToDevice, h);
                TEST(std::string(be_name) + " nonzero memcpy null dst fails",
                     rc != 0);
            }

            // 11e. Invalid copy kind returns failure
            {
                char buf1[1] = {0}, buf2[1] = {0};
                int rc = ddla_memcpy(buf1, buf2, 1,
                                     static_cast<DdlaMemoryCopyKind>(999), h);
                TEST(std::string(be_name) + " invalid copy kind fails",
                     rc != 0);
            }

            // Cleanup: free nullptr (valid no-op)
            {
                int rc = ddla_free(nullptr, h);
                TEST(std::string(be_name) + " free nullptr succeeds", rc == 0);
            }

            ddla_destroy(h);
        }
    }

    if (myid == 0) std::cout << "ALL BACKEND HANDLE TESTS PASSED" << std::endl;
    MPI_Finalize();
    return 0;
}
