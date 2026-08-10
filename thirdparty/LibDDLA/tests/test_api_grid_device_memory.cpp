#include "api_grid_test_common.h"

using namespace api_grid_test;

void check_device_memory(const ddla::DdlaHandle_t& handle, const Shape&)
{
    runtimeStream_t stream = reinterpret_cast<runtimeStream_t>(ddla_get_stream(handle));
    double host_value = 0.0;

    double* typed_sync = &host_value;
    RUNTIME_CHECK(runtimeMalloc(&typed_sync, 0));
    require_close(handle, "runtimeMalloc(typed, zero)",
                  typed_sync == nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFree(typed_sync));

    void* raw_sync = &host_value;
    RUNTIME_CHECK(runtimeMalloc(&raw_sync, 0));
    require_close(handle, "runtimeMalloc(void, zero)",
                  raw_sync == nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFree(raw_sync));

    double* typed_async = &host_value;
    RUNTIME_CHECK(runtimeMallocAsync(&typed_async, 0, stream));
    require_close(handle, "runtimeMallocAsync(typed, zero)",
                  typed_async == nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFreeAsync(typed_async, stream));

    void* raw_async = &host_value;
    RUNTIME_CHECK(runtimeMallocAsync(&raw_async, 0, stream));
    require_close(handle, "runtimeMallocAsync(void, zero)",
                  raw_async == nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFreeAsync(raw_async, stream));

    RUNTIME_CHECK(runtimeFree(nullptr));
    RUNTIME_CHECK(runtimeFreeAsync(nullptr, stream));

    const runtimeError_t null_sync_status = runtimeMalloc(nullptr, 0);
    require_close(handle, "runtimeMalloc(null output)",
                  null_sync_status != runtimeSuccess ? 0.0 : 1.0, 0.0);
    (void)runtimeGetLastError();

    const runtimeError_t null_async_status = runtimeMallocAsync(nullptr, 0, stream);
    require_close(handle, "runtimeMallocAsync(null output)",
                  null_async_status != runtimeSuccess ? 0.0 : 1.0, 0.0);
    (void)runtimeGetLastError();

    double* d_sync = nullptr;
    RUNTIME_CHECK(runtimeMalloc(&d_sync, sizeof(double)));
    require_close(handle, "runtimeMalloc(typed, nonzero)",
                  d_sync != nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFree(d_sync));

    void* d_raw_sync = nullptr;
    RUNTIME_CHECK(runtimeMalloc(&d_raw_sync, sizeof(double)));
    require_close(handle, "runtimeMalloc(void, nonzero)",
                  d_raw_sync != nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFree(d_raw_sync));

    double* d_async = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(&d_async, sizeof(double), stream));
    require_close(handle, "runtimeMallocAsync(typed, nonzero)",
                  d_async != nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFreeAsync(d_async, stream));

    void* d_raw_async = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(&d_raw_async, sizeof(double), stream));
    require_close(handle, "runtimeMallocAsync(void, nonzero)",
                  d_raw_async != nullptr ? 0.0 : 1.0, 0.0);
    RUNTIME_CHECK(runtimeFreeAsync(d_raw_async, stream));
    check_ddla_sync(handle);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_device_memory", check_device_memory);
}
