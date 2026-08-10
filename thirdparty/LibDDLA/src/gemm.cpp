#include <ddla/gemm.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ddla_stream_impl.h"

#if DDLA_HAS_CPU
extern "C" {
void sgemm_(const char*, const char*, const int*, const int*, const int*,
            const float*, const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
void cgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<float>*, const std::complex<float>*, const int*,
            const std::complex<float>*, const int*,
            const std::complex<float>*, std::complex<float>*, const int*);
void zgemm_(const char*, const char*, const int*, const int*, const int*,
            const std::complex<double>*, const std::complex<double>*, const int*,
            const std::complex<double>*, const int*,
            const std::complex<double>*, std::complex<double>*, const int*);
}
#endif

namespace ddla {

inline const char* backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

template <DdlaBackend Backend, typename T>
void gemm(
    const DdlaHandle_t& handle,
    char transa, char transb,
    int m, int n, int k,
    const T& alpha,
    const T* A, int lda,
    const T* B, int ldb,
    const T& beta,
    T* C, int ldc)
{
    static_assert(Backend != DdlaBackend::CPU || DDLA_HAS_CPU,
                  "CPU gemm is not available in this LibDDLA build");
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU gemm is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("gemm: null handle");
    }
    const DdlaBackend actual = ddla_get_backend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("gemm: template backend ") + backend_name(Backend) +
            " does not match handle backend " + backend_name(actual));
    }

    if constexpr (Backend == DdlaBackend::CPU) {
#if DDLA_HAS_CPU
        if constexpr (std::is_same_v<T, float>)
            sgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else if constexpr (std::is_same_v<T, double>)
            dgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else if constexpr (std::is_same_v<T, std::complex<float>>)
            cgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else if constexpr (std::is_same_v<T, std::complex<double>>)
            zgemm_(&transa, &transb, &m, &n, &k,
                   &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        else
            static_assert(sizeof(T) == 0, "gemm: unsupported scalar type T");
#endif
    } else {
#if DDLA_HAS_GPU
        const deblasOperation_t opA = transa == 'N' ? DEBLAS_OP_N :
                                      transa == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
        const deblasOperation_t opB = transb == 'N' ? DEBLAS_OP_N :
                                      transb == 'T' ? DEBLAS_OP_T : DEBLAS_OP_C;
        if constexpr (std::is_same_v<T, float>) {
#if defined(DDLA_USE_CUDA)
            BLAS_CHECK(cublasSgemm(handle->blasH, opA, opB, m, n, k,
                                   &alpha, A, lda, B, ldb, &beta, C, ldc));
#elif defined(DDLA_USE_HIP)
            BLAS_CHECK(hipblasSgemm(handle->blasH, opA, opB, m, n, k,
                                    &alpha, A, lda, B, ldb, &beta, C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else if constexpr (std::is_same_v<T, double>) {
#if defined(DDLA_USE_CUDA)
            BLAS_CHECK(cublasDgemm(handle->blasH, opA, opB, m, n, k,
                                   &alpha, A, lda, B, ldb, &beta, C, ldc));
#elif defined(DDLA_USE_HIP)
            BLAS_CHECK(hipblasDgemm(handle->blasH, opA, opB, m, n, k,
                                    &alpha, A, lda, B, ldb, &beta, C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
#if defined(DDLA_USE_CUDA)
            BLAS_CHECK(cublasCgemm(handle->blasH, opA, opB, m, n, k,
                                   (cuFloatComplex*)&alpha, (cuFloatComplex*)A, lda,
                                   (cuFloatComplex*)B, ldb,
                                   (cuFloatComplex*)&beta, (cuFloatComplex*)C, ldc));
#elif defined(DDLA_USE_HIP)
            BLAS_CHECK(hipblasCgemm(handle->blasH, opA, opB, m, n, k,
                                    (hipblasComplex*)&alpha, (hipblasComplex*)A, lda,
                                    (hipblasComplex*)B, ldb,
                                    (hipblasComplex*)&beta, (hipblasComplex*)C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
#if defined(DDLA_USE_CUDA)
            BLAS_CHECK(cublasZgemm(handle->blasH, opA, opB, m, n, k,
                                   (cuDoubleComplex*)&alpha, (cuDoubleComplex*)A, lda,
                                   (cuDoubleComplex*)B, ldb,
                                   (cuDoubleComplex*)&beta, (cuDoubleComplex*)C, ldc));
#elif defined(DDLA_USE_HIP)
            BLAS_CHECK(hipblasZgemm(handle->blasH, opA, opB, m, n, k,
                                    (hipblasDoubleComplex*)&alpha, (hipblasDoubleComplex*)A, lda,
                                    (hipblasDoubleComplex*)B, ldb,
                                    (hipblasDoubleComplex*)&beta, (hipblasDoubleComplex*)C, ldc));
#else
            throw std::runtime_error(
                "gemm: GPU backend requires DDLA_USE_CUDA or DDLA_USE_HIP");
#endif
        } else {
            static_assert(sizeof(T) == 0, "gemm: unsupported scalar type T");
        }
#endif
    }
}

#define INSTANTIATE_GEMM(BACKEND, TYPE)                                      \
    template void gemm<BACKEND, TYPE>(                               \
        const DdlaHandle_t&, char, char, int, int, int,                       \
        const TYPE&, const TYPE*, int, const TYPE*, int,                      \
        const TYPE&, TYPE*, int)

#if DDLA_HAS_CPU
INSTANTIATE_GEMM(DdlaBackend::CPU, float);
INSTANTIATE_GEMM(DdlaBackend::CPU, double);
INSTANTIATE_GEMM(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_GEMM(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_GEMM(DdlaBackend::GPU, float);
INSTANTIATE_GEMM(DdlaBackend::GPU, double);
INSTANTIATE_GEMM(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_GEMM(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_GEMM

} // namespace ddla
