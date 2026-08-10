#ifndef HERK_H
#define HERK_H

#include "ddla_connector.h"

namespace ddla{

inline deblasStatus_t deblasHerk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const double& alpha,
    const std::complex<double> *A, int lda,
    const double& beta,
    std::complex<double> *C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasZherk(
        handle, uplo, trans, n, k, 
        &alpha, (hipblasDoubleComplex*)A, lda,
        &beta, (hipblasDoubleComplex*)C, ldc
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasZherk(
        handle, uplo, trans, n, k, 
        &alpha, (cuDoubleComplex*)A, lda, 
        &beta, (cuDoubleComplex*)C, ldc
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}


inline deblasStatus_t deblasHerk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const float& alpha,
    const float* A, int lda,
    const float& beta,
    float* C, int ldc
)
{
    // Real HERK == SYRK.  For real data a conjugate transpose is identical
    // to a plain transpose, so map DEBLAS_OP_C -> transpose as well.
    deblasOperation_t trans_real = (trans == DEBLAS_OP_N) ? DEBLAS_OP_N : DEBLAS_OP_T;
    #if defined(DDLA_USE_HIP)
    return hipblasSsyrk(handle, uplo, trans_real, n, k,
        &alpha, A, lda, &beta, C, ldc);
    #elif defined(DDLA_USE_CUDA)
    return cublasSsyrk(handle, uplo, trans_real, n, k,
        &alpha, A, lda, &beta, C, ldc);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasHerk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const double& alpha,
    const double* A, int lda,
    const double& beta,
    double* C, int ldc
)
{
    deblasOperation_t trans_real = (trans == DEBLAS_OP_N) ? DEBLAS_OP_N : DEBLAS_OP_T;
    #if defined(DDLA_USE_HIP)
    return hipblasDsyrk(handle, uplo, trans_real, n, k,
        &alpha, A, lda, &beta, C, ldc);
    #elif defined(DDLA_USE_CUDA)
    return cublasDsyrk(handle, uplo, trans_real, n, k,
        &alpha, A, lda, &beta, C, ldc);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

inline deblasStatus_t deblasHerk(
    deblasHandle_t handle,
    deblasFillMode_t uplo, deblasOperation_t trans,
    int n, int k,
    const float& alpha,
    const std::complex<float> *A, int lda,
    const float& beta,
    std::complex<float> *C, int ldc
)
{
    #if defined(DDLA_USE_HIP)
    return hipblasCherk(
        handle, uplo, trans, n, k,
        &alpha, (hipblasComplex*)A, lda,
        &beta, (hipblasComplex*)C, ldc
    );
    #elif defined(DDLA_USE_CUDA)
    return cublasCherk(
        handle, uplo, trans, n, k, 
        &alpha, (cuFloatComplex*)A, lda,
        &beta, (cuFloatComplex*)C, ldc
    );
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
}

}

#endif // HERK_H