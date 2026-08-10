#ifndef GETRF_H
#define GETRF_H

#include "ddla_connector.h"
#include <stdexcept>

namespace ddla{

inline desolverStatus_t desolverGetrf(
    desolverHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    double *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverDgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif
    
    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork * sizeof(double), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverDgetrf(handle, m, n, A, lda, Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

inline desolverStatus_t desolverGetrf(
    desolverHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    float *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverSgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork * sizeof(float), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverSgetrf(handle, m, n, A, lda, Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

inline desolverStatus_t desolverGetrf(
    desolverHandle_t handle,
    int m,
    int n,
    std::complex<double> *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    std::complex<double> *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnZgetrf_bufferSize(handle, m, n, (cuDoubleComplex*)A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverZgetrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif
    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork * sizeof(std::complex<double>), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnZgetrf(handle, m, n, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)Workspace, devIpiv, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverZgetrf(handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif
    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

inline desolverStatus_t desolverGetrf(
    desolverHandle_t handle,
    int m,
    int n,
    std::complex<float> *A,
    int lda,
    int *devIpiv,
    int *devInfo
)
{
    std::complex<float> *Workspace = nullptr;
    int Lwork;

    runtimeStream_t stream;
    SOLVER_CHECK(desolverGetStream(handle, &stream));

    #if defined(DDLA_USE_CUDA)
    SOLVER_CHECK(cusolverDnCgetrf_bufferSize(handle, m, n, (cuFloatComplex*)A, lda, &Lwork));
    #elif defined(DDLA_USE_HIP)
    SOLVER_CHECK(hipsolverCgetrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, &Lwork));
    #else
    throw std::runtime_error("not ENABLE CUDA and ENABLE HIP\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeMallocAsync(&Workspace, Lwork * sizeof(std::complex<float>), stream));

    desolverStatus_t status{};
    #if defined(DDLA_USE_CUDA)
    status = cusolverDnCgetrf(handle, m, n, (cuFloatComplex*)A, lda, (cuFloatComplex*)Workspace, devIpiv, devInfo);
    #elif defined(DDLA_USE_HIP)
    status = hipsolverCgetrf(handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)Workspace, Lwork, devIpiv, devInfo);
    #else
    throw std::runtime_error("ENABLE CUDA or ENABLE HIP not enable\n");
    #endif

    if(Lwork > 0)
        RUNTIME_CHECK(runtimeFreeAsync(Workspace, stream));
    return status;
}

}

#endif
