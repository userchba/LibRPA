
#ifndef HELPERS_H
#define HELPERS_H

#pragma once
#include <mpi.h>
#include <iostream>
#ifdef ENABLE_NVHPC
#include <cal.h>
#include <cublasmp.h>
#include <cusolverMp.h>
#endif


typedef enum{
    LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE,
    LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT,
    LIBRPA_COMPUTE_TYPE_DOUBLE,
    LIBRPA_COMPUTE_TYPE_FLOAT
}LIBRPA_DEVICE_COMPUTE_TYPE;


#ifdef ENABLE_NVHPC
#define NVHPC_MPI_CHECK(call)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != MPI_SUCCESS)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "MPI error at %s:%d : %d\n", __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#define NVSHMEM_CHECK(call)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            fprintf(stderr, "NVSHMEM error at %s:%d : %d\n", __FILE__, __LINE__, status);                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)


#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(status));             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUBLASMP_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            fprintf(stderr, "cuBLASMp error at %s:%d : %d\n", __FILE__, __LINE__, status);                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#define CUBLAS_CHECK(err)                                                                                              \
    do {                                                                                                               \
        cublasStatus_t err_ = (err);                                                                                   \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                                           \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#define CAL_CHECK(call)                                                                                                 \
    do                                                                                                                  \
    {                                                                                                                   \
        calError_t status = call;                                                                                       \
        if (status != CAL_OK)                                                                                           \
        {                                                                                                               \
            fprintf(stderr, "CAL error at %s:%d : %d\n", __FILE__, __LINE__, status);                                   \
            exit(EXIT_FAILURE);                                                                                         \
        }                                                                                                               \
    }while(0)

#define CUSOLVERMP_CHECK(call)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        cusolverStatus_t status = call;                                                                                \
        if (status != CUSOLVER_STATUS_SUCCESS)                                                                         \
        {                                                                                                              \
            fprintf(stderr, "cuSOLVERMp error at %s:%d : %d\n", __FILE__, __LINE__, status);                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#endif
#define ORDER_CHECK(order)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if(order !='C'&&order !='c'&&order !='R'&&order !='r')                                                         \
        {                                                                                                              \
            fprintf(stderr, "Order should be either 'C' or 'R', order error at %s:%d:%s\n", __FILE__, __LINE__, order);\
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }while (0)

#endif // HELPERS_H