#pragma once
//=================hbchen 2025-05-11=========================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex>
//=================hbchen 2025-05-11=========================
#include <iostream>
#include <device_launch_parameters.h>
#include <omp.h>
#include <mpi.h>
#ifdef ENABLE_NVHPC
#include <cusolverMp.h>
#include <cublasmp.h>
#include "helpers.h"
#endif

class GpuDeviceStream{
private:
    cal_comm_create_params_t params;
    static inline calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request)
    {
        MPI_Request req;
        int         err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, (MPI_Comm)(data), &req);
        if (err != MPI_SUCCESS)
        {
            return CAL_ERROR;
        }
        *request = (void*)(req);
        return CAL_OK;
    }

    static inline calError_t request_test(void* request)
    {
        MPI_Request req = (MPI_Request)(request);
        int         completed;
        int         err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS)
        {
            return CAL_ERROR;
        }
        return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
    }

    static inline calError_t request_free(void* request)
    {
        return CAL_OK;
    }
    static inline int getLocalDevice()
    {
        int localRank;
        MPI_Comm localComm;

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        return localRank % deviceCount;
    }


public:  
    int rank;
    int nranks;
    int local_device;
    cudaStream_t stream = nullptr;
    cal_comm_t cal_comm = nullptr;
 
    cusolverMpHandle_t cusolver_handle = nullptr;
    cublasMpHandle_t cublas_handle = nullptr;
    GpuDeviceStream(){
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        local_device = GpuDeviceStream::getLocalDevice();
        // printf("myrank:%d, local_device:%d\n", rank, local_device);
        CUDA_CHECK(cudaSetDevice(local_device));
        CUDA_CHECK(cudaFree(nullptr));
        {
            params.allgather    = GpuDeviceStream::allgather;
            params.req_test     = GpuDeviceStream::request_test;
            params.req_free     = GpuDeviceStream::request_free;
            params.data         = (void*)(MPI_COMM_WORLD);
            params.rank         = rank;
            params.nranks       = nranks;
            params.local_device = local_device;

            CAL_CHECK(cal_comm_create(params, &cal_comm));
        }
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUSOLVERMP_CHECK(cusolverMpCreate(&cusolver_handle, local_device, stream));
        CUBLASMP_CHECK(cublasMpCreate(&cublas_handle, stream));

    }
    ~GpuDeviceStream(){
        if(stream!=nullptr){
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream=nullptr;
        }
        if(cal_comm!=nullptr){
            CAL_CHECK(cal_comm_destroy(cal_comm));
            cal_comm=nullptr;
        }
        if(cusolver_handle!=nullptr){
            CUSOLVERMP_CHECK(cusolverMpDestroy(cusolver_handle));
            cusolver_handle=nullptr;
        }
        if(cublas_handle!=nullptr){
            CUBLASMP_CHECK(cublasMpDestroy(cublas_handle));
            cublas_handle=nullptr;
        }
    }
    void cudaSync() const {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    void calSync() const {
        CAL_CHECK(cal_stream_sync(cal_comm,stream));
    }
};