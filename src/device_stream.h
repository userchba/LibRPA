#ifndef DEVICE_STREAM_H
#define DEVICE_STREAM_H

#include <iostream>
#ifdef ENABLE_NVHPC
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverMp.h>
#include <cublasmp.h>
#endif
#include "helpers.h"
#include <mpi.h>
class DeviceStream{
private:
    #ifdef ENABLE_NVHPC
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
    #endif
    static inline int getLocalDevice()
    {
        int localRank;
        MPI_Comm localComm;

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);

        int deviceCount = 0;
        #ifdef ENABLE_NVHPC
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        #endif

        return localRank % deviceCount;
    }
public:
    int local_device;
    #ifdef ENABLE_NVHPC
    cudaStream_t stream = nullptr;
    cal_comm_t cal_comm = nullptr;
    cusolverMpHandle_t cusolverMp_handle = nullptr;
    cublasMpHandle_t cublasMp_handle = nullptr;
    #endif
    DeviceStream(){}
    void check_memory();
    void init();
    void finalize(){
        if(stream!=nullptr){
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaStreamDestroy((cudaStream_t)stream));
            #endif
            stream=nullptr;
        }
        #ifdef ENABLE_NVHPC
        if(cusolverMp_handle!=nullptr){
            CUSOLVERMP_CHECK(cusolverMpDestroy(cusolverMp_handle));
            cusolverMp_handle=nullptr;
        }
        if(cublasMp_handle!=nullptr){
            CUBLASMP_CHECK(cublasMpDestroy(cublasMp_handle));
            cublasMp_handle=nullptr;
        }
        if(cal_comm!=nullptr){
            CAL_CHECK(cal_comm_destroy(cal_comm));
            cal_comm=nullptr;
        }
        #endif
    }
    ~DeviceStream(){
        if(stream!=nullptr){
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaStreamDestroy((cudaStream_t)stream));
            #endif
            stream=nullptr;
        }
        #ifdef ENABLE_NVHPC
        if(cusolverMp_handle!=nullptr){
            CUSOLVERMP_CHECK(cusolverMpDestroy(cusolverMp_handle));
            cusolverMp_handle=nullptr;
        }
        if(cublasMp_handle!=nullptr){
            CUBLASMP_CHECK(cublasMpDestroy(cublasMp_handle));
            cublasMp_handle=nullptr;
        }
        if(cal_comm!=nullptr){
            CAL_CHECK(cal_comm_destroy(cal_comm));
            cal_comm=nullptr;
        }
        #endif
    }
    void sync() const{
        #ifdef ENABLE_NVHPC
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        #endif
    }
    #ifdef ENABLE_NVHPC
    void cudaSync() const{
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    }
    void calSync() const {
        CAL_CHECK(cal_stream_sync(cal_comm,(cudaStream_t)stream));
    }
    #endif
};

extern DeviceStream device_stream;
#endif // DEVICE_STREAM_H