#include "device_stream.h"
#include "envs_mpi.h"
using LIBRPA::envs::mpi_comm_global_h;
void DeviceStream::init(){
    local_device = DeviceStream::getLocalDevice();
    #ifdef ENABLE_NVHPC
    CUDA_CHECK(cudaSetDevice(local_device));
    CUDA_CHECK(cudaFree(nullptr));
    {
        params.allgather    = DeviceStream::allgather;
        params.req_test     = DeviceStream::request_test;
        params.req_free     = DeviceStream::request_free;
        params.data         = (void*)(MPI_COMM_WORLD);
        params.rank         = mpi_comm_global_h.myid;
        params.nranks       = mpi_comm_global_h.nprocs;
        params.local_device = local_device;

        CAL_CHECK(cal_comm_create(params, &cal_comm));
    }
    CUDA_CHECK(cudaStreamCreate((cudaStream_t*)&stream));
    CUSOLVERMP_CHECK(cusolverMpCreate(&cusolverMp_handle, local_device, (cudaStream_t)stream));
    CUBLASMP_CHECK(cublasMpCreate(&cublasMp_handle, (cudaStream_t)stream));
    #endif
}
void DeviceStream::check_memory(){
    #ifdef ENABLE_NVHPC
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));  // 直接查询驱动
    printf("rank:%d, Used: %f GiB / %f GiB\n",mpi_comm_global_h.myid, (total - free) / (1024.0*1024.0*1024.0), total / (1024.0*1024.0*1024.0));
    #endif
}


DeviceStream device_stream = DeviceStream();