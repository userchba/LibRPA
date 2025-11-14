#include "matrix_device.h"
#include "device_stream.h"

template <typename T>
void MatrixDevice<T>::set_data_device(const int& m, const int& n, const void* d_A, const void* stream){
    set_data(m,n,stream);
    if(stream==nullptr){
        #ifdef ENABLE_NVHPC
        CUDA_CHECK(cudaMemcpyPeer(this->d_data, device_stream.local_device, d_A, device_stream.local_device, m * n * sizeof(T)));
        #endif
    }else{
        #ifdef ENABLE_NVHPC
        CUDA_CHECK(cudaMemcpyPeerAsync(this->d_data, device_stream.local_device, d_A, device_stream.local_device, m * n * sizeof(T), (cudaStream_t)stream));
        #endif
    }
}