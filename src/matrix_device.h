#ifndef MATRIX_DEVICE_H
#define MATRIX_DEVICE_H

#include<iostream>
#ifdef ENABLE_NVHPC
#include <cuda_runtime.h>
#include "helpers.h"
#endif

template <typename T>
class MatrixDevice{
private:
    T* d_data=nullptr;
    int m=0;
    int n=0;
public:
    MatrixDevice(){}
    MatrixDevice(const int& m, const int& n, const void* stream){
        this->m=m;
        this->n=n;
        if(stream==nullptr){
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaMalloc((void**)&d_data, m * n * sizeof(T)));
            #endif
        }else{
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaMallocAsync((void**)&d_data, m * n * sizeof(T), (cudaStream_t)stream));
            #endif
        }
    }
    MatrixDevice(const int& m, const int& n,const void* c_data, const void* stream){
        this->m=m;
        this->n=n;
        if(stream==nullptr){
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaMalloc((void**)&d_data, m * n * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(d_data, c_data, m * n * sizeof(T), cudaMemcpyHostToDevice));
            #endif
        }else{
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaMallocAsync((void**)&d_data, m * n * sizeof(T), (cudaStream_t)stream));
            CUDA_CHECK(cudaMemcpyAsync(d_data, c_data, m * n * sizeof(T), cudaMemcpyHostToDevice, (cudaStream_t)stream));
            #endif
        }
    }
    void set_data(const int& m, const int& n, const void* stream){
        if(m!=this->m || n!=this->n){
            clean(stream);
            this->m=m;
            this->n=n;
            if(stream==nullptr){
                #ifdef ENABLE_NVHPC
                CUDA_CHECK(cudaMalloc((void**)&d_data, m * n * sizeof(T)));
                #endif
            }else{
                #ifdef ENABLE_NVHPC
                CUDA_CHECK(cudaMallocAsync((void**)&d_data, m * n * sizeof(T), (cudaStream_t)stream));
                #endif
            }
        }
    }
    void set_data(const int& m, const int& n,const void* c_data, const void* stream){
        set_data(m,n,stream);
        if(stream==nullptr){
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaMemcpy(d_data, c_data, m * n * sizeof(T), cudaMemcpyHostToDevice));
            #endif
        }else{
            #ifdef ENABLE_NVHPC
            CUDA_CHECK(cudaMemcpyAsync(d_data, c_data, m * n * sizeof(T), cudaMemcpyHostToDevice, (cudaStream_t)stream));
            #endif
        }
    }
    void set_data_device(const int& m, const int& n, const void* d_A, const void* stream);
    
    
    void clean(const void* stream){
        if(d_data!=nullptr){
            if(stream==nullptr){
                #ifdef ENABLE_NVHPC
                cudaFree(d_data);
                #endif
            }else{
                #ifdef ENABLE_NVHPC
                cudaFreeAsync(d_data, (cudaStream_t)stream);
                #endif
            }
            d_data=nullptr;
        }
        this->m=0;
        this->n=0;
    }
    T* ptr(){
        return d_data;
    }
    const T* ptr() const {
        return d_data;
    }
    int nr() const {
        return m;
    }
    int nc() const {
        return n;
    }
    ~MatrixDevice(){
        clean(nullptr);
    }

};

#endif // MATRIX_DEVICE_H