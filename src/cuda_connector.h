#ifndef CUDA_CONNECTOR_H
#define CUDA_CONNECTOR_H

#pragma once
//=================hbchen 2025-05-11=========================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex>
#include <magma_v2.h>
//=================hbchen 2025-05-11=========================
#include <iostream>
#include <device_launch_parameters.h>
#include <omp.h>
#include <mpi.h>
#include "base_blacs.h"
#include <fstream>
#ifdef ENABLE_NVHPC
#include <cusolverMp.h>
#include <cublasmp.h>
#include "helpers.h"
#include "scalapack_connector.h"
#include <curand.h>
#include <chrono>
#include "gpu_device_stream.h"
#endif
using LIBRPA::Array_Desc;
#include "array_desc_device.h"
#include "matrix_device.h"
#ifdef ENABLE_NVHPC


class ComplexMatrixDevice{
private:
    int m=0;
    int n=0;
public:
    cuDoubleComplex* d_data=nullptr;
    
    cusolverMpGrid_t grid_cusolver=nullptr;
    cusolverMpMatrixDescriptor_t desc_cusolver=nullptr;

    cublasMpGrid_t grid_cublas=nullptr;
    cublasMpMatrixDescriptor_t desc_cublas=nullptr;

    bool is_cusolver_init=false;
    bool is_cublas_init=false;

    ComplexMatrixDevice(){

    }
    ComplexMatrixDevice(const int& m, const int& n){
        ComplexMatrixDevice();
        this->m=m;
        this->n=n;
        CUDA_CHECK(cudaMalloc((void**)&d_data, m * n * sizeof(cuDoubleComplex)));
    }
    ComplexMatrixDevice(const int& m, const int& n,void* c_data){
        this->m=m;
        this->n=n;
        CUDA_CHECK(cudaMalloc((void**)&d_data, m * n * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_data, c_data, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    void set_data(const int& m, const int& n, const cudaStream_t& stream=nullptr){
        if(m!=this->m || n!=this->n){
            clean(stream);
            this->m=m;
            this->n=n;
            if(stream==nullptr){
                CUDA_CHECK(cudaMalloc((void**)&d_data, m * n * sizeof(cuDoubleComplex)));
            }else{
                CUDA_CHECK(cudaMallocAsync((void**)&d_data, m * n * sizeof(cuDoubleComplex),stream));
            }
        }
    }
    void set_data(const int& m, const int& n,const void* c_data, const cudaStream_t& stream=nullptr){
        set_data(m,n,stream);
        if(stream==nullptr){
            CUDA_CHECK(cudaMemcpy(d_data, c_data, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }else{
            CUDA_CHECK(cudaMemcpyAsync(d_data, c_data, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice,stream));
        }
    }
    void set_data_device(const int& m, const int& n,const void* d_A, const GpuDeviceStream& gpu_dev_stream)
    {
        if(m!=this->m || n!=this->n){
            set_data(m,n,gpu_dev_stream.stream);
        }
        
        CUDA_CHECK(cudaMemcpyPeerAsync(this->d_data,gpu_dev_stream.local_device, d_A, gpu_dev_stream.local_device, m * n * sizeof(cuDoubleComplex), gpu_dev_stream.stream));

    }
    void cusolver_init(cusolverMpHandle_t handle, cal_comm_t cal_comm, const Array_Desc& arrdesc){
        if(!is_cusolver_init){
            CUSOLVERMP_CHECK(cusolverMpCreateDeviceGrid(handle,&grid_cusolver,cal_comm,arrdesc.nprows(),arrdesc.npcols(),CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));
            CUSOLVERMP_CHECK(cusolverMpCreateMatrixDesc(&desc_cusolver,grid_cusolver,CUDA_C_64F,m,n,arrdesc.mb(),arrdesc.nb(),0,0,arrdesc.m_loc()));
            is_cusolver_init=true;
        }
    }
    void cublas_init(cublasMpHandle_t handle, cal_comm_t cal_comm, const Array_Desc& arrdesc){
        if(!is_cublas_init){
            CUBLASMP_CHECK(cublasMpGridCreate(handle,arrdesc.nprows(),arrdesc.npcols(),CUBLASMP_GRID_LAYOUT_ROW_MAJOR,cal_comm,&grid_cublas));
            CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(handle,m,n,arrdesc.mb(),arrdesc.nb(),0,0,arrdesc.m_loc(),CUDA_C_64F,grid_cublas,&desc_cublas));
            is_cublas_init=true;
        }
    }
    cuDoubleComplex* ptr(){
        return d_data;
    }
    const cuDoubleComplex* ptr() const {
        return d_data;
    }
    int nr(){
        return this->m;
    }
    int nc(){
        return this->n;
    }
    void set_nr(int m){
        this->m=m;
    }
    void set_nc(int n){
        this->n=n;
    }
    void set_ptr(cuDoubleComplex* d_data){
        if(this->d_data!=nullptr){
            cudaFree(this->d_data);
            this->d_data=nullptr;
        }
        this->d_data=d_data;
    }
    void clean(const cudaStream_t& stream=nullptr){
        if(d_data!=nullptr){
            if(stream==nullptr)
                cudaFree(d_data);
            else
                cudaFreeAsync(d_data,stream);
            d_data=nullptr;
        }
        this->m=0;
        this->n=0;
    }
    void set_as_zero(const cudaStream_t& stream=nullptr);
    void set_as_identity(const GpuDeviceStream& gpu_dev_stream, const Array_Desc_Device&);
    void cublasClean(cublasMpHandle_t handle){
        if(desc_cublas!=nullptr){
            CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,desc_cublas));
            desc_cublas=nullptr;
        }
        if(grid_cublas!=nullptr){
            CUBLASMP_CHECK(cublasMpGridDestroy(handle,grid_cublas));
            grid_cublas=nullptr;
        }
        is_cublas_init=false;
    }
    ~ComplexMatrixDevice(){
        if(is_cublas_init){
            fprintf(stderr, "Error: ComplexMatrixDevice not cleaned cublasMp resources before destructing!\n");
            std::abort();
        }
        // CUDA_CHECK(cudaDeviceSynchronize());
        if(d_data!=nullptr){
            CUDA_CHECK(cudaFree(d_data));
            d_data=nullptr;
        }
        if(desc_cusolver!=nullptr){
            CUSOLVERMP_CHECK(cusolverMpDestroyMatrixDesc(desc_cusolver));
            desc_cusolver=nullptr;
        }
        if(grid_cusolver!=nullptr){
            CUSOLVERMP_CHECK(cusolverMpDestroyGrid(grid_cusolver));
            grid_cusolver=nullptr;
        }
    }
    
};


#endif

class CudaConnector
{
public:
    static
    void write_file(double* A,int M,int N,const char* name){
        std::fstream out;
        out.open(name, std::ios::out);
        if (!out.is_open()) {
            std::cerr << "Failed to open file: " << name << std::endl;
            return;
        }
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                out<<A[i+j*M]<<" ";
            }
            out<<std::endl;
        }
        out.close();
    }
    static
    void write_file(cuDoubleComplex* A,int M,int N,const char* name){
        std::fstream out;
        std::ifstream file(name);
        if(file.good()){
            return;
        }
        out.open(name, std::ios::out);
        if (!out.is_open()) {
            std::cerr << "Failed to open file: " << name << std::endl;
            return;
        }
        
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                if(std::abs(A[i+j*M].y)<1e-10)
                    A[i+j*M].y=0;
                if(std::abs(A[i+j*M].x)<1e-10)
                    A[i+j*M].x=0;
                out<<A[i+j*M].x<<"+"<<A[i+j*M].y<<"i ";
            }
            out<<std::endl;
        }
        out.close();
    }
    static inline
    void check_memory(const GpuDeviceStream& gpu_dev_stream) {
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));  // 直接查询驱动
        printf("rank:%d, Used: %f GiB / %f GiB\n",gpu_dev_stream.rank, (total - free) / (1024.0*1024.0*1024.0), total / (1024.0*1024.0*1024.0));
    }
    static inline 
    cuDoubleComplex* transpose(const cublasHandle_t &handle,const cuDoubleComplex* d_a, const int n, const int lda,bool conjugate=false)
    {
        cuDoubleComplex* a_fort;
        cudaMalloc((void**)&a_fort, n * lda * sizeof(cuDoubleComplex)); // 分配内存
        cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
        // 调用 cublasZgeam 实现转置
        if (conjugate)
            cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_C,
                    n, lda, // 转置后的维度
                    &alpha,
                    d_a, lda, // 输入矩阵的列数作为 leading dimension
                    &beta,
                    d_a, lda,       // 第二个矩阵设为空
                    a_fort, n // 输出矩阵的 leading dimension
            );
        else
        cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    n, lda, // 转置后的维度
                    &alpha,
                    d_a, lda, // 输入矩阵的列数作为 leading dimension
                    &beta,
                    d_a, lda,       // 第二个矩阵设为空
                    a_fort, n // 输出矩阵的 leading dimension
        );
        return a_fort;
    }
    static inline
    void doubleComplexTocuDoubleComplex_host(const std::complex<double> *a, cuDoubleComplex *b, const int n, const int lda,bool is_transpose=false)
    {
        if (is_transpose)
        {
            #pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < lda; j++)
                {
                    b[i*lda+j].x = a[j*n+i].real();
                    b[i*lda+j].y = a[j*n+i].imag();
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int i = 0; i < n*lda; i++)
            {
                b[i].x = a[i].real();
                b[i].y = a[i].imag();
            }
        }
    }
    static inline
    void cuDoubleComplexToDoubleComplex_host(std::complex<double> *a,const cuDoubleComplex *b,const int n,const int lda,bool is_transpose=false)
    {
        if (is_transpose)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < lda; j++)
                {
                    a[j*n+i].real(b[i*lda+j].x);
                    a[j*n+i].imag(b[i*lda+j].y);
                }
            }
        }
        else
        {
            for (int i = 0; i < n*lda; i++)
            {
                a[i].real(b[i].x);
                a[i].imag(b[i].y);
            }
        }
    }
    // update by chenhaobo in 2025-05-19
    static inline
    void cuZgetrf_f(const int& m, const int& n, cuDoubleComplex *d_a, const int& lda, int *d_ipiv, int *d_info, cusolverDnHandle_t cusolverH)
    {
        cudaStream_t stream;
        cusolverStatus_t status = cusolverDnGetStream(cusolverH, &stream);
        // printf("cusolverDnGetStream status:%d\n",status);
        // printf("Stream stream:%d\n",stream);
        // printf("cusolverDnhandle:%d\n",cusolverH);
        int lwork;
        // auto start = std::chrono::high_resolution_clock::now();
        cusolverDnZgetrf_bufferSize(cusolverH, m, n, d_a, lda, &lwork);
        cuDoubleComplex *d_work;
        cudaMallocAsync((void**)&d_work, lwork * sizeof(cuDoubleComplex), stream);
        // cudaStreamSynchronize(stream);
        cusolverDnZgetrf(cusolverH, m, n, d_a, lda, d_work, d_ipiv, d_info);
        // cudaStreamSynchronize(stream);
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration= std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // int info;
        // cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("stream:%d, info:%d,Time taken of the cudaStreamSynchronize on GPU: %lld microseconds\n",stream,info, duration.count());
        cudaFreeAsync(d_work, stream);
    }
    static
    cuDoubleComplex det_cuDoubleComplex(const cuDoubleComplex *d_a,const int *d_ipiv, const int& n, const cudaStream_t stream);
    static
    void cuDoubleComplex_minus_identity_Async(cuDoubleComplex *d_a,const int& h_n, const cudaStream_t stream);
    static inline void cuDoubleComplex_minus_identity_host(cuDoubleComplex* h_a, const int& n)
    {
        #pragma omp parallel for
        for(int i=0;i<n;i++){
            h_a[i*n+i].x -= 1.0; // 减去单位矩阵的对角线元素
        }
    }
    static inline cuDoubleComplex det_cuZgetrf_f_from_host(const int& m, const int& n, cuDoubleComplex *h_a, const int& lda, int *ipiv, int *info, cusolverDnHandle_t cusolverH,const int &deviceId)
    {
        cudaSetDevice(deviceId);
        cudaStream_t stream;
        cusolverStatus_t status = cusolverDnGetStream(cusolverH, &stream);
        int lwork;
        
        int *d_ipiv;
        cudaMallocAsync((void**)&d_ipiv, n * sizeof(int), stream);
        int *d_info;
        cudaMallocAsync((void**)&d_info, sizeof(int), stream);
        cudaMemcpyAsync(d_ipiv, ipiv, n * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_info, info, sizeof(int), cudaMemcpyHostToDevice, stream);
        cuDoubleComplex *d_a;
        cudaMallocAsync((void**)&d_a, m * n * sizeof(cuDoubleComplex), stream);
        cusolverDnZgetrf_bufferSize(cusolverH, m, n, d_a, lda, &lwork);
        cuDoubleComplex *d_work;
        cudaMallocAsync((void**)&d_work, lwork * sizeof(cuDoubleComplex), stream);
        cudaMemcpyAsync(d_a, h_a, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
        CudaConnector::cuDoubleComplex_minus_identity_Async(d_a,n, stream);
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
            double start_time_zgetrf = omp_get_wtime();
        #endif
        cusolverDnZgetrf(cusolverH, m, n, d_a, lda, d_work, d_ipiv, d_info);
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
            printf("cusolverDnZgetrf time: %f seconds\n", omp_get_wtime() - start_time_zgetrf);
        #endif
        // Check for errors in the info variable
        cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cuDoubleComplex det = det_cuDoubleComplex(d_a, d_ipiv, n, stream);
        // cudaStreamSynchronize(stream);
        cudaMemcpyAsync(ipiv, d_ipiv, n * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(d_work, stream);
        cudaFreeAsync(d_ipiv, stream);
        cudaFreeAsync(d_info, stream);
        cudaFreeAsync(d_a, stream);
        cudaStreamSynchronize(stream);
        return det;
    }
    static inline cuDoubleComplex det_magmaZgetrf_f_mgpu_from_host(const int& m, const int& n, cuDoubleComplex *h_a, const int& lda, int *ipiv, int *info, const int& ngpu, magma_queue_t* queues)
    {
        cuDoubleComplex_minus_identity_host(h_a, m);
        int nb= magma_get_zgetrf_nb( m, n );
        // printf("nb=%d\n", nb);
        int ldda= magma_roundup( m, nb );  // multiple of 32 by default
        int ldn_local,n_local;
        // printf("ldda:%d\n", ldda);
        cuDoubleComplex **d_lA = new cuDoubleComplex*[ngpu];
        for(int dev=0; dev < ngpu; dev++ ) {
            n_local = ((n/nb)/ngpu)*nb;
            if (dev < (n/nb) % ngpu)
                n_local += nb;
            else if (dev == (n/nb) % ngpu)
                n_local += n % nb;
            ldn_local = magma_roundup( n_local, nb );  // multiple of 32 by default
            // printf("dev=%d, n_local=%d, ldn_local=%d\n", dev, n_local, ldn_local);
            magma_setdevice( dev );
            magma_zmalloc( &d_lA[dev], ldda*ldn_local );
        }
        // // show h_a
        // printf("h_a\n");
        // for(int i=0;i<m;i++){
        //     for(int j=0;j<n;j++){
        //         printf("(%f,%f) ", h_a[i*lda+j].x, h_a[i*lda+j].y);
        //     }
        //     printf("\n");
        // }
        magma_zsetmatrix_1D_col_bcyclic( ngpu, m, n, nb, h_a, lda, d_lA, ldda, queues );
        // // show d_lA
        // printf("d_la before LU composition\n");
        // for(int dev=0; dev < ngpu; dev++ ) {
        //     n_local = ((n/nb)/ngpu)*nb;
        //     if (dev < (n/nb) % ngpu)
        //         n_local += nb;
        //     else if (dev == (n/nb) % ngpu)
        //         n_local += n % nb;
        //     ldn_local = magma_roundup( n_local, nb );  // multiple of 32 by default
        //     magma_setdevice( dev );
        //     printf("Device %d d_lA:\n", dev);
        //     cuDoubleComplex *h_lA=new cuDoubleComplex[ldda*ldn_local];
        //     cudaMemcpy(h_lA, d_lA[dev], ldda*ldn_local*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        //     for(int i=0;i<ldn_local;i++){
        //         for(int j=0;j<ldda;j++){
        //             printf("(%f,%f) ", h_lA[i*ldda+j].x, h_lA[i*ldda+j].y);
        //         }
        //         printf("\n");
        //     }
        //     delete[] h_lA;
        // }
        magma_zgetrf_mgpu( ngpu, m, n, d_lA, ldda, ipiv, info );
        // if (*info != 0) {
        //     printf("magma_zgetrf_mgpu returned error %lld: %s.\n",
        //            (long long) *info, magma_strerror( *info ));
        // }
        // *info=1;
        // printf("ipiv after magma_zgetrf_mgpu:");
        // for(int i=0;i<m;i++){
        //     printf("%d ", ipiv[i]);
        //     ipiv[i]=0;
        // }
        // printf("\n");
        // // lapack test
        // lapackf77_zgetrf( &m, &n, h_a, &lda, ipiv, info );
        // if (*info != 0) {
        //     printf("magma_zgetrf_mgpu returned error %lld: %s.\n",
        //            (long long) *info, magma_strerror( *info ));
        // }
        // printf("ipiv after lapack zgetrf:");
        // for(int i=0;i<m;i++){
        //     printf("%d ", ipiv[i]);
        //     // ipiv[i]=0;
        // }
        // printf("\n");
        // // show h_a
        // printf("h_a after LU composition\n");
        // for(int i=0;i<m;i++){
        //     for(int j=0;j<n;j++){
        //         printf("(%f,%f) ", h_a[i*lda+j].x, h_a[i*lda+j].y);
        //         h_a[i*lda+j].x = 0.0; // reset h_a to zero
        //         h_a[i*lda+j].y = 0.0; // reset h_a to zero
        //     }
        //     printf("\n");
        // }
        // // lapack test over
        magma_zgetmatrix_1D_col_bcyclic( ngpu, m, n, nb, d_lA, ldda, h_a, lda, queues );
        for( int dev=0; dev < ngpu; dev++ ) {
            magma_setdevice( dev );
            magma_free( d_lA[dev] );
        }
        delete[] d_lA;
        // Calculate the determinant from the diagonal elements and the pivot indices
        cuDoubleComplex det = cuDoubleComplex{1.0, 0.0};
        // printf("det=%f+%fi\n", det.x, det.y);
        for(int i=0;i<m;i++){
            // printf("h_a[%d][%d]=(%f,%f)\n", i, i, h_a[i*lda+i].x, h_a[i*lda+i].y);
            det=cuCmul(det, h_a[i*lda+i]);
            if( ipiv[i] != i + 1 ) {
                det.x = -det.x; // Adjust sign for row swaps
                det.y = -det.y; // Adjust sign for row swaps
            }
        }
        return det;
    }
    #ifdef ENABLE_NVHPC
    // static
    // void pzgetrf_cusolverMp(const int &, const int &, std::complex<double> *, const int &,
    //                             const int &, const LIBRPA::Array_Desc &, int *, int &,const char order='C');
    static 
    void pzgetrf_nvhpc(const GpuDeviceStream&, ComplexMatrixDevice &, const int &,
                                const int &, const LIBRPA::Array_Desc &, int64_t *, int *,const char& order='C');
    static 
    void pgetrf_nvhpc_mixed_precision(
        const GpuDeviceStream&, void *, 
        const int &, const int &,
        const LIBRPA::Array_Desc &, int64_t *, int *,
        const cudaDataType_t &,const char &order='C'
    );
    static 
    void pgetrf_nvhpc_mixed_precision(
        void *, const int &, const int &,
        const LIBRPA::Array_Desc &, int64_t *, int *,
        const cudaDataType_t &,const char &order='C'
    );
    static
    void pgetrs_nvhpc_mixed_precision(
        const GpuDeviceStream&, const cublasOperation_t&,
        const void* d_A, const int64_t& IA, const int64_t& JA, const LIBRPA::Array_Desc &,
        const int64_t* d_ipiv,
        void* d_B, const int64_t& IB, const int64_t& JB, const LIBRPA::Array_Desc &,
        int* d_info,const cudaDataType_t& compute_type,
        const char& order='C'
    );
    static void pgetrf_trs_nvhpc_mixed_precision(
        const GpuDeviceStream&, const cublasOperation_t&,
        void* d_A, const int64_t& IA, const int64_t& JA, const LIBRPA::Array_Desc &,
        void* d_B, const int64_t& IB, const int64_t& JB, const LIBRPA::Array_Desc &,
        const cudaDataType_t& compute_type, const char& order='C'
    );
    static
    std::complex<double> det_ComplexMatrixDevice_blacs(const cudaStream_t&, const ComplexMatrixDevice&, const LIBRPA::Array_Desc &);
    static
    std::complex<double> trace_ComplexMatrixDevice_blacs(const cudaStream_t&, const ComplexMatrixDevice&, const LIBRPA::Array_Desc &);
    static
    void trace_matrix_device_blacs(void* trace, const void* d_A, const Array_Desc &arrdesc_A, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type);
    static
    void det_matrix_device_blacs(void* det, const void* d_A, const Array_Desc &arrdesc_A, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type);
    
    // static
    // void pgemm_cublasMp(const char &transa, const char &transb, const int &m, const int &n, const int &k,
    //                     const double &alphaD, const std::complex<double> *A, const int &ia, const int &ja, const LIBRPA::Array_Desc &arrdesc_A,
    //                     const std::complex<double> *B, const int &ib, const int &jb, const LIBRPA::Array_Desc &arrdesc_B,
    //                     const double &betaD, std::complex<double> *C, const int &ic, const int &jc, const LIBRPA::Array_Desc &arrdesc_C);
    
    static 
    void pgemm_device(cublasMpHandle_t,cublasOperation_t,cublasOperation_t,const int &,const int &,const int &,
                        const void *,
                        const ComplexMatrixDevice &,int64_t,int64_t,
                        const ComplexMatrixDevice &,int64_t,int64_t,
                        const void *,
                        ComplexMatrixDevice &,int64_t,int64_t,
                        cublasComputeType_t);
       
    static 
    void pgemm_nvhpc(const GpuDeviceStream&,cublasOperation_t,cublasOperation_t,const int &,const int &,const int &,
                        const void *,
                        const ComplexMatrixDevice &,int64_t,int64_t,const Array_Desc&,
                        const ComplexMatrixDevice &,int64_t,int64_t,const Array_Desc&,
                        const void *,
                        ComplexMatrixDevice &,int64_t,int64_t,const Array_Desc&,
                        cublasComputeType_t);
    // static
    // void pgemm_nvhpc_cuFloatComplex(const GpuDeviceStream& gpu_dev_stream,cublasOperation_t transA,cublasOperation_t transB,const int & m,const int & n,const int & k,
    //                     const void *alpha,
    //                     const cuFloatComplex* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
    //                     const cuFloatComplex* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
    //                     const void *beta,
    //                     cuFloatComplex * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
    //                     cublasComputeType_t cublas_compute_type);
    static
    void pgemm_nvhpc_mixed_precision(
        const GpuDeviceStream& gpu_dev_stream,cublasOperation_t transA,cublasOperation_t transB,
        const int & m,const int & n,const int & k,
        const void *alpha,
        const void* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
        const void* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
        const void *beta,
        void * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
        cublasComputeType_t cublas_compute_type
    );
    static
    void pgemm_nvhpc_mixed_precision(
        cublasOperation_t transA,cublasOperation_t transB,
        const int & m,const int & n,const int & k,
        const void *alpha,
        const void* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
        const void* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
        const void *beta,
        void * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
        cublasComputeType_t cublas_compute_type
    );
    static
    void pgeadd_nvhpc(
        const GpuDeviceStream& gpu_dev_stream,const cublasOperation_t& trans,
        const void *alpha,
        const void* d_A, const int64_t& ia, const int64_t& ja, const Array_Desc& array_descA,
        const void* beta,
        void* d_B, const int64_t& ib, const int64_t& jb, const Array_Desc& array_descB,
        const cudaDataType_t&,
        const char& order = 'c'
    );
    static
    void pgemr2d_nvhpc(
        const GpuDeviceStream& gpu_dev_stream,const int&,const int&,
        const void* d_A, const int64_t& ia, const int64_t& ja, const Array_Desc& array_descA,
        void* d_B, const int64_t& ib, const int64_t& jb, const Array_Desc& array_descB,
        const cudaDataType_t&
    );
    static
    void cuFloatComplex_to_cuDoubleComplex_Async(const cuFloatComplex* d_a, cuDoubleComplex* d_b, const int64_t& len, const cudaStream_t& stream);
    static 
    void cuDoubleComplex_to_cuFloatComplex_Async(const cuDoubleComplex* d_a, cuFloatComplex* d_b, const int64_t& len, const cudaStream_t& stream);
    static 
    void float_to_double_device(const float* d_a, double* d_b, const int64_t& len);
    static
    void double_to_float_device(const double* d_a, float* d_b, const int64_t& len);
    static
    void multiply_number_for_ComplexMatrixDevice(ComplexMatrixDevice& mat, const cuDoubleComplex& num, const cudaStream_t& stream);                    
    static
    void diag_add_ComplexMatrixDevice(ComplexMatrixDevice& mat, const double& num, const Array_Desc& arrdesc,const cudaStream_t& stream);
    static
    void diag_add_matrix_device_blacs(
        const void* num, void* d_A, const Array_Desc& array_desc,const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
    );
    static 
    void transpose_ComplexMatrixDevice(const GpuDeviceStream& gpu_dev_stream, ComplexMatrixDevice& in ,bool is_conjugate=false);

    #endif 
    
};

#endif // CUDA_CONNECTOR_H