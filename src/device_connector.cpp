#include "device_connector.h"
#ifdef ENABLE_NVHPC
#include "cuda_connector.h"
#endif
#include "device_stream.h"

void DeviceConnector::pgetrf_device_mixed_precision(
    void* d_A, const int& m, const int& n,
    const LIBRPA::Array_Desc& arrdesc_pi,
    int64_t* d_ipiv, int* d_info,
    const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type,
    const char& order
){
    {
        #ifdef ENABLE_NVHPC
        cudaDataType_t cuda_compute_type;
        if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE){
            cuda_compute_type = CUDA_C_64F;
        }else if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT){
            cuda_compute_type = CUDA_C_32F;
        }else if(compute_type==LIBRPA_COMPUTE_TYPE_DOUBLE){
            cuda_compute_type = CUDA_R_64F;
        }else if(compute_type==LIBRPA_COMPUTE_TYPE_FLOAT){
            cuda_compute_type = CUDA_R_32F;
        }else{
            fprintf(stderr, "Error: Unsupported compute type in pgetrf_device_mixed_precision\n");
            exit(1);
        }
        CudaConnector::pgetrf_nvhpc_mixed_precision(
            d_A, m, n,
            arrdesc_pi,
            d_ipiv, d_info,
            cuda_compute_type,
            order
        );
        #else
        fprintf(stderr, "Error: NVHPC is not enabled in this build, cannot call pgetrf_device_mixed_precision\n");
        exit(1);
        #endif
    }
}

void DeviceConnector::pgemm_device_mixed_precision(
    const char& transa, const char& transb,
    const int & m,const int & n,const int & k,
    const void *alpha,
    const void* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
    const void* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
    const void *beta,
    void * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
    const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
){
    #ifdef ENABLE_NVHPC
    cublasComputeType_t cublas_compute_type;
    cublasOperation_t transA, transB;
    if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE){
        cublas_compute_type = CUBLAS_COMPUTE_64F_PEDANTIC;
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT){
        cublas_compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_DOUBLE){
        cublas_compute_type = CUBLAS_COMPUTE_64F;
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_FLOAT){
        cublas_compute_type = CUBLAS_COMPUTE_32F;
    }else{
        fprintf(stderr, "Error: Unsupported compute type in pgemm_device_mixed_precision\n");
        exit(1);
    }
    if(transa=='N'||transa=='n'){
        transA = CUBLAS_OP_N;
    }else if(transa=='T'||transa=='t'){
        transA = CUBLAS_OP_T;
    }else if(transa=='C'||transa=='c'){
        transA = CUBLAS_OP_C;
    }else{
        fprintf(stderr, "Error: Unsupported transa in pgemm_device_mixed_precision\n");
        exit(1);
    }
    if(transb=='N'||transb=='n'){
        transB = CUBLAS_OP_N;
    }else if(transb=='T'||transb=='t'){
        transB = CUBLAS_OP_T;
    }else if(transb=='C'||transb=='c'){
        transB = CUBLAS_OP_C;
    }else{
        fprintf(stderr, "Error: Unsupported transb in pgemm_device_mixed_precision\n");
        exit(1);
    }
    CudaConnector::pgemm_nvhpc_mixed_precision(
        transA, transB,
        m, n, k,
        alpha,
        d_A, ia, ja, array_descA,
        d_B, ib, jb, array_descB,
        beta,
        d_C, ic, jc, array_descC,
        cublas_compute_type
    );
    #endif
}

void DeviceConnector::transpose_device_blas(
    const void* d_A, 
    const int& m, const int& n,
    void* d_B,
    const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type,
    const bool& is_conjugate
){
    #ifdef ENABLE_NVHPC
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, (cudaStream_t)device_stream.stream));
    cublasOperation_t trans = CUBLAS_OP_T;
    if(is_conjugate)
        trans = CUBLAS_OP_C;
    if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE){
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        CUBLAS_CHECK(cublasZgeam(
            cublasH, trans, CUBLAS_OP_N,
            m, n, 
            &alpha,
            (cuDoubleComplex*)d_A, n,
            &beta,
            (cuDoubleComplex*)d_B, m,
            (cuDoubleComplex*)d_B, m));
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT){
        cuFloatComplex alpha = make_cuFloatComplex(1.0, 0.0);
        cuFloatComplex beta = make_cuFloatComplex(0.0, 0.0);
        CUBLAS_CHECK(cublasCgeam(
            cublasH, trans, CUBLAS_OP_N,
            m, n, 
            &alpha,
            (cuFloatComplex*)d_A, n,
            &beta,
            (cuFloatComplex*)d_B, m,
            (cuFloatComplex*)d_B, m));
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_DOUBLE){
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgeam(
            cublasH, trans, CUBLAS_OP_N,
            m, n, 
            &alpha,
            (double*)d_A, n,
            &beta,
            (double*)d_B, m,
            (double*)d_B, m));
        
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_FLOAT){
        float alpha = 1.0;
        float beta = 0.0;
        CUBLAS_CHECK(cublasSgeam(
            cublasH, trans, CUBLAS_OP_N,
            m, n, 
            &alpha,
            (float*)d_A, n,
            &beta,
            (float*)d_B, m,
            (float*)d_B, m));
    }else{
        fprintf(stderr, "Error: Unsupported compute type in transpose_device_blas\n");
        exit(1);
    }
    CUBLAS_CHECK(cublasDestroy(cublasH));
    #endif
}

void DeviceConnector::num_multiply_matrix_device(
    const int& n, const void* num, void* d_A, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
){
    #ifdef ENABLE_NVHPC
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, (cudaStream_t)device_stream.stream));
    if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE){
        CUBLAS_CHECK(cublasZscal(
            cublasH, n,
            (cuDoubleComplex*)num,
            (cuDoubleComplex*)d_A, 1
        ));
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT){
        CUBLAS_CHECK(cublasCscal(
            cublasH, n,
            (cuFloatComplex*)num,
            (cuFloatComplex*)d_A, 1
        )); 
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_DOUBLE){
        CUBLAS_CHECK(cublasDscal(
            cublasH, n,
            (double*)num,
            (double*)d_A, 1
        ));
    }else if(compute_type==LIBRPA_COMPUTE_TYPE_FLOAT){
        CUBLAS_CHECK(cublasSscal(
            cublasH, n,
            (float*)num,
            (float*)d_A, 1
        ));
    }else{
        fprintf(stderr, "Error: Unsupported compute type in num_multiply_matrix_device\n");
        exit(1);
    }
    #endif
}

void DeviceConnector::diag_add_matrix_device_blacs(
    const void* num, void* d_A, const Array_Desc& array_desc, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
){
    #ifdef ENABLE_NVHPC
    CudaConnector::diag_add_matrix_device_blacs(
        num, d_A, array_desc, compute_type
    );
    #endif
}

void DeviceConnector::trace_matrix_device_blacs(
    void* trace, const void* d_A, const Array_Desc& array_desc, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
){
    #ifdef ENABLE_NVHPC
    CudaConnector::trace_matrix_device_blacs(
        trace, d_A, array_desc, compute_type
    );
    #endif
}

void DeviceConnector::det_matrix_device_blacs(
    void* det, const void* d_A, const Array_Desc& array_desc, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
){
    #ifdef ENABLE_NVHPC
    CudaConnector::det_matrix_device_blacs(
        det, d_A, array_desc, compute_type
    );
    #endif
}

void DeviceConnector::float_to_double_device(float* d_A, double* d_B, const int64_t& n)
{
    #ifdef ENABLE_NVHPC
    CudaConnector::float_to_double_device(d_A, d_B, n);
    #endif
}

void DeviceConnector::double_to_float_device(double* d_A, float* d_B, const int64_t& n)
{
    #ifdef ENABLE_NVHPC
    CudaConnector::double_to_float_device(d_A, d_B, n);
    #endif
}
