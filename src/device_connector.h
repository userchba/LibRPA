#ifndef DEVICE_CONNECTOR_H
#define DEVICE_CONNECTOR_H
#include "base_blacs.h"
#include  "helpers.h"
using LIBRPA::Array_Desc;

class DeviceConnector{
public:
    static void pgetrf_device_mixed_precision(
        void* d_A, const int& m, const int& n,
        const LIBRPA::Array_Desc& arrdesc_pi,
        int64_t* d_ipiv, int* d_info,
        const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type,
        const char& order='C'
    );
    static void pgemm_device_mixed_precision(
        const char& transa, const char& transb,
        const int & m,const int & n,const int & k,
        const void *alpha,
        const void* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
        const void* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
        const void *beta,
        void * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
        const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
    );
    static void transpose_device_blas(
        const void* d_A, 
        const int& m, const int& n,
        void* d_B,
        const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type,
        const bool& is_conjugate=false
    );
    static void num_multiply_matrix_device(
        const int& n, const void* num, void* d_A, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
    );
    static void diag_add_matrix_device_blacs(
        const void* num, void* d_A, const Array_Desc& array_desc, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
    );
    
    static void trace_matrix_device_blacs(
        void* trace, const void* d_A, const Array_Desc& array_desc, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
    );
    static void det_matrix_device_blacs(
        void* det, const void* d_A, const Array_Desc& array_desc, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
    );
    static void float_to_double_device(float* d_A, double* d_B, const int64_t& n);
    static void double_to_float_device(double* d_A, float* d_B, const int64_t& n);

    
};



#endif // DEVICE_CONNECTOR_H