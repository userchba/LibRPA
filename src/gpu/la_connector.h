#ifndef LA_CONNECTOR_H
#define LA_CONNECTOR_H

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include "device_connector.h"
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/scal.h>
#include <ddla/axpy.h>
#endif
#include "../math/scalapack_connector.h"
#include "../math/matrix_m.h"
#include "../mpi/base_blacs.h"
#ifdef LIBRPA_USE_ELPA
#include "../elpa/elpa_connector.h"
#endif

namespace librpa_int
{

namespace LaConnector
{

template <typename T>
inline void pgemm(
    const char& transa, const char& transb,
    const int & m, const int & n,const int & k,
    const T & alpha,
    const T* A, const int64_t& ia, const int64_t& ja, const ArrayDesc& array_descA,
    const T* B, const int64_t& ib, const int64_t& jb, const ArrayDesc& array_descB,
    const T & beta,
    T* C,const int64_t& ic,const int64_t& jc,const ArrayDesc& array_descC
)
{
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)A)){
        ddla::pgemm(
            transa, transb,
            m, n, k,
            alpha,
            A, array_descA.ddla_desc(),
            B, array_descB.ddla_desc(),
            beta,
            C, array_descC.ddla_desc()
        );
    }
    else
    #endif
    ScalapackConnector::pgemm_f(
        transa, transb,
        m, n, k,
        alpha,
        A, ia, ja, array_descA.desc,
        B, ib, jb, array_descB.desc,
        beta,
        C, ic, jc, array_descC.desc
    );

}

template <typename T1, typename T2>
inline void scal(
    const int& N,
    const T1& alpha,
    T2* X,
    const int& incX,
    const ArrayDesc &array_desc
){
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)X)){
        // ddla::scal requires alpha and x to share type T, and the new LibDDLA
        // exposes no public accessor for the raw BLAS handle (the mixed-type
        // deblasScal overloads used to be reached through
        // array_desc.ddla_desc().ddla_handle()->blasH, which is no longer
        // valid now that DdlaStream is opaque). Promote a real alpha to T2
        // (e.g. the ELPA sqrt-Coulomb eigenvalue scaling of a complex buffer
        // by a real factor); this is numerically identical to zdscal/csscal.
        ddla::scal(array_desc.ddla_desc().ddla_handle(), N, static_cast<T2>(alpha), X, incX);
    }else
    #endif
    {
        LapackConnector::scal(N, alpha, X, incX);
    }
}


template <typename T1, typename T2>
inline void pdam(const T1& num, T2* A, const ArrayDesc& array_desc, int n = -1)
{
    if(array_desc.m() != array_desc.n()){
        throw std::runtime_error("In LaConnector::pdam, only square matrix is supported!");
    }
    // n < 0 means the whole (descriptor-sized) matrix, matching ddla::pdam's
    // own default; n >= 0 addresses only the leading n x n logical
    // sub-matrix, e.g. epsilon.cpp's n_nonsingular-order eigenbasis solve,
    // where the descriptor may be larger than the logical matrix.
    const int n_eff = (n < 0) ? array_desc.m() : n;
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)A)){
        ddla::pdam(num, A, array_desc.ddla_desc(), n);
    }else
    #endif
    {
        #pragma omp parallel for
        for (int i = 0; i != n_eff; i++)
        {
            const int ilo = array_desc.indx_g2l_r(i);
            if (ilo < 0) continue;
            const int jlo = array_desc.indx_g2l_c(i);
            if (jlo < 0) continue;
            A[ilo + array_desc.lld() * jlo] += num;
        }
    }
}

template <typename T>
inline void axpy(
    const int& N,
    const T& alpha,
    const T* X, const int& incX,
    T* Y, const int& incY,
    const BlacsCtxtHandler &blacs_h
){
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)X)){
        ddla::axpy(
            blacs_h.ddla_handle,
            N,
            alpha,
            X, incX,
            Y, incY
        );
    }else
    #endif
    {
        LapackConnector::axpy(
            N,
            alpha,
            X, incX,
            Y, incY
        );
    }
}

template <typename T>
inline void pgetrf_bpiv(
    const int& m, const int& n,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    int* ipiv, // host or device
    int& info // host
)
{
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(ia!=1 || ja!=1){
        throw std::runtime_error("In LaConnector::pgetrf_bpiv, only support ia=ja=1 for device implementation!");
    }
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        ddla::pgetrf_bpiv(m, n, d_A, array_descA.ddla_desc(), ipiv, info);
    }else
    #endif
    {
        ScalapackConnector::pgetrf_f(m, n, d_A, ia, ja, array_descA.desc, ipiv, info);
    }
}

template <typename T>
inline void pgesv(
    const int& n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const ArrayDesc& array_descB,
    int& info
)
{
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(ia!=1 || ja!=1 || ib!=1 || jb!=1){
        throw std::runtime_error("In LaConnector::pgesv, only support ia=ja=ib=jb=1 for device implementation!");
    }
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        // ddla::pgesv gained leading (side, trans) parameters; LaConnector's
        // own pgesv is always a plain left-hand solve of A*X=B.
        ddla::pgesv(
            'L', 'N',
            n, nrhs,
            d_A, array_descA.ddla_desc(),
            d_B, array_descB.ddla_desc()
        );
    }else
    #endif
    {
        std::vector<int> ipiv(array_descA.m_loc() + array_descA.mb());
        ScalapackConnector::pgesv_f(
            n, nrhs,
            d_A, ia, ja, array_descA.desc,
            ipiv.data(),
            d_B, ib, jb, array_descB.desc,
            info
        );
        if (info != 0){
            printf("Error in ScalapackConnector::pgesv_f, info = %d\n", info);
            throw std::runtime_error("info !=0\n");
        }
    }
}

template <typename T>
inline void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const ArrayDesc& array_descB,
    int& info, // host
    bool is_head = false, int location = -1
)
{
    assert(side == 'L');
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(ia!=1 || ja!=1 || ib!=1 || jb!=1){
        throw std::runtime_error("In LaConnector::pposv, only support ia=ja=ib=jb=1 for device implementation!");
    }
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        ddla::pposv(
            side, uplo, trans,
            n, nrhs,
            d_A, 1, 1, array_descA.ddla_desc(),
            d_B, 1, 1, array_descB.ddla_desc(),
            info,
            is_head, location
        );
    }else
    #endif
    {
        // ScaLAPACK's PZPOSV has no head-aware equivalent: it has no notion
        // of a tolerated indefinite pivot, so a head-corrected epsilon that
        // is not globally positive definite simply fails here with info>0.
        // Callers that need is_head on the host must fall back to pgesv (LU)
        // themselves; this path never attempts the head correction.
        if (is_head)
        {
            printf("Warning: LaConnector::pposv is_head=true requested on "
                   "the CPU/ScaLAPACK path, which has no head-aware "
                   "Cholesky; falling back to LU via LaConnector::pgesv.\n");
            pgesv(n, nrhs, d_A, ia, ja, array_descA, d_B, ib, jb, array_descB, info);
            return;
        }
        assert(trans == 'N' && side == 'L');
        ScalapackConnector::pposv_f(
            uplo, n, nrhs,
            d_A, ia, ja, array_descA.desc,
            d_B, ib, jb, array_descB.desc,
            info
        );
        if (info != 0){
            printf("the matrix is not positive definite info:%d\n", info);
            throw std::runtime_error("info !=0\n");
        }
    }
}

template <typename T>
inline matrix_m<std::complex<T>> power_hemat_la(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, 
    matrix_m<std::complex<T>> &Z_local, const ArrayDesc &ad_Z,
    size_t &n_filtered, T *W, T power, const T &threshold = -1.e5,
    bool use_gpu_replace_scalapack = false, bool use_elpa_sqrt_coulomb = false, std::complex<T>* d_A = nullptr, 
    std::complex<T>* d_Z = nullptr, std::complex<T>* d_power = nullptr)
{
    #if defined(LIBRPA_USE_ELPA)
    if(use_elpa_sqrt_coulomb){
        return ElpaConnector::power_hemat_elpa(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold,
            use_gpu_replace_scalapack, d_A, d_Z, d_power);
    }else
    #endif
    {
        return power_hemat_blacs(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold);
    }
}

template <typename T>
inline matrix_m<std::complex<T>> power_hemat_la_real(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, 
    matrix_m<std::complex<T>> &Z_local, const ArrayDesc &ad_Z,
    size_t &n_filtered, T *W, T power, const T &threshold = -1.e5,
    bool use_gpu_replace_scalapack = false, bool use_elpa_sqrt_coulomb = false,
    T* d_A = nullptr, T* d_Z = nullptr, T* d_power = nullptr)
{
    #if defined(LIBRPA_USE_ELPA)
    if(use_elpa_sqrt_coulomb){
        return ElpaConnector::power_hemat_elpa_real(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold,
            use_gpu_replace_scalapack, d_A, d_Z, d_power);
    }else
    #endif
    {
        return power_hemat_blacs_real(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold);
    }
}

}  // namespace LaConnector

}  // namespace librpa_int

#endif // LA_CONNECTOR_H
