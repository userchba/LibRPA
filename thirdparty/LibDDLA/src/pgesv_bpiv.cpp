#include <ddla/ddla.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla {

/**
 * @brief Distributed linear-system solver using block-partial-pivoting LU
 *        (driver): solve op(A) * X = B (side='L') or X * op(A) = B
 *        (side='R').
 *
 * Convenience wrapper: pgetrf_bpiv (block LU with partial pivoting within
 * each block row) + pgetrs_bpiv (solve).  Corresponds to the same
 * ScaLAPACK-style interface as pgesv, but with block-partial pivoting.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of square matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to A (input: coefficient; output: LU factors).
 * @param array_descA  DdlaDesc for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 * @throws std::runtime_error if LU factorization fails (info != 0).
 */
template <typename T>
void pgesv_bpiv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "pgesv_bpiv");
    int* d_ipiv = nullptr;
    RUNTIME_CHECK(runtimeMallocAsync(reinterpret_cast<void**>(&d_ipiv),
                                     sizeof(int) * std::max(1, array_descA.m_loc()),
                                     ddla_handle->stream));
    int info = 0;
    pgetrf_bpiv(n, n, d_A, array_descA, d_ipiv, info);
    if(info != 0){
        RUNTIME_CHECK(runtimeFreeAsync(d_ipiv, ddla_handle->stream));
        throw std::runtime_error("pgesv_bpiv: pgetrf_bpiv returned info = " + std::to_string(info));
    }
    pgetrs_bpiv(side, trans, n, nrhs, d_A, array_descA, d_ipiv, d_B, array_descB);
    RUNTIME_CHECK(runtimeFreeAsync(d_ipiv, ddla_handle->stream));
}

template void pgesv_bpiv<float>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);
template void pgesv_bpiv<double>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);
template void pgesv_bpiv<std::complex<float>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);
template void pgesv_bpiv<std::complex<double>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

} // namespace ddla
