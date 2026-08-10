#include <ddla/ddla.h>
#include <cassert>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla {

/**
 * @brief Distributed LU solve without pivoting.
 *
 * Solves A * X = B using the factors from pgetrf_nopiv.  Because no
 * pivoting was performed, the solve is simply two triangular solves:
 *   1) L * Y = B  (forward, lower triangular, unit diagonal)
 *   2) U * X = Y  (backward, upper triangular, non-unit diagonal)
 *
 * Only trans='N' is supported.
 *
 * @tparam T   Scalar type.
 * @param trans   'N' -- no transpose (only 'N' supported).
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf_nopiv).
 * @param array_descA  DdlaDesc for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 */
template<typename T>
void pgetrs_nopiv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "pgetrs_nopiv");
    assert(side == 'L' || side == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;

    if(side == 'L'){
        if(trans == 'N'){
            // A = L*U => X = U^-1 * L^-1 * B: solve L then U.
            ptrtrs('L', 'L', 'N', 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs('L', 'U', 'N', 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }else{
            // A^T = U^T * L^T => X = L^-T * U^-T * B: solve U^T then L^T.
            ptrtrs('L', 'U', trans, 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs('L', 'L', trans, 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }
    }else{
        if(trans == 'N'){
            // X * L * U = B => X = B * U^-1 * L^-1: solve U then L.
            ptrtrs('R', 'U', 'N', 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs('R', 'L', 'N', 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }else{
            // X * U^T * L^T = B => X = B * L^-T * U^-T: solve L^T then U^T.
            ptrtrs('R', 'L', trans, 'U', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
            ptrtrs('R', 'U', trans, 'N', b_rows, b_cols,
                   d_A, array_descA, d_B, array_descB);
        }
    }
}

template void pgetrs_nopiv<float>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);
template void pgetrs_nopiv<double>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);
template void pgetrs_nopiv<std::complex<float>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);
template void pgetrs_nopiv<std::complex<double>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

} // namespace ddla
