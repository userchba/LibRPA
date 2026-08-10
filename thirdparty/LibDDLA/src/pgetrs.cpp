#include <ddla/ddla.h>
#include <cassert>
#include "ddla_stream_impl.h"
#include "require_gpu.h"
namespace ddla{

template<typename T>
void pgetrs(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "pgetrs");
    assert(side == 'L' || side == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    // side='L': B is n x nrhs, solves op(A)*X = B;
    // side='R': B is nrhs x n, solves X*op(A) = B.
    if(side == 'L'){
        assert(array_descA.m() == array_descB.m());
    }else{
        assert(array_descA.m() == array_descB.n());
    }
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;

    // Apply the pgetrf pivot permutation to B via plapiv.  With
    // A = P*L*U and P = P(0)*...*P(n-2), P(k) swapping rows k and
    // ipiv(k)-1, forward application computes P^T*B (rows) or B*P (columns),
    // backward application computes P*Z (rows) or Z*P^T (columns).  rowcol
    // selects rows of B (side='L') or columns of B (side='R'); direc selects
    // the application order.  The pivot vector follows array_descA's row
    // distribution, so array_descIP is array_descA.
    if(side == 'L'){
        if(trans == 'N'){
            // A = P*L*U => X = U^-1 * L^-1 * (P^T * B)
            plapiv('F', 'R', 'C', n, nrhs, d_B, array_descB, ipiv, array_descA, nullptr);
            ptrtrs('L', 'L', 'N', 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('L', 'U', 'N', 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
        }else{
            // op(A)^T = U^T * L^T * P^T => X = P * L^-T * U^-T * B
            ptrtrs('L', 'U', trans, 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('L', 'L', trans, 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            plapiv('B', 'R', 'C', n, nrhs, d_B, array_descB, ipiv, array_descA, nullptr);
        }
    }else{
        if(trans == 'N'){
            // X * P * L * U = B => X = B * U^-1 * L^-1 * P^T
            ptrtrs('R', 'U', 'N', 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('R', 'L', 'N', 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            plapiv('B', 'C', 'C', n, nrhs, d_B, array_descB, ipiv, array_descA, nullptr);
        }else{
            // X * U^T * L^T * P^T = B => X = B * P * L^-T * U^-T
            plapiv('F', 'C', 'C', n, nrhs, d_B, array_descB, ipiv, array_descA, nullptr);
            ptrtrs('R', 'L', trans, 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('R', 'U', trans, 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
        }
    }
}

template void pgetrs<double>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    double* d_B, const DdlaDesc& array_descB
);

template void pgetrs<float>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    float* d_B, const DdlaDesc& array_descB
);

template void pgetrs<std::complex<double>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

template void pgetrs<std::complex<float>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    std::complex<float>* d_B, const DdlaDesc& array_descB
);


} // namespace ddla
