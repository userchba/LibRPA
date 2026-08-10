#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/swap.h>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla {

/**
 * @brief Distributed solve using the block LU factors from pgetrf_bpiv.
 *
 * pgetrf_bpiv factors each nb x nb diagonal block with a local getrf and
 * applies each block's row permutation (block-partial pivoting) to the right
 * panel with laswp in forward order.  The stored d_ipiv entries are 1-based
 * offsets *within* each diagonal block, kept in device memory on the owning
 * process row only.  The block permutations act on disjoint row sets, so they
 * commute and each block's swaps are local to one process row (side='L') or
 * process column (side='R').
 *
 * For a side='L', trans='N' solve the permutation Q (the composite of the
 * block swaps in the same forward order laswp applied them) must be applied
 * to B before the two triangular solves; the remaining side/trans
 * combinations mirror the trsm order and apply Q^T (backward block-local
 * order) on the opposite side.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf_bpiv).
 * @param array_descA  DdlaDesc for A.
 * @param d_ipiv  Device pivot array from pgetrf_bpiv (block-local, 1-based).
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 */
template<typename T>
void pgetrs_bpiv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    int* d_ipiv, // device
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "pgetrs_bpiv");
    assert(side == 'L' || side == 'R');
    assert(trans == 'N' || trans == 'T' || trans == 'C');
    assert(array_descA.m() == array_descA.n());
    assert(array_descA.mb() == array_descA.nb());
    if(side == 'L'){
        assert(array_descA.m() == array_descB.m());
    }else{
        assert(array_descA.m() == array_descB.n());
    }
    const int nb = array_descA.mb();
    const int b_rows = (side == 'L') ? n : nrhs;
    const int b_cols = (side == 'L') ? nrhs : n;

    // Copy the local block pivots to host.  Valid entries live only on the
    // (owner_row, owner_col) process of each block (pgetrf_bpiv broadcasts
    // them along the process row for every block except the last, which
    // breaks before the broadcast).  apply_pivots below re-broadcasts each
    // block's pivots over the full grid from that single owner, so every
    // process gets correct values regardless of block.
    std::vector<int> h_ipiv(array_descA.m_loc());
    RUNTIME_CHECK(runtimeMemcpyAsync(h_ipiv.data(), d_ipiv,
                                     h_ipiv.size() * sizeof(int),
                                     runtimeMemcpyDeviceToHost, ddla_handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));

    // Apply the block-partial pivot permutation to B: rows of B for side='L',
    // columns of B for side='R'.  Each block's swaps are local to the owning
    // process row (columns) so no data movement is required.  `forward`
    // applies each block's swaps in the same order laswp applied them,
    // `backward` in reverse order.  Note the direction composes differently
    // for rows and columns: forward rows compute P^T*X (left multiply),
    // backward rows compute P*X, forward columns compute X*P (right
    // multiply), backward columns compute X*P^T, where A = P*L*U.
    //
    // Valid block pivots live only on the (owner_row, owner_col) process:
    // pgetrf_bpiv broadcasts them along the process row for every block
    // *except* the last (the loop breaks before the broadcast), and the
    // processes of the owning process column need them for a side='R' solve.
    // A single broadcast over the full grid from rc_to_rank(owner_row,
    // owner_col) gives every process the block pivots unconditionally.
    auto apply_pivots = [&](bool columns, bool forward){
        const int lldB = array_descB.lld();
        const int nprows = array_descA.nprows();
        const int npcols = array_descA.npcols();
        std::vector<int> piv(nb);
        for(int n_s = 0; n_s < n; n_s += nb){
            const int nb_real = std::min(nb, n - n_s);
            const int owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
            const int owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);
            if(array_descA.myprow() == owner_row && array_descA.mypcol() == owner_col){
                const int mm_row_start =
                    num_loc(n_s, nb, array_descA.myprow(), array_descA.irsrc(), nprows);
                for(int i = 0; i < nb_real; ++i)
                    piv[i] = h_ipiv[mm_row_start + i];
            }
            MPI_Bcast(piv.data(), nb_real, MPI_INT,
                      ddla_handle->rc_to_rank(owner_row, owner_col), ddla_handle->comm);
            const int begin = forward ? 1 : nb_real;
            const int end   = forward ? nb_real + 1 : 0;
            const int step  = forward ? 1 : -1;
            for(int i = begin; i != end; i += step){
                const int t = piv[i - 1] - 1; // 0-based target inside the block
                if(t == i - 1)
                    continue;
                if(columns){
                    if(array_descB.mypcol() == owner_col){
                        const int j1 = array_descB.indx_g2l_c(n_s + i - 1);
                        const int j2 = array_descB.indx_g2l_c(n_s + t);
                        BLAS_CHECK(deblasSwap(ddla_handle->blasH,
                                              array_descB.m_loc(), d_B + j1 * lldB, 1,
                                              d_B + j2 * lldB, 1));
                    }
                }else{
                    if(array_descB.myprow() == owner_row){
                        const int i1 = array_descB.indx_g2l_r(n_s + i - 1);
                        const int i2 = array_descB.indx_g2l_r(n_s + t);
                        BLAS_CHECK(deblasSwap(ddla_handle->blasH,
                                              array_descB.n_loc(), d_B + i1, lldB,
                                              d_B + i2, lldB));
                    }
                }
            }
        }
    };

    if(side == 'L'){
        if(trans == 'N'){
            // A = Q^T * L * U => X = U^-1 * L^-1 * (Q * B)
            apply_pivots(/*columns=*/false, /*forward=*/true);
            ptrtrs('L', 'L', 'N', 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('L', 'U', 'N', 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
        }else{
            // op(A)^T = U^T * L^T * Q => X = Q^T * L^-T * U^-T * B
            ptrtrs('L', 'U', trans, 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('L', 'L', trans, 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            apply_pivots(/*columns=*/false, /*forward=*/false);
        }
    }else{
        if(trans == 'N'){
            // X * P * L * U = B => X = B * U^-1 * L^-1 * P^T
            ptrtrs('R', 'U', 'N', 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('R', 'L', 'N', 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            // Columns compose opposite to rows: column swaps applied in the
            // laswp order right-multiply to Z*P, not Z*P^T; apply backward.
            apply_pivots(/*columns=*/true, /*forward=*/false);
        }else{
            // X * U^T * L^T * P^T = B => X = B * P * L^-T * U^-T
            // Column swaps in the laswp order give B*P; apply forward.
            apply_pivots(/*columns=*/true, /*forward=*/true);
            ptrtrs('R', 'L', trans, 'U', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
            ptrtrs('R', 'U', trans, 'N', b_rows, b_cols, d_A, array_descA, d_B, array_descB);
        }
    }
}

template void pgetrs_bpiv<float>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    int* d_ipiv,
    float* d_B, const DdlaDesc& array_descB
);
template void pgetrs_bpiv<double>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    int* d_ipiv,
    double* d_B, const DdlaDesc& array_descB
);
template void pgetrs_bpiv<std::complex<float>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    int* d_ipiv,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);
template void pgetrs_bpiv<std::complex<double>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    int* d_ipiv,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

} // namespace ddla
