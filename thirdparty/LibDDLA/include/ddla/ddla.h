#ifndef DDLA_H
#define DDLA_H

#include <ddla/ddla_config.h>
#include "ddla_desc.h"
#include "write_matrix.h"
#include "random_generate.h"
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
#include "gemmVbatched.h"
#endif
#include <complex>

namespace ddla{

#if DDLA_HAS_GPU

/**
 * @brief Distributed triangular solve: B := op(A)^{-1} * B    (side='L')
 *                                     B := B * op(A)^{-1}    (side='R').
 *
 * Solves a triangular system on a distributed GPU matrix using block-cyclic
 * data distribution and NCCL/RCCL communication.  Corresponds to the
 * ScaLAPACK PZTRTRS / PDTRTRS routine.
 *
 * @tparam T  Scalar type (float, double, complex<float>, complex<double>).
 * @param side   'L' (left) or 'R' (right) -- which side of op(A) multiplies B.
 * @param uplo   'U' or 'L' -- which triangle of A is stored.
 * @param trans  'N' (no transpose), 'T' (transpose), 'C' (conjugate-transpose).
 * @param diag   'U' (unit diagonal) or 'N' (non-unit diagonal).
 * @param m      Number of rows of B.
 * @param n      Number of columns of B.
 * @param d_A    Device pointer to distributed triangular matrix A.
 * @param array_descA  DdlaDesc for A (must be square, mb == nb).
 * @param d_B    Device pointer to RHS / solution B.
 * @param array_descB  DdlaDesc for B.
 */
template<typename T>
void ptrtrs(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Apply a pivot permutation to a distributed matrix: swap rows or
 *        columns of A according to a column-cyclic pivot vector.
 *
 * Implements the pivoting applied after LU factorization (ScaLAPACK-style
 * PZLASWP).  For a permutation P = P(0)*...*P(m-2) where P(k) swaps row/col
 * k with row/col ipiv(k)-1, direc='F' applies pivots in ascending k order
 * (computing P^T*A for rows / A*P for columns) and direc='B' in descending
 * order (computing P*A for rows / A*P^T for columns).
 *
 * @tparam T   Scalar type.
 * @param direc   'F' -- forward pivoting order; 'B' -- backward.
 * @param rowcol  'R' -- pivot rows; 'C' -- pivot columns.
 * @param pivroc  'C' -- column-cyclic pivot distribution (only 'C').
 * @param m       Number of pivots (rows/columns of A to pivot).
 * @param n       For rowcol='R': number of columns in A; for rowcol='C':
 *                number of rows in A (the fixed segment length).
 * @param d_A     Device pointer to distributed matrix A.
 * @param array_descA   DdlaDesc for A.
 * @param ipiv    Host array of pivot indices (1-based, length >= m).
 * @param array_descIP  DdlaDesc for pivot vector (same row distribution as A).
 * @param iwork   Workspace (unused, pass nullptr).
 */
template <typename T>
void plapiv(
    const char& direc, const char& rowcol, const char& pivroc,
    const int& m, const int& n,
    T* d_A,const DdlaDesc& array_descA,
    const int* ipiv, const DdlaDesc& array_descIP,
    int* iwork
);

/**
 * @brief Swap two rows or two columns between distributed matrices.
 *
 * Exchanges the segment of length N starting at (ia,ja) in A with the
 * segment starting at (ib,jb) in B.  inca=1 swaps columns, inca=m swaps rows.
 * Communication occurs only when the source and target rows/columns reside on
 * different processes.
 *
 * @tparam Backend  Compile-time execution backend. The build default is CPU
 *                  for CPU-only builds and GPU otherwise.
 * @tparam T        Scalar type.
 * @param N     Length of the segment to swap.
 * @param A     Device pointer to distributed matrix A.
 * @param ia    Starting global row index in A (1-based).
 * @param ja    Starting global column index in A (1-based).
 * @param array_descA  DdlaDesc for A.
 * @param inca  1 (swap columns) or m_A (swap rows).
 * @param B     Device pointer to distributed matrix B.
 * @param ib    Starting global row index in B (1-based).
 * @param jb    Starting global column index in B (1-based).
 * @param array_descB  DdlaDesc for B.
 * @param incb  Increment for B (must match inca when inca == 1).
 */
template <typename T>
void pswap(
    const int& N, 
    T* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    T* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
);

/**
 * @brief Internal unblocked panel LU factorization for distributed matrices.
 *
 * Factors the panel starting at global column n_s with width nb_real.  This
 * is the inner kernel called by pgetrf to factor each diagonal block.
 * Outputs pivot indices into ipiv (1-based).
 *
 * @tparam T   Scalar type.
 * @param m        Total rows of A.
 * @param nb_real  Actual width of this panel (<= nb).
 * @param d_A      Device pointer to matrix A (input/output).
 * @param n_s      Global starting column index of the panel.
 * @param array_descA  DdlaDesc for A.
 * @param ipiv     Host pivot array (output, 1-based).
 * @param info     0 on success, >0 if singular.
 */
template <typename T>
void pgetf2(
    const int& m, const int& nb_real,
    T* d_A, const int& n_s, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Alternative panel LU factorization (rank-revealing variant).
 *
 * Uses a slightly different communication pattern than pgetf2 for
 * pivot selection within a panel.
 *
 * @tparam T   Scalar type.
 * @param m        Total rows of A.
 * @param nb_real  Actual panel width.
 * @param d_A      Device pointer to matrix A.
 * @param n_start  Global starting column of the panel.
 * @param array_descA  DdlaDesc for A.
 * @param ipiv     Host pivot array (output).
 * @param info     0 on success.
 */
template <typename T>
void pgetf2_panel(
    const int& m, const int& nb_real,
    T* d_A, const int& n_start, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Distributed LU factorization with partial (row) pivoting:
 *        A = P * L * U.
 *
 * Factors a distributed m-by-n matrix panel by panel:
 *   1. Factor the diagonal block (pgetf2).
 *   2. Broadcast L and U factors along process rows / columns.
 *   3. Solve for the U panel (trsm).
 *   4. Update the trailing submatrix (gemm: C -= L*U).
 *
 * Requires square blocks (mb == nb).  Corresponds to ScaLAPACK
 * PZGETRF / PDGETRF.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param array_descA  DdlaDesc for A (mb == nb required).
 * @param ipiv     Host pivot array (output, 1-based, length >= m_loc).
 * @param info     0 on success, >0 if singular.
 */
template <typename T>
void pgetrf(
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
);

/**
 * @brief Block LU factorization with partial pivoting within each block row.
 *
 * Computes A = P*L*U where pivoting is applied at the block level: within each
 * block column the diagonal block is factored with getrf (producing P1 A = L1 U1),
 * then the pivot is applied to the full rows (the already-factored columns as
 * well as the right panel), the U and L panels are computed via trsm, and the
 * trailing submatrix updated via gemm.  The output is a standard LU
 * factorization: each block's row swaps cover every column, including the L
 * part, so the factors can be used with any standard triangular solve.
 *
 * This is a right-looking block algorithm.  Corresponds to the block-wise
 * derivation in README.md (Experimental Routines).
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param array_descA  DdlaDesc for A (mb == nb required).
 * @param ipiv     device pivot array (output, 1-based block-local offsets,
 *                 length >= m_loc).
 * @param info     host info 0 on success, >0 if singular. 
 */
template <typename T>
void pgetrf_bpiv(const int& m, const int& n, T* d_A, const DdlaDesc& array_descA, int* d_ipiv, int& info);

/**
 * @brief Distributed LU factorization without pivoting.
 *
 * Factors a distributed m-by-n matrix A in-place as A = L * U without
 * pivoting.  L has implicit unit diagonal and is stored strictly below the
 * diagonal; U is stored on and above the diagonal.
 *
 * Uses a right-looking block algorithm: at each step the nb×nb diagonal
 * block is factored with the local getrf_nopiv, the right/lower panels are
 * computed via trsm, and the trailing submatrix is updated via gemm.  The
 * communication pattern mirrors pgetrf_bpiv, but no row pivots are applied.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param array_descA  DdlaDesc for A (mb == nb required).
 * @param info     host info: 0 on success, >0 if U(k,k) is exactly zero.
 */
template <typename T>
void pgetrf_nopiv(const int& m, const int& n, T* d_A, const DdlaDesc& array_descA, int& info);

/**
 * @brief Local LU factorization without pivoting.
 *
 * Factors a local (single-process, single-GPU) m-by-n matrix A in-place:
 * A = L * U.  L has implicit unit diagonal, stored strictly below diagonal.
 * U is stored on and above diagonal.
 *
 * Uses a right-looking block algorithm with block size nb=32:
 *   1. Panel factorization via custom getf2_nopiv_kernel.
 *   2. Solve for U panel via deblasTrsm.
 *   3. Update trailing submatrix via gemm.
 *
 * @tparam T   Scalar type.
 * @param m        Number of rows of A.
 * @param n        Number of columns of A.
 * @param d_A      Device pointer to matrix A (input/output -- L+U factors).
 * @param lda      Leading dimension of A.
 * @param d_info      Device pointer to info (0 on success, k > 0 if U(k,k) is exactly zero).
 * @param ddla_handle DDLA handle (provides stream and BLAS handle).
 */
template <typename T>
void getrf_nopiv(int m, int n, T* d_A, int lda, int* d_info, const DdlaHandle_t& ddla_handle);

/**
 * @brief Distributed LU solve: solve op(A) * X = B (side='L') or
 *        X * op(A) = B (side='R') using the factors from pgetrf.
 *
 * Steps for side='L', trans='N': apply row pivots (plapiv), forward solve
 * L*Y=B (ptrtrs), backward solve U*X=Y (ptrtrs).  Other side/trans
 * combinations apply the trsm sequence in the mirrored order and apply the
 * pivot permutation on the solution side (rows for side='L', columns for
 * side='R').
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf).
 * @param array_descA  DdlaDesc for A.
 * @param ipiv    Host pivot array from pgetrf.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 */
template <typename T>
void pgetrs(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    const int* ipiv, // host
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Distributed LU solve without pivoting: solve op(A) * X = B
 *        (side='L') or X * op(A) = B (side='R') using the LU factors
 *        produced by pgetrf_nopiv.
 *
 * Because no pivoting is used, the solution is obtained by two triangular
 * solves in the mirrored order for side='R' / trans='T','C'.
 *
 * @tparam T   Scalar type.
 * @param side    'L' -- solve op(A)*X = B (B is n x nrhs);
 *                'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param trans   'N', 'T' or 'C' -- operation applied to A.
 * @param n       Order of matrix A.
 * @param nrhs    Number of right-hand sides.
 * @param d_A     Device pointer to LU factors (from pgetrf_nopiv).
 * @param array_descA  DdlaDesc for A.
 * @param d_B     Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 */
template <typename T>
void pgetrs_nopiv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Distributed linear-system solver (driver): solve op(A) * X = B
 *        (side='L') or X * op(A) = B (side='R').
 *
 * Convenience wrapper: pgetrf (LU) + pgetrs (solve).  Corresponds to
 * ScaLAPACK PZGESV / PDGESV.
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
void pgesv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Distributed linear-system solver without pivoting (driver): solve
 *        op(A) * X = B (side='L') or X * op(A) = B (side='R').
 *
 * Convenience wrapper: pgetrf_nopiv (LU) + pgetrs_nopiv (solve).
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
void pgesv_nopiv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Distributed solve using the block LU factors from pgetrf_bpiv:
 *        solve op(A) * X = B (side='L') or X * op(A) = B (side='R').
 *
 * pgetrf_bpiv performs block-partial pivoting: each nb x nb diagonal block is
 * factored with a local getrf and its row permutation is applied to the full
 * rows (already-factored columns and right panel) with laswp in forward
 * order, yielding a standard A = P*L*U.  Its pivot array @p d_ipiv is a
 * device array holding 1-based offsets *within* each diagonal block (kept on
 * the owning process row).  The block permutations act on disjoint row sets,
 * so they commute and each block's swaps are local to one process row
 * (side='L') or process column (side='R'); pgetrs_bpiv applies them without
 * any data movement.
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
template <typename T>
void pgetrs_bpiv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    int* d_ipiv, // device
    T* d_B, const DdlaDesc& array_descB
);

/**
 * @brief Distributed linear-system solver using block-partial-pivoting LU
 *        (driver): solve op(A) * X = B (side='L') or X * op(A) = B
 *        (side='R').
 *
 * Convenience wrapper: pgetrf_bpiv (block LU with partial pivoting within
 * each block row) + pgetrs_bpiv (solve).
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
);

#endif // DDLA_HAS_GPU

/**
 * @brief Distributed matrix-matrix multiplication:
 *        C := alpha * op(A) * op(B) + beta * C.
 *
 * Uses a 2D block-cyclic data distribution and NCCL/RCCL broadcast of
 * panel columns of A and panel rows of B (AB-path).  Supports all standard
 * transpose options:
 *   - 'N': op(X) = X
 *   - 'T': op(X) = X^T
 *   - 'C': op(X) = X^H  (conjugate-transpose)
 *
 * When transa or transb is not 'N', the descriptors must be
 * ScaLAPACK-compatible (e.g. for A^T, mb(C) == nb(A) and irsrc(C) == icsrc(A)).
 * The process grid may be rectangular.
 *
 * All three descriptors (@p array_descA, @p array_descB, @p array_descC)
 * must share the same DdlaHandle_t (same backend and process grid).
 * CPU handles require host pointers; GPU handles require device pointers
 * allocated in the selected-accelerator memory space.  No implicit
 * migration between address spaces is performed.
 *
 * @tparam T    Scalar type.
 * @param transa   Operation applied to A ('N','T','C').
 * @param transb   Operation applied to B ('N','T','C').
 * @param m        Rows of op(A) and C.
 * @param n        Cols of op(B) and C.
 * @param k        Cols of op(A) / rows of op(B).
 * @param alpha    Scalar multiplier for A*B.
 * @param d_A      Pointer to distributed A (host for CPU, device for GPU).
 * @param array_descA  DdlaDesc for A.
 * @param d_B      Pointer to distributed B (host for CPU, device for GPU).
 * @param array_descB  DdlaDesc for B.
 * @param beta     Scalar multiplier for C.
 * @param d_C      Pointer to distributed C (input/output; host for CPU, device for GPU).
 * @param array_descC  DdlaDesc for C.
 */
template <DdlaBackend Backend = default_backend_v, typename T>
void pgemm(
    const char& transa, const char& transb,
    const int& m, const int& n, const int& k,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T* d_B, const DdlaDesc& array_descB,
    const T& beta,
    T* d_C, const DdlaDesc& array_descC
);

#if DDLA_HAS_GPU

/**
 * @brief Distributed matrix addition: C := alpha * op(A) + beta * op(B).
 *
 * Element-wise addition of two distributed matrices with optional transpose
 * operations.  Communication between processes is required when the data
 * distribution of op(A) differs from that of op(B) (e.g. one is transposed
 * and the other is not).
 *
 * @tparam T    Scalar type.
 * @param transa   Operation for A ('N','T','C').
 * @param transb   Operation for B ('N','T','C').
 * @param m        Rows of C.
 * @param n        Cols of C.
 * @param alpha    Scalar multiplier for op(A).
 * @param d_A      Device pointer to distributed A.
 * @param array_descA  DdlaDesc for A.
 * @param beta     Scalar multiplier for op(B).
 * @param d_B      Device pointer to distributed B.
 * @param array_descB  DdlaDesc for B.
 * @param d_C      Device pointer to result C (output).
 * @param array_descC  DdlaDesc for C.
 */
template <typename T>
void pgeadd(
    const char& transa, const char& transb,
    const int& m, const int& n,
    const T& alpha,
    const T* d_A, const DdlaDesc& array_descA,
    const T& beta,
    const T* d_B, const DdlaDesc& array_descB,
    T* d_C, const DdlaDesc& array_descC
);

/**
 * @brief Add a scalar to the diagonal of a distributed square matrix.
 *
 * For every global diagonal element A(i,i) with 0 <= i < n, add alpha.
 * Only the locally owned portion of the 2D block-cyclic distribution is
 * updated; no inter-process communication is required.
 *
 * Supported combinations match LibRPA's DeviceConnector::pdam:
 *   (float,float), (double,double),
 *   (float, complex<float>), (complex<float>, complex<float>),
 *   (double, complex<double>), (complex<double>, complex<double>).
 *
 * @tparam T1  Scalar type of the value to add.
 * @tparam T2  Element type of the distributed matrix A.
 * @param alpha  Scalar to add to each diagonal element.
 * @param d_A    Device pointer to distributed matrix A (input/output).
 * @param array_descA  DdlaDesc for A (must be square).
 */
template <typename T1, typename T2>
void pdam(const T1& alpha, T2* d_A, const DdlaDesc& array_descA);

/**
 * @brief Distributed Cholesky factorization.
 *
 * Factors a Hermitian positive-definite distributed matrix using GPU solver
 * libraries (cusolverDn / hipsolver).  Algorithm: factor diagonal block
 * (potrf), broadcast factor, solve off-diagonal (trsm), update trailing
 * submatrix via gemm/herk.  With uplo='L', computes A = L * L^H.
 * With uplo='U', computes A = U^H * U.
 *
 * @note Only complex<float> and complex<double> are instantiated.
 *
 * @tparam T   Scalar type (complex<float> or complex<double>).
 * @param uplo     'L' or 'U' -- triangle of A to store and factor.
 * @param n        Order of A.
 * @param A        Device pointer to A (input: Hermitian pos-def; output: Cholesky factor).
 * @param ia       Global starting row (1-based).
 * @param ja       Global starting col (1-based).
 * @param array_descA  DdlaDesc for A (mb == nb required).
 * @param info     0 on success, >0 if not positive-definite.
 * @param is_head  Internal flag for multi-head Cholesky (default false).
 * @param location Internal row/col rearrangement index (default -1).
 * @return true if the last diagonal element needed a sign correction,
 *         false otherwise.
 */
template<typename T>
bool ppotrf(
    const char& uplo, const int& n,
    T* A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    int& info, // host pointer
    bool is_head = false, int location = -1
);

/**
 * @brief Single-GPU Cholesky factorization from the bottom-right corner.
 *
 * With uplo='U', computes A = U * U^H and overwrites the upper triangle with
 * U.  With uplo='L', computes A = L^H * L and overwrites the lower triangle
 * with L.  In either mode the opposite triangle is not referenced or written.
 * These are bottom-right factorizations and therefore reverse the product
 * order used by the corresponding standard LAPACK POTRF convention.
 *
 * @tparam T      Scalar type (float, double, complex<float>, complex<double>).
 * @param uplo    'U' for A = U * U^H or 'L' for A = L^H * L.
 * @param n       Order of A.
 * @param d_A     Device pointer to A.
 * @param lda     Leading dimension of A.
 * @param info    0 on success; i > 0 identifies the failed reverse pivot.
 * @param handle  DDLA handle providing the GPU stream and BLAS handle.
 */
template <typename T>
void potrf_bottom_right(
    const char& uplo, const int& n, T* d_A, const int& lda,
    int& info, const DdlaHandle_t& handle
);

/**
 * @brief Distributed Cholesky factorization from the bottom-right corner.
 *
 * With uplo='U', computes A = U * U^H and overwrites the upper triangle with
 * U.  With uplo='L', computes A = L^H * L and overwrites the lower triangle
 * with L.  The matrix descriptor must use square blocks on a square process
 * grid; row and column source processes may differ.
 *
 * @tparam T            Scalar type (float, double, complex<float>,
 *                      complex<double>).
 * @param uplo          'U' for A = U * U^H or 'L' for A = L^H * L.
 * @param n             Order of A.
 * @param d_A           Device pointer to the local block-cyclic storage of A.
 * @param array_descA   Descriptor for the distributed matrix A.
 * @param info          0 on success; i > 0 identifies the failed global pivot.
 */
template <typename T>
void ppotrf_bottom_right(
    const char& uplo, const int& n, T* d_A,
    const DdlaDesc& array_descA, int& info
);

/**
 * @brief Distributed solve using Cholesky factorization: solve
 *        op(A) * X = B (side='L') or X * op(A) = B (side='R').
 *
 * Solves a Hermitian positive-definite system using the factor from
 * ppotrf.  For side='L', uplo='L' it applies L then L^H; the trsm order
 * is mirrored for side='R'.  Because A is Hermitian, op(A) == A for both
 * trans='N' and trans='C' (identical code path); trans='T' is not
 * supported.
 *
 * When `location` is a head-correction index (the same value passed to
 * ppotrf with is_head=true), ppotrs applies the matching permutation to B
 * -- rows for side='L', columns for side='R' -- around the solve and undoes
 * it afterward, so direct ppotrf + ppotrs users (and pposv) do not need to
 * permute B themselves.
 *
 * @tparam T   Scalar type.
 * @param side     'L' -- solve op(A)*X = B (B is n x nrhs);
 *                 'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param uplo     'L' or 'U' -- triangle containing the Cholesky factor.
 * @param trans    'N' or 'C' (equivalent for Hermitian A).
 * @param n        Order of A.
 * @param nrhs     Number of right-hand sides.
 * @param d_A      Device pointer to Cholesky factor (from ppotrf).
 * @param array_descA  DdlaDesc for A.
 * @param d_B      Device pointer to RHS / solution B (input/output).
 * @param array_descB  DdlaDesc for B.
 * @param is_nega  Diagonal sign-correction flag (from ppotrf return).
 * @param location Head-correction index forwarded from ppotrf; -1 (or == n)
 *                 means no B permutation.
 */
template <typename T>
void ppotrs(
    const char& side, const char& uplo, const char& trans,
    const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB,
    bool is_nega = false, int location = -1
);

/**
 * @brief Distributed solver for Hermitian positive-definite systems
 *        (driver): solve op(A) * X = B (side='L') or X * op(A) = B
 *        (side='R') via Cholesky factorization.
 *
 * Convenience wrapper:  ppotrf + ppotrs.  Corresponds to ScaLAPACK PZPOSV.
 *
 * @tparam T   Scalar type.
 * @param side     'L' -- solve op(A)*X = B (B is n x nrhs);
 *                 'R' -- solve X*op(A) = B (B is nrhs x n).
 * @param uplo     'L' or 'U' -- triangle of A to store and factor.
 * @param trans    'N' or 'C' (equivalent for Hermitian A).
 * @param n        Order of A.
 * @param nrhs     Number of right-hand sides.
 * @param d_A      Device pointer to A (input: pos-def; output: Cholesky factor).
 * @param ia       Global starting row of A (1-based).
 * @param ja       Global starting col of A (1-based).
 * @param array_descA  DdlaDesc for A.
 * @param d_B      Device pointer to RHS / solution B (input/output).
 * @param ib       Global starting row of B (1-based).
 * @param jb       Global starting col of B (1-based).
 * @param array_descB  DdlaDesc for B.
 * @param info     Output: 0 on success, >0 if not positive-definite.
 * @param is_head  Forwarded to ppotrf.
 * @param location Forwarded to ppotrf.
 */
template <typename T>
void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const DdlaDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const DdlaDesc& array_descB,
    int& info, // host pointer
    bool is_head = false, int location = -1
);

#endif // DDLA_HAS_GPU

} // namespace ddla

#endif // DDLA_H
