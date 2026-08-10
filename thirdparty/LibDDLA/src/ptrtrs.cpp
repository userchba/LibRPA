#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include <ddla/trsm.h>
#include <ddla/transport_block.h>
#include <ddla/ddla_comm.h>
#include <ddla/gemm.h>
#ifdef DDLA_USE_GPU_CPU_TUNNEL
#include <vector>
#endif
namespace ddla{


template<typename T>
void ptrtrs(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "ptrtrs");
    
    assert(array_descA.m() == array_descA.n());
    assert(array_descA.mb()==array_descA.nb());
    assert(array_descA.mb()==array_descB.mb());
    assert(side=='L'||side=='R');
    if(side=='L'){
        assert(array_descA.m() == array_descB.m());
    }else{
        assert(array_descA.m() == array_descB.n());
    }
    assert(uplo=='L'||uplo=='U');
    assert(diag=='U'||diag=='N');
    assert(trans=='N'||trans=='T'||trans=='C');
    int nb = array_descA.mb();
    int lldA = array_descA.lld();
    int lldB = array_descB.lld();

    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    // printf("nprows:%d, npcols:%d\n",nprows,npcols);

    // 初始化 NCCL  
    #ifdef DDLA_USE_CCL
    ncclComm_t row_comm=ddla_handle->nccl_row_comm;
    ncclComm_t col_comm=ddla_handle->nccl_col_comm;
    #else
    MPI_Comm row_comm=ddla_handle->row_comm;
    MPI_Comm col_comm=ddla_handle->col_comm;
    #endif
    runtimeStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;

    deblasFillMode_t uplo_device = (uplo == 'U') ? DEBLAS_FILL_MODE_UPPER : DEBLAS_FILL_MODE_LOWER;
    deblasDiagType_t diag_device = (diag == 'U') ? DEBLAS_DIAG_UNIT : DEBLAS_DIAG_NON_UNIT;
    deblasOperation_t trans_device;
    if(trans == 'N'){
        trans_device = DEBLAS_OP_N;
    }else if(trans == 'T'){
        trans_device = DEBLAS_OP_T;
    }else{
        trans_device = DEBLAS_OP_C;
    }
    deblasSideMode_t side_device = (side == 'L') ? DEBLAS_SIDE_LEFT : DEBLAS_SIDE_RIGHT;
    // Order of the triangular system: op(A) is n_solve x n_solve, matching the
    // rows of B for side='L' (B is m x n) and the columns of B for side='R'.
    const int n_solve = (side == 'L') ? m : n;
    
    T* d_block_diag,*d_block_A,*d_block_B;
    RUNTIME_CHECK(runtimeMallocAsync(&d_block_diag, nb * nb * sizeof(T), stream));
    // side='L' stages an nb x n_loc block row of B; side='R' an m_loc x nb block column.
    RUNTIME_CHECK(runtimeMallocAsync(&d_block_B, nb * std::max(array_descB.m_loc(), array_descB.n_loc()) * sizeof(T), stream));
    RUNTIME_CHECK(runtimeMallocAsync(&d_block_A, std::max(array_descA.m_loc(), array_descA.n_loc()) * nb * sizeof(T), stream));

    #ifdef DDLA_USE_GPU_CPU_TUNNEL
    std::vector<T> h_temp(nb * std::max({array_descB.m_loc(), array_descB.n_loc(), nb}));
    #endif

    int owner_row, owner_col;
    int mm_row_start, mm_col_start, mm_row_step, mm_col_step;
    int64_t A_offset, B_offset;

    if(side == 'L'){
        // Left solve: B := op(A)^{-1} * B. Block forward/backward substitution:
        // at each diagonal block, trsm on the owner row of B, broadcast the
        // solved block row, then gemm-update the remaining rows of B.
        const bool solve_backward = (uplo == 'U' && trans == 'N') || (uplo == 'L' && trans != 'N');
        const char panel_direction = (trans == 'N') ? 'C' : 'R';
        int n_s_start,n_s_end,n_s_step;
        if(solve_backward){
            n_s_start = n_solve % nb == 0 ? n_solve - nb : n_solve - n_solve % nb;
            n_s_end = -nb;
            n_s_step = -nb;
        }else{
            n_s_start = 0;
            n_s_end = n_solve % nb == 0 ? n_solve : n_solve - n_solve % nb + nb;
            n_s_step = nb;
        }
        for(int n_s = n_s_start; n_s != n_s_end; n_s += n_s_step){
            int nb_real = std::min(nb, n_solve - n_s);
            // printf("n_s=%d, nb_real=%d\n",n_s, nb_real);

            mm_row_start = num_loc(n_s, nb, array_descA.myprow(), array_descA.irsrc(), nprows);
            mm_col_start = num_loc(n_s, nb, array_descA.mypcol(), array_descA.icsrc(), npcols);

            owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
            owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

            if(array_descA.myprow() == owner_row)
                mm_row_step = nb_real;
            else
                mm_row_step = 0;
            if(array_descA.mypcol() == owner_col)
                mm_col_step = nb_real;
            else 
                mm_col_step = 0;
            // printf("owner_row:%d,owner_col:%d\n",owner_row,owner_col);

            if(array_descA.myprow() == owner_row && array_descA.mypcol() == owner_col){
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_block_diag, nb_real * sizeof(T),
                    d_A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                    nb_real * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            // 广播当前块行
            if(array_descA.myprow() == owner_row){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_temp.data(), d_block_diag, nb_real * nb_real, owner_col, ddla_handle->row_comm, stream));
                #else
                CCL_CHECK(cclBcast(d_block_diag, nb_real * nb_real, owner_col, row_comm, stream));
                #endif
                BLAS_CHECK(deblasTrsm(
                    blasH, side_device, uplo_device, trans_device, diag_device,
                    nb_real, array_descB.n_loc(), 1.0,
                    d_block_diag, nb_real,
                    d_B + mm_row_start, lldB
                ));
            }
            transport_block(
                'R', 'N', 
                nb_real, array_descB.n(),
                d_B, n_s, 0, array_descB,
                d_block_B
            );
            int length_block_A;
            int g_m, g_n;
            int g_ia, g_ja;
            if(trans != 'N'){
                g_m = nb_real;
                g_ia = n_s;
                if(uplo == 'L'){
                    A_offset = mm_row_start;
                    length_block_A = mm_col_start;
                    g_n = n_s;
                    g_ja = 0;
                }else{
                    // U^H solve: gather the row panel to the right of the diagonal.
                    A_offset = mm_row_start + (mm_col_start + mm_col_step) * array_descA.lld();
                    length_block_A = array_descA.n_loc() - mm_col_start - mm_col_step;
                    g_n = array_descA.n() - n_s - nb_real;
                    g_ja = n_s + nb_real;
                }
            }else{
                g_ja = n_s;
                g_n = nb_real;
                if(uplo == 'L'){
                    length_block_A = array_descA.m_loc() - mm_row_start - mm_row_step;
                    A_offset = mm_row_start + mm_row_step + mm_col_start * array_descA.lld();
                    g_m = array_descA.m() - n_s - nb_real;
                    g_ia = n_s + nb_real;
                }else{
                    // U solve: gather the column panel above the diagonal.
                    length_block_A = mm_row_start;
                    A_offset = mm_col_start * array_descA.lld();
                    g_m = n_s;
                    g_ia = 0;
                }
            }
            transport_block(
                panel_direction, trans,
                g_m, g_n,
                d_A, g_ia, g_ja, array_descA,
                d_block_A
            );
            if(solve_backward){
                length_block_A = mm_row_start;
                B_offset = 0;
            }else{
                length_block_A = array_descA.m_loc() - mm_row_start - mm_row_step;
                B_offset = mm_row_start + mm_row_step;
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            if(length_block_A > 0){
                gemm<DdlaBackend::GPU, T>(ddla_handle, trans, 'N',
                    length_block_A, array_descB.n_loc(), nb_real,
                    (T)-1.0,
                    d_block_A, trans == 'N' ? length_block_A : nb_real,
                    d_block_B, nb_real,
                    (T)1.0,
                    d_B + B_offset, lldB);
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
        }
    }else{
        // Right solve: X := B * op(A)^{-1}, i.e. X * op(A) = B. Column-mirror of
        // the left solve: trsm on the owner column of B, broadcast the solved
        // block column, then gemm-update the remaining columns of B with the
        // row panel of A on the far side of the diagonal block.
        const bool solve_backward = (uplo == 'L' && trans == 'N') || (uplo == 'U' && trans != 'N');
        const bool far_right = !solve_backward;
        int n_s_start,n_s_end,n_s_step;
        if(solve_backward){
            n_s_start = n_solve % nb == 0 ? n_solve - nb : n_solve - n_solve % nb;
            n_s_end = -nb;
            n_s_step = -nb;
        }else{
            n_s_start = 0;
            n_s_end = n_solve % nb == 0 ? n_solve : n_solve - n_solve % nb + nb;
            n_s_step = nb;
        }
        for(int n_s = n_s_start; n_s != n_s_end; n_s += n_s_step){
            int nb_real = std::min(nb, n_solve - n_s);

            mm_row_start = num_loc(n_s, nb, array_descA.myprow(), array_descA.irsrc(), nprows);
            mm_col_start = num_loc(n_s, nb, array_descA.mypcol(), array_descA.icsrc(), npcols);

            owner_row = indxg2p(n_s, nb, array_descA.irsrc(), nprows);
            owner_col = indxg2p(n_s, nb, array_descA.icsrc(), npcols);

            if(array_descA.myprow() == owner_row)
                mm_row_step = nb_real;
            else
                mm_row_step = 0;
            if(array_descA.mypcol() == owner_col)
                mm_col_step = nb_real;
            else 
                mm_col_step = 0;

            if(array_descA.myprow() == owner_row && array_descA.mypcol() == owner_col){
                RUNTIME_CHECK(runtimeMemcpy2DAsync(
                    d_block_diag, nb_real * sizeof(T),
                    d_A + mm_row_start + mm_col_start * lldA, lldA * sizeof(T),
                    nb_real * sizeof(T), nb_real,
                    runtimeMemcpyDeviceToDevice, stream
                ));
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            // broadcast the diagonal block within the column, then solve the
            // block column of B on the owner column's processes
            if(array_descA.mypcol() == owner_col){
                #ifdef DDLA_USE_GPU_CPU_TUNNEL
                MPI_CHECK(cclBcast(h_temp.data(), d_block_diag, nb_real * nb_real, owner_row, ddla_handle->col_comm, stream));
                #else
                CCL_CHECK(cclBcast(d_block_diag, nb_real * nb_real, owner_row, col_comm, stream));
                #endif
                BLAS_CHECK(deblasTrsm(
                    blasH, DEBLAS_SIDE_RIGHT, uplo_device, trans_device, diag_device,
                    array_descB.m_loc(), nb_real, 1.0,
                    d_block_diag, nb_real,
                    d_B + mm_col_start * lldB, lldB
                ));
            }
            // gather the solved block column of B (local part: m_loc x nb_real)
            transport_block(
                'C', 'N',
                array_descB.m(), nb_real,
                d_B, 0, n_s, array_descB,
                d_block_B
            );
            // gather the far-side panel of A: for trans='N' the row panel of A
            // on the far side of the diagonal block; for trans!='N' the column
            // panel of A on the far side, transposed during the gather.
            int length_block_A;
            int g_m, g_n;
            int g_ia, g_ja;
            char panel_dir;
            if(trans == 'N'){
                panel_dir = 'R';
                g_m = nb_real;
                g_ia = n_s;
                if(far_right){
                    g_n = array_descA.n() - n_s - nb_real;
                    g_ja = n_s + nb_real;
                    A_offset = mm_row_start + (mm_col_start + mm_col_step) * array_descA.lld();
                    length_block_A = array_descA.n_loc() - mm_col_start - mm_col_step;
                    B_offset = (mm_col_start + mm_col_step) * lldB;
                }else{
                    g_n = n_s;
                    g_ja = 0;
                    A_offset = mm_row_start;
                    length_block_A = mm_col_start;
                    B_offset = 0;
                }
            }else{
                panel_dir = 'C';
                g_n = nb_real;
                g_ja = n_s;
                if(far_right){
                    g_m = array_descA.m() - n_s - nb_real;
                    g_ia = n_s + nb_real;
                    length_block_A = array_descA.n_loc() - mm_col_start - mm_col_step;
                    B_offset = (mm_col_start + mm_col_step) * lldB;
                }else{
                    g_m = n_s;
                    g_ia = 0;
                    length_block_A = mm_col_start;
                    B_offset = 0;
                }
            }
            transport_block(
                panel_dir, trans,
                g_m, g_n,
                d_A, g_ia, g_ja, array_descA,
                d_block_A
            );
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
            if(length_block_A > 0 && array_descB.m_loc() > 0){
                gemm<DdlaBackend::GPU, T>(ddla_handle, 'N', 'N',
                    array_descB.m_loc(), length_block_A, nb_real,
                    (T)-1.0,
                    d_block_B, array_descB.m_loc(),
                    d_block_A, nb_real,
                    (T)1.0,
                    d_B + B_offset, lldB);
            }
            RUNTIME_CHECK(runtimeStreamSynchronize(stream));
        }
    }
    RUNTIME_CHECK(runtimeFreeAsync(d_block_A,stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_block_B,stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_block_diag,stream));
}


template void ptrtrs<float>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);

template void ptrtrs<double>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);

template void ptrtrs<std::complex<float>>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);

template void ptrtrs<std::complex<double>>
(
    const char& side, const char& uplo, const char& trans, const char& diag,
    const int& m, const int& n,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);


}
