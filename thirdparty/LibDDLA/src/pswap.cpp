#include <ddla/ddla.h>
#include <cassert>
#include <ddla/ddla_connector.h>
#include "ddla_stream_impl.h"
#include "require_gpu.h"
#include <ddla/swap.h>
#include "comm_traits.h"
namespace ddla{

template <typename T>
void pswap(
    const int& N, 
    T* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    T* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
)
{
    ia--;
    ja--;
    ib--;
    jb--;
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "pswap");

    assert(inca == 1 || inca == array_descA.m());
    assert(incb == 1 || incb == array_descB.m());
    if(inca == 1)
        assert(inca == incb);
    const int myprow = array_descA.myprow();
    const int mypcol = array_descA.mypcol();
    const int ia_loc = num_loc(ia, array_descA.mb(), myprow, array_descA.irsrc(), array_descA.nprows());
    const int ja_loc = num_loc(ja, array_descA.nb(), mypcol, array_descA.icsrc(), array_descA.npcols());
    const int ib_loc = num_loc(ib, array_descB.mb(), myprow, array_descB.irsrc(), array_descB.nprows());
    const int jb_loc = num_loc(jb, array_descB.nb(), mypcol, array_descB.icsrc(), array_descB.npcols());
    
    const size_t a_offset = ia_loc + ja_loc * array_descA.lld();
    const size_t b_offset = ib_loc + jb_loc * array_descB.lld();

    T* temp_swap;
    RUNTIME_CHECK(runtimeMallocAsync((void**)&temp_swap, sizeof(T) * std::max(array_descA.m_loc(), array_descA.n_loc()), ddla_handle->stream));
    if(inca == 1){
        const int Na_loc = num_loc(N + ia, array_descA.mb(), myprow, array_descA.irsrc(), array_descA.nprows());
        const int Nb_loc = num_loc(N + ib, array_descB.mb(), myprow, array_descB.irsrc(), array_descB.nprows());
        const int owner_col_A = indxg2p (ja, array_descA.nb(), array_descA.icsrc(), array_descA.npcols());
        const int owner_col_B = indxg2p (jb, array_descB.nb(), array_descB.icsrc(), array_descB.npcols());
        int length_v = Na_loc - ia_loc;
        assert(length_v == Nb_loc - ib_loc);
        if(length_v > 0){
            if(owner_col_A == owner_col_B){
                if(mypcol == owner_col_A)
                    BLAS_CHECK(deblasSwap(ddla_handle->blasH, length_v, A + a_offset, 1, B + b_offset, 1));
            }else{
                if(mypcol == owner_col_A){
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        temp_swap, sizeof(T), 
                        A + a_offset, sizeof(T),
                        sizeof(T), length_v,
                        runtimeMemcpyDeviceToDevice, ddla_handle->stream
                    ));
                    commSend(ddla_handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                    commRecv(ddla_handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_B);
                    BLAS_CHECK(deblasSwap(ddla_handle->blasH, length_v, temp_swap, 1, A + a_offset, 1));
                }else if(mypcol == owner_col_B){
                    commRecv(ddla_handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                    BLAS_CHECK(deblasSwap(ddla_handle->blasH, length_v, temp_swap, 1, B + b_offset, 1));
                    commSend(ddla_handle, CommScope::Row, temp_swap, (std::size_t)length_v, owner_col_A);
                }
            }
        }
    }else{
        const int Na_loc = num_loc(N + ja, array_descA.nb(), mypcol, array_descA.icsrc(), array_descA.npcols());
        const int Nb_loc = num_loc(N + jb, array_descB.nb(), mypcol, array_descB.icsrc(), array_descB.npcols());
        const int owner_row_A = indxg2p (ia, array_descA.mb(), array_descA.irsrc(), array_descA.nprows());
        const int owner_row_B = indxg2p (ib, array_descB.mb(), array_descB.irsrc(), array_descB.nprows());
        int length_v = Na_loc - ja_loc;
        assert(length_v == Nb_loc - jb_loc);
        if(length_v > 0){
            if(owner_row_A == owner_row_B){
                if(myprow == owner_row_A)
                    BLAS_CHECK(deblasSwap(ddla_handle->blasH, length_v, A + a_offset, array_descA.lld(), B + b_offset, array_descB.lld()));
            }else{
                if(myprow == owner_row_A){
                    RUNTIME_CHECK(runtimeMemcpy2DAsync(
                        temp_swap, sizeof(T), 
                        A + a_offset, array_descA.lld() * sizeof(T),
                        sizeof(T), length_v,
                        runtimeMemcpyDeviceToDevice, ddla_handle->stream
                    ));
                    commSend(ddla_handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_B);
                    commRecv(ddla_handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_B);
                    BLAS_CHECK(deblasSwap(ddla_handle->blasH, length_v, temp_swap, 1, A + a_offset, array_descA.lld()));
                }else if(myprow == owner_row_B){
                    commRecv(ddla_handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_A);
                    BLAS_CHECK(deblasSwap(ddla_handle->blasH, length_v, temp_swap, 1, B + b_offset, array_descB.lld()));
                    commSend(ddla_handle, CommScope::Col, temp_swap, (std::size_t)length_v, owner_row_A);
                }
            }
        }
    }
    RUNTIME_CHECK(runtimeFreeAsync(temp_swap, ddla_handle->stream));
}

template void pswap<std::complex<double>>(
    const int& N, 
    std::complex<double>* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    std::complex<double>* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
);

template void pswap<std::complex<float>>(
    const int& N, 
    std::complex<float>* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    std::complex<float>* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
);

template void pswap<float>(
    const int& N, 
    float* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    float* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
);

template void pswap<double>(
    const int& N, 
    double* A, int ia, int ja, const DdlaDesc& array_descA, const int& inca,
    double* B, int ib, int jb, const DdlaDesc& array_descB, const int& incb
);
} // DDLA
