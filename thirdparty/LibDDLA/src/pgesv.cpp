#include <ddla/ddla.h>
#include <cassert>
#include <vector>
#include <ddla/ddla_connector.h>
#include <stdexcept>
#include <string>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla{

template <typename T>
void pgesv(
    const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const DdlaDesc& array_descA,
    T* d_B, const DdlaDesc& array_descB
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();
    detail::require_gpu_backend(ddla_handle, "pgesv");
    std::vector<int> ipiv(array_descA.m_loc());
    int info = 0;
    pgetrf(
        n, n,
        d_A, array_descA,
        ipiv.data(),
        info
    );
    if(info !=0){
        throw std::runtime_error("pgesv: pgetrf returned info = " + std::to_string(info));
    }
    pgetrs(
        side, trans, n, nrhs,
        d_A, array_descA,
        ipiv.data(),
        d_B, array_descB
    );
}

template void pgesv<float>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const DdlaDesc& array_descA,
    float* d_B, const DdlaDesc& array_descB
);

template void pgesv<double>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const DdlaDesc& array_descA,
    double* d_B, const DdlaDesc& array_descB
);

template void pgesv<std::complex<float>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const DdlaDesc& array_descA,
    std::complex<float>* d_B, const DdlaDesc& array_descB
);

template void pgesv<std::complex<double>>(
    const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const DdlaDesc& array_descA,
    std::complex<double>* d_B, const DdlaDesc& array_descB
);

}
