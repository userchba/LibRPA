#pragma once
#include "epsilon.h"
#include "device_connector.h"
#include "matrix_device.h"

#ifdef LIBRPA_USE_CUDA
CorrEnergy compute_RPA_correlation_cuda(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat);
#endif
#ifdef ENABLE_NVHPC
CorrEnergy compute_RPA_correlation_blacs_2d_cuda(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat);

map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
compute_Wc_freq_q_blacs_cuda(Chi0 &, const atpair_k_cplx_mat_t &,
                        atpair_k_cplx_mat_t &,
                        const vector<std::complex<double>> &);
complex<double> compute_pi_det_blacs_2d_nvhpc(
    MatrixDevice<std::complex<double>> &, const LIBRPA::Array_Desc &arrdesc_pi, int64_t *d_ipiv, int *d_info,char order='C');
#endif
