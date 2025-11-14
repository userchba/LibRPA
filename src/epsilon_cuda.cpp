#include "epsilon_cuda.h"
#include "epsilon.h"
#include "device_stream.h"
#include <magma_v2.h>
#include <random>
#include <math.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <valarray>

#include "atoms.h"
#include "constants.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "lapack_connector.h"
#include "libri_utils.h"
#include "matrix_m_parallel_utils.h"
#include "parallel_mpi.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "scalapack_connector.h"
#include "stl_io_helper.h"
#include "utils_blacs.h"
#include "utils_io.h"
#include "utils_mem.h"
#include "utils_mpi_io.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Tensor.h>
using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
#endif
#ifdef ENABLE_NVHPC
#include "helpers.h"
#include <fstream> 
#include <string>
#endif

using LIBRPA::Array_Desc;
using LIBRPA::envs::blacs_ctxt_global_h;
using LIBRPA::envs::mpi_comm_global_h;
using LIBRPA::envs::ofs_myid;
using LIBRPA::utils::lib_printf;
CorrEnergy compute_RPA_correlation_blacs_2d_cuda(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat)
{
    lib_printf("Begin to compute_RPA_correlation_blacs_2d_nvhpc  myid: %d\n", mpi_comm_global_h.myid);
    system("free -m");
    CorrEnergy corr;
    if (mpi_comm_global_h.myid == 0) lib_printf("Calculating EcRPA with BLACS/ScaLAPACK_nvhpc 2D\n");
    const auto &mf = chi0.mf;
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    if (mpi_comm_global_h.myid == 0) lib_printf("n_abf = %d\n", n_abf);
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    mpi_comm_global_h.barrier();

    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    const auto set_IJ_nabf_nabf = LIBRPA::utils::get_necessary_IJ_from_block_2D_sy(
        'U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    #ifdef ENABLE_NVHPC
    MatrixDevice<std::complex<double>> d_chi0_block, d_coul_block, d_coul_chi0_block;
    #endif
    vector<Vector3_Order<double>> qpts;
    for(const auto &q : chi0.klist)
    {
        qpts.push_back(q);
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        // printf("processId:%d, q: (%f, %f, %f)\n", mpi_comm_global_h.myid, q.x, q.y, q.z);
        #endif
    }
    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    if (mpi_comm_global_h.is_root()) lib_printf("Finish init RPA blacs 2d\n");
    #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
    // printf("success before for loop processid:%d\n", mpi_comm_global_h.myid);
    #endif
#ifdef LIBRPA_USE_LIBRI
    
    for (const auto &q : qpts)
    {
        coul_block.zero_out();

        int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
        std::array<double, 3> qa = {q.x, q.y, q.z};
        // collect the block elements of coulomb matrices
        {
            double vq_begin = omp_get_wtime();
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
                coul_libri;
            coul_libri.clear();
            for (const auto &Mu_Nu : local_atpair)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;

                if (coulmat.count(Mu) == 0 || coulmat.at(Mu).count(Nu) == 0 ||
                    coulmat.at(Mu).at(Nu).count(q) == 0)
                    continue;
                const auto &Vq = coulmat.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                coul_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            double arr_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            double comm_begin = omp_get_wtime();
            const auto IJq_coul =
                comm_map2_first(mpi_comm_global_h.comm, coul_libri, s0_s1.first, s0_s1.second);
            double comm_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            double block_begin = omp_get_wtime();
            collect_block_from_ALL_IJ_Tensor(coul_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);
            double block_end = omp_get_wtime();
            lib_printf(
                "Vq Time  myid: %d  arr_time: %f  comm_time: %f   block_time: %f   pair_size: %d\n",
                mpi_comm_global_h.myid, arr_end - vq_begin, comm_end - comm_begin,
                block_end - block_begin, set_IJ_nabf_nabf.size());
            mpi_comm_global_h.barrier();
            double vq_end = omp_get_wtime();

            if (mpi_comm_global_h.myid == 0)
                lib_printf(" | Total vq time: %f  lri_coul: %f   comm_vq: %f   block_vq: %f\n",
                           vq_end - vq_begin, comm_begin - vq_begin, block_begin - comm_begin,
                           vq_end - block_begin);
        }
        double chi_arr_time = 0.0;
        double chi_comm_time = 0.0;
        double chi_2d_time = 0.0;
        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            double pi_freq_begin = omp_get_wtime();
            chi0_block.zero_out();
            {
                double chi_begin_arr = omp_get_wtime();
                std::map<int,
                         std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
                    chi0_libri;
                atom_mapping<ComplexMatrix>::pair_t_old chi0_wq;
                if(!chi0.get_chi0_q().empty())
                    chi0_wq = chi0.get_chi0_q().at(freq).at(q);
                chi0_libri.clear();
                if(!chi0.get_chi0_q().empty())
                for (const auto &M_Nchi : chi0_wq)
                {
                    const auto &M = M_Nchi.first;
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    for (const auto &N_chi : M_Nchi.second)
                    {
                        const auto &N = N_chi.first;
                        const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                        const auto &chi = N_chi.second;
                        std::valarray<complex<double>> chi_va(chi.c, chi.size);
                        auto pchi = std::make_shared<std::valarray<complex<double>>>();
                        *pchi = chi_va;
                        chi0_libri[M][{N, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pchi);
                    }
                }
                if (mpi_comm_global_h.is_root())
                {
                    lib_printf("Begin to clean chi0 !!! \n");
                    LIBRPA::utils::display_free_mem();
                    lib_printf("chi0_freq_q size: %d,  freq: %f, q:( %f, %f, %f )\n",
                               chi0_wq.size(), freq, q.x, q.y, q.z);
                }
                if(!chi0.get_chi0_q().empty())
                    chi0.free_chi0_q(freq, q);
                
                LIBRPA::utils::release_free_mem();
                mpi_comm_global_h.barrier();
                double chi_end_arr = omp_get_wtime();
                const auto IJq_chi0 =
                    comm_map2_first(mpi_comm_global_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                // ofs_myid << "IJq_chi0" << endl << IJq_chi0;
                double chi_end_comm = omp_get_wtime();
                collect_block_from_ALL_IJ_Tensor(chi0_block, desc_nabf_nabf,
                                                 LIBRPA::atomic_basis_abf, qa, true, CONE, IJq_chi0,
                                                 MAJOR::ROW);
                mpi_comm_global_h.barrier();
                double chi_end_2d = omp_get_wtime();

                chi_arr_time = (chi_end_arr - chi_begin_arr);
                chi_comm_time = (chi_end_comm - chi_end_arr);
                chi_2d_time = (chi_end_2d - chi_end_comm);
            }

            double pi_begin = omp_get_wtime();
            d_coul_block.set_data(coul_block.nr(), coul_block.nc(), coul_block.ptr(),device_stream.stream);
            d_chi0_block.set_data(chi0_block.nr(), chi0_block.nc(), chi0_block.ptr(),device_stream.stream);
            d_coul_chi0_block.set_data(coul_chi0_block.nr(), coul_chi0_block.nc(), coul_chi0_block.ptr(),device_stream.stream);
            std::complex<double> calpha(1.0,0.0),cbeta(0.0,0.0);
            bool is_mixed_precision = false;
            if(is_mixed_precision)
            {
                MatrixDevice<std::complex<float>> d_coul_block_f(coul_block.nr(),coul_block.nc(),device_stream.stream);
                MatrixDevice<std::complex<float>> d_chi0_block_f(chi0_block.nr(),chi0_block.nc(),device_stream.stream);
                MatrixDevice<std::complex<float>> d_coul_chi0_block_f(coul_chi0_block.nr(),coul_chi0_block.nc(),device_stream.stream);
                DeviceConnector::double_to_float_device((double*)d_coul_block.ptr(),(float*)d_coul_block_f.ptr(),d_coul_block.nr()*d_coul_block.nc()*2);
                DeviceConnector::double_to_float_device((double*)d_chi0_block.ptr(),(float*)d_chi0_block_f.ptr(),d_chi0_block.nr()*d_chi0_block.nc()*2);
                DeviceConnector::double_to_float_device((double*)d_coul_chi0_block.ptr(),(float*)d_coul_chi0_block_f.ptr(),d_coul_chi0_block.nr()*d_coul_chi0_block.nc()*2);
                std::complex<float> calpha_f(1.0f,0.0f),cbeta_f(0.0f,0.0f);
                DeviceConnector::pgemm_device_mixed_precision(
                    'N', 'N', n_abf, n_abf, n_abf,
                    &calpha_f,
                    d_coul_block_f.ptr(), 1, 1, desc_nabf_nabf,
                    d_chi0_block_f.ptr(), 1, 1, desc_nabf_nabf,
                    &cbeta_f,
                    d_coul_chi0_block_f.ptr(), 1, 1, desc_nabf_nabf,
                    LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT
                );
                DeviceConnector::float_to_double_device((float*)d_coul_chi0_block_f.ptr(),(double*)d_coul_chi0_block.ptr(),d_coul_chi0_block.nr()*d_coul_chi0_block.nc()*2);
            }else{
                DeviceConnector::pgemm_device_mixed_precision(
                    'N', 'N', n_abf, n_abf, n_abf,
                    &calpha,
                    d_coul_block.ptr(), 1, 1, desc_nabf_nabf,
                    d_chi0_block.ptr(), 1, 1, desc_nabf_nabf,
                    &cbeta,
                    d_coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf,
                    LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE
                );
            }    
            double pi_end = omp_get_wtime();
            complex<double> trace_pi(0.0, 0.0);
            complex<double> trace_pi_loc(0.0, 0.0);
            DeviceConnector::trace_matrix_device_blacs(&trace_pi_loc,d_coul_chi0_block.ptr(),desc_nabf_nabf,LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE);
            cuDoubleComplex calpha1;
            calpha1.x = -1.0;
            calpha1.y = 0.0;
            DeviceConnector::num_multiply_matrix_device(d_coul_chi0_block.nr()*d_coul_chi0_block.nc(),&calpha1,d_coul_chi0_block.ptr(),LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE);
            
            DeviceConnector::diag_add_matrix_device_blacs(&calpha,d_coul_chi0_block.ptr(),desc_nabf_nabf, LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE);
            
            int64_t *d_ipiv;
            int *d_info ;
            CUDA_CHECK(cudaMallocAsync((void**)&d_ipiv, sizeof(int64_t)*max(desc_nabf_nabf.m_loc(), desc_nabf_nabf.n_loc()),device_stream.stream));
            CUDA_CHECK(cudaMallocAsync((void**)&d_info, sizeof(int),device_stream.stream));
            
            complex<double> ln_det =
                compute_pi_det_blacs_2d_nvhpc(d_coul_chi0_block, desc_nabf_nabf, d_ipiv, d_info, 'c');
            CUDA_CHECK(cudaFreeAsync(d_ipiv,device_stream.stream));
            CUDA_CHECK(cudaFreeAsync(d_info,device_stream.stream));

            double det_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            MPI_Allreduce(&trace_pi_loc, &trace_pi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                          mpi_comm_global_h.comm);
            double pi_freq_end = omp_get_wtime();
            if (mpi_comm_global_h.myid == 0)
            {
                lib_printf(
                    "| TIME of DET-freq-q:  %f,  q: ( %f, %f, %f)  TOT: %f  CHI_arr: %f  CHI_comm: "
                    "%f, CHI_2d: %f, Pi: %f, Det: %f\n",
                    freq, q.x, q.y, q.z, pi_freq_end - pi_freq_begin, chi_arr_time, chi_comm_time,
                    chi_2d_time, pi_end - pi_begin, det_end - pi_end);
                complex<double> rpa_for_omega_q = trace_pi + ln_det;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;  //! check
                tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
            }
        }
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    if (mpi_comm_global_h.myid == 0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
            // cout << q_crpa.first << q_crpa.second << endl;
        }
        // cout << "gx_num_" << chi0.tfg.size() << "  tot_RPA_energy:  " << setprecision(8)
        // <<tot_RPA_energy << endl;
    }
    mpi_comm_global_h.barrier();
    corr.value = tot_RPA_energy;

    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_RPA_correlation_cuda(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    if (mpi_comm_global_h.myid == 0) lib_printf("Calculating EcRPA without BLACS/ScaLAPACK(cuda)\n");
    // lib_printf("Begin cal cRPA , pid:  %d\n", mpi_comm_global_h.myid);
    const auto &mf = chi0.mf;

    // freq, q
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>>
        pi_freq_q_Mu_Nu;
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::ATOM_PAIR ||
        LIBRPA::parallel_routing == LIBRPA::ParallelRouting::LIBRI)
        pi_freq_q_Mu_Nu = compute_Pi_q_MPI(chi0, coulmat);
    else
        pi_freq_q_Mu_Nu = compute_Pi_q(chi0, coulmat);
    lib_printf("Finish Pi freq on Proc %4d, size %zu\n", mpi_comm_global_h.myid,
               pi_freq_q_Mu_Nu.size());
    // mpi_comm_global_h.barrier();

    int range_all = N_all_mu;

    vector<int> part_range;
    part_range.resize(atom_mu.size());
    part_range[0] = 0;
    int count_range = 0;
    
    for (int I = 0; I != atom_mu.size() - 1; I++)
    {
        count_range += atom_mu[I];
        part_range[I + 1] = count_range;
    }
    

    // pi_freq_q contains all atoms
    map<double, map<Vector3_Order<double>, ComplexMatrix>> pi_freq_q;
    
    for(const auto &freq : chi0.tfg.get_freq_nodes())
    {
        // printf("| process %d, freq: %f\n", mpi_comm_global_h.myid, freq);
        map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old> freq_q_MuNupi;
        if(!chi0.get_chi0_q().empty())
            freq_q_MuNupi=pi_freq_q_Mu_Nu.at(freq);
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        // printf("success before freq_q_MuNupi processid:%d, freq_q_MuNupi.size(): %zu\n",
        //        mpi_comm_global_h.myid, freq_q_MuNupi.size());
        #endif
        for(const auto &q:chi0.klist){
            atom_mapping<ComplexMatrix>::pair_t_old q_MuNupi;
            if(!chi0.get_chi0_q().empty())
                q_MuNupi = freq_q_MuNupi.at(q);
            const auto MuNupi = q_MuNupi;
            pi_freq_q[freq][q].create(range_all, range_all);

            ComplexMatrix pi_munu_tmp(range_all, range_all);
            pi_munu_tmp.zero_out();
            if(!chi0.get_chi0_q().empty())
            for (const auto &Mu_Nupi : MuNupi)
            {
                const auto Mu = Mu_Nupi.first;
                const auto Nupi = Mu_Nupi.second;
                const size_t n_mu = atom_mu[Mu];
                for (const auto &Nu_pi : Nupi)
                {
                    const auto Nu = Nu_pi.first;
                    const auto pimat = Nu_pi.second;
                    const size_t n_nu = atom_mu[Nu];

                    for (size_t mu = 0; mu != n_mu; ++mu)
                    {
                        for (size_t nu = 0; nu != n_nu; ++nu)
                        {
                            pi_munu_tmp(part_range[Mu] + mu, part_range[Nu] + nu) += pimat(mu, nu);
                        }
                    }
                }
            }
            if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::ATOM_PAIR ||
                LIBRPA::parallel_routing == LIBRPA::ParallelRouting::LIBRI)
            {
                mpi_comm_global_h.reduce_ComplexMatrix(pi_munu_tmp, pi_freq_q.at(freq).at(q), 0);
            }
            else
            {
                pi_freq_q.at(freq).at(q) = std::move(pi_munu_tmp);
            }
        }
    }
    // lib_printf("Finish Pi communicate %4d, size %zu\n", mpi_comm_global_h.myid,
    // pi_freq_q_Mu_Nu.size());
    mpi_comm_global_h.barrier();
    // if (mpi_comm_global_h.myid == 0)
    {
        complex<double> tot_RPA_energy(0.0, 0.0);
        map<Vector3_Order<double>, complex<double>> cRPA_q;
        int deviceCount;
        cudaError_t err= cudaGetDeviceCount(&deviceCount);
        
        // if(err==cudaSuccess&&deviceCount>0&&deviceCount!=4)
        //     printf("cudaSuccess:%d\n",err==cudaSuccess&&deviceCount>0);
        // ==============================test the velocity of cuda stream========================================
        
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
            range_all=TEST_LU_NUMBER;
            complex<double>* temp_complex=new complex<double>[range_all*range_all];
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            double temp_begin_time=omp_get_wtime();
            // #pragma omp parallel for
            for(int i=0;i<range_all*range_all;i++){
                temp_complex[i].real(dis(gen));
                temp_complex[i].imag(dis(gen));
            }
            printf("time for generate random complex matrix: %f\n",omp_get_wtime()-temp_begin_time);
            for (const auto &freq_qpi : pi_freq_q)
            {
                const auto freq = freq_qpi.first;
                for (const auto &q_pi : freq_qpi.second)
                {
                    const auto q= q_pi.first;
                    printf("test freq: %f, q: (%f, %f, %f)\n",freq,q.x,q.y,q.z);
                    pi_freq_q[freq][q].c=new complex<double>[range_all*range_all];
                    complex<double>* c_ptr=pi_freq_q[freq][q].c;
                    
                    double temp_begin_time=omp_get_wtime();
                    #pragma omp parallel for
                    for(int i=0;i<range_all*range_all;i++){
                        c_ptr[i]=temp_complex[i];
                        
                    }
                    printf("time for copy complex matrix: %f\n",omp_get_wtime()-temp_begin_time);
                }
            }
            delete[] temp_complex;
        #endif
        printf("the size of matrix:%d,%d\n",range_all,range_all);
        // ==============================end test, the data is a fault data, but it can be used to test the speed of cuda stream========================================

        int NUM_STREAMS = 4;
        printf("deviceCount:%d\n",deviceCount);
        cudaStream_t* streams=new cudaStream_t[deviceCount*NUM_STREAMS];
        int ngpu= deviceCount;
        magma_queue_t *queues=new magma_queue_t[ngpu];
        // create streams
        double start_time=omp_get_wtime();
        int max_range_all=60000;
        printf("range_all:%d, max_range_all:%d\n",range_all,max_range_all);
        if(range_all<max_range_all){
            // #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
            printf("success before create streams");
            // #endif
            for(int i=0;i<deviceCount;i++){
                cudaSetDevice(i);//set the device
                for(int j=0;j<NUM_STREAMS;j++){
                    cudaStreamCreate(&streams[i*NUM_STREAMS+j]);
                    printf("the stream was created,device:%d,stream:%d\n",i,j);
                }
            }
            // #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
            printf("success after create streams");
            // #endif
        }else{
            printf("the range_all is too large, LU decomposition will be done with the function magma_zgetrf_mgpu");
            magma_init();
            for( int dev = 0; dev < ngpu; ++dev ) {

                magma_queue_create( dev, &queues[dev] );
            }
        }
        // #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        printf("time for create streams: %f\n",omp_get_wtime()-start_time);
        // #endif
        int processSize;
        int processId= mpi_comm_global_h.myid;
        MPI_Comm_size(MPI_COMM_WORLD, &processSize);
        printf("processSize:%d, processId:%d\n",processSize,processId);
        int num_iteration=0;
        double time_calculate=omp_get_wtime();
        #ifdef OPEN_OMP_FOR_LU_DECOMPOSITION
        #pragma omp parallel
        #pragma omp single
        #endif
        {
        
        for (const auto &freq_qpi : pi_freq_q)
        {
            const auto freq = freq_qpi.first;
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            for (const auto &q_pi : freq_qpi.second)
            {
                // if(processId!=num_iteration%processSize){
                //     num_iteration++;
                //     continue;
                // }
                #ifndef OPEN_OMP_FOR_LU_DECOMPOSITION
                    double one_task_begin = omp_get_wtime();
                #endif
                const auto q = q_pi.first;
                const auto pimat = q_pi.second;
                // double test_startTime=omp_get_wtime();
                #ifdef OPEN_OMP_FOR_LU_DECOMPOSITION
                #pragma omp task firstprivate(num_iteration,freq,q,freq_weight) 
                #endif
                {
                    complex<double> trace_pi = trace(pi_freq_q.at(freq).at(q));
                    complex<double> rpa_for_omega_q(0.0, 0.0);
                    int info_LU = 1;
                    int *ipiv = new int[range_all];
                    cuDoubleComplex det_test;
                    if(range_all<max_range_all){
                        int deviceId=(num_iteration/NUM_STREAMS)%deviceCount;
                        int streamId=num_iteration%NUM_STREAMS;
                        cudaSetDevice(deviceId);//set the device
                        #ifndef OPEN_OMP_FOR_LU_DECOMPOSITION
                            printf("one task time after set device:%f\n",omp_get_wtime()-one_task_begin);
                        #endif
                        cudaStream_t stream=streams[deviceId*NUM_STREAMS+streamId];
                        cusolverDnHandle_t handle;
                        cusolverDnCreate(&handle);
                        cusolverDnSetStream(handle, stream);
                        cuDoubleComplex* h_pi=(cuDoubleComplex*)pi_freq_q[freq][q].c;
                        // // show h_pi
                        // printf("h_pi matrix:\n");
                        // for(int i=0;i<range_all;i++){
                        //     for(int j=0;j<range_all;j++){
                        //         printf("%f+%fi ",h_pi[i*range_all+j].x,h_pi[i*range_all+j].y);
                        //     }
                        //     printf("\n");
                        // }
                        det_test=CudaConnector::det_cuZgetrf_f_from_host(range_all,range_all,(cuDoubleComplex*)pi_freq_q[freq][q].c,range_all,ipiv,&info_LU,handle,deviceId);
                        cusolverDnDestroy(handle);
                        // printf("iteration:%d",num_iteration);
                        // printf("det_test.x:%f,det_test.y:%f,info_LU:%d\n",det_test.x,det_test.y,info_LU);
                        
                    }else{
                        // magma_init();
                        // magma_queue_t* queues = new magma_queue_t[ngpu];
                        // for( int dev = 0; dev < ngpu; ++dev ) {
                        //     magma_setdevice( dev );
                        //     magma_queue_create( dev, &queues[dev] );
                        // }
                        // cuDoubleComplex* h_pi=new cuDoubleComplex[range_all*range_all];
                        // for(int i=0;i<range_all*range_all;i++){
                        //     h_pi[i].x = pi_freq_q[freq][q].c[i].real();
                        //     h_pi[i].y = pi_freq_q[freq][q].c[i].imag();
                        // }
                        det_test=CudaConnector::det_magmaZgetrf_f_mgpu_from_host(range_all,range_all,(cuDoubleComplex*)pi_freq_q[freq][q].c,range_all,ipiv,&info_LU,ngpu,queues);
                        // for( int dev = 0; dev < ngpu; ++dev ) {
                        //     magma_setdevice( dev );
                        //     magma_queue_destroy( queues[dev] );
                        // }
                        // printf("iteration:%d",num_iteration);
                        // printf("det_test.x:%f,det_test.y:%f,info_LU:%d\n",det_test.x,det_test.y,info_LU);
                        // delete[] queues;
                        // delete[] h_pi;
                        // magma_finalize();
                    }
                    if (info_LU != 0)
                    {
                        printf("Error in LU decomposition, info_LU: %d\n", info_LU);
                        exit(1);
                    }
                    // printf("ipiv:");
                    // for(int i=0;i<range_all;i++)
                    //     printf("%d ",ipiv[i]);
                    // printf("\n");
                    
                    delete[] ipiv;
                    if(range_all%2==1){
                        det_test.x=-det_test.x;
                        det_test.y=-det_test.y;
                    }
                    complex<double> det_for_rpa(det_test.x, det_test.y);
                    // printf("det_for_rpa_gpu: %f+%fi\n",det_test.x,det_test.y);
                    
                    // auto end_time = std::chrono::high_resolution_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                    // printf("LU time by lapack: %lld us\n", duration.count());
                    // complex<double> trace_pi;
                    complex<double> ln_det;
                    ln_det = std::log(det_for_rpa);
                    
                    // printf("in_det: %f+%fi, trace_pi: %f+%fi\n", ln_det.real(), ln_det.imag(),
                    //        trace_pi.real(), trace_pi.imag());
                    // cout << "PI trace vector:" << endl;
                    // cout << endl;
                    rpa_for_omega_q = ln_det + trace_pi;
                    // cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "
                    // lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                    // printf("tot_RPA_energy_gpu: %f+%fi,num_iteration:%d\n",
                    //        tot_RPA_energy.real(), tot_RPA_energy.imag(), num_iteration);
                    // printf("rpa_for_omega_q:%f+%fi,freq_weight:%f,irk_weight[q]:%f\n",
                    //        rpa_for_omega_q.real(), rpa_for_omega_q.imag(), freq_weight,
                    //        irk_weight[q]);
                    #ifdef OPEN_OMP_FOR_LU_DECOMPOSITION
                    #pragma omp critical
                    #endif
                    {
                        cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
                    
                        tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
                    }
                    // printf("freq: %f, q: (%f, %f, %f), rpa_for_omega_q: %f+%fi,tot_RPA_energy_gpu: %f+%fi,num_iteration:%d,deviceId:%d,streamId:%d\n",
                    //        freq, q.x, q.y, q.z, rpa_for_omega_q.real(), rpa_for_omega_q.imag(),tot_RPA_energy.real(), tot_RPA_energy.imag(), num_iteration, deviceId, streamId);
                    
                    // printf("tot_RPA_energy_gpu: %f+%fi,num_iteration:%d\n",tot_RPA_energy.real(),tot_RPA_energy.imag(),num_iteration);
                }
                // double test_endTime=omp_get_wtime();
                // printf("task_time:%f(aimed to test whether task is congested)\n",test_endTime-test_startTime);
                num_iteration++;
                #ifndef OPEN_OMP_FOR_LU_DECOMPOSITION
                    printf("one task time:%f\n",omp_get_wtime()-one_task_begin);
                #endif
            }
        }
        
        }
        printf("time for calculate: %f\n",omp_get_wtime()-time_calculate);
        printf("mpi_comm_global_h.myid:%d,num_iteration:%d\n",mpi_comm_global_h.myid,num_iteration);
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        // printf("time for all tasks: %f\n",omp_get_wtime()-start_time);
        #endif
        printf("tot_RPA_energy_gpu: %f+%fi\n",tot_RPA_energy.real(),tot_RPA_energy.imag());
        if(range_all<max_range_all){
            for(int i=0;i<deviceCount;i++){
                cudaSetDevice(i);//set the device
                for(int j=0;j<NUM_STREAMS;j++){
                    cudaStreamDestroy(streams[i*NUM_STREAMS+j]);
                }
            }
        }else{
            for( int dev = 0; dev < ngpu; ++dev ) {
                magma_queue_destroy( queues[dev] );
            }
            magma_finalize();
        }
        double end_time=omp_get_wtime();
        printf("time=%f\n", end_time-start_time);
        delete[] queues;
        delete[] streams;
        
        // lib_printf("Finish EcRPA %4d, size %zu\n", mpi_comm_global_h.myid,
        // pi_freq_q_Mu_Nu.size());
        mpi_comm_global_h.barrier();
        map<Vector3_Order<double>, complex<double>> global_cRPA_q;
        for (auto q_weight : irk_weight)
        {
            MPI_Reduce(&cRPA_q[q_weight.first], &global_cRPA_q[q_weight.first], 1,
                       MPI_DOUBLE_COMPLEX, MPI_SUM, 0, mpi_comm_global_h.comm);
        }

        for (auto &q_crpa : global_cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
        }
        complex<double> gather_tot_RPA_energy(0.0, 0.0);
        MPI_Reduce(&tot_RPA_energy, &gather_tot_RPA_energy, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0,
                   mpi_comm_global_h.comm);
        corr.value = gather_tot_RPA_energy;
    }
    // printf("gather_tot_RPA_energy_gpu: %f+%fi\n",corr.value.real(),corr.value.imag());
    corr.etype = CorrEnergy::type::RPA;
    return corr;
}
#ifdef ENABLE_NVHPC

complex<double> compute_pi_det_blacs_2d_nvhpc(
    MatrixDevice<std::complex<double>> &d_A, const LIBRPA::Array_Desc &arrdesc_pi, int64_t *d_ipiv, int *d_info,char order)
{
    MatrixDevice<std::complex<double>> d_A_T;
    
    ORDER_CHECK(order);  
    if(order=='C'||order=='c'){
        d_A_T.set_data(d_A.nc(), d_A.nr(), device_stream.stream);
        DeviceConnector::transpose_device_blas(d_A.ptr(),d_A_T.nr(), d_A_T.nc(),d_A_T.ptr(), LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE);
    }
    int ia=1,ja=1;
    bool is_mixed_precision=false;
    if(is_mixed_precision){
        MatrixDevice<std::complex<float>> d_A_f(d_A.nr(),d_A.nc(),device_stream.stream);
        
        DeviceConnector::double_to_float_device((order=='C'||order=='c')?(double*)d_A_T.ptr():(double*)d_A.ptr(),(float*)d_A_f.ptr(),d_A.nr()*d_A.nc()*2);
        
        DeviceConnector::pgetrf_device_mixed_precision(
            d_A_f.ptr(), ia, ja, arrdesc_pi,
            d_ipiv, d_info,
            LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT,
            order
        );
        DeviceConnector::float_to_double_device((float*)d_A_f.ptr(),(order=='C'||order=='c')?(double*)d_A_T.ptr():(double*)d_A.ptr(),d_A.nr()*d_A.nc()*2);
    }else{
        DeviceConnector::pgetrf_device_mixed_precision(
            (order=='C'||order=='c')?(void*)d_A_T.ptr():(void*)d_A.ptr(), ia, ja, arrdesc_pi, 
            d_ipiv, d_info, 
            LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE, 
            order);
    }
    if(order=='C'||order=='c'){
        DeviceConnector::transpose_device_blas(d_A_T.ptr(),d_A.nr(),d_A.nc(), d_A.ptr(), LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE);
        d_A_T.clean(device_stream.stream);
    }
    complex<double> ln_det_loc(0.0, 0.0);
    complex<double> ln_det_all(0.0, 0.0);
    std::complex<double> det_loc;
    
    DeviceConnector::det_matrix_device_blacs(
        &det_loc, d_A.ptr(), arrdesc_pi, LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE
    );
    if(det_loc.real() > 0)
    {
        ln_det_loc = std::log(det_loc);
    }
    else
    {
        ln_det_loc = std::log(-det_loc);
    }
    
    MPI_Allreduce(&ln_det_loc, &ln_det_all, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_global_h.comm);
    return ln_det_all;
}
// Done: converge compute_Wc_freq_q_blacs and compute_Wc_freq_q_blacs_wing
map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
compute_Wc_freq_q_blacs_cuda(Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps,
                        atpair_k_cplx_mat_t &coulmat_wc,
                        const vector<std::complex<double>> &epsmac_LF_imagfreq)
{
    map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_freq_q;
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    if (mpi_comm_global_h.myid == 0)
    {
        cout << "Calculating Wc using NVHPC" << endl;
    }
    mpi_comm_global_h.barrier();

    Profiler::start("compute_Wc_freq_q_blacs_init");
    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    // Use a square blocksize instead max block, otherwise heev and inversion will complain about
    // illegal parameter Maximal blocksize ensure that atom indices related to the rows/columns of a
    // local matrix is minimized.
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    // This, however, is not optimal for matrix operations, and may lead to segment fault during
    // MPI operations with parallel linear algebra subroutine. Thus we define an optimal blocksize
    Array_Desc desc_nabf_nabf_opt(blacs_ctxt_global_h);
    const int nb_opt = min(128, desc_nabf_nabf.nb());
    desc_nabf_nabf_opt.init(n_abf, n_abf, nb_opt, nb_opt, 0, 0);
    // obtain the indices of atom-pair block necessary to build 2D block of a Hermitian/symmetric
    // matrix
    const auto set_IJ_nabf_nabf = LIBRPA::utils::get_necessary_IJ_from_block_2D_sy(
        'U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    // temp_block is used to collect data from IJ-pair data structure with comm_map2_first
    auto temp_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    #ifdef ENABLE_NVHPC
    GpuDeviceStream gpu_dev_stream;
    ComplexMatrixDevice d_temp_block;
    ComplexMatrixDevice d_coul_block;
    #endif
    // Below are the working arrays for matrix operations
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_eigen_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    #ifdef ENABLE_NVHPC
    ComplexMatrixDevice d_chi0_block;
    ComplexMatrixDevice d_coul_chi0_block;
    ComplexMatrixDevice d_coul_eigen_block;
    #endif
    auto coulwc_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    #ifdef ENABLE_NVHPC
    ComplexMatrixDevice d_coulwc_block;
    #endif

    const double mem_blocks = (chi0_block.size() + coul_block.size() + coul_eigen_block.size() +
                               coul_chi0_block.size() + coulwc_block.size()) *
                              16.0e-6;
    ofs_myid << get_timestamp()
             << " Memory consumption of task-local blocks for screened Coulomb [MB]: " << mem_blocks
             << endl;

    const auto atpair_local = dispatch_upper_trangular_tasks(
        natom, blacs_ctxt_global_h.myid, blacs_ctxt_global_h.nprows, blacs_ctxt_global_h.npcols,
        blacs_ctxt_global_h.myprow, blacs_ctxt_global_h.mypcol);
#ifdef LIBRPA_DEBUG
    ofs_myid << get_timestamp() << " atpair_local " << atpair_local << endl;
    ofs_myid << get_timestamp() << " s0_s1 " << s0_s1 << endl;
#endif

    // IJ pair of Wc to be returned
    pair<set<int>, set<int>> Iset_Jset_Wc;
    for (const auto &ap : atpair_local)
    {
        Iset_Jset_Wc.first.insert(ap.first);
        Iset_Jset_Wc.second.insert(ap.second);
    }

    // Prepare local basis indices for 2D->IJ map
    int I, iI;
    map<int, vector<int>> map_lor_v;
    map<int, vector<int>> map_loc_v;
    for (int i_lo = 0; i_lo != desc_nabf_nabf.m_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf.indx_l2g_r(i_lo);
        LIBRPA::atomic_basis_abf.get_local_index(i_glo, I, iI);
        map_lor_v[I].push_back(iI);
    }
    for (int i_lo = 0; i_lo != desc_nabf_nabf.n_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf.indx_l2g_c(i_lo);
        LIBRPA::atomic_basis_abf.get_local_index(i_glo, I, iI);
        map_loc_v[I].push_back(iI);
    }

    vector<Vector3_Order<double>> qpts;
    for (const auto &q_weight : irk_weight) qpts.push_back(q_weight.first);

    vec<double> eigenvalues(n_abf);
    Profiler::cease("compute_Wc_freq_q_blacs_init");
    LIBRPA::utils::lib_printf_root("Time for Wc initialization (seconds, Wall/CPU): %f %f\n",
                                   Profiler::get_wall_time_last("compute_Wc_freq_q_blacs_init"),
                                   Profiler::get_cpu_time_last("compute_Wc_freq_q_blacs_init"));

    Profiler::start("compute_Wc_freq_q_work");
#ifdef LIBRPA_USE_LIBRI
    for (const auto &q : qpts)
    {
        const int iq = std::distance(qpts.cbegin(), std::find(qpts.cbegin(), qpts.cend(), q));
        const int iq_in_k =
            std::distance(klist.cbegin(), std::find(klist.cbegin(), klist.cend(), q));
        // q-point in fractional coordinates
        const auto &qf = kfrac_list[iq_in_k];
        LIBRPA::utils::lib_printf_root("Computing Wc(q), %d / %d, q=(%f, %f, %f)\n", iq + 1,
                                       qpts.size(), qf.x, qf.y, qf.z);
        coul_block.zero_out();
        coulwc_block.zero_out();
        // lib_printf("coul_block\n%s", str(coul_block).c_str());

        // q-array for LibRI object
        std::array<double, 3> qa = {q.x, q.y, q.z};

        // collect the block elements of truncated coulomb matrices first
        // as we reuse coul_eigen_block to reduce memory usage
        Profiler::start("epsilon_prepare_coulwc_sqrt", "Prepare sqrt of truncated Coulomb");
        {
            size_t n_singular_coulwc;
            // LibRI tensor for communication, release once done
            std::map<int,
                     std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
                couleps_libri;
            Profiler::start("epsilon_prepare_coulwc_sqrt_1", "Setup libRI object");
            for (const auto &Mu_Nu : atpair_local)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                // ofs_myid << "Mu " << Mu << " Nu " << Nu << endl;
                if (coulmat_wc.count(Mu) == 0 || coulmat_wc.at(Mu).count(Nu) == 0 ||
                    coulmat_wc.at(Mu).at(Nu).count(q) == 0)
                    continue;
                const auto &Vq = coulmat_wc.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            Profiler::stop("epsilon_prepare_coulwc_sqrt_1");

            Profiler::start("epsilon_prepare_coulwc_sqrt_2", "libRI Communicate");
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                mpi_comm_global_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
            Profiler::stop("epsilon_prepare_coulwc_sqrt_2");

            Profiler::start("epsilon_prepare_coulwc_sqrt_3", "Collect 2D-block from IJ");

            collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                          coulwc_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                          blacs_ctxt_global_h.ictxt);
            Profiler::stop("epsilon_prepare_coulwc_sqrt_3");
            Profiler::start("epsilon_prepare_coulwc_sqrt_4", "Perform square root");
            power_hemat_blacs(coulwc_block, desc_nabf_nabf_opt, coul_eigen_block,
                              desc_nabf_nabf_opt, n_singular_coulwc, eigenvalues.c, 0.5,
                              Params::sqrt_coulomb_threshold);
            Profiler::stop("epsilon_prepare_coulwc_sqrt_4");
        }
        Profiler::stop("epsilon_prepare_coulwc_sqrt");
        LIBRPA::utils::lib_printf_root(
            "Time to prepare sqrt root of Coulomb for Wc(q) (seconds, Wall/CPU): %f %f\n",
            Profiler::get_wall_time_last("epsilon_prepare_coulwc_sqrt"),
            Profiler::get_cpu_time_last("epsilon_prepare_coulwc_sqrt"));
        ofs_myid << get_timestamp() << " Done coulwc sqrt" << endl;

        Profiler::start("epsilon_prepare_couleps_sqrt", "Prepare sqrt of bare Coulomb");
        // collect the block elements of coulomb matrices
        {
            // LibRI tensor for communication, release once done
            std::map<int,
                     std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
                couleps_libri;
            ofs_myid << get_timestamp() << " Start build couleps_libri" << endl;
            for (const auto &Mu_Nu : atpair_local)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                // ofs_myid << "Mu " << Mu << " Nu " << Nu << endl;
                if (coulmat_eps.count(Mu) == 0 || coulmat_eps.at(Mu).count(Nu) == 0 ||
                    coulmat_eps.at(Mu).at(Nu).count(q) == 0)
                    continue;
                const auto &Vq = coulmat_eps.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            ofs_myid << get_timestamp() << " Done build couleps_libri" << endl;
            // ofs_myid << "Couleps_libri" << endl << couleps_libri;
            // if (couleps_libri.size() == 0)
            //     throw std::logic_error("data at q-point not found in coulmat_eps");

            // perform communication
            ofs_myid << get_timestamp() << " Start collect couleps_libri, targets" << endl;
#ifdef LIBRPA_DEBUG
            ofs_myid << set_IJ_nabf_nabf << endl;
            ofs_myid << "Extended blocks" << endl;
            ofs_myid << "atom 1: " << s0_s1.first << endl;
            ofs_myid << "atom 2: " << s0_s1.second << endl;
#endif
            // ofs_myid << "Owned blocks\n";
            // print_keys(ofs_myid, couleps_libri);
            // mpi_comm_global_h.barrier();
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                mpi_comm_global_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
            ofs_myid << get_timestamp() << " Done collect couleps_libri, collected blocks" << endl;

            ofs_myid << get_timestamp() << " Start construct couleps 2D block" << endl;
            collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);
            #ifndef ENABLE_NVHPC
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                          coul_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                          blacs_ctxt_global_h.ictxt);
            #else
            d_temp_block.set_data(temp_block.nr(),temp_block.nc(),temp_block.ptr());
            d_coul_block.set_data(coul_block.nr(),coul_block.nc());
            CudaConnector::pgemr2d_nvhpc(
                gpu_dev_stream, n_abf, n_abf,
                d_temp_block.ptr(), 1, 1, desc_nabf_nabf,
                d_coul_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                CUDA_C_64F
            );
            gpu_dev_stream.cudaSync();
            CUDA_CHECK(cudaMemcpy(coul_block.ptr(), d_coul_block.ptr(), sizeof(cuDoubleComplex)*coul_block.nr()*coul_block.nc(), cudaMemcpyDeviceToHost));
            #endif
            ofs_myid << get_timestamp() << " Done construct couleps 2D block" << endl;
        }
        
        size_t n_singular;
        ofs_myid << get_timestamp() << " Start power hemat couleps\n";
        matrix_m<std::complex<double>> sqrtveig_blacs;
        #ifdef ENABLE_NVHPC
        ComplexMatrixDevice d_sqrtveig_blacs;
        #endif
        if (is_gamma_point(q))
        {
            // choice of power_hemat_blacs_real/power_hemat_blacs_desc
            // leads to sub-meV difference
            sqrtveig_blacs = power_hemat_blacs_real(
                coul_block, desc_nabf_nabf_opt, coul_eigen_block, desc_nabf_nabf_opt, n_singular,
                eigenvalues.c, 0.5, Params::sqrt_coulomb_threshold);
            if (Params::replace_w_head && Params::option_dielect_func == 3)
            {
                df_headwing.wing_mu_to_lambda(sqrtveig_blacs, desc_nabf_nabf_opt);
            }
        }
        else
        {
            sqrtveig_blacs = power_hemat_blacs(coul_block, desc_nabf_nabf_opt, coul_eigen_block,
                                               desc_nabf_nabf_opt, n_singular, eigenvalues.c, 0.5,
                                               Params::sqrt_coulomb_threshold);
        }
        ofs_myid << get_timestamp() << " Done power hemat couleps\n";
        // lib_printf("nabf %d nsingu %lu\n", n_abf, n_singular);
        // release sqrtv when the q-point is not Gamma, or macroscopic dielectric constant at
        // imaginary frequency is not prepared
        if (epsmac_LF_imagfreq.empty() || !is_gamma_point(q)) sqrtveig_blacs.clear();
        const size_t n_nonsingular = n_abf - n_singular;
        if(gpu_dev_stream.rank==0){
            printf("n_abf:%lu,n_nonsingular:%lu,n_singular:%lu\n",n_abf,n_nonsingular,n_singular);
        }
        Profiler::stop("epsilon_prepare_couleps_sqrt");
        LIBRPA::utils::lib_printf_root(
            "Time to prepare sqrt root of Coulomb for Epsilon(q) (seconds, Wall/CPU): %f %f\n",
            Profiler::get_wall_time_last("epsilon_prepare_couleps_sqrt"),
            Profiler::get_cpu_time_last("epsilon_prepare_couleps_sqrt"));
        ofs_myid << get_timestamp() << " Done couleps sqrt\n";
        std::flush(ofs_myid);

        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            Profiler::start("epsilon_wc_work_q_omega");
            Profiler::start("epsilon_prepare_chi0_2d", "Prepare Chi0 2D block");
            chi0_block.zero_out();
            {
                std::map<int, std::map<std::pair<int, std::array<double, 3>>,
                                       RI::Tensor<complex<double>>>>
                    chi0_libri;
                if (chi0.get_chi0_q().count(freq) > 0 && chi0.get_chi0_q().at(freq).count(q) > 0)
                {
                    const auto &chi0_wq = chi0.get_chi0_q().at(freq).at(q);
                    for (const auto &M_Nchi : chi0_wq)
                    {
                        const auto &M = M_Nchi.first;
                        const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                        for (const auto &N_chi : M_Nchi.second)
                        {
                            const auto &N = N_chi.first;
                            const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                            const auto &chi = N_chi.second;
                            std::valarray<complex<double>> chi_va(chi.c, chi.size);
                            auto pchi = std::make_shared<std::valarray<complex<double>>>();
                            *pchi = chi_va;
                            chi0_libri[M][{N, qa}] =
                                RI::Tensor<complex<double>>({n_mu, n_nu}, pchi);
                        }
                    }
                    // Release the chi0 block for this frequency and q to reduce memory load,
                    // as they will not be used again
                    chi0.free_chi0_q(freq, q);
                }
                // ofs_myid << "chi0_libri" << endl << chi0_libri;
                Profiler::start("epsilon_prepare_chi0_2d_comm_map2");
                const auto IJq_chi0 = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                    mpi_comm_global_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                Profiler::stop("epsilon_prepare_chi0_2d_comm_map2");
                Profiler::start("epsilon_prepare_chi0_2d_collect_block");
                collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf,
                                                 LIBRPA::atomic_basis_abf, qa, true, CONE, IJq_chi0,
                                                 MAJOR::ROW);
                #ifndef ENABLE_NVHPC
                ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1,
                                              desc_nabf_nabf.desc, chi0_block.ptr(), 1, 1,
                                              desc_nabf_nabf_opt.desc, blacs_ctxt_global_h.ictxt);
                #else
                d_temp_block.set_data(temp_block.nr(),temp_block.nc(),temp_block.ptr());
                d_chi0_block.set_data(chi0_block.nr(),chi0_block.nc());
                CudaConnector::pgemr2d_nvhpc(
                    gpu_dev_stream, n_abf, n_abf,
                    d_temp_block.ptr(), 1, 1, desc_nabf_nabf,
                    d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                    CUDA_C_64F
                );
                #endif
                Profiler::stop("epsilon_prepare_chi0_2d_collect_block");
            }
            Profiler::stop("epsilon_prepare_chi0_2d");

            Profiler::start("epsilon_compute_eps", "Compute dielectric matrix");
            if(gpu_dev_stream.rank==0)
                printf("is_gamma_point(q):%d\n",is_gamma_point(q));
            const std::complex<double> calpha(1.0,0.0),cbeta(0.0,0.0);
            if (epsmac_LF_imagfreq.size() > 0 && is_gamma_point(q))
            {
                ofs_myid << get_timestamp() << " Entering dielectric matrix head overwrite" << endl;
                // rotate to Coulomb-eigenvector basis
                // descending order
                
                d_sqrtveig_blacs.set_data(sqrtveig_blacs.nr(), sqrtveig_blacs.nc(), sqrtveig_blacs.ptr());
                d_coul_chi0_block.set_data(coul_chi0_block.nr(), coul_chi0_block.nc());
                gpu_dev_stream.cudaSync();
                CudaConnector::pgemm_nvhpc(
                    gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_nonsingular, n_abf,
                    &calpha,
                    d_chi0_block,1,1,desc_nabf_nabf_opt,
                    d_sqrtveig_blacs,1,1,desc_nabf_nabf_opt,
                    &cbeta,
                    d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                    CUBLAS_COMPUTE_64F_PEDANTIC);
                
                CudaConnector::pgemm_nvhpc(
                    gpu_dev_stream,CUBLAS_OP_C,CUBLAS_OP_N,n_nonsingular, n_nonsingular, n_abf,
                    &calpha,
                    d_sqrtveig_blacs,1,1,desc_nabf_nabf_opt,
                    d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                    &cbeta,
                    d_chi0_block,1,1,desc_nabf_nabf_opt,
                    CUBLAS_COMPUTE_64F_PEDANTIC);
                
                if (Params::option_dielect_func == 3)
                {
                    
                    cuDoubleComplex calpha1;
                    calpha1.x = -1.0;
                    calpha1.y = 0.0;
                    CudaConnector::multiply_number_for_ComplexMatrixDevice(d_chi0_block,calpha1,gpu_dev_stream.stream);
                    CudaConnector::diag_add_ComplexMatrixDevice(d_chi0_block,1.0,desc_nabf_nabf_opt,gpu_dev_stream.stream);
                    gpu_dev_stream.cudaSync();
                    // {
                    //     std::string filename = "gpu_";
                    //     filename += to_string(gpu_dev_stream.nranks);
                    //     filename += "_";
                    //     filename += to_string(gpu_dev_stream.rank);
                    //     gpu_dev_stream.cudaSync();
                    //     CUDA_CHECK(cudaMemcpy(coul_chi0_block.ptr(),d_chi0_block.ptr(),sizeof(cuDoubleComplex)*coul_chi0_block.nr()*coul_chi0_block.nc(),cudaMemcpyDeviceToHost));
                    //     CudaConnector::write_file((cuDoubleComplex*)coul_chi0_block.ptr(),coul_chi0_block.nr(),coul_chi0_block.nc(),filename.data());
                    // }
                    // CUDA_CHECK(cudaMemcpy(chi0_block.ptr(),d_chi0_block.ptr(),sizeof(cuDoubleComplex)*chi0_block.nr()*chi0_block.nc(),cudaMemcpyDeviceToHost));
                    
                    ofs_myid << get_timestamp() << "Perform the head & wing element overwrite"
                             << endl;
                    // df_headwing.rewrite_eps(chi0_block, ifreq, desc_nabf_nabf_opt);
                    df_headwing.rewrite_eps_nvhpc(gpu_dev_stream, d_chi0_block, ifreq, desc_nabf_nabf_opt);
                    // d_chi0_block.set_data(chi0_block.nr(), chi0_block.nc(), chi0_block.ptr());
                    
                    d_coul_eigen_block.set_data(coul_eigen_block.nr(), coul_eigen_block.nc(), coul_eigen_block.ptr());
                    // rotate back to ABF
                    // descending order
                    CudaConnector::pgemm_nvhpc(
                        gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_nonsingular, n_nonsingular,
                        &calpha,
                        d_coul_eigen_block,1,1,desc_nabf_nabf_opt,
                        d_chi0_block,1,1,desc_nabf_nabf_opt,
                        &cbeta,
                        d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                        CUBLAS_COMPUTE_64F_PEDANTIC);
                    
                    CudaConnector::pgemm_nvhpc(
                        gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_C,n_abf, n_abf, n_nonsingular,
                        &calpha,
                        d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                        d_coul_eigen_block,1,1,desc_nabf_nabf_opt,
                        &cbeta,
                        d_chi0_block,1,1,desc_nabf_nabf_opt,
                        CUBLAS_COMPUTE_64F_PEDANTIC);
                    // subtract 1 from diagonal
                    CudaConnector::diag_add_ComplexMatrixDevice(d_chi0_block,-1.0,desc_nabf_nabf_opt,gpu_dev_stream.stream);
                    Profiler::start("epsilon_multiply_coulwc", "Multiply truncated Coulomb");
                    d_coulwc_block.set_data(coulwc_block.nr(), coulwc_block.nc(), coulwc_block.ptr());
                    d_coul_chi0_block.set_data(coul_chi0_block.nr(), coul_chi0_block.nc());
                    CudaConnector::pgemm_nvhpc(
                        gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_abf, n_abf,
                        &calpha,
                        d_coulwc_block,1,1,desc_nabf_nabf_opt,
                        d_chi0_block,1,1,desc_nabf_nabf_opt,
                        &cbeta,
                        d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                        CUBLAS_COMPUTE_64F_PEDANTIC);
                    CudaConnector::pgemm_nvhpc(
                        gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_abf, n_abf,
                        &calpha,
                        d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                        d_coulwc_block,1,1,desc_nabf_nabf_opt,
                        &cbeta,
                        d_chi0_block,1,1,desc_nabf_nabf_opt,
                        CUBLAS_COMPUTE_64F_PEDANTIC);
                    Profiler::stop("epsilon_multiply_coulwc");
                }
                else
                {
                    const int ilo = desc_nabf_nabf_opt.indx_g2l_r(0);
                    const int jlo = desc_nabf_nabf_opt.indx_g2l_c(0);
                    if (ilo >= 0 && jlo >= 0)
                    {
                        ofs_myid << get_timestamp() << "Perform the head element overwrite" << endl;
                        std::complex<double> temp_element = 1.0 - epsmac_LF_imagfreq[ifreq];
                        CUDA_CHECK(cudaMemcpy(d_chi0_block.ptr() + ilo + jlo * d_chi0_block.nr(),
                                               &temp_element, sizeof(cuDoubleComplex),
                                               cudaMemcpyHostToDevice));
                    }
                    d_coul_eigen_block.set_data(coul_eigen_block.nr(), coul_eigen_block.nc(), coul_eigen_block.ptr());
                    CudaConnector::pgemm_nvhpc(
                        gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_nonsingular, n_nonsingular,
                        &calpha,
                        d_coul_eigen_block,1,1,desc_nabf_nabf_opt,
                        d_chi0_block,1,1,desc_nabf_nabf_opt,
                        &cbeta,
                        d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                        CUBLAS_COMPUTE_64F_PEDANTIC);
                    CudaConnector::pgemm_nvhpc(
                        gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_C,n_abf, n_abf, n_nonsingular,
                        &calpha,
                        d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                        d_coul_eigen_block,1,1,desc_nabf_nabf_opt,
                        &cbeta,
                        d_chi0_block,1,1,desc_nabf_nabf_opt,
                        CUBLAS_COMPUTE_64F_PEDANTIC);
                    
                    cuDoubleComplex calpha1;
                    calpha1.x = -1.0;
                    calpha1.y = 0.0;
                    CudaConnector::multiply_number_for_ComplexMatrixDevice(d_chi0_block,calpha1,gpu_dev_stream.stream);
                    CudaConnector::diag_add_ComplexMatrixDevice(d_chi0_block,1.0,desc_nabf_nabf_opt,gpu_dev_stream.stream);
                    gpu_dev_stream.cudaSync();
                    Profiler::start("epsilon_invert_eps and epsilon_multiply_coulwc", "Invert dielectric matrix and Multiply truncated Coulomb");
                    char order = 'c';
                    d_coulwc_block.set_data(coulwc_block.nr(), coulwc_block.nc(), coulwc_block.ptr(),gpu_dev_stream.stream);
                    d_coul_chi0_block.set_data_device(coulwc_block.nr(), coulwc_block.nc(), d_coulwc_block.ptr(),gpu_dev_stream);
                    int64_t* d_ipiv;
                    int* d_info;
                    CUDA_CHECK(cudaMallocAsync(&d_info,sizeof(int),gpu_dev_stream.stream));
                    if(order == 'c'||order == 'C'){
                        CUDA_CHECK(cudaMallocAsync(&d_ipiv,sizeof(int64_t)*desc_nabf_nabf_opt.n_loc(),gpu_dev_stream.stream));
                        CudaConnector::transpose_ComplexMatrixDevice(gpu_dev_stream,d_coul_chi0_block);
                        CudaConnector::transpose_ComplexMatrixDevice(gpu_dev_stream,d_chi0_block);
                    }else{
                        CUDA_CHECK(cudaMallocAsync(&d_ipiv,sizeof(int64_t)*desc_nabf_nabf_opt.m_loc(),gpu_dev_stream.stream));
                    }
                    CudaConnector::pgetrf_nvhpc_mixed_precision(
                        gpu_dev_stream, d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                        d_ipiv, d_info,
                        CUDA_C_64F, order
                    );
                    CudaConnector::pgetrs_nvhpc_mixed_precision(
                        gpu_dev_stream, CUBLAS_OP_N,
                        d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                        d_ipiv,
                        d_coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                        d_info,
                        CUDA_C_64F, order
                    );
                    if(order == 'c'||order == 'C'){
                        CudaConnector::transpose_ComplexMatrixDevice(gpu_dev_stream,d_coul_chi0_block);
                    }
                    CUDA_CHECK(cudaFreeAsync(d_info, gpu_dev_stream.stream));
                    CUDA_CHECK(cudaFreeAsync(d_ipiv, gpu_dev_stream.stream));
                    d_chi0_block.set_data_device(coulwc_block.nr(), coulwc_block.nc(), d_coulwc_block.ptr(), gpu_dev_stream);
                    
                    CudaConnector::pgeadd_nvhpc(
                        gpu_dev_stream, CUBLAS_OP_N,
                        &calpha, 
                        d_coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                        &calpha1,
                        d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                        CUDA_C_64F, order
                    );
                    d_coul_chi0_block.set_data_device(d_chi0_block.nr(), d_chi0_block.nc(), d_chi0_block.ptr(), gpu_dev_stream);
                    CudaConnector::pgemm_nvhpc(
                        gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_abf, n_abf,
                        &calpha,
                        (order=='r'||order=='R')?d_coulwc_block:d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                        (order=='r'||order=='R')?d_coul_chi0_block:d_coulwc_block,1,1,desc_nabf_nabf_opt,
                        &cbeta,
                        d_chi0_block,1,1,desc_nabf_nabf_opt,
                        CUBLAS_COMPUTE_64F_PEDANTIC);
                    Profiler::stop("epsilon_invert_eps and epsilon_multiply_coulwc");
                }
            }
            else
            {
                Profiler::start("epsilon_compute_eps_pgemm_1");
                // d_chi0_block.set_data(chi0_block.nr(), chi0_block.nc(), chi0_block.ptr());
                d_coul_block.set_data(coul_block.nr(), coul_block.nc(), coul_block.ptr());
                d_coul_chi0_block.set_data(coul_chi0_block.nr(), coul_chi0_block.nc());
                
                CudaConnector::pgemm_nvhpc(
                    gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_abf, n_abf,
                    &calpha,
                    d_coul_block,1,1,desc_nabf_nabf_opt,
                    d_chi0_block,1,1,desc_nabf_nabf_opt,
                    &cbeta,
                    d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                    CUBLAS_COMPUTE_64F_PEDANTIC);
                Profiler::cease("epsilon_compute_eps_pgemm_1");
                Profiler::start("epsilon_compute_eps_pgemm_2");
                CudaConnector::pgemm_nvhpc(
                    gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_abf, n_abf,
                    &calpha,
                    d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                    d_coul_block,1,1,desc_nabf_nabf_opt,
                    &cbeta,
                    d_chi0_block,1,1,desc_nabf_nabf_opt,
                    CUBLAS_COMPUTE_64F_PEDANTIC);
                // d_coul_block.cublasClean(gpu_dev_stream.cublas_handle);
                Profiler::cease("epsilon_compute_eps_pgemm_2");
                // now chi0_block is actually v1/2 chi v1/2
                cuDoubleComplex calpha1;
                calpha1.x = -1.0;
                calpha1.y = 0.0;
                CudaConnector::multiply_number_for_ComplexMatrixDevice(d_chi0_block,calpha1,gpu_dev_stream.stream);
                gpu_dev_stream.cudaSync();
                CudaConnector::diag_add_ComplexMatrixDevice(d_chi0_block,1.0,desc_nabf_nabf_opt,gpu_dev_stream.stream);
                Profiler::stop("epsilon_compute_eps");
                // now chi0_block is actually the dielectric matrix
                // perform inversion
                Profiler::start("epsilon_invert_eps and epsilon_multiply_coulwc", "Invert dielectric matrix and Multiply truncated Coulomb");
                char order = 'c';
                d_coulwc_block.set_data(coulwc_block.nr(), coulwc_block.nc(), coulwc_block.ptr(),gpu_dev_stream.stream);
                d_coul_chi0_block.set_data_device(coulwc_block.nr(), coulwc_block.nc(), d_coulwc_block.ptr(),gpu_dev_stream);
                int64_t* d_ipiv;
                int* d_info;
                CUDA_CHECK(cudaMallocAsync(&d_info,sizeof(int),gpu_dev_stream.stream));
                if(order == 'c'||order == 'C'){
                    CUDA_CHECK(cudaMallocAsync(&d_ipiv,sizeof(int64_t)*desc_nabf_nabf_opt.n_loc(),gpu_dev_stream.stream));
                    CudaConnector::transpose_ComplexMatrixDevice(gpu_dev_stream,d_coul_chi0_block);
                    CudaConnector::transpose_ComplexMatrixDevice(gpu_dev_stream,d_chi0_block);
                }else{
                    CUDA_CHECK(cudaMallocAsync(&d_ipiv,sizeof(int64_t)*desc_nabf_nabf_opt.m_loc(),gpu_dev_stream.stream));
                }
                CudaConnector::pgetrf_nvhpc_mixed_precision(
                    gpu_dev_stream, d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                    d_ipiv, d_info,
                    CUDA_C_64F, order
                );
                
                CudaConnector::pgetrs_nvhpc_mixed_precision(
                    gpu_dev_stream, CUBLAS_OP_N,
                    d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                    d_ipiv,
                    d_coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                    d_info,
                    CUDA_C_64F, order
                );
                if(order == 'c'||order == 'C'){
                    CudaConnector::transpose_ComplexMatrixDevice(gpu_dev_stream,d_coul_chi0_block);
                }
                CUDA_CHECK(cudaFreeAsync(d_info, gpu_dev_stream.stream));
                CUDA_CHECK(cudaFreeAsync(d_ipiv, gpu_dev_stream.stream));
                d_chi0_block.set_data_device(coulwc_block.nr(), coulwc_block.nc(), d_coulwc_block.ptr(), gpu_dev_stream);
                
                CudaConnector::pgeadd_nvhpc(
                    gpu_dev_stream, CUBLAS_OP_N,
                    &calpha, 
                    d_coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                    &calpha1,
                    d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                    CUDA_C_64F, order
                );
                d_coul_chi0_block.set_data_device(d_chi0_block.nr(), d_chi0_block.nc(), d_chi0_block.ptr(), gpu_dev_stream);
                CudaConnector::pgemm_nvhpc(
                    gpu_dev_stream,CUBLAS_OP_N,CUBLAS_OP_N,n_abf, n_abf, n_abf,
                    &calpha,
                    (order=='r'||order=='R')?d_coulwc_block:d_coul_chi0_block,1,1,desc_nabf_nabf_opt,
                    (order=='r'||order=='R')?d_coul_chi0_block:d_coulwc_block,1,1,desc_nabf_nabf_opt,
                    &cbeta,
                    d_chi0_block,1,1,desc_nabf_nabf_opt,
                    CUBLAS_COMPUTE_64F_PEDANTIC);
                Profiler::stop("epsilon_invert_eps and epsilon_multiply_coulwc");
            }
            // Array_Desc_Device array_desc_device(desc_nabf_nabf_opt);
            // printf("successful create object array_desc_device\n");
            #ifndef ENABLE_NVHPC
            gpu_dev_stream.calSync();
            CUDA_CHECK(cudaMemcpy(chi0_block.ptr(),d_chi0_block.ptr(),sizeof(cuDoubleComplex)*chi0_block.nr()*chi0_block.nc(),cudaMemcpyDeviceToHost));
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, chi0_block.ptr(), 1, 1,
                                          desc_nabf_nabf_opt.desc, temp_block.ptr(), 1, 1,
                                          desc_nabf_nabf.desc, blacs_ctxt_global_h.ictxt);
            #else
            d_temp_block.set_data(temp_block.nr(),temp_block.nc());
            CudaConnector::pgemr2d_nvhpc(
                gpu_dev_stream, n_abf, n_abf,
                d_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt,
                d_temp_block.ptr(), 1, 1, desc_nabf_nabf,
                CUDA_C_64F
            );
            CUDA_CHECK(cudaMemcpyAsync(temp_block.ptr(), d_temp_block.ptr(), sizeof(cuDoubleComplex)*temp_block.nr()*temp_block.nc(), cudaMemcpyDeviceToHost, gpu_dev_stream.stream));
            gpu_dev_stream.cudaSync();
            #endif
            
            Profiler::start("epsilon_convert_wc_2d_to_ij", "Convert Wc, 2D -> IJ");
            Profiler::start("epsilon_convert_wc_map_block", "Initialize Wc atom-pair map");
            map<int, map<int, matrix_m<complex<double>>>> Wc_MNmap;
            // map_block_to_IJ_storage(Wc_MNmap, LIBRPA::atomic_basis_abf,
            //                         LIBRPA::atomic_basis_abf, chi0_block,
            //                         desc_nabf_nabf, MAJOR::ROW);
            map_block_to_IJ_storage_new(Wc_MNmap, LIBRPA::atomic_basis_abf, map_lor_v, map_loc_v,
                                        temp_block, desc_nabf_nabf, MAJOR::ROW);
            Profiler::stop("epsilon_convert_wc_map_block");

            Profiler::start("epsilon_convert_wc_communicate", "Communicate");
            {
                std::map<int, std::map<std::pair<int, std::array<double, 3>>,
                                       RI::Tensor<complex<double>>>>
                    Wc_libri;
                Profiler::start("epsilon_convert_wc_communicate_1");
                for (const auto &M_NWc : Wc_MNmap)
                {
                    const auto &M = M_NWc.first;
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    for (const auto &N_Wc : M_NWc.second)
                    {
                        const auto &N = N_Wc.first;
                        const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                        const auto &Wc = N_Wc.second;
                        // std::valarray<complex<double>> Wc_va(Wc.ptr(), Wc.size());
                        // auto pWc = std::make_shared<std::valarray<complex<double>>>();
                        // *pWc = Wc_va;
                        /*if (iq == 10 && ifreq == 10)
                        {
                            char fn[100];
                            sprintf(fn, "Wc_M_%zu_N_%zu.dat", M, N);
                            print_matrix_mm_file(Wc, Params::output_dir + "/" + fn);
                        }*/
                        Wc_libri[M][{N, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, Wc.sptr());
                    }
                }
                Profiler::stop("epsilon_convert_wc_communicate_1");
                Profiler::start("epsilon_convert_wc_communicate_2");
                // main timing
                // cout << Wc_libri;
                const auto IJq_Wc = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                    mpi_comm_global_h.comm, Wc_libri, Iset_Jset_Wc.first, Iset_Jset_Wc.second);
                Profiler::stop("epsilon_convert_wc_communicate_2");
                Profiler::start("epsilon_convert_wc_communicate_3");
                // parse collected to
                for (const auto &MN : atpair_local)
                {
                    const auto &M = MN.first;
                    const auto &N = MN.second;
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                    // Use row major for later usage in LibRI
                    Wc_freq_q[freq][M][N][q] = matrix_m<complex<double>>(
                        n_mu, n_nu, IJq_Wc.at(M).at({N, qa}).data, MAJOR::ROW);
                }
                Profiler::stop("epsilon_convert_wc_communicate_3");
                // for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                //     for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                //     {
                //     }
            }
            Profiler::stop("epsilon_convert_wc_communicate");
            Profiler::stop("epsilon_convert_wc_2d_to_ij");
            Profiler::cease("epsilon_wc_work_q_omega");
            LIBRPA::utils::lib_printf_root(
                "Time for Wc(i_q=%d, i_omega=%d) (seconds, Wall/CPU): %f %f\n", iq + 1, ifreq + 1,
                Profiler::get_wall_time_last("epsilon_wc_work_q_omega"),
                Profiler::get_cpu_time_last("epsilon_wc_work_q_omega"));
        }
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    Profiler::cease("compute_Wc_freq_q_work");
    LIBRPA::utils::lib_printf_root("Time for Wc computation (seconds, Wall/CPU): %f %f\n",
                                   Profiler::get_wall_time_last("compute_Wc_freq_q_work"),
                                   Profiler::get_cpu_time_last("compute_Wc_freq_q_work"));

    return Wc_freq_q;
}

#endif