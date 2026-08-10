#include <cassert>
#include <cmath>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <string>
#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <random>
#include "ddla_stream_impl.h"

using namespace ddla;


void check_pzgetrf(int n, const DdlaHandle_t& ddla_handle)
{

    DdlaDesc matrix_desc(ddla_handle);
    matrix_desc.init_square_blk(n, n, 0, 0);
    int nb = std::min(128, matrix_desc.mb());
    matrix_desc.init(n, n, nb, nb, 0, 0);

    int myid = matrix_desc.mypcol() + matrix_desc.myprow()*matrix_desc.npcols();
    printf("myid:%d, m_loc:%d, n_loc:%d, mb:%d, nb:%d, m:%d, n:%d\n", myid, matrix_desc.m_loc(), matrix_desc.n_loc(), matrix_desc.mb(), matrix_desc.nb(), matrix_desc.m(), matrix_desc.n());
    bool verbose = false;
    
    std::vector<std::complex<double>> a(matrix_desc.m_loc()*matrix_desc.n_loc());
    std::vector<std::complex<double>> b(matrix_desc.m_loc()*matrix_desc.n_loc());

    std::complex<double>* d_A,* d_A_copy,* d_identity;
    std::vector<std::complex<double>> h_identity(matrix_desc.m_loc()*matrix_desc.n_loc());
    memset(h_identity.data(),0,sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc());
    for(int i=0;i<matrix_desc.m();i++){
        int i_loc = matrix_desc.indx_g2l_r(i);
        if(i_loc<0) continue;
        int j_loc = matrix_desc.indx_g2l_c(i);
        if(j_loc<0) continue;
        h_identity[i_loc+j_loc*matrix_desc.lld()] = {1.0, 0.0};
    }
    // printf("before syn 1:%d\n",myid);
    // RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t size = matrix_desc.m_loc()*matrix_desc.n_loc()*sizeof(std::complex<double>);

    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_A, size, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_A_copy, size, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMallocAsync((void**)&d_identity, size, ddla_handle->stream));
    
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    ddla_handle->check_memory();
    MPI_Barrier(MPI_COMM_WORLD);
    random_generate(d_A, matrix_desc.m_loc()*matrix_desc.n_loc());
    
    RUNTIME_CHECK(runtimeMemcpyAsync(d_A_copy, d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), runtimeMemcpyDeviceToDevice, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMemcpyAsync(a.data(), d_A, matrix_desc.m_loc() * matrix_desc.n_loc()* sizeof(std::complex<double>), runtimeMemcpyDeviceToHost, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMemcpyAsync(b.data(), d_A, matrix_desc.m_loc() * matrix_desc.n_loc()* sizeof(std::complex<double>), runtimeMemcpyDeviceToHost, ddla_handle->stream));
    RUNTIME_CHECK(runtimeMemcpyAsync(d_identity, h_identity.data(), sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), runtimeMemcpyHostToDevice, ddla_handle->stream));

    if(verbose)
    {
        std::string filename = "before_trf_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    printf("myid:%d, start sv\n",myid);
    double start_time_sv = MPI_Wtime();
    pgesv(
        'L', 'N',
        n, n,
        d_A, matrix_desc,
        d_identity, matrix_desc
    );
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    printf("myid:%d, pzgesv time:%lf\n",myid,MPI_Wtime()-start_time_sv);
    // RUNTIME_CHECK(runtimeMemcpyAsync(d_A, d_A_copy, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), runtimeMemcpyDeviceToDevice, ddla_handle->stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_gemm = MPI_Wtime();
    pgemm(
        'N', 'N',
        n, n, n,
        {1.0,0.0},
        d_A_copy, matrix_desc,
        d_identity, matrix_desc,
        {0.0,0.0},
        d_A, matrix_desc
    );
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    double end_time_gemm = MPI_Wtime();
    printf("myid:%d, pzgemm time:%lf\n",myid,end_time_gemm-start_time_gemm);
    if(verbose)
    { 
        RUNTIME_CHECK(runtimeMemcpyAsync(a.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        std::string filename = "identity_myid_";
        filename += std::to_string(myid);
        filename += ".txt";
        write_matrix<DdlaBackend::CPU>(a.data(), matrix_desc.m_loc(), matrix_desc.n_loc(), filename.c_str());
    }
    {
        // check the data from scalapack and bcast
        printf("myid:%d, start check pztrtrs result between scalapack and bcast\n",myid);        
        std::vector<std::complex<double>> temp_bcast(matrix_desc.m_loc()*matrix_desc.n_loc());
        RUNTIME_CHECK(runtimeMemcpyAsync(temp_bcast.data(), d_A, sizeof(std::complex<double>)*matrix_desc.m_loc()*matrix_desc.n_loc(), runtimeMemcpyDeviceToHost, ddla_handle->stream));
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        for(int i=0;i<matrix_desc.m();i++){
            int i_loc = matrix_desc.indx_g2l_r(i);
            if(i_loc<0)
                continue;
            for(int j=0;j<matrix_desc.n();j++){
                int j_loc = matrix_desc.indx_g2l_c(j);
                if(j_loc<0)
                    continue;
                double diff_abs;
                if(i==j)
                    diff_abs = std::abs(1. - temp_bcast[i_loc+j_loc*matrix_desc.m_loc()]);
                else
                    diff_abs = std::abs(temp_bcast[i_loc+j_loc*matrix_desc.m_loc()]);
                if(diff_abs>1e-10){
                    printf("myid:%d, check failed at global index (%d,%d), bcast value=(%lf,%lf)\n",
                        myid,
                        i,j,
                        temp_bcast[i_loc+j_loc*matrix_desc.m_loc()].real(),temp_bcast[i_loc+j_loc*matrix_desc.m_loc()].imag()
                    );
                    break;
                }
                
            }
        }
        printf("end check pztrtrs result between scalapack and bcast\n");
    }
    RUNTIME_CHECK(runtimeFreeAsync(d_identity, ddla_handle->stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_A, ddla_handle->stream));
    RUNTIME_CHECK(runtimeFreeAsync(d_A_copy, ddla_handle->stream));
    RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
}
int main(int argc, char* argv[]) {  
    MPI_Init(&argc, &argv);
    printf("before stream init\n");
    DdlaHandle_t ddla_handle = nullptr;
    ddla_init(ddla_handle);
    ddla_set(ddla_handle);
    // RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
    printf("after stream init\n");
    check_pzgetrf(5000, ddla_handle);
    for(int i = 5000; i <= 4 * 5000; i += 5000){
        RUNTIME_CHECK(runtimeStreamSynchronize(ddla_handle->stream));
        MPI_Barrier(MPI_COMM_WORLD);
        printf("testing matrix size: %d\n",i);
        check_pzgetrf(i, ddla_handle);
    }
    ddla_destroy(ddla_handle);
    MPI_Finalize();
    return 0;
}