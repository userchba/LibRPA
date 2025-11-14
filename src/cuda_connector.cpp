#include "cuda_connector.h"
#include <assert.h>
// cusolverMp include
#include "helpers.h"
#include <cusolverMp.h>
#include "lapack_connector.h"
#include "envs_mpi.h"
using LIBRPA::envs::mpi_comm_global_h;
#include "device_stream.h"
// #define CUSOLVERMP_MPI_GRID_COL_MAJOR
// #define OPEN_TEST_FOR_LU_DECOMPOSITION
// #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// #include <fstream> 
// #include <string>
// #endif
// #ifdef ENABLE_CUBLASMP
// // #include<curand.h>
// // #include<chrono>
// #endif
// #include <vector>
// #ifdef ENABLE_CUSOLVERMP
// void CudaConnector::pzgetrf_cusolverMp(const int &m, const int &n, std::complex<double> *h_C_A, const int &ia,
//                                 const int &ja, const LIBRPA::Array_Desc &arrdesc_pi, int *ipiv, int &h_info_getrf,const char order)
//     {
//         const int64_t M = arrdesc_pi.m();
//         const int64_t N = arrdesc_pi.n();

//         const int64_t IA = ia;
//         const int64_t JA = ja;

//         /* Tile sizes */
//         const int64_t MA = arrdesc_pi.mb();
//         const int64_t NA = arrdesc_pi.nb();
//         int numRowDevices, numColDevices;
//         if(order=='C'){
//             numRowDevices = arrdesc_pi.npcols();
//             numColDevices = arrdesc_pi.nprows();
//         }
//         else if(order=='R'){
//             numRowDevices = arrdesc_pi.nprows();
//             numColDevices = arrdesc_pi.npcols();
//         }else{
//             fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
//         }
        
        
//         const uint32_t RSRCA = 0;
//         const uint32_t CSRCA = 0;

//         int mpiCommSize, mpiRank;
//         MPI_Comm_size(MPI_COMM_WORLD, &mpiCommSize);
//         MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

//         int local_device = getLocalDevice();
//         int numDevices = 0;
        
//         // printf("Number of devices = %d\n", numDevices);
//         // printf("local_device = %d, mpiRank = %d, mpiCommSize = %d\n", local_device, mpiRank, mpiCommSize);
//         #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
//         double time_start_set_device=omp_get_wtime();
//         #endif
//         cudaError_t cudaStat  = cudaSetDevice(local_device);
//         #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
//         printf("time for set device:%f seconds, rank:%d\n", omp_get_wtime() - time_start_set_device,mpiRank);
//         #endif
//         assert(cudaStat == cudaSuccess);
//         cudaStat = cudaFree(0);
//         assert(cudaStat == cudaSuccess);
//         // const int mpiRank_T=arrdesc_pi.myprow()+ arrdesc_pi.mypcol()*arrdesc_pi.npcols();
//         const int rank     = mpiRank;
//         // printf("mpiRank=%d,mpiRank_T=%d\n", mpiRank, mpiRank_T);
//         const int commSize = mpiCommSize;
//         /* Library handles */
//         cusolverMpHandle_t cusolverMpHandle = NULL;
//         cal_comm_t         cal_comm         = NULL;

//         /* Error codes */
//         cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
//         calError_t       calStat      = CAL_OK;
//         cudaStat     = cudaSuccess;

//         /* User defined stream */
//         cudaStream_t localStream = NULL;
//         cal_comm_create_params_t params;
//         params.allgather    = allgather;
//         params.req_test     = request_test;
//         params.req_free     = request_free;
//         params.data         = (void*)(MPI_COMM_WORLD);
//         params.rank         = rank;
//         params.nranks       = commSize;
//         params.local_device = local_device;

//         calStat = cal_comm_create(params, &cal_comm);
//         assert(calStat == CAL_OK);

//         /* Create local stream */
//         #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
//         double time_start_create_stream=omp_get_wtime();
//         #endif
//         cudaStat = cudaStreamCreate(&localStream);
//         #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
//         printf("time for create stream:%f seconds, rank:%d\n", omp_get_wtime() - time_start_create_stream,mpiRank);
//         #endif
//         assert(cudaStat == cudaSuccess);

//         /* Initialize cusolverMp library handle */
//         #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
//         double time_start_cusolverMpCreate=omp_get_wtime();
//         #endif
//         cusolverStat = cusolverMpCreate(&cusolverMpHandle, local_device, localStream);
//         #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
//         printf("time for cusolverMpCreate:%f seconds, rank:%d\n", omp_get_wtime() - time_start_cusolverMpCreate,mpiRank);
//         #endif
//         assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

//         /* cusolverMp grids */
//         cusolverMpGrid_t gridA_C = NULL;

//         /* cusolverMp matrix descriptors */
//         cusolverMpMatrixDescriptor_t descrA_C = NULL;

//         /* Distributed matrices */
//         void*    d_C_A  = NULL; // for cuDoubleComplex
//         int64_t* d_ipiv = NULL;

//         /* Distributed device workspace */
//         void* d_work_getrf_C = NULL;

//         /* Distributed host workspace */
//         void* h_work_getrf_C = NULL;

//         /* size of workspace on device */
//         size_t workspaceInBytesOnDevice_getrf_C = 0;

//         /* size of workspace on host */
//         size_t workspaceInBytesOnHost_getrf_C = 0;

//         /* error codes from cusolverMp (device) */
//         int* d_info_getrf = NULL;

//         /* error codes from cusolverMp (host) */
//         // int h_info_getrf = 0;

//         /* Single process per device */
//         assert((numRowDevices * numColDevices) == commSize);

//         /* =========================================== */
//         /*          Create inputs on master rank       */
//         /* =========================================== */

//         const int64_t lda   = (IA - 1) + N;
//         const int64_t colsA = (JA - 1) + N;
//         // cuDoubleComplex* h_C_A = NULL;
//         int64_t LLDA, localColsA;
//         if(order=='C'){
//             localColsA =arrdesc_pi.m_loc();
//             LLDA=arrdesc_pi.n_loc();
//         }else if(order=='R'){
//             localColsA =arrdesc_pi.n_loc();
//             LLDA=arrdesc_pi.m_loc();
//         }else{
//             fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
//         }
        
//         /* Allocate global d_A */
//         cudaStat = cudaMalloc((void**)&d_C_A, localColsA * LLDA * sizeof(cuDoubleComplex));
//         assert(cudaStat == cudaSuccess);

//         /* =========================================== */
//         /*          CREATE GRID DESCRIPTORS            */
//         /* =========================================== */
//         if(order=='C'){
//         cusolverStat = cusolverMpCreateDeviceGrid(
//                 cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_COL_MAJOR);
//         }else if(order=='R'){
//         cusolverStat = cusolverMpCreateDeviceGrid(
//                 cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);
//         }else{
//             fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
//         }
//         assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

//         /* =========================================== */
//         /*        CREATE MATRIX DESCRIPTORS            */
//         /* =========================================== */
//         cusolverStat = cusolverMpCreateMatrixDesc(
//                 &descrA_C, gridA_C, CUDA_C_64F, (IA - 1) + M, (JA - 1) + N, MA, NA, RSRCA, CSRCA, LLDA);
//         assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

//         /* Allocate global d_ipiv */
//         /* REMARK : ipiv overlaps A[IA, JA:JA+N] as in Netlib's ScaLAPACK */
//         cudaStat = cudaMalloc((void**)&d_ipiv, arrdesc_pi.m_loc() * sizeof(int64_t));
//         assert(cudaStat == cudaSuccess);

//         /* =========================================== */
//         /*             ALLOCATE D_INFO                 */
//         /* =========================================== */

//         cudaStat = cudaMalloc((void**)&d_info_getrf, sizeof(int));
//         assert(cudaStat == cudaSuccess);

//         /* =========================================== */
//         /*                RESET D_INFO                 */
//         /* =========================================== */

//         cudaStat = cudaMemset(d_info_getrf, 1, sizeof(int));
//         assert(cudaStat == cudaSuccess);

//         /* =========================================== */
//         /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
//         /* =========================================== */
//         cusolverStat = cusolverMpGetrf_bufferSize(cusolverMpHandle,
//                                                   N,
//                                                   N,
//                                                   d_C_A,
//                                                   IA,
//                                                   JA,
//                                                   descrA_C,
//                                                   d_ipiv,
//                                                   CUDA_C_64F,
//                                                   &workspaceInBytesOnDevice_getrf_C,
//                                                   &workspaceInBytesOnHost_getrf_C);
//         assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

//         /* =========================================== */
//         /*         ALLOCATE PGETRF WORKSPACE            */
//         /* =========================================== */
//         cudaStat = cudaMalloc((void**)&d_work_getrf_C, workspaceInBytesOnDevice_getrf_C);
//         assert(cudaStat == cudaSuccess);
//         h_work_getrf_C = (void*)malloc(workspaceInBytesOnHost_getrf_C);
//         assert(h_work_getrf_C != NULL);

//         // copy matrix from h_C_A to d_C_A
//         std::complex<double>* h_C_A_temp = NULL;
//         size_t temp_size = (int64_t)localColsA * (int64_t)LLDA * sizeof(cuDoubleComplex);
//         if(order=='C'){
//             h_C_A_temp = LapackConnector::transpose(h_C_A, arrdesc_pi.m_loc(), arrdesc_pi.n_loc());
//             cudaStat = cudaMemcpy(d_C_A, h_C_A_temp, temp_size, cudaMemcpyHostToDevice);
//         }else if(order=='R'){
//             cudaStat = cudaMemcpy(d_C_A, h_C_A, temp_size, cudaMemcpyHostToDevice);
//         }else{
//             fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
//         }
//         assert(cudaStat == cudaSuccess);
        

//         /* sync wait for data to arrive to device */
//         calStat = cal_stream_sync(cal_comm, localStream);
//         assert(calStat == CAL_OK);


//         /* =========================================== */
//         /*                   CALL PGETRF               */
//         /* =========================================== */
//         h_info_getrf=1;
//         // printf("h_info_getrf before LU composition(cuDoubleComplex) : %d\n", h_info_getrf);
//         // printf("LU decomposition begin(cuDoubleComplex)\n");
//         double start_time_C = omp_get_wtime();
//         cusolverStat = cusolverMpGetrf(cusolverMpHandle,
//                                        N,
//                                        N,
//                                        d_C_A,
//                                        IA,
//                                        JA,
//                                        descrA_C,
//                                        d_ipiv,
//                                        CUDA_C_64F,
//                                        d_work_getrf_C,
//                                        workspaceInBytesOnDevice_getrf_C,
//                                        h_work_getrf_C,
//                                        workspaceInBytesOnHost_getrf_C,
//                                        d_info_getrf);

//         /* sync after cusolverMpGetrf */
//         calStat = cal_stream_sync(cal_comm, localStream);
//         assert(calStat == CAL_OK);
//         // printf("LU decomposition end(cuDoubleComplex), time = %f seconds,rand=%d\n", omp_get_wtime() - start_time_C,rank);

//         /* copy d_info_getrf to host */
//         cudaStat = cudaMemcpyAsync(&h_info_getrf, d_info_getrf, sizeof(int), cudaMemcpyDeviceToHost, localStream);
//         assert(cudaStat == cudaSuccess);
//         /* wait for d_info_getrf copy */
//         cudaStat = cudaStreamSynchronize(localStream);
//         assert(cudaStat == cudaSuccess);
//         // printf("h_info_getrf after composition(cuDoubleComplex) : %d\n", h_info_getrf);
//         /* check return value of cusolverMpGetrf */
//         assert(h_info_getrf == 0);
//         // copy d_ipiv to ipiv
//         cudaStat = cudaMemcpy(ipiv, d_ipiv, arrdesc_pi.m_loc() * sizeof(int64_t), cudaMemcpyDeviceToHost);
//         assert(cudaStat == cudaSuccess);
//         // copy matrix from d_C_A to h_C_A
//         if(order=='C'){
//             cudaStat = cudaMemcpy(h_C_A_temp, d_C_A, temp_size, cudaMemcpyDeviceToHost);
//             LapackConnector::transpose(h_C_A_temp, h_C_A, arrdesc_pi.m_loc(), arrdesc_pi.n_loc());
//         }else if(order=='R'){
//             cudaStat = cudaMemcpy(h_C_A, d_C_A, temp_size, cudaMemcpyDeviceToHost);
//         }else{
//             fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
//         }
//         assert(cudaStat == cudaSuccess);

//         calStat = cal_stream_sync(cal_comm, localStream);
//         assert(calStat == CAL_OK);

//         cudaStat = cudaStreamSynchronize(localStream);
//         assert(cudaStat == cudaSuccess);


//         /* sync wait for data to arrive to host */
//         calStat = cal_stream_sync(cal_comm, localStream);
//         assert(calStat == CAL_OK);


//         /* =========================================== */
//         /*            CHECK RESIDUAL ON MASTER         */
//         /* =========================================== */

//         /* =========================================== */
//         /*        CLEAN UP HOST WORKSPACE ON MASTER    */
//         /* =========================================== */
//         cusolverStat = cusolverMpDestroyMatrixDesc(descrA_C);
//         assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

//         cusolverStat = cusolverMpDestroyGrid(gridA_C);
//         assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);
//         if (d_C_A != NULL)
//         {
//             cudaStat = cudaFree(d_C_A);
//             assert(cudaStat == cudaSuccess);
//             d_C_A = NULL;
//         }
//         if (d_ipiv != NULL)
//         {
//             cudaStat = cudaFree(d_ipiv);
//             assert(cudaStat == cudaSuccess);
//             d_ipiv = NULL;
//         }

//         if(h_C_A_temp != NULL)
//         {
//             delete [] h_C_A_temp;
//             h_C_A_temp = NULL;
//         }

        
//         if (d_work_getrf_C != NULL)
//         {
//             cudaStat = cudaFree(d_work_getrf_C);
//             assert(cudaStat == cudaSuccess);
//             d_work_getrf_C = NULL;
//         }

//         if (d_info_getrf != NULL)
//         {
//             cudaStat = cudaFree(d_info_getrf);
//             assert(cudaStat == cudaSuccess);
//             d_info_getrf = NULL;
//         }
//         if (h_work_getrf_C)
//         {
//             free(h_work_getrf_C);
//             h_work_getrf_C = NULL;
//         }
//         /* Destroy cusolverMp handle */
//         cusolverStat = cusolverMpDestroy(cusolverMpHandle);
//         assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

//         /* sync before cal_comm_destroy */
//         calStat = cal_comm_barrier(cal_comm, localStream);
//         assert(calStat == CAL_OK);

//         /* destroy CAL communicator */
//         calStat = cal_comm_destroy(cal_comm);
//         assert(calStat == CAL_OK);

//         /* destroy user stream */
//         cudaStat = cudaStreamDestroy(localStream);
//         assert(cudaStat == cudaSuccess);

//         /* MPI barrier before MPI_Finalize */
//         MPI_Barrier(MPI_COMM_WORLD);
//         // printf("success in test\n");
//         return;
//     }
// #endif

#ifdef ENABLE_NVHPC
void CudaConnector::pzgetrf_nvhpc(const GpuDeviceStream&gpu_dev_stream, ComplexMatrixDevice &d_A, const int &ia,
                                const int &ja, const LIBRPA::Array_Desc &arrdesc_pi, int64_t *d_ipiv, int *d_info_getrf, const char& order)
{
        const int64_t M = arrdesc_pi.m();
        const int64_t N = arrdesc_pi.n();

        const int64_t IA = ia;
        const int64_t JA = ja;

        /* Tile sizes */
        const int64_t MA = arrdesc_pi.mb();
        const int64_t NA = arrdesc_pi.nb();
        int numRowDevices, numColDevices;
        if(order=='C'){
            numRowDevices = arrdesc_pi.npcols();
            numColDevices = arrdesc_pi.nprows();
        }
        else if(order=='R'){
            numRowDevices = arrdesc_pi.nprows();
            numColDevices = arrdesc_pi.npcols();
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
        }
        
        
        const uint32_t RSRCA = 0;
        const uint32_t CSRCA = 0;

        int mpiRank = gpu_dev_stream.rank;
        int mpiCommSize = gpu_dev_stream.nranks;

        int local_device = gpu_dev_stream.local_device;

        const int rank     = mpiRank;
        const int commSize = mpiCommSize;
        /* Library handles */
        cusolverMpHandle_t cusolverMpHandle = gpu_dev_stream.cusolver_handle;
        cal_comm_t         cal_comm         = gpu_dev_stream.cal_comm;

        /* Error codes */
        cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
        cudaError_t cudaStat     = cudaSuccess;
        cudaStream_t localStream = gpu_dev_stream.stream;
        /* Initialize cusolverMp library handle */
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        double time_start_cusolverMpCreate=omp_get_wtime();
        #endif
        // cusolverStat = cusolverMpCreate(&cusolverMpHandle, local_device, localStream);
        #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        // printf("time for cusolverMpCreate:%f seconds, rank:%d\n", omp_get_wtime() - time_start_cusolverMpCreate,mpiRank);
        #endif
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* cusolverMp grids */
        cusolverMpGrid_t gridA_C = NULL;

        /* cusolverMp matrix descriptors */
        cusolverMpMatrixDescriptor_t descrA_C = NULL;

        /* Distributed matrices */
        void*    d_C_A  = d_A.ptr(); // for cuDoubleComplex
        

        /* Distributed device workspace */
        void* d_work_getrf_C = NULL;

        /* Distributed host workspace */
        void* h_work_getrf_C = NULL;

        /* size of workspace on device */
        size_t workspaceInBytesOnDevice_getrf_C = 0;

        /* size of workspace on host */
        size_t workspaceInBytesOnHost_getrf_C = 0;

        /* Single process per device */
        assert((numRowDevices * numColDevices) == commSize);

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        const int64_t lda   = (IA - 1) + N;
        const int64_t colsA = (JA - 1) + N;
        // cuDoubleComplex* h_C_A = NULL;
        int64_t LLDA, localColsA;
        if(order=='C'){
            localColsA =arrdesc_pi.m_loc();
            LLDA=arrdesc_pi.n_loc();
        }else if(order=='R'){
            localColsA =arrdesc_pi.n_loc();
            LLDA=arrdesc_pi.m_loc();
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
        }

        /* =========================================== */
        /*          CREATE GRID DESCRIPTORS            */
        /* =========================================== */
        if(order=='C'){
        cusolverStat = cusolverMpCreateDeviceGrid(
                cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_COL_MAJOR);
        }else if(order=='R'){
        cusolverStat = cusolverMpCreateDeviceGrid(
                cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C' or 'R'\n");
        }
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*        CREATE MATRIX DESCRIPTORS            */
        /* =========================================== */
        cusolverStat = cusolverMpCreateMatrixDesc(
                &descrA_C, gridA_C, CUDA_C_64F, (IA - 1) + M, (JA - 1) + N, MA, NA, RSRCA, CSRCA, LLDA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);


        /* =========================================== */
        /*                RESET D_INFO                 */
        /* =========================================== */

        CUDA_CHECK(cudaMemset(d_info_getrf, 1, sizeof(int)));
        

        /* =========================================== */
        /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
        /* =========================================== */
        cusolverStat = cusolverMpGetrf_bufferSize(cusolverMpHandle,
                                                  N,
                                                  N,
                                                  d_C_A,
                                                  IA,
                                                  JA,
                                                  descrA_C,
                                                  d_ipiv,
                                                  CUDA_C_64F,
                                                  &workspaceInBytesOnDevice_getrf_C,
                                                  &workspaceInBytesOnHost_getrf_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*         ALLOCATE PGETRF WORKSPACE            */
        /* =========================================== */
        cudaStat = cudaMalloc((void**)&d_work_getrf_C, workspaceInBytesOnDevice_getrf_C);
        assert(cudaStat == cudaSuccess);
        h_work_getrf_C = (void*)malloc(workspaceInBytesOnHost_getrf_C);
        assert(h_work_getrf_C != NULL);        
        

        /* sync wait for data to arrive to device */
        CAL_CHECK(cal_stream_sync(cal_comm, localStream));


        /* =========================================== */
        /*                   CALL PGETRF               */
        /* =========================================== */
        double start_time_C = omp_get_wtime();
        cusolverStat = cusolverMpGetrf(cusolverMpHandle,
                                       N,
                                       N,
                                       d_C_A,
                                       IA,
                                       JA,
                                       descrA_C,
                                       d_ipiv,
                                       CUDA_C_64F,
                                       d_work_getrf_C,
                                       workspaceInBytesOnDevice_getrf_C,
                                       h_work_getrf_C,
                                       workspaceInBytesOnHost_getrf_C,
                                       d_info_getrf);

        /* sync after cusolverMpGetrf */
        CAL_CHECK(cal_stream_sync(cal_comm, localStream));
        // printf("LU decomposition end(cuDoubleComplex), time = %f seconds,rand=%d\n", omp_get_wtime() - start_time_C,rank);

        /* copy d_info_getrf to host */
        int h_info_getrf=1;
        cudaStat = cudaMemcpyAsync(&h_info_getrf, d_info_getrf, sizeof(int), cudaMemcpyDeviceToHost, localStream);
        assert(cudaStat == cudaSuccess);
        /* wait for d_info_getrf copy */
        gpu_dev_stream.cudaSync();
        assert(cudaStat == cudaSuccess);
        assert(h_info_getrf == 0);


        /* sync wait for data to arrive to host */
        CAL_CHECK(cal_stream_sync(cal_comm, localStream));

        /* =========================================== */
        /*        CLEAN UP HOST WORKSPACE ON MASTER    */
        /* =========================================== */
        cusolverStat = cusolverMpDestroyMatrixDesc(descrA_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyGrid(gridA_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        
        if (d_work_getrf_C != NULL)
        {
            cudaStat = cudaFree(d_work_getrf_C);
            assert(cudaStat == cudaSuccess);
            d_work_getrf_C = NULL;
        }

        if (h_work_getrf_C)
        {
            free(h_work_getrf_C);
            h_work_getrf_C = NULL;
        }
        /* Destroy cusolverMp handle */
        // cusolverStat = cusolverMpDestroy(cusolverMpHandle);
        // assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* MPI barrier before MPI_Finalize */
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("success in test\n");
        return;
}
void CudaConnector::pgetrf_nvhpc_mixed_precision(
        const GpuDeviceStream&gpu_dev_stream, void *d_A, 
        const int &ia, const int &ja,
        const LIBRPA::Array_Desc &arrdesc_pi, int64_t *d_ipiv, int *d_info_getrf,
        const cudaDataType_t &computeType,const char &order
    )
{
        const int64_t M = arrdesc_pi.m();
        const int64_t N = arrdesc_pi.n();

        const int64_t IA = ia;
        const int64_t JA = ja;

        /* Tile sizes */
        const int64_t MA = arrdesc_pi.mb();
        const int64_t NA = arrdesc_pi.nb();
        int numRowDevices, numColDevices;
        if(order=='C'||order=='c'){
            numRowDevices = arrdesc_pi.npcols();
            numColDevices = arrdesc_pi.nprows();
        }
        else if(order=='R'||order=='r'){
            numRowDevices = arrdesc_pi.nprows();
            numColDevices = arrdesc_pi.npcols();
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C'('c') or 'R'('r')\n");
        }
        
        
        const uint32_t RSRCA = 0;
        const uint32_t CSRCA = 0;

        int local_device = gpu_dev_stream.local_device;

        const int rank     = gpu_dev_stream.rank;
        const int commSize =  gpu_dev_stream.nranks;
        /* Library handles */
        cusolverMpHandle_t cusolverMpHandle = gpu_dev_stream.cusolver_handle;
        cal_comm_t         cal_comm         = gpu_dev_stream.cal_comm;

        /* Error codes */
        cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
        cudaError_t cudaStat     = cudaSuccess;
        cudaStream_t localStream = gpu_dev_stream.stream;

        /* cusolverMp grids */
        cusolverMpGrid_t gridA_C = NULL;

        /* cusolverMp matrix descriptors */
        cusolverMpMatrixDescriptor_t descrA_C = NULL;

        /* Distributed matrices */
        void*    d_C_A  = d_A; //
        

        /* Distributed device workspace */
        void* d_work_getrf_C = NULL;

        /* Distributed host workspace */
        void* h_work_getrf_C = NULL;

        /* size of workspace on device */
        size_t workspaceInBytesOnDevice_getrf_C = 0;

        /* size of workspace on host */
        size_t workspaceInBytesOnHost_getrf_C = 0;

        /* Single process per device */
        assert((numRowDevices * numColDevices) == commSize);

        const int64_t lda   = (IA - 1) + N;
        const int64_t colsA = (JA - 1) + N;
        // cuDoubleComplex* h_C_A = NULL;
        int64_t LLDA, localColsA;
        if(order=='C'||order=='c'){
            localColsA =arrdesc_pi.m_loc();
            LLDA=arrdesc_pi.n_loc();
        }else if(order=='R'||order=='r'){
            localColsA =arrdesc_pi.n_loc();
            LLDA=arrdesc_pi.m_loc();
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C'('c') or 'R'('r')\n");
        }

        /* =========================================== */
        /*          CREATE GRID DESCRIPTORS            */
        /* =========================================== */
        if(order=='C'||order=='c'){
        cusolverStat = cusolverMpCreateDeviceGrid(
                cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_COL_MAJOR);
        }else if(order=='R'||order=='r'){
        cusolverStat = cusolverMpCreateDeviceGrid(
                cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C'('c') or 'R'('r')\n");
        }
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*        CREATE MATRIX DESCRIPTORS            */
        /* =========================================== */
        cusolverStat = cusolverMpCreateMatrixDesc(
                &descrA_C, gridA_C, computeType, (IA - 1) + M, (JA - 1) + N, MA, NA, RSRCA, CSRCA, LLDA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);


        /* =========================================== */
        /*                RESET D_INFO                 */
        /* =========================================== */

        CUDA_CHECK(cudaMemset(d_info_getrf, 1, sizeof(int)));
        

        /* =========================================== */
        /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
        /* =========================================== */
        cusolverStat = cusolverMpGetrf_bufferSize(cusolverMpHandle,
                                                  N,
                                                  N,
                                                  d_C_A,
                                                  IA,
                                                  JA,
                                                  descrA_C,
                                                  d_ipiv,
                                                  computeType,
                                                  &workspaceInBytesOnDevice_getrf_C,
                                                  &workspaceInBytesOnHost_getrf_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*         ALLOCATE PGETRF WORKSPACE            */
        /* =========================================== */
        CUDA_CHECK(cudaMallocAsync((void**)&d_work_getrf_C, workspaceInBytesOnDevice_getrf_C, localStream));
        h_work_getrf_C = (void*)malloc(workspaceInBytesOnHost_getrf_C);
        assert(h_work_getrf_C != NULL);        

        /* =========================================== */
        /*                   CALL PGETRF               */
        /* =========================================== */
        double start_time_C = omp_get_wtime();
        CUSOLVERMP_CHECK(cusolverMpGetrf(
            cusolverMpHandle, N, N,
            d_C_A, IA, JA, descrA_C,
            d_ipiv, computeType,
            d_work_getrf_C, workspaceInBytesOnDevice_getrf_C,
            h_work_getrf_C, workspaceInBytesOnHost_getrf_C,
            d_info_getrf));
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync after cusolverMpGetrf */
        CAL_CHECK(cal_stream_sync(cal_comm, localStream));

        /* copy d_info_getrf to host */
        int h_info_getrf=1;
        cudaStat = cudaMemcpyAsync(&h_info_getrf, d_info_getrf, sizeof(int), cudaMemcpyDeviceToHost, localStream);
        assert(cudaStat == cudaSuccess);
        /* wait for d_info_getrf copy */
        gpu_dev_stream.cudaSync();
        assert(cudaStat == cudaSuccess);
        assert(h_info_getrf == 0);

        /* =========================================== */
        /*        CLEAN UP HOST WORKSPACE ON MASTER    */
        /* =========================================== */
        cusolverStat = cusolverMpDestroyMatrixDesc(descrA_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyGrid(gridA_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        
        if (d_work_getrf_C != NULL)
        {
            CUDA_CHECK(cudaFreeAsync(d_work_getrf_C, localStream));
            d_work_getrf_C = NULL;
        }

        if (h_work_getrf_C)
        {
            free(h_work_getrf_C);
            h_work_getrf_C = NULL;
        }

        /* MPI barrier before MPI_Finalize */
        MPI_Barrier(MPI_COMM_WORLD);
        return;
}
void CudaConnector::pgetrf_nvhpc_mixed_precision(
        void *d_A, const int &ia, const int &ja,
        const LIBRPA::Array_Desc &arrdesc_pi, int64_t *d_ipiv, int *d_info_getrf,
        const cudaDataType_t &computeType,const char &order
    )
{
        const int64_t M = arrdesc_pi.m();
        const int64_t N = arrdesc_pi.n();

        const int64_t IA = ia;
        const int64_t JA = ja;

        /* Tile sizes */
        const int64_t MA = arrdesc_pi.mb();
        const int64_t NA = arrdesc_pi.nb();
        int numRowDevices, numColDevices;
        if(order=='C'||order=='c'){
            numRowDevices = arrdesc_pi.npcols();
            numColDevices = arrdesc_pi.nprows();
        }
        else if(order=='R'||order=='r'){
            numRowDevices = arrdesc_pi.nprows();
            numColDevices = arrdesc_pi.npcols();
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C'('c') or 'R'('r')\n");
        }
        
        
        const uint32_t RSRCA = 0;
        const uint32_t CSRCA = 0;

        int local_device = device_stream.local_device;

        const int rank     = mpi_comm_global_h.myid;
        const int commSize =  mpi_comm_global_h.nprocs;
        /* Library handles */
        cusolverMpHandle_t cusolverMpHandle = device_stream.cusolverMp_handle;
        cal_comm_t         cal_comm         = device_stream.cal_comm;

        /* Error codes */
        cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
        cudaError_t cudaStat     = cudaSuccess;
        cudaStream_t localStream = (cudaStream_t)device_stream.stream;

        /* cusolverMp grids */
        cusolverMpGrid_t gridA_C = NULL;

        /* cusolverMp matrix descriptors */
        cusolverMpMatrixDescriptor_t descrA_C = NULL;

        /* Distributed matrices */
        void*    d_C_A  = d_A; //
        

        /* Distributed device workspace */
        void* d_work_getrf_C = NULL;

        /* Distributed host workspace */
        void* h_work_getrf_C = NULL;

        /* size of workspace on device */
        size_t workspaceInBytesOnDevice_getrf_C = 0;

        /* size of workspace on host */
        size_t workspaceInBytesOnHost_getrf_C = 0;

        /* Single process per device */
        assert((numRowDevices * numColDevices) == commSize);

        const int64_t lda   = (IA - 1) + N;
        const int64_t colsA = (JA - 1) + N;
        // cuDoubleComplex* h_C_A = NULL;
        int64_t LLDA, localColsA;
        if(order=='C'||order=='c'){
            localColsA =arrdesc_pi.m_loc();
            LLDA=arrdesc_pi.n_loc();
        }else if(order=='R'||order=='r'){
            localColsA =arrdesc_pi.n_loc();
            LLDA=arrdesc_pi.m_loc();
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C'('c') or 'R'('r')\n");
        }

        /* =========================================== */
        /*          CREATE GRID DESCRIPTORS            */
        /* =========================================== */
        if(order=='C'||order=='c'){
        cusolverStat = cusolverMpCreateDeviceGrid(
                cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_COL_MAJOR);
        }else if(order=='R'||order=='r'){
        cusolverStat = cusolverMpCreateDeviceGrid(
                cusolverMpHandle, &gridA_C, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);
        }else{
            fprintf(stderr, "Error: cusolverMpgetrf order must be 'C'('c') or 'R'('r')\n");
        }
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*        CREATE MATRIX DESCRIPTORS            */
        /* =========================================== */
        cusolverStat = cusolverMpCreateMatrixDesc(
                &descrA_C, gridA_C, computeType, (IA - 1) + M, (JA - 1) + N, MA, NA, RSRCA, CSRCA, LLDA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);


        /* =========================================== */
        /*                RESET D_INFO                 */
        /* =========================================== */

        CUDA_CHECK(cudaMemset(d_info_getrf, 1, sizeof(int)));
        

        /* =========================================== */
        /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
        /* =========================================== */
        cusolverStat = cusolverMpGetrf_bufferSize(cusolverMpHandle,
                                                  N,
                                                  N,
                                                  d_C_A,
                                                  IA,
                                                  JA,
                                                  descrA_C,
                                                  d_ipiv,
                                                  computeType,
                                                  &workspaceInBytesOnDevice_getrf_C,
                                                  &workspaceInBytesOnHost_getrf_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*         ALLOCATE PGETRF WORKSPACE            */
        /* =========================================== */
        CUDA_CHECK(cudaMallocAsync((void**)&d_work_getrf_C, workspaceInBytesOnDevice_getrf_C, localStream));
        h_work_getrf_C = (void*)malloc(workspaceInBytesOnHost_getrf_C);
        assert(h_work_getrf_C != NULL);        

        /* =========================================== */
        /*                   CALL PGETRF               */
        /* =========================================== */
        double start_time_C = omp_get_wtime();
        CUSOLVERMP_CHECK(cusolverMpGetrf(
            cusolverMpHandle, N, N,
            d_C_A, IA, JA, descrA_C,
            d_ipiv, computeType,
            d_work_getrf_C, workspaceInBytesOnDevice_getrf_C,
            h_work_getrf_C, workspaceInBytesOnHost_getrf_C,
            d_info_getrf));
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync after cusolverMpGetrf */
        CAL_CHECK(cal_stream_sync(cal_comm, localStream));

        /* copy d_info_getrf to host */
        int h_info_getrf=1;
        cudaStat = cudaMemcpyAsync(&h_info_getrf, d_info_getrf, sizeof(int), cudaMemcpyDeviceToHost, localStream);
        assert(cudaStat == cudaSuccess);
        /* wait for d_info_getrf copy */
        device_stream.cudaSync();
        assert(cudaStat == cudaSuccess);
        assert(h_info_getrf == 0);

        /* =========================================== */
        /*        CLEAN UP HOST WORKSPACE ON MASTER    */
        /* =========================================== */
        cusolverStat = cusolverMpDestroyMatrixDesc(descrA_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyGrid(gridA_C);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        
        if (d_work_getrf_C != NULL)
        {
            CUDA_CHECK(cudaFreeAsync(d_work_getrf_C, localStream));
            d_work_getrf_C = NULL;
        }

        if (h_work_getrf_C)
        {
            free(h_work_getrf_C);
            h_work_getrf_C = NULL;
        }

        /* MPI barrier before MPI_Finalize */
        MPI_Barrier(MPI_COMM_WORLD);
        return;
}
void CudaConnector::pgetrs_nvhpc_mixed_precision(
    const GpuDeviceStream& gpu_dev_stream, const cublasOperation_t& trans,
    const void* d_A, const int64_t& IA, const int64_t& JA, const LIBRPA::Array_Desc &arrdesc_A,
    const int64_t* d_ipiv,
    void* d_B, const int64_t& IB, const int64_t& JB, const LIBRPA::Array_Desc &arrdesc_B,
    int* d_info, const cudaDataType_t& compute_type,
    const char& order
)
{
    const int64_t N = arrdesc_A.m();
    const int64_t NRHS = arrdesc_B.n();
    assert(arrdesc_A.m()==arrdesc_A.n());

    int h_info = 1;
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    void* h_work = NULL;
    void* d_work = NULL;
    const int64_t mbA = arrdesc_A.mb();
    const int64_t nbA = arrdesc_A.nb();
    const int64_t mbB = arrdesc_B.mb();
    const int64_t nbB = arrdesc_B.nb();
    assert(arrdesc_A.npcols() == arrdesc_B.npcols());
    assert(arrdesc_A.nprows() == arrdesc_B.nprows());
    int numRowDevices, numColDevices;
    ORDER_CHECK(order);
    if(order=='C'||order=='c'){
        assert(N==NRHS);
        numRowDevices = arrdesc_A.npcols();
        numColDevices = arrdesc_A.nprows();
    }
    else{
        numRowDevices = arrdesc_A.nprows();
        numColDevices = arrdesc_A.npcols();
    }
    int64_t LLDA, localColsA, LLDB, localColsB;
    if(order=='C'||order=='c'){
        LLDA=arrdesc_A.n_loc();
        localColsA =arrdesc_A.m_loc();
        LLDB=arrdesc_B.n_loc();
        localColsB =arrdesc_B.m_loc();
    }else{
        LLDA=arrdesc_A.m_loc();
        localColsA =arrdesc_A.n_loc();
        LLDB=arrdesc_B.m_loc();
        localColsB =arrdesc_B.n_loc();
    }
    int mpiCommSize = gpu_dev_stream.nranks;
    int rank = gpu_dev_stream.rank;

    cusolverMpHandle_t cusolverMpHandle = gpu_dev_stream.cusolver_handle;

    cusolverMpGrid_t grid = NULL;
    cusolverMpMatrixDescriptor_t descrA = NULL;
    cusolverMpMatrixDescriptor_t descrB = NULL;

    CUSOLVERMP_CHECK(cusolverMpCreateDeviceGrid(
        cusolverMpHandle, &grid, gpu_dev_stream.cal_comm, 
        numRowDevices, numColDevices, 
        (order=='c'||order=='C')?CUSOLVERMP_GRID_MAPPING_COL_MAJOR:CUSOLVERMP_GRID_MAPPING_ROW_MAJOR)
    );
    
    CUSOLVERMP_CHECK(cusolverMpCreateMatrixDesc(
            &descrA, grid, compute_type, 
            (IA - 1) + N, (JA - 1) + N, 
            mbA, nbA, 0, 0, LLDA)
    );
    CUSOLVERMP_CHECK(cusolverMpCreateMatrixDesc(
            &descrB, grid, compute_type, 
            (IB - 1) + N, (JB - 1) + NRHS, 
            mbB, nbB, 0, 0, LLDB)
    );

    CUSOLVERMP_CHECK(cusolverMpGetrs_bufferSize(
        cusolverMpHandle, trans, N, NRHS,
        d_A, IA, JA, descrA, d_ipiv,
        d_B, IB, JB, descrB,
        compute_type,
        &workspaceInBytesOnDevice,&workspaceInBytesOnHost)
    );
    gpu_dev_stream.calSync();
    if (workspaceInBytesOnHost > 0)
    {
        h_work = (void*)malloc(workspaceInBytesOnHost);
        assert(h_work != NULL);
    }
    if (workspaceInBytesOnDevice > 0)
    {
        CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, gpu_dev_stream.stream));
    }
    gpu_dev_stream.calSync();
    CUSOLVERMP_CHECK(cusolverMpGetrs(
        cusolverMpHandle, trans, 
        N, NRHS,
        d_A, IA, JA, descrA, d_ipiv,
        d_B, IB, JB, descrB,
        compute_type,
        d_work, workspaceInBytesOnDevice,
        h_work, workspaceInBytesOnHost,
        d_info)
    );
    gpu_dev_stream.calSync();

    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_info!=0)
    {
        fprintf(stderr, "Error: cusolverMpgetrs failed with info=%d\n", h_info);
        exit(1);
    }
    if(h_work!=NULL)
    {
        free(h_work);
    }
    if(d_work!=NULL){
        CUDA_CHECK(cudaFreeAsync(d_work, gpu_dev_stream.stream));
    }
}
void CudaConnector::pgetrf_trs_nvhpc_mixed_precision(
    const GpuDeviceStream& gpu_dev_stream, const cublasOperation_t& trans,
    void* d_A, const int64_t& IA, const int64_t& JA, const LIBRPA::Array_Desc &arrdesc_A,
    void* d_B, const int64_t& IB, const int64_t& JB, const LIBRPA::Array_Desc &arrdesc_B,
    const cudaDataType_t& compute_type, const char& order
)
{
    int64_t* d_ipiv;
    int* d_info;
    CUDA_CHECK(cudaMallocAsync(&d_info,sizeof(int),gpu_dev_stream.stream));
    if(order == 'c'||order == 'C'){
        CUDA_CHECK(cudaMallocAsync(&d_ipiv,sizeof(int64_t)*arrdesc_A.n_loc(),gpu_dev_stream.stream));
    }else{
        CUDA_CHECK(cudaMallocAsync(&d_ipiv,sizeof(int64_t)*arrdesc_A.m_loc(),gpu_dev_stream.stream));
    }
    pgetrf_nvhpc_mixed_precision(
        gpu_dev_stream, d_A, 1, 1, arrdesc_A,
        d_ipiv, d_info,
        CUDA_C_64F, order
    );
    pgetrs_nvhpc_mixed_precision(
        gpu_dev_stream, trans,
        d_A, 1, 1, arrdesc_A,
        d_ipiv,
        d_B, 1, 1, arrdesc_B,
        d_info,
        CUDA_C_64F, order
    );
    CUDA_CHECK(cudaFreeAsync(d_info, gpu_dev_stream.stream));
    CUDA_CHECK(cudaFreeAsync(d_ipiv, gpu_dev_stream.stream));
}
// void CudaConnector::pgemm_cublasMp(const char &transa, const char &transb, const int &m, const int &n, const int &k,
//                         const double &alphaD, const std::complex<double> *A, const int &ia, const int &ja, const LIBRPA::Array_Desc &arrdesc_A,
//                         const std::complex<double> *B, const int &ib, const int &jb, const LIBRPA::Array_Desc &arrdesc_B,
//                         const double &betaD, std::complex<double> *C, const int &ic, const int &jc, const LIBRPA::Array_Desc &arrdesc_C)
// {
//     using input_t = cuDoubleComplex;
//     using output_t = cuDoubleComplex;
//     using compute_t = cuDoubleComplex;
//     const cudaDataType_t cuda_input_type = CUDA_C_64F;
//     const cudaDataType_t cuda_output_type = CUDA_C_64F;
//     cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_64F_PEDANTIC;

//     int64_t mbA = arrdesc_A.mb();
//     int64_t nbA = arrdesc_A.nb();
//     int64_t mbB = arrdesc_B.mb();
//     int64_t nbB = arrdesc_B.nb();
//     int64_t mbC = arrdesc_C.mb();
//     int64_t nbC = arrdesc_C.nb();
//     int nprow = arrdesc_A.nprows();
//     int npcol = arrdesc_A.npcols();
//     char grid_layout = 'r';
//     bool verbose = false;
//     int rank, nranks;
//     MPI_Comm_size(MPI_COMM_WORLD, &nranks);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     // if(rank==0)
//     //     printf("sizeof(input_t)=%zu sizeof(output_t)=%zu sizeof(compute_t)=%zu\n", sizeof(input_t), sizeof(output_t), sizeof(compute_t));
//     const int myprow = arrdesc_A.myprow();
//     const int mypcol = arrdesc_A.mypcol();
//     if(rank==0)
//     {
//         printf("m:%" PRId64 ", n:%" PRId64 ", k:%" PRId64 "\n", m, n, k);
//         printf("nprow:%d, npcol:%d\n", nprow, npcol);
//         printf("mbA:%" PRId64 ", nbA:%" PRId64 "\n", mbA, nbA);
//     }
//     const int local_device = getLocalDevice();
//     printf("myrank:%d, myprow:%d, mypcol:%d, local_device:%d\n", rank, myprow, mypcol, local_device);
//     CUDA_CHECK(cudaSetDevice(local_device));
//     CUDA_CHECK(cudaFree(nullptr));

//     cal_comm_t cal_comm;
//     cal_comm_create_params_t params;
//     {
//         params.allgather    = allgather;
//         params.req_test     = request_test;
//         params.req_free     = request_free;
//         params.data         = (void*)(MPI_COMM_WORLD);
//         params.rank         = rank;
//         params.nranks       = nranks;
//         params.local_device = local_device;

//         CAL_CHECK(cal_comm_create(params, &cal_comm));
//     }
//     cudaStream_t stream = nullptr;
//     CUDA_CHECK(cudaStreamCreate(&stream));

//     cublasMpHandle_t handle = nullptr;
//     CUBLASMP_CHECK(cublasMpCreate(&handle, stream));

//     cublasMpGrid_t grid = nullptr;

//     cublasMpMatrixDescriptor_t descA = nullptr;
//     cublasMpMatrixDescriptor_t descB = nullptr;
//     cublasMpMatrixDescriptor_t descC = nullptr;

//     input_t* d_A = nullptr;
//     input_t* d_B = nullptr;
//     output_t* d_C = nullptr;

//     void* d_work = nullptr;

//     compute_t alpha = {alphaD,0.0};
//     compute_t beta = {betaD,0.0};


//     size_t workspaceInBytesOnDevice = 0;
//     size_t workspaceInBytesOnHost = 0;

//     const int64_t global_m_a = (ia - 1) + m;
//     const int64_t global_n_a = (ja - 1) + k;
//     const int64_t global_m_b = (ib - 1) + k;
//     const int64_t global_n_b = (jb - 1) + n;
//     const int64_t global_m_c = (ic - 1) + m;
//     const int64_t global_n_c = (jc - 1) + n;

//     const int64_t llda = cublasMpNumroc(global_m_a, mbA, myprow, 0, nprow);
//     const int64_t loc_n_a = cublasMpNumroc(global_n_a, nbA, mypcol, 0, npcol);
//     printf("rank:%d, llda=%" PRId64 ", loc_n_a=%" PRId64 "\n", rank, llda, loc_n_a);
//     const int64_t lldb = cublasMpNumroc(global_m_b, mbB, myprow, 0, nprow);
//     const int64_t loc_n_b = cublasMpNumroc(global_n_b, nbB, mypcol, 0, npcol);

//     const int64_t lldc = cublasMpNumroc(global_m_c, mbC, myprow, 0, nprow);
//     const int64_t loc_n_c = cublasMpNumroc(global_n_c, nbC, mypcol, 0, npcol);

//     CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(input_t), stream));
//     CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b * sizeof(input_t), stream));
//     CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c * sizeof(output_t), stream));

//     CUDA_CHECK(cudaMemcpyAsync(d_A, A, llda * loc_n_a * sizeof(input_t), cudaMemcpyHostToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_B, B, lldb * loc_n_b * sizeof(input_t), cudaMemcpyHostToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_C, C, lldc * loc_n_c * sizeof(output_t), cudaMemcpyHostToDevice, stream));

//     CUBLASMP_CHECK(cublasMpGridCreate(
//         handle,
//         nprow,
//         npcol,
//         grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
//         cal_comm,
//         &grid));

//     CUBLASMP_CHECK(
//         cublasMpMatrixDescriptorCreate(handle,global_m_a, global_n_a, mbA, nbA, 0, 0, llda, cuda_input_type, grid, &descA));
//     CUBLASMP_CHECK(
//         cublasMpMatrixDescriptorCreate(handle,global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, cuda_input_type, grid, &descB));
//     CUBLASMP_CHECK(
//         cublasMpMatrixDescriptorCreate(handle,global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, cuda_output_type, grid, &descC));
//     cublasOperation_t transA,transB;
//     if(transa=='N')
//         transA=CUBLAS_OP_N;
//     else if(transa=='T')
//         transA=CUBLAS_OP_T;
//     else if(transa=='C')
//         transA=CUBLAS_OP_C;
//     else{
//         if(rank==0)
//             printf("transa=%c is not supported\n", transa);
//         exit(1);
//     }
//     if(transb=='N')
//         transB=CUBLAS_OP_N;
//     else if(transb=='T')
//         transB=CUBLAS_OP_T;
//     else if(transb=='C')
//         transB=CUBLAS_OP_C;
//     else{
//         if(rank==0)
//             printf("transb=%c is not supported\n", transb);
//         exit(1);
//     }



//     CUBLASMP_CHECK(cublasMpGemm_bufferSize(
//         handle,
//         transA,
//         transB,
//         m,
//         n,
//         k,
//         &alpha,
//         d_A,
//         ia,
//         ja,
//         descA,
//         d_B,
//         ib,
//         jb,
//         descB,
//         &beta,
//         d_C,
//         ic,
//         jc,
//         descC,
//         cublas_compute_type,
//         &workspaceInBytesOnDevice,
//         &workspaceInBytesOnHost));

//     CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
//     printf("workspaceInBytesOnDevice=%zu, workspaceInBytesOnHost=%zu, rank:%d\n", workspaceInBytesOnDevice, workspaceInBytesOnHost, rank);
//     std::vector<int8_t> h_work(workspaceInBytesOnHost);

//     CUDA_CHECK(cudaStreamSynchronize(stream));

//     const double begin = MPI_Wtime();

//     CUBLASMP_CHECK(cublasMpGemm(
//         handle,
//         transA,
//         transB,
//         m,
//         n,
//         k,
//         &alpha,
//         d_A,
//         ia,
//         ja,
//         descA,
//         d_B,
//         ib,
//         jb,
//         descB,
//         &beta,
//         d_C,
//         ic,
//         jc,
//         descC,
//         cublas_compute_type,
//         d_work,
//         workspaceInBytesOnDevice,
//         h_work.data(),
//         workspaceInBytesOnHost));
//     printf("Duration(before synchronize): %lf GFlops: %lf rank:%d\n", MPI_Wtime() - begin, (2 * m * n * k * 1e-9) / (MPI_Wtime() - begin), rank);
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     printf("Duration(after synchronize): %lf GFlops: %lf rank:%d\n", MPI_Wtime() - begin, (2 * m * n * k * 1e-9) / (MPI_Wtime() - begin), rank);
//     if(verbose){
//     }
//     CUDA_CHECK(cudaMemcpyAsync(C, d_C, lldc * loc_n_c * sizeof(output_t), cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaStreamSynchronize(stream));

//     CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descA));
//     CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descB));
//     CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descC));

//     CUBLASMP_CHECK(cublasMpGridDestroy(handle,grid));

//     CUBLASMP_CHECK(cublasMpDestroy(handle));

//     CUDA_CHECK(cudaFreeAsync(d_A, stream));
//     CUDA_CHECK(cudaFreeAsync(d_B, stream));
//     CUDA_CHECK(cudaFreeAsync(d_C, stream));
//     CUDA_CHECK(cudaFreeAsync(d_work, stream));

//     CAL_CHECK(cal_comm_destroy(cal_comm));
//     CUDA_CHECK(cudaStreamDestroy(stream));

//     MPI_Barrier(MPI_COMM_WORLD);

// }
void CudaConnector::pgemm_device(cublasMpHandle_t handle,cublasOperation_t transA,cublasOperation_t transB,const int &m,const int &n,const int &k,
                                    const void *alpha,
                                    const ComplexMatrixDevice &d_A,int64_t ia,int64_t ja,
                                    const ComplexMatrixDevice &d_B,int64_t ib,int64_t jb,
                                    const void *beta,
                                    ComplexMatrixDevice &d_C,int64_t ic,int64_t jc,
                                    cublasComputeType_t cublas_compute_type)
{
    void* d_work;
    
    size_t workspaceInBytesOnDevice,workspaceInBytesOnHost;
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(handle,transA,transB,m,n,k,
                                            alpha,
                                            (const void *)(d_A.ptr()),ia,ja,d_A.desc_cublas,
                                            (const void *)d_B.ptr(),ib,jb,d_B.desc_cublas,
                                            beta,
                                            d_C.ptr(),ic,jc,d_C.desc_cublas,
                                            cublas_compute_type,
                                            &workspaceInBytesOnDevice,
                                            &workspaceInBytesOnHost));
    CUDA_CHECK(cudaMalloc((void**)&d_work,workspaceInBytesOnDevice));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);
    CUBLASMP_CHECK(cublasMpGemm(handle,transA,transB,m,n,k,
                                alpha,
                                (const void *)d_A.ptr(),ia,ja,d_A.desc_cublas,
                                (const void *)d_B.ptr(),ib,jb,d_B.desc_cublas,
                                beta,
                                d_C.ptr(),ic,jc,d_C.desc_cublas,
                                cublas_compute_type,
                                d_work,workspaceInBytesOnDevice,
                                h_work.data(),workspaceInBytesOnHost));
    
    CUDA_CHECK(cudaFree(d_work));
}    

void CudaConnector::pgemm_nvhpc(const GpuDeviceStream& gpu_dev_stream,cublasOperation_t transA,cublasOperation_t transB,const int & m,const int & n,const int & k,
                        const void *alpha,
                        const ComplexMatrixDevice &d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
                        const ComplexMatrixDevice &d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
                        const void *beta,
                        ComplexMatrixDevice & d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
                        cublasComputeType_t cublas_compute_type)
{
    using input_t = cuDoubleComplex;
    using output_t = cuDoubleComplex;
    using compute_t = cuDoubleComplex;
    const cudaDataType_t cuda_input_type = CUDA_C_64F;
    const cudaDataType_t cuda_output_type = CUDA_C_64F;

    int64_t mbA = array_descA.mb();
    int64_t nbA = array_descA.nb();
    int64_t mbB = array_descB.mb();
    int64_t nbB = array_descB.nb();
    int64_t mbC = array_descC.mb();
    int64_t nbC = array_descC.nb();
    int nprow = array_descA.nprows();
    int npcol = array_descA.npcols();
    int llda= array_descA.lld();
    int lldb= array_descB.lld();
    int lldc= array_descC.lld();
    cublasMpGrid_t grid = nullptr;

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    cublasMpMatrixDescriptor_t descC = nullptr;
    char grid_layout = 'r';
    int rank=gpu_dev_stream.rank;
    int nranks=gpu_dev_stream.nranks;
    
    const int myprow = (grid_layout == 'c' ? rank % nprow : rank / npcol);
    const int mypcol = (grid_layout == 'c' ? rank / nprow : rank % npcol);
    
    cublasMpHandle_t handle = nullptr;
    CUBLASMP_CHECK(cublasMpCreate(&handle, gpu_dev_stream.stream));
    void* d_work = nullptr;


    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    int64_t global_m_a, global_n_a, global_m_b, global_n_b;
    if(transA == CUBLAS_OP_N){
        global_m_a = (ia - 1) + m;
        global_n_a = (ja - 1) + k;
    }else{
        global_m_a = (ia - 1) + k;
        global_n_a = (ja - 1) + m;
    }
    if(transB == CUBLAS_OP_N){
        global_m_b = (ib - 1) + k;
        global_n_b = (jb - 1) + n;
    }else{
        global_m_b = (ib - 1) + n;
        global_n_b = (jb - 1) + k;
    }
    const int64_t global_m_c = (ic - 1) + m;
    const int64_t global_n_c = (jc - 1) + n;
    
    CUBLASMP_CHECK(cublasMpGridCreate(
        handle, nprow, npcol,
        grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        gpu_dev_stream.cal_comm, &grid)
    );
    
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_a, global_n_a, mbA, nbA, 0, 0, llda, cuda_input_type, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, cuda_input_type, grid, &descB));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, cuda_output_type, grid, &descC));
    
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(
        handle, transA, transB,
        m, n, k,
        alpha,
        d_A.ptr(), ia, ja, descA,
        d_B.ptr(), ib, jb, descB,
        beta,
        d_C.ptr(), ic, jc, descC,
        cublas_compute_type,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost)
    );
    gpu_dev_stream.cudaSync();
    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, gpu_dev_stream.stream));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CUBLASMP_CHECK(cublasMpGemm(
        handle, transA, transB, 
        m, n, k,
        alpha,
        d_A.ptr(), ia, ja, descA,
        d_B.ptr(), ib, jb, descB,
        beta,
        d_C.ptr(), ic, jc, descC,
        cublas_compute_type,
        d_work, workspaceInBytesOnDevice,
        h_work.data(), workspaceInBytesOnHost
    ));
    
    gpu_dev_stream.cudaSync();
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descB));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descC));

    CUBLASMP_CHECK(cublasMpGridDestroy(handle,grid));

    CUBLASMP_CHECK(cublasMpDestroy(handle));
    CUDA_CHECK(cudaFree(d_work));

    MPI_Barrier(MPI_COMM_WORLD);
}
// void CudaConnector::pgemm_nvhpc_cuFloatComplex(const GpuDeviceStream& gpu_dev_stream,cublasOperation_t transA,cublasOperation_t transB,const int & m,const int & n,const int & k,
//                         const void *alpha,
//                         const cuFloatComplex* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
//                         const cuFloatComplex* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
//                         const void *beta,
//                         cuFloatComplex * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
//                         cublasComputeType_t cublas_compute_type)
// {
//     using input_t = cuFloatComplex;
//     using output_t = cuFloatComplex;
//     using compute_t = cuFloatComplex;
//     const cudaDataType_t cuda_input_type = CUDA_C_32F;
//     const cudaDataType_t cuda_output_type = CUDA_C_32F;

//     int64_t mbA = array_descA.mb();
//     int64_t nbA = array_descA.nb();
//     int64_t mbB = array_descB.mb();
//     int64_t nbB = array_descB.nb();
//     int64_t mbC = array_descC.mb();
//     int64_t nbC = array_descC.nb();
//     int nprow = array_descA.nprows();
//     int npcol = array_descA.npcols();
//     int llda= array_descA.lld();
//     int lldb= array_descB.lld();
//     int lldc= array_descC.lld();
//     cublasMpGrid_t grid = nullptr;

//     cublasMpMatrixDescriptor_t descA = nullptr;
//     cublasMpMatrixDescriptor_t descB = nullptr;
//     cublasMpMatrixDescriptor_t descC = nullptr;
//     char grid_layout = 'r';
//     int rank=gpu_dev_stream.rank;
//     int nranks=gpu_dev_stream.nranks;
//     // if(rank==0)
//     //     printf("sizeof(input_t)=%zu sizeof(output_t)=%zu sizeof(compute_t)=%zu\n", sizeof(input_t), sizeof(output_t), sizeof(compute_t));
//     const int myprow = (grid_layout == 'c' ? rank % nprow : rank / npcol);
//     const int mypcol = (grid_layout == 'c' ? rank / nprow : rank % npcol);
//     const int local_device = gpu_dev_stream.local_device;
//     cudaStream_t stream = gpu_dev_stream.stream;

//     cublasMpHandle_t handle = nullptr;
//     CUBLASMP_CHECK(cublasMpCreate(&handle, stream));
//     void* d_work = nullptr;


//     size_t workspaceInBytesOnDevice = 0;
//     size_t workspaceInBytesOnHost = 0;

//     const int64_t global_m_a = (ia - 1) + m;
//     const int64_t global_n_a = (ja - 1) + k;
//     const int64_t global_m_b = (ib - 1) + k;
//     const int64_t global_n_b = (jb - 1) + n;
//     const int64_t global_m_c = (ic - 1) + m;
//     const int64_t global_n_c = (jc - 1) + n;

//     gpu_dev_stream.cudaSync();
//     // printf("before create grid, rank:%d\n", rank);
//     CUBLASMP_CHECK(cublasMpGridCreate(
//         handle,
//         nprow,
//         npcol,
//         grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
//         gpu_dev_stream.cal_comm,
//         &grid));
//     // printf("after create grid, rank:%d\n", rank);
//     CUBLASMP_CHECK(
//         cublasMpMatrixDescriptorCreate(handle,global_m_a, global_n_a, mbA, nbA, 0, 0, llda, cuda_input_type, grid, &descA));
//     CUBLASMP_CHECK(
//         cublasMpMatrixDescriptorCreate(handle,global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, cuda_input_type, grid, &descB));
//     CUBLASMP_CHECK(
//         cublasMpMatrixDescriptorCreate(handle,global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, cuda_output_type, grid, &descC));
//     // printf("after create desc, rank:%d\n", rank);
//     CUBLASMP_CHECK(cublasMpGemm_bufferSize(
//         handle,
//         transA,
//         transB,
//         m,
//         n,
//         k,
//         alpha,
//         d_A,
//         ia,
//         ja,
//         descA,
//         d_B,
//         ib,
//         jb,
//         descB,
//         beta,
//         d_C,
//         ic,
//         jc,
//         descC,
//         cublas_compute_type,
//         &workspaceInBytesOnDevice,
//         &workspaceInBytesOnHost));
//     // printf("workspaceInBytesOnDevice=%zu, workspaceInBytesOnHost=%zu, rank:%d\n", workspaceInBytesOnDevice, workspaceInBytesOnHost, rank);
//     CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
//     std::vector<int8_t> h_work(workspaceInBytesOnHost);

//     gpu_dev_stream.cudaSync();

//     // const double begin = MPI_Wtime();
//     // printf("before gemm, rank:%d\n", rank);
//     CUBLASMP_CHECK(cublasMpGemm(
//         handle,
//         transA,
//         transB,
//         m,
//         n,
//         k,
//         alpha,
//         d_A,
//         ia,
//         ja,
//         descA,
//         d_B,
//         ib,
//         jb,
//         descB,
//         beta,
//         d_C,
//         ic,
//         jc,
//         descC,
//         cublas_compute_type,
//         d_work,
//         workspaceInBytesOnDevice,
//         h_work.data(),
//         workspaceInBytesOnHost));
//     // printf("after gemm, rank:%d\n", rank);
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descA));
//     CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descB));
//     CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descC));

//     CUBLASMP_CHECK(cublasMpGridDestroy(handle,grid));

//     CUBLASMP_CHECK(cublasMpDestroy(handle));
//     CUDA_CHECK(cudaFree(d_work));

//     MPI_Barrier(MPI_COMM_WORLD);
// }
void CudaConnector::pgemm_nvhpc_mixed_precision(
    const GpuDeviceStream& gpu_dev_stream,cublasOperation_t transA,cublasOperation_t transB,const int & m,const int & n,const int & k,
        const void *alpha,
        const void* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
        const void* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
        const void *beta,
        void * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
        cublasComputeType_t cublas_compute_type
)
{
    cudaDataType_t cuda_compute_type;
    if(cublas_compute_type == CUBLAS_COMPUTE_64F_PEDANTIC){
        cuda_compute_type = CUDA_C_64F;
    }else if(cublas_compute_type == CUBLAS_COMPUTE_32F_PEDANTIC){
        cuda_compute_type = CUDA_C_32F;
    }else{
        fprintf(stderr, "Unsupported cublas_compute_type\n");
    }
    int64_t mbA = array_descA.mb();
    int64_t nbA = array_descA.nb();
    int64_t mbB = array_descB.mb();
    int64_t nbB = array_descB.nb();
    int64_t mbC = array_descC.mb();
    int64_t nbC = array_descC.nb();
    int nprow = array_descA.nprows();
    int npcol = array_descA.npcols();
    int llda= array_descA.lld();
    int lldb= array_descB.lld();
    int lldc= array_descC.lld();
    cublasMpGrid_t grid = nullptr;

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    cublasMpMatrixDescriptor_t descC = nullptr;
    char grid_layout = 'r';
    int rank=gpu_dev_stream.rank;
    int nranks=gpu_dev_stream.nranks;
    // if(rank==0)
    //     printf("sizeof(input_t)=%zu sizeof(output_t)=%zu sizeof(compute_t)=%zu\n", sizeof(input_t), sizeof(output_t), sizeof(compute_t));
    const int myprow = (grid_layout == 'c' ? rank % nprow : rank / npcol);
    const int mypcol = (grid_layout == 'c' ? rank / nprow : rank % npcol);
    const int local_device = gpu_dev_stream.local_device;
    cudaStream_t stream = gpu_dev_stream.stream;

    cublasMpHandle_t handle = nullptr;
    // printf("before create handle, rank:%d\n");
    // CudaConnector::check_memory(gpu_dev_stream);
    CUBLASMP_CHECK(cublasMpCreate(&handle, stream));
    // printf("after create handle, rank:%d\n");
    // CudaConnector::check_memory(gpu_dev_stream);
    void* d_work = nullptr;


    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    int64_t global_m_a, global_n_a, global_m_b, global_n_b;
    if(transA == CUBLAS_OP_N){
        global_m_a = (ia - 1) + m;
        global_n_a = (ja - 1) + k;
    }else{
        global_m_a = (ia - 1) + k;
        global_n_a = (ja - 1) + m;
    }
    if(transB == CUBLAS_OP_N){
        global_m_b = (ib - 1) + k;
        global_n_b = (jb - 1) + n;
    }else{
        global_m_b = (ib - 1) + n;
        global_n_b = (jb - 1) + k;
    }
    const int64_t global_m_c = (ic - 1) + m;
    const int64_t global_n_c = (jc - 1) + n;

    // gpu_dev_stream.cudaSync();
    // printf("before create grid, rank:%d\n", rank);
    // CudaConnector::check_memory(gpu_dev_stream);
    CUBLASMP_CHECK(cublasMpGridCreate(
        handle,
        nprow,
        npcol,
        grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        gpu_dev_stream.cal_comm,
        &grid));
    // printf("after create grid, rank:%d\n", rank);
    // CudaConnector::check_memory(gpu_dev_stream);
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_a, global_n_a, mbA, nbA, 0, 0, llda, cuda_compute_type, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, cuda_compute_type, grid, &descB));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, cuda_compute_type, grid, &descC));
    
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(
        handle, transA, transB,
        m, n, k,
        alpha,
        d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        beta,
        d_C, ic, jc, descC,
        cublas_compute_type,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost)
    );
    // printf("workspaceInBytesOnDevice=%zu GiB, workspaceInBytesOnHost=%zu GiB, rank:%d\n", workspaceInBytesOnDevice, workspaceInBytesOnHost, rank);
    gpu_dev_stream.cudaSync();
    // printf("before malloc d_work, rank:%d\n");
    // CudaConnector::check_memory(gpu_dev_stream);
    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    // printf("after malloc d_work, rank:%d\n");
    // CudaConnector::check_memory(gpu_dev_stream);
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    // gpu_dev_stream.cudaSync();

    // const double begin = MPI_Wtime();
    CUBLASMP_CHECK(cublasMpGemm(
        handle, transA, transB,
        m, n, k,
        alpha,
        d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        beta,
        d_C, ic, jc, descC,
        cublas_compute_type,
        d_work, workspaceInBytesOnDevice,
        h_work.data(), workspaceInBytesOnHost)
    );
    // gpu_dev_stream.cudaSync();
    // printf("before free d_work, rank:%d\n");
    // CudaConnector::check_memory(gpu_dev_stream);
    CUDA_CHECK(cudaFreeAsync(d_work, stream));
    // printf("after free d_work, rank:%d\n");
    // CudaConnector::check_memory(gpu_dev_stream);
    gpu_dev_stream.cudaSync();
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descB));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descC));

    CUBLASMP_CHECK(cublasMpGridDestroy(handle,grid));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

    MPI_Barrier(MPI_COMM_WORLD);
}


void CudaConnector::pgemm_nvhpc_mixed_precision(
    cublasOperation_t transA,cublasOperation_t transB,const int & m,const int & n,const int & k,
    const void *alpha,
    const void* d_A,int64_t ia,int64_t ja,const Array_Desc& array_descA,
    const void* d_B,int64_t ib,int64_t jb,const Array_Desc& array_descB,
    const void *beta,
    void * d_C,int64_t ic,int64_t jc,const Array_Desc& array_descC,
    cublasComputeType_t cublas_compute_type
)
{
    cudaDataType_t cuda_compute_type;
    if(cublas_compute_type == CUBLAS_COMPUTE_64F_PEDANTIC){
        cuda_compute_type = CUDA_C_64F;
    }else if(cublas_compute_type == CUBLAS_COMPUTE_32F_PEDANTIC){
        cuda_compute_type = CUDA_C_32F;
    }else{
        fprintf(stderr, "Unsupported cublas_compute_type\n");
    }
    int64_t mbA = array_descA.mb();
    int64_t nbA = array_descA.nb();
    int64_t mbB = array_descB.mb();
    int64_t nbB = array_descB.nb();
    int64_t mbC = array_descC.mb();
    int64_t nbC = array_descC.nb();
    int nprow = array_descA.nprows();
    int npcol = array_descA.npcols();
    int llda= array_descA.lld();
    int lldb= array_descB.lld();
    int lldc= array_descC.lld();
    cublasMpGrid_t grid = nullptr;

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    cublasMpMatrixDescriptor_t descC = nullptr;
    char grid_layout = 'r';
    int rank=mpi_comm_global_h.myid;
    int nranks=mpi_comm_global_h.nprocs;
    // if(rank==0)
    //     printf("sizeof(input_t)=%zu sizeof(output_t)=%zu sizeof(compute_t)=%zu\n", sizeof(input_t), sizeof(output_t), sizeof(compute_t));
    const int myprow = (grid_layout == 'c' ? rank % nprow : rank / npcol);
    const int mypcol = (grid_layout == 'c' ? rank / nprow : rank % npcol);
    const int local_device = device_stream.local_device;
    cudaStream_t stream = (cudaStream_t)device_stream.stream;

    cublasMpHandle_t handle = nullptr;
    CUBLASMP_CHECK(cublasMpCreate(&handle, stream));
    void* d_work = nullptr;


    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    int64_t global_m_a, global_n_a, global_m_b, global_n_b;
    if(transA == CUBLAS_OP_N){
        global_m_a = (ia - 1) + m;
        global_n_a = (ja - 1) + k;
    }else{
        global_m_a = (ia - 1) + k;
        global_n_a = (ja - 1) + m;
    }
    if(transB == CUBLAS_OP_N){
        global_m_b = (ib - 1) + k;
        global_n_b = (jb - 1) + n;
    }else{
        global_m_b = (ib - 1) + n;
        global_n_b = (jb - 1) + k;
    }
    const int64_t global_m_c = (ic - 1) + m;
    const int64_t global_n_c = (jc - 1) + n;

    CUBLASMP_CHECK(cublasMpGridCreate(
        handle,
        nprow,
        npcol,
        grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        device_stream.cal_comm,
        &grid));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_a, global_n_a, mbA, nbA, 0, 0, llda, cuda_compute_type, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, cuda_compute_type, grid, &descB));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle,global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, cuda_compute_type, grid, &descC));
    
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(
        handle, transA, transB,
        m, n, k,
        alpha,
        d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        beta,
        d_C, ic, jc, descC,
        cublas_compute_type,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost)
    );
    device_stream.cudaSync();
    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    // const double begin = MPI_Wtime();
    CUBLASMP_CHECK(cublasMpGemm(
        handle, transA, transB,
        m, n, k,
        alpha,
        d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        beta,
        d_C, ic, jc, descC,
        cublas_compute_type,
        d_work, workspaceInBytesOnDevice,
        h_work.data(), workspaceInBytesOnHost)
    );
    
    CUDA_CHECK(cudaFreeAsync(d_work, stream));
    
    device_stream.cudaSync();
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descB));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle,descC));

    CUBLASMP_CHECK(cublasMpGridDestroy(handle,grid));

    CUBLASMP_CHECK(cublasMpDestroy(handle));
}
void CudaConnector::pgeadd_nvhpc(
    const GpuDeviceStream& gpu_dev_stream,const cublasOperation_t& trans,
    const void *alpha,
    const void* d_A, const int64_t& ia, const int64_t& ja, const Array_Desc& array_descA,
    const void* beta,
    void* d_B, const int64_t& ib, const int64_t& jb, const Array_Desc& array_descB,
    const cudaDataType_t& compute_type,
    const char& order
)
{
    cudaStream_t stream = gpu_dev_stream.stream;
    cublasMpHandle_t handle = nullptr;
    const int64_t mbA = array_descA.mb();
    const int64_t nbA = array_descA.nb();
    const int64_t mbB = array_descB.mb();
    const int64_t nbB = array_descB.nb();
    
    int nprow, npcol;
    assert(array_descA.nprows() == array_descB.nprows());
    assert(array_descA.npcols() == array_descB.npcols());
    ORDER_CHECK(order);
    if(order == 'c'||order == 'C'){
        nprow = array_descA.npcols();
        npcol = array_descA.nprows();
    }else{
        nprow = array_descA.nprows();
        npcol = array_descA.npcols();
    }
    int llda,loc_n_a,lldb,loc_n_b,mA,nA,mB,nB;
    if(order == 'c'||order == 'C'){
        llda = array_descA.n_loc();
        loc_n_a = array_descA.m_loc();
        lldb = array_descB.n_loc();
        loc_n_b = array_descB.m_loc();
        mA = array_descA.n();
        nA = array_descA.m();
        mB = array_descB.n();
        nB = array_descB.m();

    }else{
        llda = array_descA.m_loc();
        loc_n_a = array_descA.n_loc();
        lldb = array_descB.m_loc();
        loc_n_b = array_descB.n_loc();
        mA = array_descA.m();
        nA = array_descA.n();
        mB = array_descB.m();
        nB = array_descB.n();
    }
    if(trans == CUBLAS_OP_N){
        assert(mA == nA);
        assert(mB == nB);
    }else{
        assert(mA == nB);
        assert(mB == nA);
    }
    const int global_m_a = ia-1+mA;
    const int global_n_a = ja-1+nA;
    const int global_m_b = ib-1+mB;
    const int global_n_b = jb-1+nB;
    int rank = gpu_dev_stream.rank;
    int nranks = gpu_dev_stream.nranks;

    const int local_device = gpu_dev_stream.local_device;
    CUBLASMP_CHECK(cublasMpCreate(&handle, stream));
    cublasMpGrid_t grid = nullptr;
    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    void* d_work = nullptr;
    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    CUBLASMP_CHECK(cublasMpGridCreate(
        handle, nprow, npcol, 
        order=='c'||order=='C'?CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        gpu_dev_stream.cal_comm,
        &grid));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle, global_m_a, global_n_a, mbA, nbA, 0, 0, llda, compute_type, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle, global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, compute_type, grid, &descB));
    CUBLASMP_CHECK(cublasMpGeadd_bufferSize(
        handle, trans,
        mB, nB,
        alpha,
        d_A, ia, ja, descA,
        beta,
        d_B, ib, jb, descB,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost));
    gpu_dev_stream.calSync();
    
    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    gpu_dev_stream.calSync();
    std::vector<int8_t> h_work(workspaceInBytesOnHost);
    CUBLASMP_CHECK(cublasMpGeadd(
        handle, trans,
        mB, nB,
        alpha,
        d_A, ia, ja, descA,
        beta,
        d_B, ib, jb, descB,
        d_work, workspaceInBytesOnDevice,
        h_work.data(), workspaceInBytesOnHost));
    gpu_dev_stream.calSync();
    CUDA_CHECK(cudaFreeAsync(d_work, stream));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle, descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle, descB));
    CUBLASMP_CHECK(cublasMpGridDestroy(handle, grid));
    CUBLASMP_CHECK(cublasMpDestroy(handle));
}  

void CudaConnector::pgemr2d_nvhpc(
    const GpuDeviceStream& gpu_dev_stream,const int& m, const int& n,
    const void* d_A, const int64_t& ia, const int64_t& ja, const Array_Desc& array_descA,
    void* d_B, const int64_t& ib, const int64_t& jb, const Array_Desc& array_descB,
    const cudaDataType_t& compute_type
)
{
    
    cudaStream_t stream = gpu_dev_stream.stream;
    cublasMpHandle_t handle = nullptr;
    const int64_t mbA = array_descA.mb();
    const int64_t nbA = array_descA.nb();
    const int64_t mbB = array_descB.mb();
    const int64_t nbB = array_descB.nb();
    int nprow, npcol;
    assert(array_descA.nprows() == array_descB.nprows());
    assert(array_descA.npcols() == array_descB.npcols());
    nprow = array_descA.nprows();
    npcol = array_descA.npcols();
    
    int llda = array_descA.m_loc();
    int loc_n_a = array_descA.n_loc();

    int lldb = array_descB.m_loc();
    int loc_n_b = array_descB.n_loc();

    const int global_m_a = ia-1+m;
    const int global_n_a = ja-1+n;

    const int global_m_b = ib-1+m;
    const int global_n_b = jb-1+n;
    
    CUBLASMP_CHECK(cublasMpCreate(&handle, stream));
    cublasMpGrid_t grid = nullptr;
    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    void* d_work = nullptr;
    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    CUBLASMP_CHECK(cublasMpGridCreate(
        handle, nprow, npcol, 
        CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        gpu_dev_stream.cal_comm,
        &grid)
    );
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle, global_m_a, global_n_a, mbA, nbA, 0, 0, llda, compute_type, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(handle, global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, compute_type, grid, &descB));
    CUBLASMP_CHECK(cublasMpGemr2D_bufferSize(
        handle,
        m, n,
        d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost,
        gpu_dev_stream.cal_comm)
    );

    gpu_dev_stream.calSync();
    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CUBLASMP_CHECK(cublasMpGemr2D(
        handle,
        m, n,
        d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        d_work, workspaceInBytesOnDevice,
        h_work.data(), workspaceInBytesOnHost,
        gpu_dev_stream.cal_comm)
    );
    gpu_dev_stream.calSync();
    CUDA_CHECK(cudaFreeAsync(d_work, stream));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle, descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(handle, descB));
    CUBLASMP_CHECK(cublasMpGridDestroy(handle, grid));
    CUBLASMP_CHECK(cublasMpDestroy(handle));
} 
#endif

