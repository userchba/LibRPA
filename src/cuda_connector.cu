#include "cuda_connector.h"
#include "device_stream.h"
// add by hbchen in 2025-05-19
__device__ int get_indxg2p(const int &indxglob, const int &nb, const int &iproc,
                              const int &isrcproc, const int &nprocs)
{
    return (isrcproc + indxglob / nb) % nprocs;
}
__device__ int get_indxg2l(const int &indxglob, const int &nb, const int &iproc,
                              const int &isrcproc, const int &nprocs)
{
    return nb * (indxglob / (nb * nprocs)) + indxglob % nb;
}
__device__ int get_index_g2l_r(const int& gindx, const int& m, const int& mb,const int &myprow, const int& irsrc, const int& nprows)
{
    return myprow != get_indxg2p(gindx, mb, myprow, irsrc, nprows) ||
                   gindx >= m
               ? -1
               : get_indxg2l(gindx, mb, myprow, irsrc, nprows);
}
__device__ int get_index_g2l_c(const int& gindx, const int &n, const int &nb, const int &mypcol, const int &icsrc, const int &npcols)
{
    return mypcol != get_indxg2p(gindx, nb, mypcol, icsrc, npcols) ||
                   gindx >= n
               ? -1
               : get_indxg2l(gindx, nb, mypcol, icsrc, npcols);
}
__global__ void det_multiply_seq(const cuDoubleComplex *inC,const int* d_ipiv,const int& num, cuDoubleComplex* ouC){
    int f = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("f = %d\n", f);
    ouC[f]=make_cuDoubleComplex(1.0, 0.0);
    for(int i=f;i<num;i+=blockDim.x){
        ouC[f]=cuCmul(ouC[f], inC[i*num+i]);
        if(d_ipiv[i]!=i+1){
            ouC[f].x=-ouC[f].x;
            ouC[f].y=-ouC[f].y;
        }
    }
}
__global__ void cuDoubleComplex_minus_identity_kernel(cuDoubleComplex *d_a,const int& d_n){
    int f= blockIdx.x*blockDim.x + threadIdx.x;
    if(f<d_n){
        d_a[f*d_n+f].x=d_a[f*d_n+f].x-1.0;
    }
}
__global__ void multiply_number_for_ComplexMatrixDevice_kernel(cuDoubleComplex* d_a, const cuDoubleComplex& d_num, const int& d_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_len)
    {
        d_a[idx] = cuCmul(d_a[idx], d_num);
    }
}
cuDoubleComplex CudaConnector::det_cuDoubleComplex(const cuDoubleComplex *d_a,const int *d_ipiv, const int& n, const cudaStream_t stream){
    int blockSize = 256;
    // printf("blockSize = %d\n", blockSize);
    int gridSize = 1;
    int *d_n;
    cudaMallocAsync((void**)&d_n, sizeof(int), stream);
    cudaMemcpyAsync(d_n, &n, sizeof(int), cudaMemcpyHostToDevice, stream);
    cuDoubleComplex *d_ouC;
    cuDoubleComplex *h_ouC;
    cudaHostAlloc((void**)&h_ouC, blockSize*sizeof(cuDoubleComplex), cudaHostAllocDefault);
    cuDoubleComplex ouC=make_cuDoubleComplex(1.0, 0.0);
    cudaMallocAsync((void**)&d_ouC, blockSize*sizeof(cuDoubleComplex), stream);
    // printf("start det_multiply\n");
    det_multiply_seq<<<gridSize, blockSize, 0, stream>>>(d_a,d_ipiv,*d_n,d_ouC);
    cudaMemcpyAsync(h_ouC, d_ouC, blockSize*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_ouC, stream);
    cudaFreeAsync(d_n, stream);
    for(int i=0;i<blockSize;i++){
        // printf("h_ouC[%d] = (%f, %f)\n", i, h_ouC[i].x, h_ouC[i].y);
        ouC=cuCmul(ouC, h_ouC[i]);
    }
    cudaFreeHost(h_ouC);
    return ouC;
}
void CudaConnector::cuDoubleComplex_minus_identity_Async(cuDoubleComplex *d_a,const int& h_n, const cudaStream_t stream){
    int blockSize = 256;
    int gridSize = (h_n + blockSize - 1) / blockSize;
    int * d_n;
    cudaMallocAsync((void**)&d_n, sizeof(int), stream);
    cudaMemcpyAsync(d_n, &h_n, sizeof(int), cudaMemcpyHostToDevice, stream);
    cuDoubleComplex_minus_identity_kernel<<<gridSize, blockSize, 0, stream>>>(d_a,*d_n);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_n, stream);
}
void CudaConnector::multiply_number_for_ComplexMatrixDevice(ComplexMatrixDevice& mat, const cuDoubleComplex& num, const cudaStream_t& stream)
{
    int blockSize = 256;
    cuDoubleComplex * d_num;
    int *d_len;
    CUDA_CHECK(cudaMallocAsync((void**)&d_len, sizeof(int), stream));
    int len = mat.nr()*mat.nc();
    CUDA_CHECK(cudaMemcpyAsync(d_len, &len, sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_num, sizeof(cuDoubleComplex), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_num, &num, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
    int gridSize = (mat.nr()*mat.nc() + blockSize - 1) / blockSize;
    multiply_number_for_ComplexMatrixDevice_kernel<<<gridSize, blockSize, 0, stream>>>(mat.ptr(), *d_num, *d_len);
}
__global__ void diag_add_ComplexMatrixDevice_kernel(cuDoubleComplex* d_a, const double& d_num, const int& d_m, const int& d_n, const int& d_m_loc, const int& d_n_loc, const int& d_mb, const int& d_nb, const int& d_myprow, const int& d_mypcol, const int& d_irsrc, const int& d_icsrc, const int& d_nprows, const int& d_npcols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < d_m)
    {
        int col = row; // Diagonal element
        int local_row = get_index_g2l_r(row, d_m, d_mb, d_myprow, d_irsrc, d_nprows);
        int local_col = get_index_g2l_c(col, d_n, d_nb, d_mypcol, d_icsrc, d_npcols);
        if (local_row != -1 && local_col != -1 && local_row < d_m_loc && local_col < d_n_loc)
        {
            int local_index = local_row + local_col * d_m_loc; // Column-major order
            d_a[local_index].x += d_num; // Add the real number to the real part
        }
    }
}

void CudaConnector::diag_add_ComplexMatrixDevice(ComplexMatrixDevice& mat, const double& num, const Array_Desc& arrdesc,const cudaStream_t& stream)
{
    double *d_num;
    int blockSize = 256;
    CUDA_CHECK(cudaMallocAsync((void**)&d_num, sizeof(double), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_num, &num, sizeof(double), cudaMemcpyHostToDevice, stream));
    int *d_m,*d_n,*d_m_loc,*d_n_loc,*d_mb,*d_nb,*d_myprow,*d_mypcol,*d_irsrc,*d_icsrc,*d_nprows,*d_npcols;
    CUDA_CHECK(cudaMallocAsync((void**)&d_m, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_n, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_m_loc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_n_loc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_mb, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_nb, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_myprow, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_mypcol, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_irsrc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_icsrc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_nprows, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_npcols, sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_m, &arrdesc.m(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n, &arrdesc.n(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_m_loc, &arrdesc.m_loc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n_loc, &arrdesc.n_loc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mb, &arrdesc.mb(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_nb, &arrdesc.nb(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_myprow, &arrdesc.myprow(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mypcol, &arrdesc.mypcol(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_irsrc, &arrdesc.irsrc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_icsrc, &arrdesc.icsrc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_nprows, &arrdesc.nprows(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_npcols, &arrdesc.npcols(), sizeof(int), cudaMemcpyHostToDevice, stream));
    int gridSize = (arrdesc.m() + blockSize - 1) / blockSize;
    diag_add_ComplexMatrixDevice_kernel<<<gridSize, blockSize, 0, stream>>>(mat.ptr(),*d_num,*d_m,*d_n,*d_m_loc,*d_n_loc,*d_mb,*d_nb,*d_myprow,*d_mypcol,*d_irsrc,*d_icsrc,*d_nprows,*d_npcols);
    CUDA_CHECK(cudaFreeAsync(d_num, stream));
    CUDA_CHECK(cudaFreeAsync(d_m, stream));
    CUDA_CHECK(cudaFreeAsync(d_n, stream));
    CUDA_CHECK(cudaFreeAsync(d_m_loc, stream));
    CUDA_CHECK(cudaFreeAsync(d_n_loc, stream));
    CUDA_CHECK(cudaFreeAsync(d_mb, stream));
    CUDA_CHECK(cudaFreeAsync(d_nb, stream));
    CUDA_CHECK(cudaFreeAsync(d_myprow, stream));
    CUDA_CHECK(cudaFreeAsync(d_mypcol, stream));
    CUDA_CHECK(cudaFreeAsync(d_irsrc, stream));
    CUDA_CHECK(cudaFreeAsync(d_icsrc, stream));
    CUDA_CHECK(cudaFreeAsync(d_nprows, stream));
    CUDA_CHECK(cudaFreeAsync(d_npcols, stream));

}
__global__ void diag_add_matrix_device_blacs_kernel(const cuDoubleComplex* num, cuDoubleComplex* d_A, const Array_Desc_Device& array_desc_device)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=array_desc_device.m())
        return;
    int ilo = array_desc_device.indx_g2l_r(i);
    if(ilo==-1)
        return;
    int jlo = array_desc_device.indx_g2l_c(i);
    if(jlo==-1)
        return;
    if(ilo<array_desc_device.m_loc() && jlo<array_desc_device.n_loc())
    {
        d_A[ilo + jlo*array_desc_device.m_loc()] = cuCadd(d_A[ilo + jlo*array_desc_device.m_loc()], *num);
    }
}
void CudaConnector::diag_add_matrix_device_blacs(
    const void* num, void* d_A, const Array_Desc& array_desc,const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type
){
    Array_Desc_Device array_desc_device(array_desc);
    Array_Desc_Device* d_array_desc_device;
    void* d_num;
    CUDA_CHECK(cudaMallocAsync((void**)&d_array_desc_device, sizeof(Array_Desc_Device), (cudaStream_t)device_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_array_desc_device, &array_desc_device, sizeof(Array_Desc_Device), cudaMemcpyHostToDevice, (cudaStream_t)device_stream.stream));
    int blockSize = 256;
    int gridSize = (array_desc_device.m() + blockSize - 1) / blockSize;
    if(compute_type == LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE){
        CUDA_CHECK(cudaMallocAsync((void**)&d_num, sizeof(cuDoubleComplex), (cudaStream_t)device_stream.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_num, num, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, (cudaStream_t)device_stream.stream));
        
        diag_add_matrix_device_blacs_kernel<<<gridSize, blockSize, 0, (cudaStream_t)device_stream.stream>>>((cuDoubleComplex*)d_num,(cuDoubleComplex*)d_A,*d_array_desc_device);
       
    }else if(compute_type == LIBRPA_COMPUTE_TYPE_COMPLEX_FLOAT){
        CUDA_CHECK(cudaMallocAsync((void**)&d_num, sizeof(cuFloatComplex), (cudaStream_t)device_stream.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_num, num, sizeof(cuFloatComplex), cudaMemcpyHostToDevice, (cudaStream_t)device_stream.stream));
    }else if(compute_type == LIBRPA_COMPUTE_TYPE_DOUBLE){
        CUDA_CHECK(cudaMallocAsync((void**)&d_num, sizeof(double), (cudaStream_t)device_stream.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_num, num, sizeof(double), cudaMemcpyHostToDevice, (cudaStream_t)device_stream.stream));
    }else{
        CUDA_CHECK(cudaMallocAsync((void**)&d_num, sizeof(float), (cudaStream_t)device_stream.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_num, num, sizeof(float), cudaMemcpyHostToDevice, (cudaStream_t)device_stream.stream));
    }
    CUDA_CHECK(cudaFreeAsync(d_num, (cudaStream_t)device_stream.stream));
    CUDA_CHECK(cudaFreeAsync(d_array_desc_device, (cudaStream_t)device_stream.stream));

}
__global__ void det_ComplexMatrixDevice_blacs_kernel(const cuDoubleComplex* d_in,cuDoubleComplex* d_out,const int& m,const int& n,const int& m_loc,const int& n_loc,const int& mb,const int& nb,const int& myprow,const int& mypcol,const int& irsrc,const int& icsrc,const int& nprows,const int& npcols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("idx=%d\n",idx);
    d_out[idx] = make_cuDoubleComplex(1.0,0.0);
    
    for(int i=idx;i<m;i+=blockDim.x)
    {
        int local_row = get_index_g2l_r(i, m, mb, myprow, irsrc, nprows);
        int local_col = get_index_g2l_c(i, m, mb, mypcol, icsrc, npcols);
        if (local_row != -1 && local_col != -1 && local_row < m_loc && local_col < n_loc)
        {
            d_out[idx] = cuCmul(d_out[idx],d_in[local_row + local_col*m_loc]);
        }
    }
}
std::complex<double> CudaConnector::det_ComplexMatrixDevice_blacs(const cudaStream_t& stream, const ComplexMatrixDevice&d_A, const LIBRPA::Array_Desc &arrdesc_pi)
{
    std::complex<double> det_loc={1.0,0.0};
    int *d_m,*d_n,*d_m_loc,*d_n_loc,*d_mb,*d_nb,*d_myprow,*d_mypcol,*d_irsrc,*d_icsrc,*d_nprows,*d_npcols;
    CUDA_CHECK(cudaMallocAsync((void**)&d_m, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_n, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_m_loc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_n_loc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_mb, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_nb, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_myprow, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_mypcol, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_irsrc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_icsrc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_nprows, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_npcols, sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_m, &arrdesc_pi.m(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n, &arrdesc_pi.n(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_m_loc, &arrdesc_pi.m_loc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n_loc, &arrdesc_pi.n_loc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mb, &arrdesc_pi.mb(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_nb, &arrdesc_pi.nb(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_myprow, &arrdesc_pi.myprow(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mypcol, &arrdesc_pi.mypcol(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_irsrc, &arrdesc_pi.irsrc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_icsrc, &arrdesc_pi.icsrc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_nprows, &arrdesc_pi.nprows(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_npcols, &arrdesc_pi.npcols(), sizeof(int), cudaMemcpyHostToDevice, stream));
    int blockSize = 256;
    int gridSize = 1;
    cuDoubleComplex* d_detA;
    std::complex<double> *h_detA = new std::complex<double>[blockSize];
    CUDA_CHECK(cudaMallocAsync((void**)&d_detA, sizeof(cuDoubleComplex) * blockSize, stream));
    // printf("before det_ComplexMatrixDevice_blacs_kernel\n");
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    det_ComplexMatrixDevice_blacs_kernel<<<gridSize, blockSize, 0, stream>>>(d_A.ptr(),d_detA,*d_m,*d_n,*d_m_loc,*d_n_loc,*d_mb,*d_nb,*d_myprow,*d_mypcol,*d_irsrc,*d_icsrc,*d_nprows,*d_npcols);
    CUDA_CHECK(cudaFreeAsync(d_m, stream));
    CUDA_CHECK(cudaFreeAsync(d_n, stream));
    CUDA_CHECK(cudaFreeAsync(d_m_loc, stream));
    CUDA_CHECK(cudaFreeAsync(d_n_loc, stream));
    CUDA_CHECK(cudaFreeAsync(d_mb, stream));
    CUDA_CHECK(cudaFreeAsync(d_nb, stream));
    CUDA_CHECK(cudaFreeAsync(d_myprow, stream));
    CUDA_CHECK(cudaFreeAsync(d_mypcol, stream));
    CUDA_CHECK(cudaFreeAsync(d_irsrc, stream));
    CUDA_CHECK(cudaFreeAsync(d_icsrc, stream));
    CUDA_CHECK(cudaFreeAsync(d_nprows, stream));
    CUDA_CHECK(cudaFreeAsync(d_npcols, stream));
    // printf("after det_ComplexMatrixDevice_blacs_kernel\n");
    CUDA_CHECK(cudaMemcpyAsync(h_detA, d_detA, sizeof(cuDoubleComplex) * blockSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_detA, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int i=0;i<blockSize;i++)
    {
        det_loc*=h_detA[i];
    }
    delete[] h_detA;
    return det_loc;
}
__global__ void transpose_kernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, const int& d_m,const int& d_n, const bool& d_conjugate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d processing index %d\n", idx, idx);
    if (idx < d_m * d_n)
    {
        // data was stored in column-major order
        int row = idx % d_m;
        int col = idx / d_m;
        int transpose_id= col + row * d_n;
        d_out[transpose_id] = d_in[idx];
        if (d_conjugate){
            d_out[transpose_id].y = -d_out[transpose_id].y;
        }
        
    }
}

void CudaConnector::transpose_ComplexMatrixDevice(const GpuDeviceStream& gpu_dev_stream, ComplexMatrixDevice& in ,bool is_conjugate)
{
    int *d_m_ptr,*d_n_ptr;
    bool *d_conjugate_ptr;
    int m = in.nr();
    int n = in.nc();
    CUDA_CHECK(cudaMallocAsync((void**)&d_m_ptr, sizeof(int), gpu_dev_stream.stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_n_ptr, sizeof(int), gpu_dev_stream.stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_conjugate_ptr, sizeof(bool), gpu_dev_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_m_ptr, &m, sizeof(int), cudaMemcpyHostToDevice, gpu_dev_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n_ptr, &n, sizeof(int), cudaMemcpyHostToDevice, gpu_dev_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_conjugate_ptr, &is_conjugate, sizeof(bool), cudaMemcpyHostToDevice, gpu_dev_stream.stream));

    const cuDoubleComplex* d_in = in.ptr();
    cuDoubleComplex* d_out;
    CUDA_CHECK(cudaMallocAsync((void**)&d_out, sizeof(cuDoubleComplex) * n * m, gpu_dev_stream.stream));
    int blockSize = 256;
    int gridSize = (m * n + blockSize - 1) / blockSize;
    // printf("before transpose kernel, gridSize=%d, blockSize=%d, m=%d, n=%d, is_conjugate=%d\n", gridSize, blockSize, m, n, is_conjugate);
    transpose_kernel<<<gridSize, blockSize, 0, gpu_dev_stream.stream>>>(d_in, d_out, *d_m_ptr, *d_n_ptr, *d_conjugate_ptr);
    // printf("after transpose kernel\n");
    CUDA_CHECK(cudaFreeAsync(d_m_ptr, gpu_dev_stream.stream));
    CUDA_CHECK(cudaFreeAsync(d_n_ptr, gpu_dev_stream.stream));
    CUDA_CHECK(cudaFreeAsync(d_conjugate_ptr, gpu_dev_stream.stream));
    gpu_dev_stream.cudaSync();
    in.set_ptr(d_out);
    in.set_nr(n);
    in.set_nc(m);
}
__global__ void cuFloatComplex_to_cuDoubleComplex_kernel(const cuFloatComplex* d_in, cuDoubleComplex* d_out, const int64_t& d_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_len)
    {
        d_out[idx] = make_cuDoubleComplex(static_cast<double>(d_in[idx].x), static_cast<double>(d_in[idx].y));
    }
}
void CudaConnector::cuFloatComplex_to_cuDoubleComplex_Async(const cuFloatComplex* d_in, cuDoubleComplex* d_out, const int64_t& len, const cudaStream_t& stream)
{
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    // printf("gridSize = %d, blockSize = %d, len = %d\n", gridSize, blockSize, len);
    // Launch the kernel
    int64_t *d_len;
    CUDA_CHECK(cudaMallocAsync((void**)&d_len, sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_len, &len, sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    cuFloatComplex_to_cuDoubleComplex_kernel<<<gridSize, blockSize, 0, stream>>>(d_in, d_out, *d_len);
}
__global__ void cuDoubleComplex_to_cuFloatComplex_kernel(const cuDoubleComplex* d_a, cuFloatComplex* d_b, const int64_t& d_len)
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < d_len)
    {
        d_b[idx] = make_cuFloatComplex(static_cast<float>(d_a[idx].x), static_cast<float>(d_a[idx].y));
    }
}
void CudaConnector::cuDoubleComplex_to_cuFloatComplex_Async(const cuDoubleComplex* d_a, cuFloatComplex* d_b, const int64_t& len, const cudaStream_t& stream)
{
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    // printf("gridSize = %d, blockSize = %d, len = %d\n", gridSize, blockSize, len);
    // Launch the kernel
    int64_t *d_len;
    CUDA_CHECK(cudaMallocAsync((void**)&d_len, sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_len, &len, sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    // printf("start kernel function\n");
    cuDoubleComplex_to_cuFloatComplex_kernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, *d_len);
}
__global__ void float_to_double_kernel(const float* d_a, double* d_b, const int64_t& d_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_len)
    {
        d_b[idx] = d_a[idx];
    }
}
void CudaConnector::float_to_double_device(const float* d_a, double* d_b, const int64_t& len)
{
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    int64_t *d_len;
    CUDA_CHECK(cudaMallocAsync((void**)&d_len, sizeof(int64_t), device_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_len, &len, sizeof(int64_t), cudaMemcpyHostToDevice, device_stream.stream));
    float_to_double_kernel<<<gridSize, blockSize, 0, device_stream.stream>>>(d_a, d_b, *d_len);
    CUDA_CHECK(cudaFreeAsync(d_len, device_stream.stream));
}
__global__ void double_to_float_kernel(const double* d_a, float* d_b, const int64_t& d_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_len)
    {
        d_b[idx] = static_cast<float>(d_a[idx]);
    }
}
void CudaConnector::double_to_float_device(const double* d_a, float* d_b, const int64_t& len)
{
    int blockSize = 256;
    int gridSize = (len + blockSize - 1) / blockSize;
    int64_t *d_len;
    CUDA_CHECK(cudaMallocAsync((void**)&d_len, sizeof(int64_t), device_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_len, &len, sizeof(int64_t), cudaMemcpyHostToDevice, device_stream.stream));
    double_to_float_kernel<<<gridSize, blockSize, 0, device_stream.stream>>>(d_a, d_b, *d_len);
    CUDA_CHECK(cudaFreeAsync(d_len, device_stream.stream));
}
__global__ void trace_ComplexMatrixDevice_blacs_kernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, const int& d_m, const int& d_n, const int& d_m_loc, const int& d_n_loc, const int& d_mb, const int& d_nb, const int& d_myprow, const int& d_mypcol, const int& d_irsrc, const int& d_icsrc, const int& d_nprows, const int& d_npcols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[idx] = make_cuDoubleComplex(0.0, 0.0);
    for(int i=idx;i<d_m;i+=blockDim.x){
        int local_row = get_index_g2l_r(i, d_m, d_mb, d_myprow, d_irsrc, d_nprows);
        int local_col = get_index_g2l_c(i, d_n, d_nb, d_mypcol, d_icsrc, d_npcols);
        if (local_row != -1 && local_col != -1 && local_row < d_m_loc && local_col < d_n_loc)
        {
            d_out[idx] = cuCadd(d_out[idx], d_in[local_row + local_col * d_m_loc]);
        }
    }
}

__global__ void trace_matrix_device_blacs_kernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, const Array_Desc_Device& array_desc_device)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[idx] = make_cuDoubleComplex(0.0, 0.0);
    for(int i=idx;i<array_desc_device.m();i+=blockDim.x){
        int local_row = array_desc_device.indx_g2l_r(i);
        int local_col = array_desc_device.indx_g2l_c(i);
        if (local_row != -1 && local_col != -1 && local_row < array_desc_device.m_loc() && local_col < array_desc_device.n_loc())
        {
            d_out[idx] = cuCadd(d_out[idx], d_in[local_row + local_col * array_desc_device.m_loc()]);
        }
    }
}
void CudaConnector::trace_matrix_device_blacs(void* trace, const void* d_A, const Array_Desc &arrdesc_A, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type)
{
    Array_Desc_Device array_desc_device(arrdesc_A);
    Array_Desc_Device* d_array_desc_device;
    CUDA_CHECK(cudaMallocAsync((void**)&d_array_desc_device, sizeof(Array_Desc_Device), (cudaStream_t)device_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_array_desc_device, &array_desc_device, sizeof(Array_Desc_Device), cudaMemcpyHostToDevice, (cudaStream_t)device_stream.stream));
    int blockSize = 256;
    int gridSize = 1;
    void* d_trace;
    void* h_trace;
    if(compute_type == LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE){
        std::complex<double> *trace_z = (std::complex<double>*)trace;
        CUDA_CHECK(cudaMallocAsync((void**)&d_trace, blockSize*sizeof(std::complex<double>), device_stream.stream));
        h_trace = (std::complex<double>*)malloc(blockSize*sizeof(std::complex<double>));
        std::complex<double> *h_trace_z = (std::complex<double>*)h_trace;
        *(trace_z) = std::complex<double>(0.0, 0.0);
        trace_matrix_device_blacs_kernel<<<gridSize, blockSize, 0, device_stream.stream>>>((const cuDoubleComplex*)d_A, (cuDoubleComplex*)d_trace, *d_array_desc_device);
        CUDA_CHECK(cudaMemcpyAsync(h_trace, d_trace, blockSize*sizeof(std::complex<double>), cudaMemcpyDeviceToHost, device_stream.stream));
        device_stream.sync();
        for(int i=0;i<blockSize;i++){
            *(trace_z) += h_trace_z[i];
        }
    }
    CUDA_CHECK(cudaFreeAsync(d_array_desc_device,device_stream.stream));
    CUDA_CHECK(cudaFreeAsync(d_trace,device_stream.stream));
    free(h_trace);
}
__global__ void det_matrix_device_blacs_kernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, const Array_Desc_Device& array_desc_device)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row,local_col;
    d_out[idx] = make_cuDoubleComplex(1.0, 0.0);
    for(int i=idx;i<array_desc_device.m();i+=blockDim.x){
        local_row = array_desc_device.indx_g2l_r(i);
        local_col = array_desc_device.indx_g2l_c(i);
        if (local_row != -1 && local_col != -1 && local_row < array_desc_device.m_loc() && local_col < array_desc_device.n_loc())
        {
            d_out[idx] = cuCmul(d_out[idx], d_in[local_row + local_col * array_desc_device.m_loc()]);
        }
    }
}
void CudaConnector::det_matrix_device_blacs(void* det, const void* d_A, const Array_Desc &arrdesc_A, const LIBRPA_DEVICE_COMPUTE_TYPE& compute_type)
{
    Array_Desc_Device array_desc_device(arrdesc_A);
    Array_Desc_Device* d_array_desc_device;
    CUDA_CHECK(cudaMallocAsync((void**)&d_array_desc_device, sizeof(Array_Desc_Device), (cudaStream_t)device_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_array_desc_device, &array_desc_device, sizeof(Array_Desc_Device), cudaMemcpyHostToDevice, (cudaStream_t)device_stream.stream));
    int blockSize = 256;
    int gridSize = 1;
    void* d_det;
    void* h_det;
    if(compute_type==LIBRPA_COMPUTE_TYPE_COMPLEX_DOUBLE){
        std::complex<double> *det_z = (std::complex<double>*)det;
        CUDA_CHECK(cudaMallocAsync((void**)&d_det, blockSize*sizeof(std::complex<double>), device_stream.stream));
        h_det = (std::complex<double>*)malloc(blockSize*sizeof(std::complex<double>));
        std::complex<double> *h_det_z = (std::complex<double>*)h_det;
        *(det_z) = std::complex<double>(1.0, 0.0);
        det_matrix_device_blacs_kernel<<<gridSize, blockSize, 0, device_stream.stream>>>((const cuDoubleComplex*)d_A, (cuDoubleComplex*)d_det, *d_array_desc_device);
        CUDA_CHECK(cudaMemcpyAsync(h_det, d_det, blockSize*sizeof(std::complex<double>), cudaMemcpyDeviceToHost, device_stream.stream));
        device_stream.sync();
        for(int i=0;i<blockSize;i++){
            *(det_z) *= h_det_z[i];
        }
    }
    CUDA_CHECK(cudaFreeAsync(d_array_desc_device,device_stream.stream));
    CUDA_CHECK(cudaFreeAsync(d_det,device_stream.stream));
    free(h_det);
}
std::complex<double> CudaConnector::trace_ComplexMatrixDevice_blacs(const cudaStream_t& stream, const ComplexMatrixDevice& d_A, const LIBRPA::Array_Desc &arrdesc)
{
    
    int blockSize = 256;
    int *d_m,*d_n,*d_m_loc,*d_n_loc,*d_mb,*d_nb,*d_myprow,*d_mypcol,*d_irsrc,*d_icsrc,*d_nprows,*d_npcols;
    CUDA_CHECK(cudaMallocAsync((void**)&d_m, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_n, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_m_loc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_n_loc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_mb, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_nb, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_myprow, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_mypcol, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_irsrc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_icsrc, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_nprows, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_npcols, sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_m, &arrdesc.m(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n, &arrdesc.n(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_m_loc, &arrdesc.m_loc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n_loc, &arrdesc.n_loc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mb, &arrdesc.mb(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_nb, &arrdesc.nb(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_myprow, &arrdesc.myprow(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mypcol, &arrdesc.mypcol(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_irsrc, &arrdesc.irsrc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_icsrc, &arrdesc.icsrc(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_nprows, &arrdesc.nprows(), sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_npcols, &arrdesc.npcols(), sizeof(int), cudaMemcpyHostToDevice, stream));
    cuDoubleComplex *d_out;
    CUDA_CHECK(cudaMallocAsync((void**)&d_out, sizeof(cuDoubleComplex)*blockSize, stream));
    std::complex<double> *h_out;
    std::complex<double> trace_loc = {0.0, 0.0};
    h_out = new std::complex<double>[blockSize];
    int gridSize = 1;
    trace_ComplexMatrixDevice_blacs_kernel<<<gridSize, blockSize, 0, stream>>>(d_A.ptr(),d_out,*d_m,*d_n,*d_m_loc,*d_n_loc,*d_mb,*d_nb,*d_myprow,*d_mypcol,*d_irsrc,*d_icsrc,*d_nprows,*d_npcols);
    CUDA_CHECK(cudaFreeAsync(d_m, stream));
    CUDA_CHECK(cudaFreeAsync(d_n, stream));
    CUDA_CHECK(cudaFreeAsync(d_m_loc, stream));
    CUDA_CHECK(cudaFreeAsync(d_n_loc, stream));
    CUDA_CHECK(cudaFreeAsync(d_mb, stream));
    CUDA_CHECK(cudaFreeAsync(d_nb, stream));
    CUDA_CHECK(cudaFreeAsync(d_myprow, stream));
    CUDA_CHECK(cudaFreeAsync(d_mypcol, stream));
    CUDA_CHECK(cudaFreeAsync(d_irsrc, stream));
    CUDA_CHECK(cudaFreeAsync(d_icsrc, stream));
    CUDA_CHECK(cudaFreeAsync(d_nprows, stream));
    CUDA_CHECK(cudaFreeAsync(d_npcols, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, sizeof(cuDoubleComplex)*blockSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int i=0;i<blockSize;i++)
    {
        trace_loc += h_out[i];
    }
    delete[] h_out;
    return trace_loc;
}
void ComplexMatrixDevice::set_as_zero(const cudaStream_t& stream){
    if(d_data == nullptr|| m==0 || n==0){
        fprintf(stderr,"Error: ComplexMatrixDevice::set_as_zero: d_data is null or m or n is zero\n");
        std::abort();
    }
    if(stream == nullptr){
        CUDA_CHECK(cudaMemset(d_data, 0, sizeof(cuDoubleComplex)*m*n));
    }else{
        CUDA_CHECK(cudaMemsetAsync(d_data, 0, sizeof(cuDoubleComplex)*m*n, stream));
    }
}
__global__ void set_as_identity_kernel(cuDoubleComplex*d_A, const Array_Desc_Device& array_desc_device){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=array_desc_device.m())
        return;
    int ilo = array_desc_device.indx_g2l_r(i);
    if(ilo==-1)
        return;
    int jlo = array_desc_device.indx_g2l_c(i);
    if(jlo==-1)
        return;
    d_A[ilo+jlo*array_desc_device.m_loc()] = make_cuDoubleComplex(1.0, 0.0);
}
void ComplexMatrixDevice::set_as_identity(const GpuDeviceStream& gpu_dev_stream, const Array_Desc_Device& array_desc_device){
    this->set_as_zero(gpu_dev_stream.stream);
    Array_Desc_Device* d_array_desc_device;
    // printf("array_desc_device.m():%d\n", array_desc_device.m());
    CUDA_CHECK(cudaMallocAsync((void**)&d_array_desc_device, sizeof(Array_Desc_Device), gpu_dev_stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_array_desc_device, &array_desc_device, sizeof(Array_Desc_Device), cudaMemcpyHostToDevice, gpu_dev_stream.stream));
    if(array_desc_device.m()!=array_desc_device.n()){
        fprintf(stderr,"Error: ComplexMatrixDevice::set_as_identity: m and n are not equal\n");
        std::abort();
    }
    int blockSize = 256;
    int gridSize = (array_desc_device.m()+blockSize-1)/blockSize;
    set_as_identity_kernel<<<gridSize, blockSize, 0, gpu_dev_stream.stream>>>(d_data, *d_array_desc_device);
    CUDA_CHECK(cudaFreeAsync(d_array_desc_device, gpu_dev_stream.stream));
}