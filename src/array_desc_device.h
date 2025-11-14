#include "base_blacs.h"
class Array_Desc_Device{
private:
    int ictxt_;
    // int nprocs_;
    // int myid_;
    int nprows_;
    int myprow_;
    int npcols_;
    int mypcol_;

    // Array dimensions
    int m_;
    int n_;
    int mb_;
    int nb_;
    int irsrc_;
    int icsrc_;
    int lld_;
    int m_local_;
    int n_local_;
    __host__ __device__ static int indx_g2p(
        const int &indxglob, const int &nb, const int &isrcproc, const int &nprocs);
    __host__ __device__ static int indx_g2l(
        const int &indxglob, const int &nb, const int &isrcproc, const int &nprocs);
public:
    Array_Desc_Device(const LIBRPA::Array_Desc& array_desc);
    __host__ __device__ 
    int indx_g2l_r(int gindx) const;
    __host__ __device__ 
    int indx_g2l_c(int gindx) const;
    __host__ __device__ 
    const int& m() const{ return m_; }
    __host__ __device__ 
    const int& n() const{ return n_; }
    __host__ __device__ 
    const int& mb() const{ return mb_; }
    __host__ __device__ 
    const int& nb() const{ return nb_; }
    __host__ __device__ 
    const int& irsrc() const{ return irsrc_; }
    __host__ __device__ 
    const int& icsrc() const{ return icsrc_; }
    __host__ __device__ 
    const int& lld() const{ return lld_; }
    __host__ __device__ 
    const int& m_loc() const{ return m_local_; }
    __host__ __device__ 
    const int& n_loc() const{ return n_local_; }
    __host__ __device__ 
    const int& nprows() const{ return nprows_; }
    __host__ __device__ 
    const int& npcols() const{ return npcols_; }
    __host__ __device__ 
    const int& myprow() const{ return myprow_; }
    __host__ __device__ 
    const int& mypcol() const{ return mypcol_; }
    __host__ __device__ 
    const int& ictxt() const{ return ictxt_; }

};