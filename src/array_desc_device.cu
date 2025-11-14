#include "array_desc_device.h"

Array_Desc_Device::Array_Desc_Device(const LIBRPA::Array_Desc& array_desc) {
    this->ictxt_ = array_desc.ictxt();
    this->m_ = array_desc.m();
    this->n_ = array_desc.n();
    this->mb_ = array_desc.mb();
    this->nb_ = array_desc.nb();
    this->lld_ = array_desc.lld();
    this->irsrc_ = array_desc.irsrc();
    this->icsrc_ = array_desc.icsrc();
    this->m_local_ = array_desc.m_loc();
    this->n_local_ = array_desc.n_loc();
    this->myprow_ = array_desc.myprow();
    this->mypcol_ = array_desc.mypcol();
    this->nprows_ = array_desc.nprows();
    this->npcols_ = array_desc.npcols();
}
__host__ __device__ int Array_Desc_Device::indx_g2p(const int &indxglob, const int &nb, const int &isrcproc, const int &nprocs) {
    return (isrcproc + indxglob / nb) % nprocs;
}
__host__ __device__ int Array_Desc_Device::indx_g2l(const int &indxglob, const int &nb, const int &isrcproc, const int &nprocs) {
    return nb * (indxglob / (nb * nprocs)) + indxglob % nb;
}
__host__ __device__ int Array_Desc_Device::indx_g2l_r(int gindx)const{
    return this->myprow_ != indx_g2p(gindx, this->mb_, this->irsrc_, this->nprows_) ||
                   gindx >= this->m_
               ? -1
               : indx_g2l(gindx, this->mb_, this->irsrc_, this->nprows_);
}
__host__ __device__ int Array_Desc_Device::indx_g2l_c(int gindx)const{
    return this->mypcol_ != indx_g2p(gindx, this->nb_, this->icsrc_, this->npcols_) ||
                   gindx >= this->n_
               ? -1
               : indx_g2l(gindx, this->nb_, this->icsrc_, this->npcols_);
}