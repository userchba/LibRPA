#ifndef DEVICE_CONNECTOR_H
#define DEVICE_CONNECTOR_H

#include "../mpi/base_blacs.h"

namespace librpa_int{
namespace DeviceConnector{
    // static void float_to_double_device(float* d_A, double* d_B, const int64_t& n);
    // static void double_to_float_device(double* d_A, float* d_B, const int64_t& n);
    // static void
bool check_device_ptr(void* A);

template<typename T1, typename T2>
void pdam(const T1& num, T2* d_A, const ArrayDesc& array_desc);

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
// DdlaStream became an opaque forward declaration in the current LibDDLA
// (the concrete class moved to a private header), so `handle->stream` no
// longer compiles. Route every call site through this accessor instead,
// which wraps the new ddla::ddla_get_stream(handle) (returns void*).
inline ddla::runtimeStream_t stream(const ddla::DdlaHandle_t& handle)
{
    return static_cast<ddla::runtimeStream_t>(ddla::ddla_get_stream(handle));
}
#endif

}
}




#endif // DEVICE_CONNECTOR_H