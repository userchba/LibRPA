#ifndef DDLA_MPI_DATATYPE_H
#define DDLA_MPI_DATATYPE_H

// ---------------------------------------------------------------------------
// Private header: MPI_Datatype for each DDLA scalar type. Shared by any
// translation unit that issues MPI collectives or point-to-point calls.
// Always pass an element count with the matching MPI_Datatype here -- never
// a byte count with MPI_BYTE, which silently overflows `int` well before a
// realistic panel size and forgoes MPI's own type checking (F6).
// ---------------------------------------------------------------------------

#include <mpi.h>
#include <complex>

namespace ddla {
namespace detail {

template <typename T>
static inline MPI_Datatype mpi_datatype();

template <> inline MPI_Datatype mpi_datatype<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_datatype<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_datatype<std::complex<float>>() { return MPI_CXX_FLOAT_COMPLEX; }
template <> inline MPI_Datatype mpi_datatype<std::complex<double>>() { return MPI_CXX_DOUBLE_COMPLEX; }

} // namespace detail
} // namespace ddla

#endif // DDLA_MPI_DATATYPE_H
