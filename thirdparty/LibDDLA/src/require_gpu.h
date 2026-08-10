#ifndef REQUIRE_GPU_H
#define REQUIRE_GPU_H

// ---------------------------------------------------------------------------
// Internal helper: reject CPU handles for accelerator-only routines.
// Uses the public backend query (ddla_get_backend), not private handle layout.
// ---------------------------------------------------------------------------

#include <stdexcept>
#include <string>
#include <ddla/ddla_handle_t.h>

namespace ddla {
namespace detail {

inline void require_gpu_backend(const DdlaHandle_t& handle, const char* routine_name)
{
    if (handle == nullptr) {
        throw std::runtime_error(std::string(routine_name) + ": null handle");
    }
    DdlaBackend backend = ddla_get_backend(handle);
    if (backend == DdlaBackend::CPU) {
        throw std::runtime_error(
            std::string(routine_name) + " is not supported on CPU backend handles");
    }
}

} // namespace detail
} // namespace ddla

#endif // REQUIRE_GPU_H
