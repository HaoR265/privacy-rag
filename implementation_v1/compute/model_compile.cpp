#include "session.hpp"
namespace prag::model {
// Source-controlled capability, in a separate translation unit so normal
// builds still compile/link the complete worker path. No runtime override.
// A false result never certifies or enables the full numeric/model pipeline.
bool full_model_configuration_enabled() noexcept {return false;}
// The existing SEAL/SCI backend implements CPU primitives. CUDA R0/R1 code
// cannot establish this separate R2 capability. Enabling other capabilities
// must not accidentally enable complete CPU model execution.
bool private_model_gpu_backend_available() noexcept {return false;}
static_assert(sizeof(Word)==8,"64 bit storage required");
}
