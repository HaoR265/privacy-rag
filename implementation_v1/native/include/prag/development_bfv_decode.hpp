#pragma once
#ifndef PRAG_ISOLATED_DEVELOPMENT
#error "Synthetic BFV phase entry is excluded from production"
#endif
#include "prag/secure_ops.hpp"
#include <gmpxx.h>
namespace prag {
// Public synthetic diagnostics only; no symbol is linked into prag_crypto.
std::vector<Word>
development_bfv59_decode(SecureOps &,
                         const std::vector<mpz_class> &local_components);
} // namespace prag
