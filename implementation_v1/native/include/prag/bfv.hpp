#pragma once
#include <gmpxx.h>
#include <seal/seal.h>
#include <cstddef>
namespace prag {
// Public layout; enforces both no wrap and the existing ART-002 weight L1 cap.
std::size_t packed_bfv_rows_per_ciphertext(std::size_t dimension);
constexpr std::size_t MATRIX_REDUCTION_TILE = 16;
constexpr std::size_t MATRIX_BATCH_MAX = 16;
constexpr std::size_t DECODE_QK_REVERSE_BATCH_MAX = 64;
constexpr std::size_t MATRIX_OUTPUT_COORDINATES = 512;
constexpr std::size_t MATRIX_INNER_MAX = 8192;
constexpr std::size_t MATRIX_COLUMNS_MAX = 155648;
seal::EncryptionParameters locked_bfv_parameters();
mpz_class modulus_product(const std::vector<seal::Modulus> &);
mpz_class rns_coefficient(const uint64_t *, size_t, size_t,
                          const std::vector<seal::Modulus> &);
void write_bfv_context_artifact(const std::string &path);
} // namespace prag
