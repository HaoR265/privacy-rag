#pragma once
#include "prag/channel.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace prag {
constexpr uint64_t MASK = (uint64_t(1) << 59) - 1;
using Word = uint64_t;
using Bit = bool;
using Bits = std::array<Bit, 59>;
using PrivateWeightBlock = std::function<std::vector<std::vector<Word>>(
    size_t input_begin, size_t actual_columns, size_t first_row, size_t group_rows)>;
// Public descriptors are validated at the typed tensor boundary. All scalar
// Word and Bit arguments below are this party's shares, never plaintext.
struct TokenLookupStats {
  uint64_t onehot_bytes=0,initial_key_bytes=0,switch_key_bytes=0,online_bfv_gc_bytes=0;
  double onehot_seconds=0;
};
class BfvEngine;
class SecureOps {
  struct Impl;
  std::unique_ptr<Impl> p_;
  std::shared_ptr<BfvEngine> bfv_;

public:
  SecureOps(Role, Channel &, const std::string &fresh_ledger_path,
            bool isolated_development = false);
  ~SecureOps();
  SecureOps(const SecureOps &) = delete;
  Role role() const;
  Word public_word(Word v) const;
  Bit public_bit(bool v) const;
  Word add(Word a, Word b) const { return (a + b) & MASK; }
  Word sub(Word a, Word b) const { return (a - b) & MASK; }
  Word mul_public(Word a, Word b) const { return (a * b) & MASK; }
  Word mul(Word, Word);
  std::vector<Word> private_diagonal(const std::vector<Word>& x,
      const std::vector<Word>& server_weights,unsigned fx,unsigned fw);
  std::vector<Word> one_hot_shared(Word index,size_t padded_vocabulary);
  std::vector<Word> mul_many(const std::vector<Word> &,
                             const std::vector<Word> &);
  Bits a2b(Word);
  Word b2a(const Bits &);
  Word b2a_bit(Bit);
  Bit band(Bit, Bit);
  Bit bxor(Bit a, Bit b) const { return a != b; }
  Bit bnot(Bit b) const { return b != public_bit(true); }
  Bit bor(Bit a, Bit b) { return bxor(bxor(a, b), band(a, b)); }
  Bit gt(Word, Word, bool signed_values = true);
  Bit equal(Word, Word);
  Word select(Bit, Word, Word);
  Word floor(Word, unsigned);
  Word lookup(Word, const std::vector<Word> &);
  std::vector<Word> private_linear(const std::vector<Word> &,
                                   const std::vector<std::vector<Word>> &,
                                   size_t, unsigned, unsigned);
  // At most 19 public 8192-column blocks; the callback runs only at S.
  // It returns group_rows x actual_columns, before public tail zero padding.
  std::vector<Word> private_linear_streamed(const std::vector<Word> &, size_t,
                                           unsigned, unsigned,
                                           const PrivateWeightBlock &);
  // True polynomial matrix tile: X is B x K (this party's shares), W is
  // O x K (only S), result is B x O at fx+fw. B<=16, K<=8192, O<=155648.
  // A ciphertext contains up to 512 distinct output coordinates, accumulated
  // over public cost-selected power-of-two reduction blocks before protection and return.
  std::vector<std::vector<Word>> private_matrix(
      const std::vector<std::vector<Word>> &, size_t output_columns,
      unsigned fx, unsigned fw, const PrivateWeightBlock &);
  // Fixed 155648-entry shared index scan, f12 S-only table, f12 output shares.
  // Exactly 19 input ciphertexts and hidden/512 c1 returns; keys reused only
  // inside this SecureOps session. Does not apply model validity/failure policy.
  std::vector<Word> private_token_lookup(Word shared_index, size_t vocabulary,
      size_t hidden, const PrivateWeightBlock&, TokenLookupStats* stats=nullptr);
  Channel &channel();
  bool isolated_development() const;
  void reserve_material(const char *, size_t);
  // Public lifecycle only. Reservations never become reusable on close/abort.
  void abort_session() noexcept;
  void close_session() noexcept;
  bool session_active() const noexcept;
};
} // namespace prag
