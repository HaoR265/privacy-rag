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
// Explicit opt-in: owner promises one helper may invoke this callback, with
// no concurrent calls or external mutation; all referenced owners outlive join.
struct WorkerSafePrivateWeightReader {
  PrivateWeightBlock callback;
  std::vector<std::vector<Word>> operator()(size_t a,size_t b,size_t c,size_t d)const{return callback(a,b,c,d);}
};
inline PrivateWeightBlock worker_safe_private_reader(PrivateWeightBlock callback){
 return WorkerSafePrivateWeightReader{std::move(callback)};
}
inline bool private_reader_worker_safe(const PrivateWeightBlock& reader){
 return reader.target<WorkerSafePrivateWeightReader>()!=nullptr;
}
// Public descriptors are validated at the typed tensor boundary. All scalar
// Word and Bit arguments below are this party's shares, never plaintext.
struct TokenLookupStats {
  bool coordinate_pipeline_used=false;
  double pipeline_prepare_seconds=0,pipeline_tiles_wall_seconds=0;
  uint64_t onehot_bytes=0,initial_key_bytes=0,switch_key_bytes=0,online_bfv_gc_bytes=0;
  double onehot_seconds=0;
  // Opt-in, party-local wall times. Communication/waiting is included; these
  // are not CPU times and must not be added across C and S. Sequential raw
  // includes read; pipeline raw is consume-only, with helper preparation and
  // overlapped tile wall reported separately.
  // Only public schedule counts are reported, never indices or secret values.
  double initial_key_seconds=0,switch_key_seconds=0,input_seconds=0;
  double server_raw_seconds=0,server_weight_read_seconds=0,server_packing_seconds=0;
  // Nested inside raw: centered lifting + forward weight NTT, and ciphertext
  // dyadic multiply/add respectively. Neither includes the other or read.
  double server_weight_lift_ntt_seconds=0,server_dyadic_seconds=0;
  // Optional local CUDA dyadic backend. Counts/times contain no secret values.
  bool gpu_dyadic_used=false,gpu_cleanup_closed=false;
  double gpu_constructor_seconds=0,gpu_input_upload_seconds=0,gpu_weight_upload_seconds=0;
  double gpu_packing_constructor_seconds=0;
  size_t gpu_packing_device_bytes=0;
  double gpu_weight_lift_seconds=0,gpu_weight_ntt_seconds=0,gpu_dyadic_seconds=0;
  double gpu_kernel_seconds=0,gpu_output_download_seconds=0,gpu_cleanup_seconds=0;
  uint64_t gpu_device_bytes=0,gpu_wipe_synchronized_bytes=0,gpu_freed_bytes=0;
  int gpu_cleanup_error=0;
  double return_phase_seconds=0,gc_seconds=0,total_seconds=0;
  uint64_t server_weight_blocks=0,server_weight_values=0,server_weight_ntts=0;
  uint64_t server_pack_merges=0,input_ciphertexts=0,returned_tiles=0;
};
class BfvEngine;
class SecureOps {
  struct Impl;
  std::unique_ptr<Impl> p_;
  std::shared_ptr<BfvEngine> bfv_;
  friend class BatchOps;
  // Only the typed shared QK wrapper may enter this dedicated reverse layout.
  std::vector<std::vector<Word>> private_qk_reverse64(
      const std::vector<std::vector<Word>>&,const PrivateWeightBlock&);
  std::vector<std::vector<Word>> private_matrix_impl(
      const std::vector<std::vector<Word>>&,size_t,unsigned,unsigned,
      const PrivateWeightBlock&,bool qk_reverse64);

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
  // Public B in [1,4]. Each slot still scans 155648 entries, has 19 fresh input
  // ciphertexts and independent protected returns. Only S weight transforms
  // are shared inside this call. Model validity policy remains the caller's.
  std::vector<std::vector<Word>> private_token_lookup_many(
      const std::vector<Word>& shared_indices,size_t vocabulary,size_t hidden,
      const PrivateWeightBlock&,TokenLookupStats* stats=nullptr);
  Channel &channel();
  bool isolated_development() const;
  void reserve_material(const char *, size_t);
  // Public lifecycle only. Reservations never become reusable on close/abort.
  void abort_session() noexcept;
  void close_session() noexcept;
  bool session_active() const noexcept;
};
} // namespace prag
