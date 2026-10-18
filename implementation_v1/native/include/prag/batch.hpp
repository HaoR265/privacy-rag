#pragma once
#include "prag/secure_ops.hpp"
namespace prag {
// Same semi-honest two-party ring/XOR shares as SecureOps. All shapes and
// parameters are public. No output is reconstructed by this production API.
enum class BatchCode : uint64_t {
  Range = 1,
  Floor,
  Compare,
  Select,
  A2B,
  B2A,
  Or,
  Exponent,
  Shift,
  GuardedSum,
  PublicLookup,
  SoftmaxPrepare,
  ExpPrepare,
  PublicLookupRows,
  ReciprocalSiLUPrepare,
  ReciprocalSiLURestore,
  ExpLookup,
  SiLUInput = 18,
  SiLUComposite = 20,
  Floor24Range = 19,
  Floor12Range = 21
};
struct BatchSpec {
  BatchCode code;
  size_t rows = 1, cols = 1;
  // Range: signed lower, upper, dummy. Floor: shift. Compare: signed flag.
  // Exponent: signed low, high, sqrt flag. Shift: signed base.
  // GuardedSum: public positive sum limit; nonnegative bounded leaves.
  std::array<Word, 3> parameters{};
};
struct BatchResult {
  std::vector<Word> words;
  std::vector<uint8_t> bits;
};
struct BatchStats {
  uint64_t calls = 0, and_gates = 0, input_ot_choices = 0, masked_words = 0,
           masked_bits = 0;
  // Public shape routing counts; contain no operand data.
  uint64_t gpu_silu_calls = 0, gpu_silu_elements = 0;
  uint64_t topology_cache_hits = 0, topology_cache_misses = 0;
};
class BatchOps {
  struct Impl;
  std::unique_ptr<Impl> p_;

public:
  explicit BatchOps(SecureOps &);
  ~BatchOps();
  BatchOps(const BatchOps &) = delete;
  BatchResult run(const BatchSpec &, const std::vector<Word> &,
                  const std::vector<uint8_t> &bits = {});
  // Vector Beaver multiplication: public descriptor plus one opening of all
  // masked differences. Does not interpret ring shares as signed plaintext.
  std::vector<Word> multiply(const std::vector<Word> &,
                             const std::vector<Word> &);
  std::vector<Word> lookup(const std::vector<Word> &,
                           const std::vector<Word> &public_table);
  // X[B,K] and Y[K,O] are both secret shared. The caller establishes the
  // public bound for every complete and partial dot before using this API.
  std::vector<Word> matrix(const std::vector<Word> &x,
                           const std::vector<Word> &y, size_t rows,
                           size_t inner, size_t columns, unsigned fx,
                           unsigned fy, Word partial_abs_bound,
                           const std::function<void(const char *, bool)> &phase = {});
  // One complete logical multiplication. identity = unique sequence, layer,
  // head, forward/decode step, kind(1=QK,2=PV). Physical BFV rows remain <=16.
  // Every call owns fresh A/B/T and opens its D/E once; no resumable handle.
  std::vector<Word> logical_matrix(const std::vector<Word>& x,
      const std::vector<Word>& y, size_t rows, size_t inner, size_t columns,
      unsigned fx, unsigned fy, Word partial_abs_bound,
      const std::array<Word,5>& identity,
      const std::function<void(const char*,bool)>& phase = {});
  BatchResult lookup_rows(const std::vector<Word>& indices,
                          const std::vector<Word>& public_table, size_t columns);
  // Fused original ExpPrepare and three public exp-table selections.
  // words: three digit-major table values; bits: off followed by negative.
  BatchResult exp_lookup(const std::vector<Word>&,
                         const std::array<std::vector<Word>,3>&);
  BatchResult silu_composite(const std::vector<Word>& x, const std::array<std::vector<Word>,3>& tables);
  BatchResult silu_input(const std::vector<Word>&,
                         const std::array<std::vector<Word>,3>&, Word abs_bound);
  BatchStats stats() const;
};
} // namespace prag
