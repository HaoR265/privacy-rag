#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace seal {class SEALContext;class GaloisKeys;class Ciphertext;}
namespace prag::experimental_full_switch {
constexpr size_t degree=16384;
struct Timing {double setup_upload_ms=0,upload_ms=0,rns_ntt_ms=0,product_ms=0,inverse_ms=0,moddown_ms=0,download_ms=0;size_t device_bytes=0,sensitive_bytes=0;};
struct Cleanup{int first_cuda_error=0;size_t wipe_synchronized_bytes=0,freed_bytes=0;bool closed=false;};
class Switch {
 public:
 Switch(const seal::SEALContext&,const seal::GaloisKeys&,uint32_t element);
 ~Switch();Switch(const Switch&)=delete;Switch& operator=(const Switch&)=delete;
 // Includes CPU coefficient Galois permutation and complete GPU key switch.
 seal::Ciphertext apply(const seal::Ciphertext&);
 // Candidate internal API: one pair, same two independent key transitions.
 seal::Ciphertext packing_pair(const seal::Ciphertext&,const seal::Ciphertext&,Switch& difference,unsigned level);
 Timing timing() const;Cleanup close()noexcept;
 private:
 struct State;std::shared_ptr<State> state_;std::shared_ptr<seal::SEALContext> context_;uint32_t element_=0;
 void initialize(const std::vector<uint64_t>& q,const std::vector<uint64_t>& f,const std::vector<uint64_t>& r,const std::vector<uint64_t>& ni,const std::vector<uint64_t>& invp,const std::vector<uint64_t>& keys);
 // Private: apply validates canonical coefficients with SEAL is_valid_for first.
 std::vector<uint64_t> evaluate(const std::vector<uint64_t>& input);
 std::vector<uint64_t> packing_evaluate(const uint64_t* left,const uint64_t* right,Switch&,unsigned);
};
}
