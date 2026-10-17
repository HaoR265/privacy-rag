#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace seal {class SEALContext;class Ciphertext;}
namespace prag::experimental::resident {
constexpr size_t degree=16384,primes=6,blocks=19,block_width=8192;
struct Timing {
 double initial_input_upload_ms=0,weight_upload_ms=0,lift_ms=0,ntt_ms=0,dyadic_ms=0,download_ms=0;
 std::array<double,14> ntt_stages_ms{};
 size_t input_bytes=0,ring_weight_bytes=0,ntt_weight_bytes=0,output_bytes=0,device_bytes=0;
};
struct Cleanup {int first_cuda_error=0;size_t wipe_synchronized_bytes=0,freed_bytes=0;bool closed=false;};
class TokenWeights {
public:
 // Original coefficient-form ciphertexts [slot<=4][block=19].
 // Context/shape validated; original CPU SEAL input NTT remains in construction.
 TokenWeights(const seal::SEALContext&,const std::vector<std::vector<seal::Ciphertext>>&);
 ~TokenWeights();
 TokenWeights(const TokenWeights&)=delete;TokenWeights& operator=(const TokenWeights&)=delete;
 // Exactly [19][8192] canonical ring59 words, signed abs <= 2^20.
 // Returns [slot][2][6][16384] canonical NTT residues.
 std::vector<uint64_t> evaluate(const std::vector<uint64_t>& ring_weights);
 Timing timing()const;Cleanup close()noexcept;
private:
 struct State;std::shared_ptr<State> state_;
 void initialize(size_t slots,const std::vector<uint64_t>& input_ntt,
                 const std::vector<uint64_t>& moduli,const std::vector<uint64_t>& root_pairs);
};
}
