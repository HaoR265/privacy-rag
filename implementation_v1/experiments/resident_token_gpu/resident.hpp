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
 double initial_input_upload_ms=0,weight_upload_ms=0,lift_ms=0,ntt_ms=0,dyadic_ms=0,download_ms=0,finish_ms=0;
 std::array<double,14> ntt_stages_ms{};
 size_t input_bytes=0,ring_weight_bytes=0,ntt_weight_bytes=0,output_bytes=0,device_bytes=0;
};
struct Cleanup {int first_cuda_error=0;size_t wipe_synchronized_bytes=0,freed_bytes=0;bool closed=false;};
// Internal borrowed device view. All producers retain their storage through owner.
// Public shape and device are checked at every consumer; ready is a CUDA event.
// A view is consumed before the next call on its producer overwrites that buffer.
struct DeviceCipherBatch {
 const uint64_t* data=nullptr;size_t words=0;int device=-1;void* ready=nullptr;
 std::shared_ptr<void> owner;
};
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
 DeviceCipherBatch evaluate_finished(const std::vector<uint64_t>& weights,const std::vector<uint64_t>& local);
 void bind_local_shares(const std::vector<std::vector<uint64_t>>&);
 DeviceCipherBatch evaluate_local_finished(const std::vector<uint64_t>&);
 void stack_store(size_t slot,unsigned level,const DeviceCipherBatch&);
 DeviceCipherBatch stack_view(size_t slot,unsigned level);
 std::vector<uint64_t> download(const DeviceCipherBatch&);
 Timing timing()const;Cleanup close()noexcept;
private:
 struct State;std::shared_ptr<State> state_;std::shared_ptr<seal::SEALContext> context_;
 void initialize(size_t slots,const std::vector<uint64_t>& input_ntt,
                 const std::vector<uint64_t>& moduli,const std::vector<uint64_t>& root_pairs,
                 const std::vector<uint64_t>& inverse_pairs,const std::vector<uint64_t>& inv_degree);
 void initialize_local(const std::vector<uint64_t>&,const std::vector<uint64_t>&,uint64_t,uint64_t);
 DeviceCipherBatch evaluate_device(const std::vector<uint64_t>&,const std::vector<uint64_t>&);
};
}
