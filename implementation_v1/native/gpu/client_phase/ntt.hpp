#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace seal { class SEALContext; }
namespace prag::experimental::ntt {
constexpr size_t degree=16384;
struct Timing {
 double upload_ms=0,download_ms=0,kernel_ms=0;
 std::array<double,14> stage_ms{};
 size_t data_bytes=0,device_bytes=0;
};
struct Cleanup {
 int first_cuda_error=0;
 size_t wipe_synchronized_bytes=0,freed_bytes=0;
 bool closed=false;
};
// Canonical-only host API. Layout [batch][prime][coefficient].
// forward: natural -> SEAL bit-reversed; inverse: reverse -> natural.
class NttBatch {
public:
 NttBatch(const seal::SEALContext& context,bool key_level,size_t batches);
 ~NttBatch();
 NttBatch(const NttBatch&)=delete;
 NttBatch& operator=(const NttBatch&)=delete;
 void bind_client_key(const seal::SEALContext& context,const std::vector<uint64_t>& ntt_key);
 std::vector<uint64_t> client_phase(const std::vector<uint64_t>& input);
 std::vector<uint64_t> transform(const std::vector<uint64_t>& input,bool inverse);
 Timing timing()const;
 Cleanup close()noexcept;
private:
 struct State;
 std::shared_ptr<State> state_;
 void upload_client_key(const std::vector<uint64_t>& pairs);
 void initialize(size_t batches,const std::vector<uint64_t>& moduli,
   const std::vector<uint64_t>& forward_pairs,const std::vector<uint64_t>& inverse_pairs,
   const std::vector<uint64_t>& inverse_degree_pairs);
};
}
