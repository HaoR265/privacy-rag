#pragma once
#include <cstdint>
namespace seal {class SEALContext;class PublicKey;class Ciphertext;}
#include <memory>
#include <vector>
#include <string>
namespace prag::enc0_gpu {
constexpr size_t degree=16384;
struct Timing {double setup_upload_ms=0,upload_ms=0,rns_ntt_ms=0,product_ms=0,inverse_ms=0,moddown_ms=0,download_ms=0;size_t device_bytes=0,sensitive_bytes=0;};
struct Cleanup {int first_cuda_error=0;size_t wipe_synchronized_bytes=0,freed_bytes=0;bool closed=false;};
// One encryption's private ternary u then CBD e0,e1, at key level. No reuse.
struct Samples {std::vector<uint64_t> words;Samples():words(3*7*degree){} ~Samples();Samples(const Samples&)=delete;Samples& operator=(const Samples&)=delete;};
void sample(const seal::SEALContext&,Samples&);
seal::Ciphertext reference(const seal::SEALContext&,const seal::PublicKey&,const Samples&);
class PublicZero {
 struct State;std::shared_ptr<State> state_;std::shared_ptr<seal::SEALContext> context_;
 void initialize(const std::vector<uint64_t>&,const std::vector<uint64_t>&,const std::vector<uint64_t>&,const std::vector<uint64_t>&,const std::vector<uint64_t>&,const std::vector<uint64_t>&,const std::string&,const std::string&);
 std::vector<uint64_t> evaluate(const std::vector<uint64_t>&);
public:
 PublicZero(const seal::SEALContext&,const seal::PublicKey&,const std::string& uuid,const std::string& pci);
 ~PublicZero();PublicZero(const PublicZero&)=delete;PublicZero& operator=(const PublicZero&)=delete;
 seal::Ciphertext encrypt_from_samples(const Samples&); // diagnostic equality entry; caller owns one-time samples
 seal::Ciphertext encrypt_zero(); // production-shaped wrapper, samples once then wipes
 Timing timing()const;Cleanup close()noexcept;
};
}
