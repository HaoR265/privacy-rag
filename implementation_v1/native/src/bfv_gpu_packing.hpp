#pragma once
#include <seal/seal.h>
#include "switch.hpp"
#include "bfv_acyclic_pack.hpp"
#include <array>
#include <chrono>
#include <memory>
#include <stdexcept>
namespace prag::experimental_full_switch {
// One local session's nine independent key transitions. No randomness generated
// here, no key generation/decryption/network; each level retains BOTH switches.
class Packing {
 public:
 struct Stats {double constructor_ms=0,packing_ms=0,cleanup_ms=0;size_t merges=0,device_bytes=0,sensitive_bytes=0,freed_bytes=0,wipe_synchronized_bytes=0;bool closed=false;int first_cuda_error=0;};
 Packing(const seal::SEALContext& context,const std::vector<seal::GaloisKeys>& keys):context_(context){
  auto begin=Clock::now();if(keys.size()!=9)throw std::invalid_argument("E_PACK_KEY_LEVELS");
  for(unsigned l=0;l<9;++l){plans_[2*l]=std::make_unique<Switch>(context,keys[l],1);plans_[2*l+1]=std::make_unique<Switch>(context,keys[l],(1u<<(l+1))+1);}
  for(const auto& p:plans_){auto t=p->timing();stats_.device_bytes+=t.device_bytes;stats_.sensitive_bytes+=t.sensitive_bytes;}stats_.constructor_ms=ms(begin);
 }
 Packing(const Packing&)=delete;Packing& operator=(const Packing&)=delete;
 seal::Ciphertext pair(seal::Ciphertext left,seal::Ciphertext right,unsigned level){
  if(closing_)throw std::runtime_error("E_PACK_CLOSED");if(level>=9)throw std::invalid_argument("E_PACK_LEVEL");auto begin=Clock::now();
#ifdef PRAG_TOKEN_PACK_RESIDENT_PAIR
  auto result=plans_[2*level]->packing_pair(left,right,*plans_[2*level+1],level);
  ++stats_.merges;stats_.packing_ms+=ms(begin);return result;
#else
  seal::Evaluator eval(context_);prag::monomial_shift(right,16384>>(level+1),context_);
  seal::Ciphertext difference;eval.sub(left,right,difference);eval.add_inplace(left,right);
  left=plans_[2*level]->apply(left);difference=plans_[2*level+1]->apply(difference);
  eval.add_inplace(left,difference);++stats_.merges;stats_.packing_ms+=ms(begin);return left;
#endif
 }
 experimental::resident::DeviceCipherBatch pair_device(const experimental::resident::DeviceCipherBatch& left,const experimental::resident::DeviceCipherBatch& right,unsigned level){
  if(closing_||level>=9)throw std::invalid_argument("E_PACK_DEVICE_LEVEL");auto begin=Clock::now();
  auto result=plans_[2*level]->packing_device(left,right,*plans_[2*level+1],level);
  ++stats_.merges;stats_.packing_ms+=ms(begin);return result;
 }
 Stats close()noexcept{
  if(stats_.closed)return stats_;closing_=true;auto begin=Clock::now();bool all=true;stats_.freed_bytes=stats_.wipe_synchronized_bytes=0;
  for(auto& p:plans_)if(p){auto c=p->close();all&=c.closed;stats_.freed_bytes+=c.freed_bytes;stats_.wipe_synchronized_bytes+=c.wipe_synchronized_bytes;if(!stats_.first_cuda_error&&c.first_cuda_error)stats_.first_cuda_error=c.first_cuda_error;}
  stats_.closed=all;stats_.cleanup_ms+=ms(begin);return stats_;
 }
 Stats stats()const{return stats_;}
 ~Packing(){close();}
 private:
 using Clock=std::chrono::steady_clock;static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
 seal::SEALContext context_;std::array<std::unique_ptr<Switch>,18> plans_;Stats stats_;bool closing_=false;
};
}
