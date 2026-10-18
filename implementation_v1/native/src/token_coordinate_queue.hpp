#pragma once
// Uncompiled isolated preparation. No CUDA/network/material calls here.
#include <openssl/crypto.h>
#include <array>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdint>
namespace prag::candidate_token_pipeline {
struct Coordinate {
 std::vector<uint64_t> weights; // exactly 19*8192 ring59 words
 std::array<uint64_t,4> local{}; // S-only dot shares; inactive slots zero
 size_t index=0,slots=0;
 double prepare_seconds=0;double read_seconds=0;size_t blocks=0,values=0;
 Coordinate():weights(19*8192){}
 Coordinate(const Coordinate&)=delete;
 void erase()noexcept{OPENSSL_cleanse(weights.data(),weights.size()*sizeof(uint64_t));OPENSSL_cleanse(local.data(),sizeof(local));}
 ~Coordinate(){erase();}
};
// Lifetime contract: callbacks/reference owners live until run returns; prepare
// is the sole reader caller, explicitly permitted off-thread by its owner.
// consume owns CUDA/network on original thread. Neither callback retains data.
// One tile/call only; run joins before releasing any referenced input share.
class TwoCoordinates {
 struct Slot {Coordinate value;bool ready=false;};
 std::array<Slot,2> slots_;std::mutex mutex_;std::condition_variable cv_;
 bool stop_=false;std::exception_ptr failure_;std::thread producer_;
 void stop_join()noexcept{
  {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}cv_.notify_all();
  if(producer_.joinable())producer_.join();
 }
 public:
 TwoCoordinates()=default;TwoCoordinates(const TwoCoordinates&)=delete;
 ~TwoCoordinates(){stop_join();}
 // Fixed public bounds: at most one 512-coordinate tile and 1..4 slots.
 // This is a one-use object: no session/material or buffers survive run.
 void run(size_t begin,size_t count,size_t batch,
          const std::function<void(Coordinate&)>& prepare,
          const std::function<void(const Coordinate&)>& consume,
          bool reader_off_thread_authorized){
  if(stop_||producer_.joinable()||!reader_off_thread_authorized||!prepare||!consume||!count||count>512||batch<1||batch>4||begin>2048-count)throw std::invalid_argument("E_PIPELINE_CONTRACT");
  producer_=std::thread([&]{
   try{for(size_t k=0;k<count;++k){auto& slot=slots_[k%2];
    {std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return stop_||!slot.ready;});if(stop_)return;}
    slot.value.index=begin+k;slot.value.slots=batch;prepare(slot.value);
    {std::lock_guard<std::mutex> lock(mutex_);if(stop_)return;slot.ready=true;}cv_.notify_all();
   }}catch(...){std::lock_guard<std::mutex> lock(mutex_);failure_=std::current_exception();stop_=true;cv_.notify_all();}
  });
  try{for(size_t k=0;k<count;++k){auto& slot=slots_[k%2];
   {std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return stop_||slot.ready;});if(failure_)std::rethrow_exception(failure_);if(stop_)throw std::runtime_error("E_PIPELINE_CANCELLED");}
   consume(slot.value);slot.value.erase();
   {std::lock_guard<std::mutex> lock(mutex_);slot.ready=false;}cv_.notify_all();
  }stop_join();}catch(...){stop_join();for(auto& slot:slots_)slot.value.erase();throw;}
  for(auto& slot:slots_)slot.value.erase();
 }
};
}
