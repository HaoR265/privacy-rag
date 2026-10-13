#include "local_shares.hpp"
#include <iostream>
#include <stdexcept>
using namespace prag::model;
int main(){
  auto gpu=make_rtx2060_shares();constexpr Word mask=(Word{1}<<59)-1;
  // Public synthetic random-share patterns include wrap at bits 59 and 64.
  std::vector<Word>a{0,1,mask,Word{1}<<58,mask-7},b{mask,mask,2,Word{1}<<58,17};
  auto sum=gpu->add(a,b),negative=gpu->scale(a,mask);
  for(size_t i=0;i<a.size();++i)if(sum[i]!=((a[i]+b[i])&mask)||negative[i]!=((-a[i])&mask))throw std::runtime_error("LOCAL_RING_MISMATCH");
  auto reduced=gpu->sum_rows({mask,1,mask,1,mask,mask},2);if(reduced!=std::vector<Word>{0,0,mask-1})throw std::runtime_error("LOCAL_SUM_MISMATCH");
  auto arranged=gpu->layout(a,{4,1,3,0,2});if(arranged!=std::vector<Word>{a[4],a[1],a[3],a[0],a[2]})throw std::runtime_error("LOCAL_LAYOUT_MISMATCH");
  {auto cache=gpu->allocate(2*28*8*1280*128);cache->write(8192,a);if(cache->read(8192,a.size())!=a)throw std::runtime_error("GPU_CACHE_MISMATCH");cache->clear();}
  auto stats=gpu->stats();if(stats.live_bytes)throw std::runtime_error("GPU_ALLOCATION_RETAINED");
  std::cout<<"{\"device\":\"RTX2060\",\"exact_local_ring\":true,\"cache_allocated_and_cleared\":true,\"kernel_calls\":"<<stats.kernels<<",\"peak_owned_bytes\":"<<stats.peak_bytes<<",\"live_owned_bytes\":"<<stats.live_bytes<<"}\n";
}
