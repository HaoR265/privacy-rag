#include "resident.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
namespace prag::experimental::resident {
namespace {
struct Root {unsigned long long value,quotient;};
struct Modulus {unsigned long long q,r0,r1;};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("E_RESIDENT_CUDA: ")+cudaGetErrorString(e));}
// Same exact Barrett limb sequence as the separately checked BFV dyadic candidate.
__device__ unsigned long long multiply(unsigned long long a,unsigned long long b,Modulus m){
 const auto lo=a*b,hi=__umul64hi(a,b);
 const auto carry0=__umul64hi(lo,m.r0);const auto round1=lo*m.r1;const auto tmp=round1+carry0;
 const auto high1=__umul64hi(lo,m.r1)+(tmp<round1);const auto round2=hi*m.r0;const auto next=tmp+round2;
 const auto high2=__umul64hi(hi,m.r0)+(next<tmp);const auto quotient=hi*m.r1+high1+high2;
 const auto remainder=lo-quotient*m.q;return remainder>=m.q?remainder-m.q:remainder;
}
__device__ unsigned long long mul_root(unsigned long long a,Root r,unsigned long long q){
 const auto h=__umul64hi(a,r.quotient),z=a*r.value-h*q;return z>=q?z-q:z;
}
__global__ void lift(const uint64_t* ring,uint64_t* ntt,const Modulus* moduli){
 const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=blocks*primes*degree)return;
 const size_t coefficient=i%degree,k=(i/degree)%primes,block=i/(primes*degree);
 // All blocks and both halves are written on every call. No sparse skipping.
 uint64_t value=0;if(coefficient<block_width){value=ring[block*block_width+(block_width-1-coefficient)];
  value+=((moduli[k].q-(uint64_t(1)<<59))&(uint64_t(0)-(value>>58)));}
 ntt[i]=value;
}
__global__ void forward_stage(uint64_t* data,const Modulus* moduli,const Root* roots,size_t gap,size_t root_start){
 const size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(idx>=blocks*primes*degree/2)return;
 const size_t poly=idx/(degree/2),j=idx%(degree/2),group=j/gap,offset=j%gap,k=poly%primes;
 const size_t x=poly*degree+group*(gap*2)+offset,y=x+gap;
 const auto q=moduli[k].q;const auto u=data[x];const auto v=mul_root(data[y],roots[k*degree+root_start+group],q);
 const auto sum=u+v,diff=u+q-v;data[x]=sum>=q?sum-q:sum;data[y]=diff>=q?diff-q:diff;
}
__global__ void accumulate(const uint64_t* input,const uint64_t* weight,uint64_t* output,const Modulus* moduli,size_t words){
 const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=words)return;
 constexpr size_t limbs=degree*primes;const size_t coefficient=i%limbs,k=coefficient/degree;const auto m=moduli[k];
 unsigned long long sum=0;
 #pragma unroll 1
 for(size_t block=0;block<blocks;++block){sum+=multiply(input[block*words+i],weight[block*limbs+coefficient],m);sum=sum>=m.q?sum-m.q:sum;}
 output[i]=sum;
}
}
struct TokenWeights::State {
 int device=-1;size_t words=0;uint64_t *input=nullptr,*ring=nullptr,*weight=nullptr,*output=nullptr;
 Modulus* moduli=nullptr;Root* roots=nullptr;cudaStream_t stream=nullptr;std::array<cudaEvent_t,19> events{};
 Timing stats;Cleanup cleanup;bool closing=false;
 Cleanup release()noexcept{
  closing=true;if(cleanup.closed)return cleanup;
  auto ok=[&](cudaError_t e){if(e!=cudaSuccess&&!cleanup.first_cuda_error)cleanup.first_cuda_error=int(e);return e==cudaSuccess;};
  if(device<0){cleanup.closed=true;return cleanup;}if(!ok(cudaSetDevice(device)))return cleanup;
  size_t queued=0;
  for(auto item:std::array<std::pair<uint64_t*,size_t>,4>{{{input,stats.input_bytes},{ring,stats.ring_weight_bytes},{weight,stats.ntt_weight_bytes},{output,stats.output_bytes}}})
   if(item.first&&ok(cudaMemsetAsync(item.first,0,item.second,stream)))queued+=item.second;
  if(ok(cudaStreamSynchronize(stream)))cleanup.wipe_synchronized_bytes+=queued;
  auto free_data=[&](uint64_t*& p,size_t bytes){if(p&&ok(cudaFree(p))){p=nullptr;cleanup.freed_bytes+=bytes;}};
  free_data(input,stats.input_bytes);free_data(ring,stats.ring_weight_bytes);free_data(weight,stats.ntt_weight_bytes);free_data(output,stats.output_bytes);
  if(moduli&&ok(cudaFree(moduli)))moduli=nullptr;if(roots&&ok(cudaFree(roots)))roots=nullptr;
  bool events_gone=true;for(auto& e:events){if(e&&ok(cudaEventDestroy(e)))e=nullptr;events_gone&=!e;}
  if(stream&&ok(cudaStreamDestroy(stream)))stream=nullptr;
  cleanup.closed=!input&&!ring&&!weight&&!output&&!moduli&&!roots&&!stream&&events_gone;return cleanup;
 }
 ~State(){release();}
 double milliseconds(size_t a,size_t b){float ms=0;check(cudaEventElapsedTime(&ms,events[a],events[b]));return ms;}
};
void TokenWeights::initialize(size_t slots,const std::vector<uint64_t>& input,
 const std::vector<uint64_t>& q,const std::vector<uint64_t>& root_pairs){
 state_=std::make_shared<State>();auto& s=*state_;s.words=slots*2*primes*degree;
 int count=0;check(cudaGetDeviceCount(&count));for(int i=0;i<count;++i){cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,i));if(std::strstr(p.name,"RTX 2060")&&p.major==7&&p.minor==5){s.device=i;break;}}
 if(s.device<0)throw std::runtime_error("E_RTX2060_REQUIRED");check(cudaSetDevice(s.device));check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));
 for(auto& e:s.events)check(cudaEventCreate(&e));
 s.stats.input_bytes=input.size()*8;s.stats.ring_weight_bytes=blocks*block_width*8;s.stats.ntt_weight_bytes=blocks*primes*degree*8;s.stats.output_bytes=s.words*8;
 s.stats.device_bytes=s.stats.input_bytes+s.stats.ring_weight_bytes+s.stats.ntt_weight_bytes+s.stats.output_bytes+primes*sizeof(Modulus)+root_pairs.size()*8;
 check(cudaMalloc(&s.input,s.stats.input_bytes));check(cudaMalloc(&s.ring,s.stats.ring_weight_bytes));check(cudaMalloc(&s.weight,s.stats.ntt_weight_bytes));check(cudaMalloc(&s.output,s.stats.output_bytes));
 check(cudaMalloc(&s.moduli,primes*sizeof(Modulus)));check(cudaMalloc(&s.roots,root_pairs.size()*8));
 std::array<Modulus,primes> moduli;for(size_t i=0;i<primes;++i){const unsigned __int128 ratio=(~static_cast<unsigned __int128>(0))/q[i];moduli[i]={q[i],uint64_t(ratio),uint64_t(ratio>>64)};}
 try{
 check(cudaMemcpyAsync(s.moduli,moduli.data(),sizeof(moduli),cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.roots,root_pairs.data(),root_pairs.size()*8,cudaMemcpyHostToDevice,s.stream));
 check(cudaEventRecord(s.events[0],s.stream));check(cudaMemcpyAsync(s.input,input.data(),s.stats.input_bytes,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));check(cudaEventSynchronize(s.events[1]));
 s.stats.initial_input_upload_ms=s.milliseconds(0,1);
 }catch(...){s.release();throw;}
}
TokenWeights::~TokenWeights()=default;
std::vector<uint64_t> TokenWeights::evaluate(const std::vector<uint64_t>& ring){
 auto& s=*state_;if(s.closing)throw std::runtime_error("E_RESIDENT_CLOSED");
 constexpr uint64_t modulus=uint64_t(1)<<59,bound=uint64_t(1)<<20;
 if(ring.size()!=blocks*block_width)throw std::invalid_argument("E_RESIDENT_WEIGHT_SHAPE");
 for(auto v:ring)if(v>=modulus||(v>bound&&v<modulus-bound))throw std::invalid_argument("E_RESIDENT_WEIGHT_BOUND");
 check(cudaSetDevice(s.device));std::vector<uint64_t> output(s.words);
 try{
  check(cudaEventRecord(s.events[0],s.stream));check(cudaMemcpyAsync(s.ring,ring.data(),s.stats.ring_weight_bytes,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));
  lift<<<(blocks*primes*degree+255)/256,256,0,s.stream>>>(s.ring,s.weight,s.moduli);check(cudaGetLastError());check(cudaEventRecord(s.events[2],s.stream));
  for(size_t level=0;level<14;++level){forward_stage<<<(blocks*primes*degree/2+255)/256,256,0,s.stream>>>(s.weight,s.moduli,s.roots,degree>>(level+1),size_t(1)<<level);check(cudaGetLastError());check(cudaEventRecord(s.events[level+3],s.stream));}
  accumulate<<<(s.words+255)/256,256,0,s.stream>>>(s.input,s.weight,s.output,s.moduli,s.words);check(cudaGetLastError());check(cudaEventRecord(s.events[17],s.stream));
  check(cudaMemcpyAsync(output.data(),s.output,s.stats.output_bytes,cudaMemcpyDeviceToHost,s.stream));check(cudaEventRecord(s.events[18],s.stream));check(cudaEventSynchronize(s.events[18]));
  s.stats.weight_upload_ms=s.milliseconds(0,1);s.stats.lift_ms=s.milliseconds(1,2);s.stats.ntt_ms=s.milliseconds(2,16);s.stats.dyadic_ms=s.milliseconds(16,17);s.stats.download_ms=s.milliseconds(17,18);
  for(size_t i=0;i<14;++i)s.stats.ntt_stages_ms[i]=s.milliseconds(i+2,i+3);
  return output;
 }catch(...){s.release();volatile uint64_t* p=output.data();for(size_t i=0;i<output.size();++i)p[i]=0;throw;}
}
Timing TokenWeights::timing()const{return state_->stats;}
Cleanup TokenWeights::close()noexcept{return state_->release();}
}
