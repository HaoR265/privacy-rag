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

__global__ void inverse_stage(uint64_t* data,const Modulus* moduli,const Root* roots,const Root* invn,size_t words,size_t gap,size_t start,bool last){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=words/2)return;
 size_t poly=i/(degree/2),j=i%(degree/2),g=j/gap,off=j%gap,k=poly%primes;
 size_t x=poly*degree+g*2*gap+off,y=x+gap;uint64_t q=moduli[k].q,u=data[x],v=data[y];
 auto a=u+v;a=a>=q?a-q:a;auto b=u+q-v;b=b>=q?b-q:b;
 b=mul_root(b,roots[k*degree+start+g],q);if(last){a=mul_root(a,invn[k],q);b=mul_root(b,invn[k],q);}data[x]=a;data[y]=b;
}
__global__ void add_shift(const uint64_t* input,uint64_t* output,const uint64_t* scaled,const Modulus* mods,size_t words){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=words)return;
 size_t j=i%degree,p=(i/degree)%primes,c=(i/(degree*primes))%2,slot=i/(2*degree*primes);
 uint64_t q=mods[p].q,v=input[i];if(c==0&&j==block_width-1){v+=scaled[slot*primes+p];if(v>=q)v-=q;}
 size_t dest=(j+2*degree-(block_width-1))%(2*degree);output[i-j+dest%degree]=dest>=degree&&v?q-v:v;
}
constexpr size_t ring_words=blocks*block_width,ring_tiles=(ring_words+255)/256;
__global__ void local_partials(const uint64_t* shares,const uint64_t* weights,uint64_t* partial){
 __shared__ uint64_t values[256];const size_t slot=blockIdx.y,tile=blockIdx.x,j=tile*256+threadIdx.x;
 uint64_t v=j<ring_words?shares[slot*ring_words+j]*weights[j]:0;values[threadIdx.x]=v;__syncthreads();
 for(unsigned d=128;d;d>>=1){if(threadIdx.x<d)values[threadIdx.x]+=values[threadIdx.x+d];__syncthreads();}
 if(threadIdx.x==0)partial[slot*ring_tiles+tile]=values[0];
}
__global__ void local_scale(const uint64_t* partial,uint64_t* scaled,const Root* delta,const Modulus* q,uint64_t r,uint64_t half){
 __shared__ uint64_t values[256];const size_t slot=blockIdx.x;uint64_t sum=0;
 for(size_t j=threadIdx.x;j<ring_tiles;j+=256)sum+=partial[slot*ring_tiles+j];values[threadIdx.x]=sum;__syncthreads();
 for(unsigned d=128;d;d>>=1){if(threadIdx.x<d)values[threadIdx.x]+=values[threadIdx.x+d];__syncthreads();}
 if(threadIdx.x<primes){const uint64_t m=values[0]&((uint64_t(1)<<59)-1),lo=m*r,hi=__umul64hi(m,r),added=lo+half;
 const uint64_t fix=((hi+(added<lo))<<5)|(added>>59);const auto k=threadIdx.x;
 uint64_t z=mul_root(m,delta[k],q[k].q)+fix;if(z>=q[k].q)z-=q[k].q;scaled[slot*primes+k]=z;}
}

}
struct TokenWeights::State {
 int device=-1;size_t words=0;uint64_t *input=nullptr,*ring=nullptr,*weight=nullptr,*output=nullptr;
 Modulus* moduli=nullptr;Root* roots=nullptr,*inverse=nullptr,*invn=nullptr;
 uint64_t *finished=nullptr,*scaled=nullptr,*stack=nullptr;size_t slots=0;cudaStream_t stream=nullptr;std::array<cudaEvent_t,19> events{};
 uint64_t *local_shares=nullptr,*local_partial=nullptr;Root* local_delta=nullptr;uint64_t local_r=0,local_half=0;bool local_bound=false;
 Timing stats;Cleanup cleanup;bool closing=false;
 Cleanup release()noexcept{
  closing=true;if(cleanup.closed)return cleanup;
  auto ok=[&](cudaError_t e){if(e!=cudaSuccess&&!cleanup.first_cuda_error)cleanup.first_cuda_error=int(e);return e==cudaSuccess;};
  if(device<0){cleanup.closed=true;return cleanup;}if(!ok(cudaSetDevice(device)))return cleanup;
  size_t queued=0;
  for(auto item:std::array<std::pair<uint64_t*,size_t>,9>{{{input,stats.input_bytes},{ring,stats.ring_weight_bytes},{weight,stats.ntt_weight_bytes},{output,stats.output_bytes},{finished,stats.output_bytes},{scaled,slots*primes*8},{stack,10*stats.output_bytes},{local_shares,slots*ring_words*8},{local_partial,slots*ring_tiles*8}}})
   if(item.first&&ok(cudaMemsetAsync(item.first,0,item.second,stream)))queued+=item.second;
  if(ok(cudaStreamSynchronize(stream)))cleanup.wipe_synchronized_bytes+=queued;
  auto free_data=[&](uint64_t*& p,size_t bytes){if(p&&ok(cudaFree(p))){p=nullptr;cleanup.freed_bytes+=bytes;}};
  free_data(input,stats.input_bytes);free_data(ring,stats.ring_weight_bytes);free_data(weight,stats.ntt_weight_bytes);free_data(output,stats.output_bytes);
  free_data(finished,stats.output_bytes);free_data(scaled,slots*primes*8);free_data(stack,10*stats.output_bytes);
  free_data(local_shares,slots*ring_words*8);free_data(local_partial,slots*ring_tiles*8);if(local_delta&&ok(cudaFree(local_delta)))local_delta=nullptr;
  if(inverse&&ok(cudaFree(inverse)))inverse=nullptr;if(invn&&ok(cudaFree(invn)))invn=nullptr;
  if(moduli&&ok(cudaFree(moduli)))moduli=nullptr;if(roots&&ok(cudaFree(roots)))roots=nullptr;
  bool events_gone=true;for(auto& e:events){if(e&&ok(cudaEventDestroy(e)))e=nullptr;events_gone&=!e;}
  if(stream&&ok(cudaStreamDestroy(stream)))stream=nullptr;
  cleanup.closed=!input&&!ring&&!weight&&!output&&!finished&&!scaled&&!stack&&!inverse&&!invn&&!moduli&&!roots&&!stream&&!local_shares&&!local_partial&&!local_delta&&events_gone;return cleanup;
 }
 ~State(){release();}
 double milliseconds(size_t a,size_t b){float ms=0;check(cudaEventElapsedTime(&ms,events[a],events[b]));return ms;}
};
void TokenWeights::initialize(size_t slots,const std::vector<uint64_t>& input,
 const std::vector<uint64_t>& q,const std::vector<uint64_t>& root_pairs,const std::vector<uint64_t>& inverse_pairs,const std::vector<uint64_t>& inv_degree){
 state_=std::make_shared<State>();auto& s=*state_;s.words=slots*2*primes*degree;s.slots=slots;
 int count=0;check(cudaGetDeviceCount(&count));for(int i=0;i<count;++i){cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,i));if(std::strstr(p.name,"RTX 2060")&&p.major==7&&p.minor==5){s.device=i;break;}}
 if(s.device<0)throw std::runtime_error("E_RTX2060_REQUIRED");check(cudaSetDevice(s.device));check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));
 for(auto& e:s.events)check(cudaEventCreate(&e));
 s.stats.input_bytes=input.size()*8;s.stats.ring_weight_bytes=blocks*block_width*8;s.stats.ntt_weight_bytes=blocks*primes*degree*8;s.stats.output_bytes=s.words*8;
 s.stats.device_bytes=s.stats.input_bytes+s.stats.ring_weight_bytes+s.stats.ntt_weight_bytes+s.stats.output_bytes+primes*sizeof(Modulus)+root_pairs.size()*8;
 check(cudaMalloc(&s.input,s.stats.input_bytes));check(cudaMalloc(&s.ring,s.stats.ring_weight_bytes));check(cudaMalloc(&s.weight,s.stats.ntt_weight_bytes));check(cudaMalloc(&s.output,s.stats.output_bytes));
 s.stats.device_bytes+=11*s.stats.output_bytes+slots*primes*8+inverse_pairs.size()*8+inv_degree.size()*8;
 if(s.stats.device_bytes>300ULL*1024*1024)throw std::runtime_error("E_RESIDENT_PAYLOAD_BOUND");
 check(cudaMalloc(&s.finished,s.stats.output_bytes));check(cudaMalloc(&s.scaled,slots*primes*8));check(cudaMalloc(&s.stack,10*s.stats.output_bytes));
 check(cudaMalloc(&s.inverse,inverse_pairs.size()*8));check(cudaMalloc(&s.invn,inv_degree.size()*8));
 check(cudaMalloc(&s.moduli,primes*sizeof(Modulus)));check(cudaMalloc(&s.roots,root_pairs.size()*8));
 std::array<Modulus,primes> moduli;for(size_t i=0;i<primes;++i){const unsigned __int128 ratio=(~static_cast<unsigned __int128>(0))/q[i];moduli[i]={q[i],uint64_t(ratio),uint64_t(ratio>>64)};}
 try{
 check(cudaMemcpyAsync(s.moduli,moduli.data(),sizeof(moduli),cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.roots,root_pairs.data(),root_pairs.size()*8,cudaMemcpyHostToDevice,s.stream));
 check(cudaMemcpyAsync(s.inverse,inverse_pairs.data(),inverse_pairs.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.invn,inv_degree.data(),inv_degree.size()*8,cudaMemcpyHostToDevice,s.stream));
 check(cudaEventRecord(s.events[0],s.stream));check(cudaMemcpyAsync(s.input,input.data(),s.stats.input_bytes,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));check(cudaEventSynchronize(s.events[1]));
 s.stats.initial_input_upload_ms=s.milliseconds(0,1);
 }catch(...){s.release();throw;}
}
void TokenWeights::initialize_local(const std::vector<uint64_t>& shares,const std::vector<uint64_t>& delta,uint64_t r,uint64_t half){
 auto& s=*state_;if(s.closing||s.local_bound||shares.size()!=s.slots*ring_words||delta.size()!=primes*2||r>=(uint64_t(1)<<59)||half!=(uint64_t(1)<<58))throw std::invalid_argument("E_LOCAL_BINDING");
 if(s.stats.device_bytes+shares.size()*8+s.slots*ring_tiles*8+delta.size()*8>300ULL*1024*1024)throw std::invalid_argument("E_LOCAL_DEVICE_BUDGET");
 check(cudaSetDevice(s.device));try{check(cudaMalloc(&s.local_shares,shares.size()*8));check(cudaMalloc(&s.local_partial,s.slots*ring_tiles*8));check(cudaMalloc(&s.local_delta,delta.size()*8));
 check(cudaMemcpyAsync(s.local_shares,shares.data(),shares.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.local_delta,delta.data(),delta.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaStreamSynchronize(s.stream));s.local_r=r;s.local_half=half;s.local_bound=true;s.stats.device_bytes+=shares.size()*8+s.slots*ring_tiles*8+delta.size()*8;
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
DeviceCipherBatch TokenWeights::evaluate_device(const std::vector<uint64_t>& ring,const std::vector<uint64_t>& residues){
 auto& s=*state_;if(s.closing)throw std::runtime_error("E_RESIDENT_CLOSED");
 constexpr uint64_t modulus=uint64_t(1)<<59,bound=uint64_t(1)<<20;
 if(ring.size()!=blocks*block_width)throw std::invalid_argument("E_RESIDENT_WEIGHT_SHAPE");
 for(auto v:ring)if(v>=modulus||(v>bound&&v<modulus-bound))throw std::invalid_argument("E_RESIDENT_WEIGHT_BOUND");
 if(s.local_bound?!residues.empty():residues.size()!=s.slots*primes)throw std::invalid_argument("E_RESIDENT_LOCAL_SHAPE");
 check(cudaSetDevice(s.device));
 try{
  check(cudaEventRecord(s.events[0],s.stream));check(cudaMemcpyAsync(s.ring,ring.data(),s.stats.ring_weight_bytes,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));
  lift<<<(blocks*primes*degree+255)/256,256,0,s.stream>>>(s.ring,s.weight,s.moduli);check(cudaGetLastError());check(cudaEventRecord(s.events[2],s.stream));
  for(size_t level=0;level<14;++level){forward_stage<<<(blocks*primes*degree/2+255)/256,256,0,s.stream>>>(s.weight,s.moduli,s.roots,degree>>(level+1),size_t(1)<<level);check(cudaGetLastError());check(cudaEventRecord(s.events[level+3],s.stream));}
  accumulate<<<(s.words+255)/256,256,0,s.stream>>>(s.input,s.weight,s.output,s.moduli,s.words);check(cudaGetLastError());check(cudaEventRecord(s.events[17],s.stream));
  for(size_t l=0;l<14;++l){inverse_stage<<<(s.words/2+255)/256,256,0,s.stream>>>(s.output,s.moduli,s.inverse,s.invn,s.words,size_t(1)<<l,degree-(degree>>l)+1,l==13);check(cudaGetLastError());}
  if(s.local_bound){local_partials<<<dim3(ring_tiles,s.slots),256,0,s.stream>>>(s.local_shares,s.ring,s.local_partial);check(cudaGetLastError());local_scale<<<s.slots,256,0,s.stream>>>(s.local_partial,s.scaled,s.local_delta,s.moduli,s.local_r,s.local_half);check(cudaGetLastError());}
  else check(cudaMemcpyAsync(s.scaled,residues.data(),residues.size()*8,cudaMemcpyHostToDevice,s.stream));
  add_shift<<<(s.words+255)/256,256,0,s.stream>>>(s.output,s.finished,s.scaled,s.moduli,s.words);check(cudaGetLastError());
  check(cudaEventRecord(s.events[18],s.stream));check(cudaEventSynchronize(s.events[18]));
  s.stats.weight_upload_ms=s.milliseconds(0,1);s.stats.lift_ms=s.milliseconds(1,2);s.stats.ntt_ms=s.milliseconds(2,16);s.stats.dyadic_ms=s.milliseconds(16,17);s.stats.download_ms=0;s.stats.finish_ms=s.milliseconds(17,18);
  for(size_t i=0;i<14;++i)s.stats.ntt_stages_ms[i]=s.milliseconds(i+2,i+3);
  return {s.finished,s.words,s.device,s.events[18],state_};
 }catch(...){s.release();throw;}
}
void TokenWeights::stack_store(size_t slot,unsigned level,const DeviceCipherBatch& v){
 auto& s=*state_;constexpr size_t words=2*primes*degree;
 if(s.closing||slot>=s.slots||level>=10||v.words!=words||v.device!=s.device||!v.owner||!v.ready||!v.data)throw std::invalid_argument("E_DEVICE_STACK_VIEW");
 check(cudaSetDevice(s.device));try{check(cudaStreamWaitEvent(s.stream,static_cast<cudaEvent_t>(v.ready),0));
 check(cudaMemcpyAsync(s.stack+(slot*10+level)*words,v.data,words*8,cudaMemcpyDeviceToDevice,s.stream));
 check(cudaEventRecord(s.events[18],s.stream));check(cudaEventSynchronize(s.events[18]));}catch(...){s.release();throw;}
}
DeviceCipherBatch TokenWeights::stack_view(size_t slot,unsigned level){
 auto& s=*state_;constexpr size_t words=2*primes*degree;
 if(s.closing||slot>=s.slots||level>=10)throw std::invalid_argument("E_DEVICE_STACK_INDEX");
 return {s.stack+(slot*10+level)*words,words,s.device,s.events[18],state_};
}
std::vector<uint64_t> TokenWeights::download(const DeviceCipherBatch& v){
 auto& s=*state_;if(s.closing||v.device!=s.device||!v.owner||!v.ready||!v.data||v.words!=2*primes*degree)throw std::invalid_argument("E_DEVICE_DOWNLOAD_VIEW");
 std::vector<uint64_t> out(v.words);check(cudaSetDevice(s.device));
 try{check(cudaStreamWaitEvent(s.stream,static_cast<cudaEvent_t>(v.ready),0));check(cudaMemcpyAsync(out.data(),v.data,v.words*8,cudaMemcpyDeviceToHost,s.stream));check(cudaStreamSynchronize(s.stream));return out;}
 catch(...){s.release();volatile uint64_t* p=out.data();for(size_t i=0;i<out.size();++i)p[i]=0;throw;}
}

Timing TokenWeights::timing()const{return state_->stats;}
Cleanup TokenWeights::close()noexcept{return state_->release();}
}
