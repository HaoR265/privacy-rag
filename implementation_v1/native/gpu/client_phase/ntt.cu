#include "ntt.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
namespace prag::experimental::ntt {
namespace {
struct Root {unsigned long long value,quotient;};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("E_NTT_CUDA: ")+cudaGetErrorString(e));}
// Every entry restores its caller's device, including exception cleanup. The
// explicit restore reports failure; the noexcept destructor is a last fallback.
class DeviceScope {
 int previous_=-1;int& first_;cudaError_t status_=cudaSuccess;bool restore_needed_=false;
 void remember(cudaError_t e)noexcept{if(e!=cudaSuccess&&!first_)first_=int(e);}
 public:
 DeviceScope(int target,int& first)noexcept:first_(first){
  status_=cudaGetDevice(&previous_);remember(status_);
  if(status_!=cudaSuccess)return;
  if(previous_!=target){restore_needed_=true;status_=cudaSetDevice(target);remember(status_);}
 }
 cudaError_t status()const noexcept{return status_;}
 cudaError_t restore()noexcept{
  if(!restore_needed_)return cudaSuccess;
  auto e=cudaSetDevice(previous_);remember(e);
  if(e==cudaSuccess)restore_needed_=false;
  return e;
 }
 ~DeviceScope()noexcept{restore();}
};
void wipe_host(std::vector<uint64_t>& values)noexcept{
 volatile uint64_t* p=values.data();for(size_t i=0;i<values.size();++i)p[i]=0;
}
// SEAL multiply_uint_mod(a, MultiplyUIntModOperand, q), canonical a.
// quotient=floor(root*2^64/q). Remainder is <2q; one subtraction suffices.
__device__ unsigned long long mul(unsigned long long a,Root r,unsigned long long q){
 auto t=__umul64hi(a,r.quotient);auto rem=a*r.value-t*q;return rem>=q?rem-q:rem;
}
__device__ unsigned long long add(unsigned long long a,unsigned long long b,unsigned long long q){auto x=a+b;return x>=q?x-q:x;}
__device__ unsigned long long sub(unsigned long long a,unsigned long long b,unsigned long long q){auto x=a+q-b;return x>=q?x-q:x;}
__global__ void stage(uint64_t* data,const uint64_t* moduli,const Root* roots,
 const Root* invn,size_t primes,size_t butterflies,size_t gap,size_t root_start,bool inverse,bool last){
 const size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(idx>=butterflies)return;
 const size_t poly=idx/(degree/2),j=idx%(degree/2),group=j/gap,offset=j%gap;
 const size_t k=poly%primes,x=poly*degree+group*(2*gap)+offset,y=x+gap;
 const auto q=moduli[k],u=data[x],v=data[y];const auto r=roots[k*degree+root_start+group];
 if(inverse){
  auto a=add(u,v,q),b=mul(sub(u,v,q),r,q);
  if(last){a=mul(a,invn[k],q);b=mul(b,invn[k],q);}
  data[x]=a;data[y]=b;
 }else{auto t=mul(v,r,q);data[x]=add(u,t,q);data[y]=sub(u,t,q);}
}
}
__global__ void client_dyadic(uint64_t* data,const uint64_t* q,const Root* key,size_t words,size_t primes){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<words)data[i]=mul(data[i],key[i%(primes*degree)],q[(i/degree)%primes]);
}
struct NttBatch::State {
 int device=-1;size_t primes=0,words=0;std::vector<uint64_t> host_q;
 uint64_t *data=nullptr,*q=nullptr;Root *forward=nullptr,*inverse=nullptr,*invn=nullptr,*client_key=nullptr;
 cudaStream_t stream=nullptr;std::array<cudaEvent_t,17> events{};Timing stats;Cleanup cleanup;bool closing=false;
 Cleanup release()noexcept{
  closing=true;if(cleanup.closed)return cleanup;
  auto ok=[&](cudaError_t e){if(e!=cudaSuccess&&!cleanup.first_cuda_error)cleanup.first_cuda_error=int(e);return e==cudaSuccess;};
  if(device<0){cleanup.closed=true;return cleanup;}
  DeviceScope selected(device,cleanup.first_cuda_error);if(!ok(selected.status())){selected.restore();return cleanup;}
  if(data){bool wiped=ok(cudaMemsetAsync(data,0,stats.data_bytes,stream));if(ok(cudaStreamSynchronize(stream))&&wiped)cleanup.wipe_synchronized_bytes+=stats.data_bytes;
   if(ok(cudaFree(data))){data=nullptr;cleanup.freed_bytes+=stats.data_bytes;}}
  if(client_key){size_t bytes=primes*degree*sizeof(Root);bool wiped=ok(cudaMemsetAsync(client_key,0,bytes,stream));if(ok(cudaStreamSynchronize(stream))&&wiped)cleanup.wipe_synchronized_bytes+=bytes;
   if(ok(cudaFree(client_key))){client_key=nullptr;cleanup.freed_bytes+=bytes;}}
  if(q&&ok(cudaFree(q)))q=nullptr;
  if(forward&&ok(cudaFree(forward)))forward=nullptr;
  if(inverse&&ok(cudaFree(inverse)))inverse=nullptr;
  if(invn&&ok(cudaFree(invn)))invn=nullptr;
  for(auto& e:events)if(e&&ok(cudaEventDestroy(e)))e=nullptr;
  if(stream&&ok(cudaStreamDestroy(stream)))stream=nullptr;
  bool events_gone=true;for(auto e:events)events_gone&=!e;
  cleanup.closed=!data&&!client_key&&!q&&!forward&&!inverse&&!invn&&!stream&&events_gone;
  selected.restore();return cleanup;
 }
 ~State(){release();}
};
void NttBatch::initialize(size_t batches,const std::vector<uint64_t>& moduli,
 const std::vector<uint64_t>& f,const std::vector<uint64_t>& r,const std::vector<uint64_t>& ni){
 state_=std::make_shared<State>();auto& s=*state_;s.primes=moduli.size();s.words=batches*s.primes*degree;s.host_q=moduli;
 const char* expected_uuid=std::getenv("PRAG_ENC0_GPU_UUID");const char* expected_pci=std::getenv("PRAG_ENC0_GPU_PCI");
 if(!expected_uuid||!expected_pci)throw std::runtime_error("E_PHASE_DEVICE_BINDING");
 int count=0;check(cudaGetDeviceCount(&count));for(int i=0;i<count;++i){cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,i));char bus[32]{};check(cudaDeviceGetPCIBusId(bus,sizeof(bus),i));char id[64]{};const auto* u=reinterpret_cast<const unsigned char*>(p.uuid.bytes);snprintf(id,sizeof(id),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);if(std::string(expected_uuid)==id&&std::string(expected_pci)==bus&&std::strstr(p.name,"RTX 2060")&&p.major==7&&p.minor==5){s.device=i;break;}}

 if(s.device<0)throw std::runtime_error("E_RTX2060_REQUIRED");
 DeviceScope selected(s.device,s.cleanup.first_cuda_error);check(selected.status());
 try{
 check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));for(auto& e:s.events)check(cudaEventCreate(&e));
 s.stats.data_bytes=s.words*8;s.stats.device_bytes=s.stats.data_bytes+moduli.size()*8+f.size()*8+r.size()*8+ni.size()*8;
 check(cudaMalloc(&s.data,s.stats.data_bytes));check(cudaMalloc(&s.q,moduli.size()*8));
 check(cudaMalloc(&s.forward,f.size()*8));check(cudaMalloc(&s.inverse,r.size()*8));check(cudaMalloc(&s.invn,ni.size()*8));
 check(cudaMemcpyAsync(s.q,moduli.data(),moduli.size()*8,cudaMemcpyHostToDevice,s.stream));
 check(cudaMemcpyAsync(s.forward,f.data(),f.size()*8,cudaMemcpyHostToDevice,s.stream));
 check(cudaMemcpyAsync(s.inverse,r.data(),r.size()*8,cudaMemcpyHostToDevice,s.stream));
 check(cudaMemcpyAsync(s.invn,ni.data(),ni.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaStreamSynchronize(s.stream));check(selected.restore());
 }catch(...){s.release();throw;}
}
NttBatch::~NttBatch()=default;
std::vector<uint64_t> NttBatch::transform(const std::vector<uint64_t>& input,bool inverse){
 auto& s=*state_;if(s.closing)throw std::runtime_error("E_NTT_CLOSED");
 if(input.size()!=s.words)throw std::invalid_argument("E_NTT_SHAPE");
 for(size_t i=0;i<input.size();++i)if(input[i]>=s.host_q[(i/degree)%s.primes])throw std::invalid_argument("E_NTT_CANONICAL_REQUIRED");
 DeviceScope selected(s.device,s.cleanup.first_cuda_error);check(selected.status());std::vector<uint64_t> output(s.words);
 try {
 check(cudaEventRecord(s.events[0],s.stream));check(cudaMemcpyAsync(s.data,input.data(),s.stats.data_bytes,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));
 for(size_t level=0;level<14;++level){
  const size_t gap=inverse?(size_t(1)<<level):(degree>>(level+1));
  const size_t root_start=inverse?(degree-(degree>>level)+1):(size_t(1)<<level);
  stage<<<(s.words/2+255)/256,256,0,s.stream>>>(s.data,s.q,inverse?s.inverse:s.forward,s.invn,s.primes,s.words/2,gap,root_start,inverse,level==13);
  check(cudaGetLastError());check(cudaEventRecord(s.events[level+2],s.stream));
 }
 check(cudaMemcpyAsync(output.data(),s.data,s.stats.data_bytes,cudaMemcpyDeviceToHost,s.stream));check(cudaEventRecord(s.events[16],s.stream));check(cudaEventSynchronize(s.events[16]));
 auto ms=[&](size_t a,size_t b){float t=0;check(cudaEventElapsedTime(&t,s.events[a],s.events[b]));return double(t);};
 s.stats.upload_ms=ms(0,1);s.stats.download_ms=ms(15,16);s.stats.kernel_ms=ms(1,15);
 for(size_t i=0;i<14;++i)s.stats.stage_ms[i]=ms(i+1,i+2);
 check(selected.restore());return output;
 }catch(...){
  // Drain outstanding transfers before the host output buffer is destroyed.
  // close() retains observable CUDA errors; never replace the original error.
  s.release();wipe_host(output);throw;
 }
}
Timing NttBatch::timing()const{return state_->stats;}
Cleanup NttBatch::close()noexcept{return state_->release();}
}

namespace prag::experimental::ntt {
void NttBatch::upload_client_key(const std::vector<uint64_t>& pairs){
 auto& s=*state_;if(s.closing||s.client_key)throw std::runtime_error("E_CLIENT_KEY_STATE");
 if(s.primes!=6||pairs.size()!=s.primes*degree*2)throw std::invalid_argument("E_CLIENT_KEY_CONTEXT");
 DeviceScope selected(s.device,s.cleanup.first_cuda_error);check(selected.status());
 try{check(cudaMalloc(&s.client_key,pairs.size()*8));check(cudaMemcpyAsync(s.client_key,pairs.data(),pairs.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaStreamSynchronize(s.stream));s.stats.device_bytes+=pairs.size()*8;check(selected.restore());}
 catch(...){s.release();throw;}
}
std::vector<uint64_t> NttBatch::client_phase(const std::vector<uint64_t>& input){
 auto& s=*state_;if(s.closing||!s.client_key)throw std::runtime_error("E_CLIENT_KEY_STATE");
 if(input.size()!=s.words)throw std::invalid_argument("E_NTT_SHAPE");
 for(size_t i=0;i<input.size();++i)if(input[i]>=s.host_q[(i/degree)%s.primes])throw std::invalid_argument("E_NTT_CANONICAL_REQUIRED");
 DeviceScope selected(s.device,s.cleanup.first_cuda_error);check(selected.status());std::vector<uint64_t> output(s.words);
 try{
 check(cudaEventRecord(s.events[0],s.stream));check(cudaMemcpyAsync(s.data,input.data(),s.stats.data_bytes,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));
 for(size_t level=0;level<14;++level){
  stage<<<(s.words/2+255)/256,256,0,s.stream>>>(s.data,s.q,s.forward,s.invn,s.primes,s.words/2,degree>>(level+1),size_t(1)<<level,false,false);check(cudaGetLastError());
 }
 client_dyadic<<<(s.words+255)/256,256,0,s.stream>>>(s.data,s.q,s.client_key,s.words,s.primes);check(cudaGetLastError());
 for(size_t level=0;level<14;++level){
  stage<<<(s.words/2+255)/256,256,0,s.stream>>>(s.data,s.q,s.inverse,s.invn,s.primes,s.words/2,size_t(1)<<level,degree-(degree>>level)+1,true,level==13);check(cudaGetLastError());
 }
 check(cudaEventRecord(s.events[15],s.stream));check(cudaMemcpyAsync(output.data(),s.data,s.stats.data_bytes,cudaMemcpyDeviceToHost,s.stream));check(cudaEventRecord(s.events[16],s.stream));check(cudaEventSynchronize(s.events[16]));
 auto ms=[&](size_t a,size_t b){float t=0;check(cudaEventElapsedTime(&t,s.events[a],s.events[b]));return double(t);};s.stats.upload_ms=ms(0,1);s.stats.kernel_ms=ms(1,15);s.stats.download_ms=ms(15,16);
 check(selected.restore());return output;
 }catch(...){s.release();wipe_host(output);throw;}
}
}
