#include "enc0.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <array>
namespace prag::enc0_gpu {
namespace {
struct Root {unsigned long long value,quotient;};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("E_NTT_CUDA: ")+cudaGetErrorString(e));}
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
struct Modulus {unsigned long long q,r0,r1;};
// Exact unsigned-limb translation of SEAL 4.1.2 barrett_reduce_128.
// q < 2^63, inputs < q. All limb overflow is modulo 2^64 by definition.
__device__ unsigned long long multiply(unsigned long long a,unsigned long long b,Modulus m){
 const auto lo=a*b,hi=__umul64hi(a,b);
 const auto carry0=__umul64hi(lo,m.r0);
 const auto round1=lo*m.r1;
 const auto tmp=round1+carry0;
 const auto high1=__umul64hi(lo,m.r1)+(tmp<round1);
 const auto round2=hi*m.r0;
 const auto next=tmp+round2;
 const auto high2=__umul64hi(hi,m.r0)+(next<tmp);
 const auto quotient=hi*m.r1+high1+high2;
 const auto remainder=lo-quotient*m.q;
 return remainder>=m.q?remainder-m.q:remainder;
}

__global__ void product(const uint64_t* u,const uint64_t* key,uint64_t* out,const Modulus* mods){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=2*7*degree)return;
 out[i]=multiply(u[i%(7*degree)],key[i],mods[(i/degree)%7]);
}
__global__ void noise_add(uint64_t* prod,const uint64_t* samples,const uint64_t* q){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<2*7*degree)prod[i]=add(prod[i],samples[7*degree+i],q[(i/degree)%7]);
}
__global__ void moddown(const uint64_t* prod,const uint64_t* input,uint64_t* out,const uint64_t* q,const Root* invp){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=2*6*degree)return;
 size_t c=i/(6*degree),p=(i/degree)%6,k=i%degree;auto qi=q[p],special=q[6],half=special>>1;
 auto last=prod[(c*7+6)*degree+k]+half;last=last>=special?last-special:last;
 auto reduced=last>=qi?last-qi:last;auto halfmod=half>=qi?half-qi:half;
 auto correction=sub(reduced,halfmod,qi);
 auto value=mul(sub(prod[(c*7+p)*degree+k],correction,qi),invp[p],qi);
 out[i]=value;
}
}
struct PublicZero::State {
 int device=-1;bool closing=false;cudaStream_t stream=nullptr;std::array<cudaEvent_t,8> events{};
 uint64_t *input=nullptr,*expanded=nullptr,*prod=nullptr,*output=nullptr,*keys=nullptr,*q=nullptr;
 Root *forward=nullptr,*inverse=nullptr,*invn=nullptr,*invp=nullptr;Modulus* mods=nullptr;Timing stats;Cleanup cleanup;
 Cleanup release()noexcept{
  closing=true;if(cleanup.closed)return cleanup;auto ok=[&](cudaError_t e){if(e!=cudaSuccess&&!cleanup.first_cuda_error)cleanup.first_cuda_error=int(e);return e==cudaSuccess;};
  if(device<0){cleanup.closed=true;return cleanup;}if(!ok(cudaSetDevice(device)))return cleanup;
  size_t queued=0;for(auto item:std::array<std::pair<uint64_t*,size_t>,5>{{{input,3*7*degree*8},{expanded,7*degree*8},{prod,2*7*degree*8},{output,2*6*degree*8},{keys,2*7*degree*8}}})if(item.first&&ok(cudaMemsetAsync(item.first,0,item.second,stream)))queued+=item.second;
  if(ok(cudaStreamSynchronize(stream)))cleanup.wipe_synchronized_bytes+=queued;
  auto free_secret=[&](uint64_t*& p,size_t n){if(p&&ok(cudaFree(p))){p=nullptr;cleanup.freed_bytes+=n;}};
  free_secret(input,3*7*degree*8);free_secret(expanded,7*degree*8);free_secret(prod,2*7*degree*8);free_secret(output,2*6*degree*8);free_secret(keys,2*7*degree*8);
  if(q&&ok(cudaFree(q)))q=nullptr;if(mods&&ok(cudaFree(mods)))mods=nullptr;if(forward&&ok(cudaFree(forward)))forward=nullptr;if(inverse&&ok(cudaFree(inverse)))inverse=nullptr;if(invn&&ok(cudaFree(invn)))invn=nullptr;if(invp&&ok(cudaFree(invp)))invp=nullptr;
  bool gone=true;for(auto& e:events){if(e&&ok(cudaEventDestroy(e)))e=nullptr;gone&=!e;}if(stream&&ok(cudaStreamDestroy(stream)))stream=nullptr;
  cleanup.closed=!input&&!expanded&&!prod&&!output&&!keys&&!q&&!mods&&!forward&&!inverse&&!invn&&!invp&&!stream&&gone;return cleanup;
 }
 ~State(){release();}
};
void PublicZero::initialize(const std::vector<uint64_t>& qs,const std::vector<uint64_t>& f,const std::vector<uint64_t>& r,const std::vector<uint64_t>& ni,const std::vector<uint64_t>& pi,const std::vector<uint64_t>& key,const std::string& uuid,const std::string& pci){
 state_=std::make_shared<State>();auto& s=*state_;int count=0;
 if(uuid.empty()||pci.empty())throw std::runtime_error("E_ENC0_DEVICE_BINDING");
 check(cudaGetDeviceCount(&count));for(int i=0;i<count;++i){cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,i));char bus[32]{};check(cudaDeviceGetPCIBusId(bus,sizeof(bus),i));char id[64]{};const auto* u=reinterpret_cast<const unsigned char*>(p.uuid.bytes);snprintf(id,sizeof(id),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);if(uuid==id&&pci==bus&&std::strstr(p.name,"RTX 2060")&&p.major==7&&p.minor==5){s.device=i;break;}}
 if(s.device<0)throw std::runtime_error("E_RTX2060_IDENTITY");
 std::array<Modulus,7> mods;for(size_t p=0;p<7;++p){auto ratio=(~static_cast<unsigned __int128>(0))/qs[p];mods[p]={qs[p],uint64_t(ratio),uint64_t(ratio>>64)};}
 try {check(cudaSetDevice(s.device));check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));for(auto& e:s.events)check(cudaEventCreate(&e));
 check(cudaMalloc(&s.input,3*7*degree*8));check(cudaMalloc(&s.expanded,7*degree*8));check(cudaMalloc(&s.prod,2*7*degree*8));check(cudaMalloc(&s.output,2*6*degree*8));check(cudaMalloc(&s.keys,key.size()*8));check(cudaMalloc(&s.q,7*8));check(cudaMalloc(&s.mods,7*sizeof(Modulus)));check(cudaMalloc(&s.forward,f.size()*8));check(cudaMalloc(&s.inverse,r.size()*8));check(cudaMalloc(&s.invn,ni.size()*8));check(cudaMalloc(&s.invp,pi.size()*8));
 s.stats.sensitive_bytes=(3*7+7+2*7+2*6+2*7)*degree*8;s.stats.device_bytes=s.stats.sensitive_bytes+7*8+7*sizeof(Modulus)+(f.size()+r.size()+ni.size()+pi.size())*8;


 check(cudaEventRecord(s.events[0],s.stream));
 check(cudaMemcpyAsync(s.keys,key.data(),key.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.q,qs.data(),7*8,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.mods,mods.data(),sizeof(mods),cudaMemcpyHostToDevice,s.stream));
 check(cudaMemcpyAsync(s.forward,f.data(),f.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.inverse,r.data(),r.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.invn,ni.data(),ni.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaMemcpyAsync(s.invp,pi.data(),pi.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));check(cudaEventSynchronize(s.events[1]));float t;check(cudaEventElapsedTime(&t,s.events[0],s.events[1]));s.stats.setup_upload_ms=t;
 }catch(...){s.release();throw;}
}
PublicZero::~PublicZero()=default;
std::vector<uint64_t> PublicZero::evaluate(const std::vector<uint64_t>& input){
 auto& s=*state_;if(s.closing)throw std::runtime_error("E_SWITCH_CLOSED");if(input.size()!=3*7*degree)throw std::invalid_argument("E_SWITCH_SHAPE");check(cudaSetDevice(s.device));std::vector<uint64_t> output(2*6*degree);
 struct WipeFailure{std::vector<uint64_t>& v;bool ok=false;~WipeFailure(){if(!ok){volatile uint64_t* p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}}}wipe{output};
 try{
 check(cudaEventRecord(s.events[0],s.stream));check(cudaMemcpyAsync(s.input,input.data(),input.size()*8,cudaMemcpyHostToDevice,s.stream));check(cudaEventRecord(s.events[1],s.stream));
 check(cudaMemcpyAsync(s.expanded,s.input,7*degree*8,cudaMemcpyDeviceToDevice,s.stream));
 for(size_t l=0;l<14;++l){stage<<<(7*degree/2+255)/256,256,0,s.stream>>>(s.expanded,s.q,s.forward,s.invn,7,7*degree/2,degree>>(l+1),size_t(1)<<l,false,false);check(cudaGetLastError());}check(cudaEventRecord(s.events[2],s.stream));
 product<<<(2*7*degree+255)/256,256,0,s.stream>>>(s.expanded,s.keys,s.prod,s.mods);check(cudaGetLastError());check(cudaEventRecord(s.events[3],s.stream));
 for(size_t l=0;l<14;++l){stage<<<(2*7*degree/2+255)/256,256,0,s.stream>>>(s.prod,s.q,s.inverse,s.invn,7,2*7*degree/2,size_t(1)<<l,degree-(degree>>l)+1,true,l==13);check(cudaGetLastError());}check(cudaEventRecord(s.events[4],s.stream));
 noise_add<<<(2*7*degree+255)/256,256,0,s.stream>>>(s.prod,s.input,s.q);check(cudaGetLastError());
 moddown<<<(2*6*degree+255)/256,256,0,s.stream>>>(s.prod,s.input,s.output,s.q,s.invp);check(cudaGetLastError());check(cudaEventRecord(s.events[5],s.stream));
 check(cudaMemcpyAsync(output.data(),s.output,output.size()*8,cudaMemcpyDeviceToHost,s.stream));check(cudaEventRecord(s.events[6],s.stream));check(cudaEventSynchronize(s.events[6]));
 auto time=[&](size_t a,size_t b){float t;check(cudaEventElapsedTime(&t,s.events[a],s.events[b]));return double(t);};s.stats.upload_ms=time(0,1);s.stats.rns_ntt_ms=time(1,2);s.stats.product_ms=time(2,3);s.stats.inverse_ms=time(3,4);s.stats.moddown_ms=time(4,5);s.stats.download_ms=time(5,6);
 for(auto item:std::array<std::pair<uint64_t*,size_t>,4>{{{s.input,3*7*degree*8},{s.expanded,7*degree*8},{s.prod,2*7*degree*8},{s.output,2*6*degree*8}}})check(cudaMemsetAsync(item.first,0,item.second,s.stream));check(cudaStreamSynchronize(s.stream));wipe.ok=true;return output;
 }catch(...){s.release();throw;}
}
Timing PublicZero::timing()const{return state_->stats;}
Cleanup PublicZero::close()noexcept{return state_->release();}
}
