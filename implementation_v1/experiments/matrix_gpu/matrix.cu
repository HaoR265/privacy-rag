#include "matrix.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
namespace matrix_gpu { namespace {
struct Root {unsigned long long value,quotient;};
struct Modulus {unsigned long long q,r0,r1;};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
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

__global__ void lift(const uint64_t* ring,uint64_t* out,const Modulus* mods,size_t blocks){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=blocks*primes*degree)return;
 size_t b=i/(primes*degree),k=(i/degree)%primes,z=i%degree;auto v=ring[b*degree+z];out[i]=v+((mods[k].q-(uint64_t(1)<<59))&(uint64_t(0)-(v>>58)));
}
__global__ void stage(uint64_t* data,const Modulus* mods,const Root* roots,const Root* invn,size_t butterflies,size_t gap,size_t start,bool inverse,bool last){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=butterflies)return;
 size_t poly=i/(degree/2),j=i%(degree/2),group=j/gap,offset=j%gap,k=poly%primes,x=poly*degree+group*2*gap+offset,y=x+gap;
 auto q=mods[k].q;auto u=data[x],v=data[y];auto r=roots[k*degree+start+group];
 if(inverse){auto a=u+v;a=a>=q?a-q:a;auto b=u+q-v;b=b>=q?b-q:b;b=mul_root(b,r,q);if(last){a=mul_root(a,invn[k],q);b=mul_root(b,invn[k],q);}data[x]=a;data[y]=b;}
 else {auto t=mul_root(v,r,q),a=u+t,b=u+q-t;data[x]=a>=q?a-q:a;data[y]=b>=q?b-q:b;}
}
__global__ void product(const uint64_t* input,const uint64_t* weight,uint64_t* out,const Modulus* mods,size_t blocks){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=2*primes*degree)return;
 size_t k=(i/degree)%primes,z=i%(primes*degree);auto m=mods[k];unsigned long long sum=0;
 for(size_t b=0;b<blocks;++b){sum+=multiply(input[b*2*primes*degree+i],weight[b*primes*degree+z],m);sum=sum>=m.q?sum-m.q:sum;}out[i]=sum;
}
}
namespace {
__global__ void gather_c0(const uint64_t* output,const uint64_t* indices,uint64_t* selected,size_t count){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<primes*count)selected[i]=output[(i/count)*degree+indices[i%count]];
}
struct HostWipe{std::vector<uint64_t>& v;~HostWipe(){volatile uint64_t* p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}};
}
struct Matrix::State {
 int device=-1;size_t blocks=0,bytes=0;uint64_t *input=nullptr,*ring=nullptr,*weight=nullptr,*output=nullptr,*selected=nullptr,*indices=nullptr;Modulus* mods=nullptr;Root *f=nullptr,*r=nullptr,*ni=nullptr;Cleanup c;bool closing=false;
 Cleanup release()noexcept{closing=true;if(c.closed)return c;auto ok=[&](cudaError_t e){if(e!=cudaSuccess&&!c.first_cuda_error)c.first_cuda_error=int(e);return e==cudaSuccess;};if(device<0){c.closed=true;return c;}if(!ok(cudaSetDevice(device)))return c;
 std::array<std::pair<uint64_t*,size_t>,6> items={{{input,blocks*2*primes*degree*8},{ring,blocks*degree*8},{weight,blocks*primes*degree*8},{output,2*primes*degree*8},{selected,512*primes*8},{indices,512*8}}};size_t queued=0;for(auto a:items)if(a.first&&ok(cudaMemset(a.first,0,a.second)))queued+=a.second;if(ok(cudaDeviceSynchronize()))c.wiped+=queued;
 auto free=[&](auto*& ptr,size_t n){if(ptr&&ok(cudaFree(ptr))){ptr=nullptr;c.freed+=n;}};free(input,items[0].second);free(ring,items[1].second);free(weight,items[2].second);free(output,items[3].second);free(selected,items[4].second);free(indices,items[5].second);
 auto pub=[&](auto*& ptr){if(ptr&&ok(cudaFree(ptr)))ptr=nullptr;};pub(mods);pub(f);pub(r);pub(ni);c.closed=!input&&!ring&&!weight&&!output&&!selected&&!indices&&!mods&&!f&&!r&&!ni;return c;}
 ~State(){release();}
 void transform(uint64_t* ptr,size_t polys,bool inverse){size_t butterflies=polys*primes*degree/2;for(size_t l=0;l<14;++l){stage<<<(butterflies+255)/256,256>>>(ptr,mods,inverse?r:f,ni,butterflies,inverse?(size_t(1)<<l):(degree>>(l+1)),inverse?(degree-(degree>>l)+1):(size_t(1)<<l),inverse,inverse&&l==13);check(cudaGetLastError());}}
};
void Matrix::initialize(const std::vector<uint64_t>& input,const std::vector<uint64_t>& q,const std::vector<uint64_t>& f,const std::vector<uint64_t>& r,const std::vector<uint64_t>& ni){
 s=std::make_shared<State>();s->blocks=input.size()/(2*primes*degree);int count=0;check(cudaGetDeviceCount(&count));for(int i=0;i<count;++i){cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,i));if(std::strstr(p.name,"RTX 2060")&&p.major==7&&p.minor==5){s->device=i;break;}}if(s->device<0)throw std::runtime_error("RTX2060_REQUIRED");check(cudaSetDevice(s->device));
 try{check(cudaMalloc(&s->input,input.size()*8));check(cudaMalloc(&s->ring,s->blocks*degree*8));check(cudaMalloc(&s->weight,s->blocks*primes*degree*8));check(cudaMalloc(&s->output,2*primes*degree*8));check(cudaMalloc(&s->selected,512*primes*8));check(cudaMalloc(&s->indices,512*8));check(cudaMalloc(&s->mods,primes*sizeof(Modulus)));check(cudaMalloc(&s->f,f.size()*8));check(cudaMalloc(&s->r,r.size()*8));check(cudaMalloc(&s->ni,ni.size()*8));
 std::array<Modulus,primes> mods;for(size_t k=0;k<primes;++k){auto ratio=(~static_cast<unsigned __int128>(0))/q[k];mods[k]={q[k],uint64_t(ratio),uint64_t(ratio>>64)};}
 check(cudaMemcpy(s->mods,mods.data(),sizeof(mods),cudaMemcpyHostToDevice));check(cudaMemcpy(s->f,f.data(),f.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(s->r,r.data(),r.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(s->ni,ni.data(),ni.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(s->input,input.data(),input.size()*8,cudaMemcpyHostToDevice));s->transform(s->input,s->blocks*2,false);check(cudaDeviceSynchronize());
 s->bytes=512*(primes+1)*8+input.size()*8+s->blocks*(degree+primes*degree)*8+2*primes*degree*8+sizeof(mods)+(f.size()+r.size()+ni.size())*8;
 }catch(...){s->release();throw;}}
Matrix::~Matrix()=default;
std::vector<uint64_t> Matrix::evaluate(const std::vector<uint64_t>& ring){return evaluate_impl(ring,nullptr);}
std::vector<uint64_t> Matrix::evaluate_selective(const std::vector<uint64_t>& ring,const std::vector<uint64_t>& targets){
 if(targets.empty()||targets.size()>512)throw std::invalid_argument("MATRIX_GPU_TARGET_COUNT");
 for(size_t i=0;i<targets.size();++i)if(targets[i]>=degree||(i&&targets[i]<=targets[i-1]))throw std::invalid_argument("MATRIX_GPU_TARGET_INDEX");
 return evaluate_impl(ring,&targets);
}
std::vector<uint64_t> Matrix::evaluate_impl(const std::vector<uint64_t>& ring,const std::vector<uint64_t>* targets){if(!s||s->closing||ring.size()!=s->blocks*degree)throw std::invalid_argument("MATRIX_GPU_SHAPE");for(auto v:ring)if(v>=(uint64_t(1)<<59))throw std::invalid_argument("MATRIX_GPU_RING");std::vector<uint64_t> out(2*primes*degree);
 try{check(cudaSetDevice(s->device));check(cudaMemcpy(s->ring,ring.data(),ring.size()*8,cudaMemcpyHostToDevice));size_t total=s->blocks*primes*degree;lift<<<(total+255)/256,256>>>(s->ring,s->weight,s->mods,s->blocks);check(cudaGetLastError());s->transform(s->weight,s->blocks,false);product<<<(out.size()+255)/256,256>>>(s->input,s->weight,s->output,s->mods,s->blocks);check(cudaGetLastError());s->transform(s->output,2,true);if(!targets){check(cudaMemcpy(out.data(),s->output,out.size()*8,cudaMemcpyDeviceToHost));}
 else {
  const size_t count=targets->size();std::vector<uint64_t> selected(count*primes);HostWipe wipe{selected};
  check(cudaMemcpy(s->indices,targets->data(),count*8,cudaMemcpyHostToDevice));
  gather_c0<<<(count*primes+255)/256,256>>>(s->output,s->indices,s->selected,count);check(cudaGetLastError());
  check(cudaMemcpy(out.data()+primes*degree,s->output+primes*degree,primes*degree*8,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(selected.data(),s->selected,selected.size()*8,cudaMemcpyDeviceToHost));
  for(size_t k=0;k<primes;++k)for(size_t j=0;j<count;++j)out[k*degree+(*targets)[j]]=selected[k*count+j];
 }
 return out;
 }catch(...){s->release();volatile uint64_t* p=out.data();for(size_t i=0;i<out.size();++i)p[i]=0;throw;}}
size_t Matrix::device_bytes()const{return s->bytes;}Cleanup Matrix::close()noexcept{return s->release();}
}
