#include "local_shares.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
namespace prag::model {
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(std::string("E_CUDA_LOCAL: ")+cudaGetErrorString(e)); }
constexpr uint64_t ring_mask=(uint64_t{1}<<59)-1;
__global__ void add_kernel(const uint64_t*a,const uint64_t*b,uint64_t*c,size_t n){size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(i<n)c[i]=(a[i]+b[i])&ring_mask;}
__global__ void scale_kernel(const uint64_t*a,uint64_t k,uint64_t*c,size_t n){size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(i<n)c[i]=(a[i]*k)&ring_mask;}
__global__ void layout_kernel(const uint64_t*a,const size_t*map,uint64_t*c,size_t n){size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(i<n)c[i]=a[map[i]];}
__global__ void sum_kernel(const uint64_t*a,uint64_t*b,size_t rows,size_t cols){size_t row=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(row<rows){uint64_t x=0;for(size_t i=0;i<cols;++i)x+=a[row*cols+i];b[row]=x&ring_mask;}}
struct State {cudaStream_t stream{};int device;LocalShareStats stats;explicit State(int d):device(d){check(cudaSetDevice(device));check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}~State(){cudaSetDevice(device);cudaStreamDestroy(stream);}};
class Storage final:public ShareStorage {
  std::shared_ptr<State> state;size_t count;uint64_t* ptr=nullptr;
 public:
  Storage(std::shared_ptr<State>s,size_t n):state(std::move(s)),count(n){if(!n||n>SIZE_MAX/8)throw std::invalid_argument("E_CUDA_SHAPE");check(cudaSetDevice(state->device));check(cudaMalloc(&ptr,n*8));state->stats.live_bytes+=n*8;state->stats.peak_bytes=std::max(state->stats.peak_bytes,state->stats.live_bytes);try{check(cudaMemsetAsync(ptr,0,n*8,state->stream));check(cudaStreamSynchronize(state->stream));}catch(...){clear();throw;}}
  ~Storage(){clear();}
  uint64_t* data(){if(!ptr)throw std::runtime_error("E_CUDA_STORAGE_CLOSED");return ptr;}
  void write(size_t first,const std::vector<Word>&x)override{if(first>count||x.size()>count-first)throw std::invalid_argument("E_CUDA_SHAPE");for(auto v:x)if(v>ring_mask)throw std::invalid_argument("E_CUDA_WORD");check(cudaSetDevice(state->device));check(cudaMemcpyAsync(data()+first,x.data(),x.size()*8,cudaMemcpyHostToDevice,state->stream));check(cudaStreamSynchronize(state->stream));state->stats.uploaded_bytes+=x.size()*8;}
  std::vector<Word> read(size_t first,size_t n)override{if(first>count||n>count-first)throw std::invalid_argument("E_CUDA_SHAPE");std::vector<Word>x(n);check(cudaSetDevice(state->device));check(cudaMemcpyAsync(x.data(),data()+first,n*8,cudaMemcpyDeviceToHost,state->stream));check(cudaStreamSynchronize(state->stream));state->stats.downloaded_bytes+=n*8;return x;}
  void clear()noexcept override{if(!ptr)return;cudaSetDevice(state->device);cudaMemsetAsync(ptr,0,count*8,state->stream);cudaStreamSynchronize(state->stream);cudaFree(ptr);ptr=nullptr;state->stats.live_bytes-=count*8;}
};
class CudaShares final:public LocalShares {
 std::shared_ptr<State> state;
 public:
 explicit CudaShares(int d):state(std::make_shared<State>(d)){}
 std::vector<Word> add(const std::vector<Word>&a,const std::vector<Word>&b)override{if(a.empty()||a.size()!=b.size())throw std::invalid_argument("E_CUDA_SHAPE");Storage x(state,a.size()),y(state,b.size()),z(state,a.size());x.write(0,a);y.write(0,b);add_kernel<<<(a.size()+255)/256,256,0,state->stream>>>(x.data(),y.data(),z.data(),a.size());check(cudaGetLastError());++state->stats.kernels;return z.read(0,a.size());}
 std::vector<Word> scale(const std::vector<Word>&a,Word k)override{if(a.empty()||k>ring_mask)throw std::invalid_argument("E_CUDA_SHAPE");Storage x(state,a.size()),z(state,a.size());x.write(0,a);scale_kernel<<<(a.size()+255)/256,256,0,state->stream>>>(x.data(),k,z.data(),a.size());check(cudaGetLastError());++state->stats.kernels;return z.read(0,a.size());}
 std::vector<Word> layout(const std::vector<Word>&a,const std::vector<size_t>&map)override{if(a.empty()||map.empty())throw std::invalid_argument("E_CUDA_SHAPE");std::vector<Word>indices(map.begin(),map.end());for(auto i:indices)if(i>=a.size())throw std::invalid_argument("E_CUDA_PUBLIC_MAP");Storage x(state,a.size()),idx(state,map.size()),z(state,map.size());x.write(0,a);idx.write(0,indices);layout_kernel<<<(map.size()+255)/256,256,0,state->stream>>>(x.data(),reinterpret_cast<const size_t*>(idx.data()),z.data(),map.size());check(cudaGetLastError());++state->stats.kernels;return z.read(0,map.size());}
 std::vector<Word> sum_rows(const std::vector<Word>&a,size_t cols)override{if(a.empty()||!cols||a.size()%cols)throw std::invalid_argument("E_CUDA_SHAPE");size_t rows=a.size()/cols;Storage x(state,a.size()),y(state,rows);x.write(0,a);sum_kernel<<<(rows+255)/256,256,0,state->stream>>>(x.data(),y.data(),rows,cols);check(cudaGetLastError());++state->stats.kernels;return y.read(0,rows);}
 std::unique_ptr<ShareStorage> allocate(size_t words)override{return std::make_unique<Storage>(state,words);}
 LocalShareStats stats()const override{return state->stats;}
};
}
std::unique_ptr<LocalShares> make_rtx2060_shares(){int count;check(cudaGetDeviceCount(&count));for(int d=0;d<count;++d){cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,d));if(std::strstr(p.name,"RTX 2060")&&p.major==7&&p.minor==5)return std::make_unique<CudaShares>(d);}throw std::runtime_error("E_RTX2060_REQUIRED");}
}
