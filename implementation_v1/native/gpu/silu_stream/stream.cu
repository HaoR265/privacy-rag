#include "stream.hpp"
#include <exception>
// Independent exact SCI MITCCRH half-gate primitive diagnostic, not production.
#include "halfgate.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>
#include <string>
#include <limits>
#include <openssl/crypto.h>
#include <cstdio>
#include <exception>
#include <cstdlib>
namespace gpu_hg {
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e));}
// AES S-box is public. Indices are each party's own already-known labels/key bytes.
// No device memory or timing is exposed as a new protocol output by this prototype.
__device__ const uint8_t sb[256]={
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};
__device__ __forceinline__ uint32_t rol(uint32_t x,unsigned n){return (x<<n)|(x>>(32-n));}
__device__ __forceinline__ uint32_t subword(uint32_t x){return uint32_t(__ldg(sb+(x&255)))|(uint32_t(__ldg(sb+((x>>8)&255)))<<8)|(uint32_t(__ldg(sb+((x>>16)&255)))<<16)|(uint32_t(__ldg(sb+(x>>24)))<<24);}
__device__ __forceinline__ uint32_t mix(uint32_t x){uint32_t hi=x&0x80808080u;uint32_t twice=((x&0x7f7f7f7fu)<<1)^((hi>>7)*0x1bu);return twice^rol(x,8)^rol(x,16)^rol(x,24)^rol(twice,24);}
__device__ __forceinline__ Label X(Label a,Label b){Label o;for(int j=0;j<4;++j)o.w[j]=a.w[j]^b.w[j];return o;}
__device__ __forceinline__ Label mask(Label a,uint32_t bit){Label o;for(int j=0;j<4;++j)o.w[j]=a.w[j]&(0u-bit);return o;}
__device__ __forceinline__ Label aesround(Label s,Label k,bool last){Label out;for(int c=0;c<4;++c){uint32_t x=(s.w[c]&255)|(s.w[(c+1)&3]&0xff00)|(s.w[(c+2)&3]&0xff0000)|(s.w[(c+3)&3]&0xff000000);x=subword(x);out.w[c]=(last?x:mix(x))^k.w[c];}return out;}
__device__ __forceinline__ Label nextkey(Label k,uint8_t rc){uint32_t t=subword(rol(k.w[3],24))^rc;k.w[0]^=t;k.w[1]^=k.w[0];k.w[2]^=k.w[1];k.w[3]^=k.w[2];return k;}
template<int N> __device__ __forceinline__ void hash(Label (&x)[N],Label seed,uint64_t keyid){Label key=seed;key.w[2]^=uint32_t(keyid);key.w[3]^=uint32_t(keyid>>32);Label orig[N];for(int j=0;j<N;++j){orig[j]=x[j];x[j]=X(x[j],key);}uint8_t rc=1;
 for(int r=1;r<=10;++r){key=nextkey(key,rc);for(int j=0;j<N;++j)x[j]=aesround(x[j],key,r==10);rc=(rc<<1)^((0u-(rc>>7))&0x1b);}
 for(int j=0;j<N;++j)x[j]=X(x[j],orig[j]);}
struct Config {Label seed,delta;uint64_t first;Label zero,one;};
__global__ void garble_kernel(const Input*in,Garbled*out,const Config*cfg,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;Label a=in[i].a,b=in[i].b,d=cfg->delta;Label ha[2]={a,X(a,d)},hb[2]={b,X(b,d)};uint64_t id=2*(cfg->first+i);hash(ha,cfg->seed,id);hash(hb,cfg->seed,id+1);Label t0=X(X(ha[0],ha[1]),mask(d,b.w[0]&1)),tmp=X(hb[0],hb[1]);out[i].table0=t0;out[i].table1=X(tmp,a);out[i].zero=X(X(X(ha[0],mask(t0,a.w[0]&1)),hb[0]),mask(tmp,b.w[0]&1));}
__global__ void eval_kernel(const EvalInput*in,Label*out,const Config*cfg,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;Label a=in[i].a,b=in[i].b;Label ha[1]={a},hb[1]={b};uint64_t id=2*(cfg->first+i);hash(ha,cfg->seed,id);hash(hb,cfg->seed,id+1);out[i]=X(X(X(X(ha[0],hb[0]),mask(in[i].table0,a.w[0]&1)),mask(in[i].table1,b.w[0]&1)),mask(a,b.w[0]&1));}
// Explicit ownership: failed memset/sync retains all allocations; failed free
// retains that handle for retry. No global pool or device reset is used.
struct WipeGuard {
 void* data; size_t size; bool wiped=false;
 WipeGuard(void* p,size_t n) noexcept:data(p),size(n){}
 WipeGuard(const WipeGuard&)=delete;
 void clear() noexcept {if(!wiped){OPENSSL_cleanse(data,size);wiped=true;}}
 ~WipeGuard(){clear();}
};
static std::string uuid_string(const cudaUUID_t& u){
 const auto* b=reinterpret_cast<const unsigned char*>(u.bytes);char out[41];
 std::snprintf(out,sizeof(out),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);return out;
}
static std::string canonical_pci(const std::string& s){unsigned a,b,c,d;int used=0;if(std::sscanf(s.c_str(),"%x:%x:%x.%x%n",&a,&b,&c,&d,&used)!=4||used!=int(s.size())||b>255||c>31||d>7)throw std::runtime_error("DEVICE_PCI_FORMAT");char v[32];std::snprintf(v,sizeof(v),"%08x:%02x:%02x.%x",a,b,c,d);return v;}
static bool matches(int index,const DeviceBinding& expected){
 cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,index));char pci[32]{};ck(cudaDeviceGetPCIBusId(pci,sizeof(pci),index));
 return uuid_string(p.uuid)==expected.uuid && canonical_pci(pci)==canonical_pci(expected.pci_bus_id) && std::string(p.name)=="NVIDIA GeForce RTX 2060" && p.major==7 && p.minor==5;
}
[[noreturn]] static void cleanup_fail_stop(const char* public_error) noexcept {std::fputs(public_error,stderr);std::fputc('\n',stderr);std::_Exit(70);}
struct DeviceScope {
 int previous=-1,index=-1;
 explicit DeviceScope(const DeviceBinding& expected){
  if(expected.uuid.empty()||expected.pci_bus_id.empty())throw std::runtime_error("DEVICE_BINDING_REQUIRED");
  canonical_pci(expected.pci_bus_id);int n=0;ck(cudaGetDeviceCount(&n));
  for(int i=0;i<n;++i)if(matches(i,expected)){if(index!=-1)throw std::runtime_error("DEVICE_BINDING_AMBIGUOUS");index=i;}
  if(index<0)throw std::runtime_error("RTX2060_IDENTITY_NOT_FOUND");
  ck(cudaGetDevice(&previous));
  try {ck(cudaSetDevice(index));int actual=-1;ck(cudaGetDevice(&actual));
   if(actual!=index||!matches(actual,expected))throw std::runtime_error("DEVICE_BINDING_CHANGED");
  } catch (...) {if(previous!=index&&cudaSetDevice(previous)!=cudaSuccess)cleanup_fail_stop("GPU_DEVICE_RESTORE_FAILED");throw;}
 }
 void restore(){if(previous>=0&&previous!=index)ck(cudaSetDevice(previous));previous=-1;}
 DeviceScope(const DeviceScope&)=delete;
 ~DeviceScope(){if(previous>=0&&previous!=index&&cudaSetDevice(previous)!=cudaSuccess)cleanup_fail_stop("GPU_DEVICE_RESTORE_FAILED");}
};
struct Buffers {
 void *a=nullptr,*b=nullptr,*cfg=nullptr;size_t na=0,nb=0;bool cleaned=false;
 Buffers()=default;Buffers(const Buffers&)=delete;
 void close(){
  cleaned=false;if(!a&&!b&&!cfg){cleaned=true;return;}
  cudaError_t err=cudaSuccess;auto remember=[&](cudaError_t e){if(err==cudaSuccess&&e!=cudaSuccess)err=e;};
  if(a)remember(cudaMemset(a,0,na));if(b)remember(cudaMemset(b,0,nb));if(cfg)remember(cudaMemset(cfg,0,sizeof(Config)));
  remember(cudaDeviceSynchronize());ck(err); // keep every handle if wipe completion was not confirmed
  auto release=[&](void*& p){if(p){auto e=cudaFree(p);if(e==cudaSuccess)p=nullptr;remember(e);}};
  release(a);release(b);release(cfg);ck(err);cleaned=!a&&!b&&!cfg;
 }
 ~Buffers() noexcept {try{close();}catch(...){cleanup_fail_stop("GPU_RESOURCE_CLEANUP_FAILED");}}
};
static void check(size_t n,uint64_t first){if(!n||n>1048576||first>(UINT64_MAX-2*n)/2)throw std::runtime_error("SHAPE_OR_KEY_ID");}
std::string device(const DeviceBinding& expected){DeviceScope scope(expected);cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,scope.index));return std::string(p.name)+" / "+uuid_string(p.uuid)+" / "+canonical_pci(expected.pci_bus_id);}

#include "stream_impl.inc"
}
