#include "prag/silu_gpu_bridge.hpp"
#include "halfgate.hpp"
#include "stream.hpp"
#include "template_bytes.hpp"
#include <openssl/crypto.h>
#include <openssl/sha.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <type_traits>
namespace prag::silu_gpu_bridge {
namespace {
template<class T>struct WipeVector {std::vector<T> v;explicit WipeVector(size_t n=0):v(n){}~WipeVector(){if(!v.empty())OPENSSL_cleanse(v.data(),v.size()*sizeof(T));}};
std::string required(const char* name){auto p=std::getenv(name);if(!p||!*p)throw std::runtime_error("E_GPU_BINDING_REQUIRED");return p;}
gpu_hg::StreamPlan plan;gpu_hg::DeviceBinding device;Binding binding;bool ready=false;
uint32_t read32(const std::vector<unsigned char>&b,size_t&at){if(at+4>b.size())throw std::runtime_error("E_TEMPLATE_TRUNCATED");uint32_t x=0;for(unsigned i=0;i<4;++i)x|=uint32_t(b[at++])<<(8*i);return x;}
gpu_hg::Label label(sci::block128 x){gpu_hg::Label r;std::memcpy(&r,&x,16);return r;}

}
Configuration::~Configuration(){OPENSSL_cleanse(&seed,sizeof(seed));OPENSSL_cleanse(&delta,sizeof(delta));OPENSSL_cleanse(&zero,sizeof(zero));OPENSSL_cleanse(&one,sizeof(one));}
Binding prepare(){
 if(ready)return binding;
 device={required("PRAG_SILU_GPU_UUID"),required("PRAG_SILU_GPU_PCI")};
 std::vector<unsigned char>b(std::begin(compiled_plan),std::end(compiled_plan));
 std::array<unsigned char,32> hash{};SHA256(b.data(),b.size(),hash.data());
 const unsigned char expected[]={0x7,0x2b,0x8c,0x7f,0xc3,0x71,0xf5,0xd,0xaf,0xe4,0x2d,0x88,0xba,0x27,0xc,0x55,0x4c,0x8c,0x8a,0x42,0xa,0xb7,0xf0,0x9c,0xb1,0x54,0xb,0xa5,0x22,0xf6,0xa5,0xd9 };
 if(!std::equal(hash.begin(),hash.end(),expected))throw std::runtime_error("E_SILU_GPU_TEMPLATE_BINDING");
 std::istringstream bound_bytes(std::string(reinterpret_cast<const char*>(b.data()),b.size()));
 plan=gpu_hg::read_plan(bound_bytes);
 // Bind implementation, fixed tiling and wire-layout version independently of
 // each participant's local GPU identity (which need not match remotely).
 std::string public_id="exact-silu-gpu-stream-global-three-phase-v1;tile=256;frames=65536;";public_id.append(reinterpret_cast<char*>(hash.data()),32);SHA256(reinterpret_cast<const unsigned char*>(public_id.data()),public_id.size(),binding.digest.data());
 gpu_hg::device(device);ready=true;return binding;
}
std::vector<sci::block128> execute(bool sender,size_t n,const std::vector<sci::block128>&ci,const std::vector<sci::block128>&si,const Configuration&cfg,GcBatchChannel&channel){
 if(!ready||!n||n>6144||ci.size()!=59*n||si.size()!=119*n)throw std::runtime_error("E_SILU_GPU_SHAPE");
 WipeVector<sci::block128> outputs(60*n);
 auto times=gpu_hg::stream_complete(sender,device,plan,n,ci.data(),si.data(),outputs.v.data(),label(cfg.seed),label(cfg.delta),label(cfg.zero),label(cfg.one),0,[&](void*p,size_t bytes){if(sender)channel.send_tables(p,bytes);else channel.recv_tables(p,bytes);});
 if(!times.cleaned||!times.host_config_wiped)throw std::runtime_error("E_GPU_COMPLETION");

 std::vector<sci::block128> result;result.swap(outputs.v);return result;
}
}
