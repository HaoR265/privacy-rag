// Isolated fault injection before any model weight/arithmetic is used.
// The production method's exceptional path must release a caller-owned Cache.
#include "model.hpp"
#include <csignal>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
using namespace prag::model;
struct FailingWeights:Weights {
  const PrivateMatrix& token_table()override {throw std::runtime_error("EXPECTED_TOKEN_WEIGHT_FAULT");}
  LayerWeights layer(size_t)override {throw std::runtime_error("UNREACHABLE_LAYER");}
  PrivateMatrix final_norm()override {throw std::runtime_error("UNREACHABLE_FINAL_NORM");}
};
int run(Role role,int fd,const std::string& prefix,const std::string& pki,int mode){
  std::array<unsigned char,16>sid{};sid[0]=112;sid[1]=60+mode;
  std::array<unsigned char,32>cfg{};cfg[0]=112;cfg[1]=60+mode;
  Channel io(fd,sid,cfg);std::string own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,pki+"/ca.pem",pki+"/"+own+".pem",pki+"/"+own+".key","localhost");
  SecureOps ops(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  auto gpu=make_rtx2060_shares();
  std::array<Row,3>exp{Row(256),Row(256),Row(256)};
  Numerics n(ops,exp,[mode](const char*,bool begin){if(mode==1&&begin)throw std::runtime_error("EXPECTED_STAGE_FAULT");},gpu.get());
  FailingWeights weights;Transformer model(n,ModelShape::generation(),weights,Rows(64,Row(1280)),Rows(64,Row(1280)),0);
  Cache cache(ops,*gpu);cache.next_write=1024;
  cache.write(false,0,0,1023,Row(128,MASK));
  if(gpu->stats().live_bytes!=587202560)throw std::runtime_error("CACHE_ALLOCATION");
  bool fault=false;
  try{model.forward({ops.public_word(0)},{ops.public_bit(true)},{ops.public_word(1024)},&cache,1024);}
  catch(const std::runtime_error&e){fault=std::string(e.what())==(mode?"EXPECTED_STAGE_FAULT":"EXPECTED_TOKEN_WEIGHT_FAULT");}
  if(!fault||ops.session_active()||cache.usable||cache.next_write||gpu->stats().live_bytes)throw std::runtime_error("INCOMPLETE_CACHE_ABORT");
  for(auto b:cache.valid)if(b)throw std::runtime_error("RETAINED_VALIDITY");
  bool closed=false;try{cache.read(false,0,0,1);}catch(const std::runtime_error&e){closed=std::string(e.what())=="E_CUDA_STORAGE_CLOSED";}
  if(!closed)throw std::runtime_error("GPU_CACHE_REUSE");
  if(role==Role::Client)std::cout<<"{\"fault\":\""<<(mode?"stage_callback":"token_weight_loader")<<"\",\"cache_released_before_destructor\":true,\"cache_reuse_rejected\":true,\"crypto_aborted\":true,\"model_arithmetic_executed\":false}\n";
  return 0;
}
int main(int argc,char**argv){
  if(argc!=4)return 2;std::signal(SIGPIPE,SIG_IGN);int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))return 3;
  auto child=fork();if(child<0)return 3;int status=1;
  if(!child){close(pair[0]);try{status=run(Role::Server,pair[1],argv[1],argv[2],std::stoi(argv[3]));}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}close(pair[1]);_exit(status);}
  close(pair[1]);try{status=run(Role::Client,pair[0],argv[1],argv[2],std::stoi(argv[3]));}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}close(pair[0]);
  int other=0;waitpid(child,&other,0);return status||!WIFEXITED(other)||WEXITSTATUS(other);
}
