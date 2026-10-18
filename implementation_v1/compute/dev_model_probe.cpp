// Public synthetic numeric diagnostics ONLY. No production linkage or data.
#include "pipeline.hpp"
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
using namespace prag;
static int run(Role role,int fd,const std::string& prefix){
  std::array<unsigned char,16> session{};session[0]=43;std::array<unsigned char,32> config{};config[0]=1;
  Channel io(fd,session,config);SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),true);
  std::array<std::vector<uint64_t>,3> tables;for(auto& t:tables)t.resize(256);
  // This run uses no LUT calls; zero tables are unused, never reported checked.
  model::Numerics n(op,tables);
  auto share=[&](uint64_t value){return role==Role::Client?912837ULL:(value-912837ULL)&MASK;};
  std::vector<uint64_t> output;
  output.push_back(n.scale_shift(share(uint64_t(-12345)&MASK),share(4),0));
  output.push_back(n.scale_shift(share(12345),share(uint64_t(-4)&MASK),0));
  output.push_back(n.inverse_sqrt(share(2ULL<<24)));
  output.push_back(n.reciprocal(share(3ULL<<24)));
  model::Retriever retrieval(n);
  std::vector<model::Candidate> candidates{{share(uint64_t(-5)&MASK),share(0),op.public_bit(true)},
    {share(100),share(1),op.public_bit(false)}, {share(uint64_t(-5)&MASK),share(2),op.public_bit(true)}};
  auto selected=retrieval.top2(candidates);
  output.push_back(selected[0].index);output.push_back(selected[1].index);
  // Adopted Q/K caps are encoded f12 integers, not rounded real magnitudes.
  output.push_back(n.checked_encoded(share(2000000),8943813));
  output.push_back(n.checked_encoded(share(uint64_t(-8943813)&MASK),8943813));
  // DEC-012 preserves both endpoints of the floor-rescaled product domain.
  output.push_back(n.checked_encoded(op.floor(op.mul(share(256ULL<<12),share(256ULL<<12)),12),65536ULL<<12));
  output.push_back(n.checked_encoded(share(uint64_t(-(int64_t{65536}<<12))&MASK),65536ULL<<12));
  output.push_back(op.b2a_bit(n.failure()));
  output.push_back(n.checked_encoded(share(8943814),8943813));
  output.push_back(n.checked_encoded(share((65536ULL<<12)+1),65536ULL<<12));
  output.push_back(n.scale_shift(share(uint64_t(-(int64_t{1}<<56))&MASK),share(0),0));
  model::Word invalid=share(0);output.push_back(n.exponent(invalid,-10,8,true));
  output.push_back(invalid);output.push_back(op.b2a_bit(n.failure()));
  if(role==Role::Server){io.send_data(output.data(),output.size()*8);return 0;}
  std::vector<uint64_t> other(output.size());io.recv_data(other.data(),other.size()*8);
  for(size_t i=0;i<output.size();++i)output[i]=(output[i]+other[i])&MASK;
  std::vector<uint64_t> expected{uint64_t(-772)&MASK,197520,11863283,5592405,0,2,2000000,uint64_t(-8943813)&MASK,65536ULL<<12,uint64_t(-(int64_t{65536}<<12))&MASK,0,0,0,0,0,1ULL<<24,1};
  if(output!=expected)throw std::runtime_error("E_NUMERIC_DIAGNOSTIC_MISMATCH");
  std::cout<<"{\"entry\":\"isolated_public_numeric_probe\",\"two_processes\":true,\"barrel_shift_signed_floor\":\"matched\",\"six_round_inverse_sqrt\":\"matched_R1\",\"six_round_reciprocal\":\"matched_R1\",\"top2_validity_negative_scores_ties\":\"matched\",\"qk_and_swiglu_signed_boundaries\":\"matched\",\"strict_2pow56_guard_and_positive_dummy\":\"matched\",\"client_bytes\":"<<io.bytes()<<",\"LUT_exponential_not_run\":true,\"full_model_not_run\":true}\n";
  return 0;
}
int main(int argc,char** argv){
  if(argc!=2)return 2;int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets))return 3;
  pid_t child=fork();if(child<0)return 3;int rc=1;
  try{if(child==0){close(sockets[0]);rc=run(Role::Server,sockets[1],argv[1]);_exit(rc);}close(sockets[1]);rc=run(Role::Client,sockets[0],argv[1]);}
  catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(child==0)_exit(1);}
  int state=0;waitpid(child,&state,0);return rc||!WIFEXITED(state)||WEXITSTATUS(state);
}
