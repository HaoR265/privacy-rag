// Public synthetic diagnostics of the exact production guard kernel only.
// Real two-party SecureOps; reconstruction is confined to this executable.
#include "artifacts.hpp"
#include <csignal>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
using namespace prag::model;

struct Case {
  const char* name;
  std::array<int64_t,5> input;
  Word cap,limit;
  bool prior_failure=false;
};
static int run(Role role,int fd,const std::string& prefix,const std::string& constants_path){
  Constants constants(constants_path);
  std::array<unsigned char,16> session{};session[0]=71;
  std::array<unsigned char,32> configuration{};configuration[0]=14;
  Channel io(fd,session,configuration);
  SecureOps s(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),true);
  const Word cap=numeric_profile::rms_coordinate_cap(1024),limit=numeric_profile::rms_sum_limit(1024);
  const Word main=numeric_profile::l2_coordinate_cap,l2=numeric_profile::l2_sum_limit;
  // 67108863^2 + 2^2 + 6267^2 + 9743^2 is the exact RMS1024 limit.
  const std::vector<Case> cases{
    {"zero",{0,0,0,0,0},cap,limit},
    {"negative_coordinate_boundary",{-int64_t(cap),1,-2,3,0},cap,limit},
    {"exact_sum_limit",{67108863,2,6267,9743,0},cap,limit},
    {"one_above_sum_limit",{67108863,2,6267,9743,1},cap,limit},
    {"positive_coordinate_overflow",{int64_t(cap)+1,1,-2,3,4},cap,limit},
    {"negative_coordinate_overflow",{-int64_t(cap)-1,1,-2,3,4},cap,limit},
    {"large_input_before_square",{int64_t{1}<<55,1,-2,3,4},cap,limit},
    {"L2_coordinate_boundary",{int64_t(main),-int64_t(main),1,-2,0},main,l2},
    {"L2_coordinate_overflow",{int64_t(main)+1,-1,2,-3,0},main,l2},
    {"prior_failure_does_not_corrupt_row_guard",{1,-2,3,-4,5},cap,limit,true}
  };
  std::vector<uint64_t> bytes;
  for(std::size_t which=0;which<cases.size();++which){
    const auto& c=cases[which];Numerics n(s,constants.exp);
    if(c.prior_failure)n.fail(s.public_bit(true));
    Row input;
    for(std::size_t j=0;j<c.input.size();++j){
      Word client=(Word(which+1)*912837+Word(j+1)*7919)&mask;
      input.push_back(role==Role::Client?client:(Word(c.input[j])-client)&mask);
    }
    const auto before=io.bytes();
    auto result=n.guarded_square_sum(input,c.cap,c.limit);
    Row output=result.values;
    output.push_back(result.sum);output.push_back(s.b2a_bit(result.row_failed));output.push_back(s.b2a_bit(n.failure()));
    // Independent mathematical oracle: original coordinates and full exact sum.
    __int128 total=0;bool rejected=false;
    for(auto v:c.input){__int128 x=v;rejected|=x>c.cap||x<-__int128(c.cap);total+=x*x;}
    rejected|=total>c.limit;
    Row expected;
    for(auto v:c.input)expected.push_back(rejected?0:Word(v)&mask);
    expected.push_back(rejected?0:Word(total));expected.push_back(rejected);expected.push_back(rejected||c.prior_failure);
    if(role==Role::Server)s.channel().send_data(output.data(),int(output.size()*sizeof(Word)));
    else{
      Row peer(output.size());s.channel().recv_data(peer.data(),int(peer.size()*sizeof(Word)));
      for(std::size_t j=0;j<output.size();++j)if(s.add(output[j],peer[j])!=expected[j])throw std::runtime_error("E_PUBLIC_GUARD_MISMATCH");
      bytes.push_back(io.bytes()-before);
      std::cout<<"{\"entry\":\"isolated_public_numeric_guard_probe\",\"case\":\""<<c.name<<"\",\"matched\":true,\"client_channel_bytes\":"<<bytes.back()<<",\"real_two_party_SecureOps\":true,\"full_RMS_L2_or_model_run\":false}"<<std::endl;
    }
  }
  if(role==Role::Client){
    for(std::size_t j=2;j<bytes.size();++j)if(bytes[j]!=bytes[1])throw std::runtime_error("E_PUBLIC_GUARD_SHAPE_MISMATCH");
    std::cout<<"{\"entry\":\"isolated_public_numeric_guard_summary\",\"cases_matched\":"<<cases.size()<<",\"same_public_shape_same_steady_frame_bytes\":true,\"client_bytes\":"<<io.bytes()<<",\"production_TLS_not_run\":true}"<<std::endl;
  }
  return 0;
}
int main(int argc,char** argv){
  std::signal(SIGPIPE,SIG_IGN);
  if(argc!=3)return 2;
  int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets))return 3;
  pid_t child=fork();if(child<0){close(sockets[0]);close(sockets[1]);return 3;}
  int rc=1;
  try{
    if(child==0){close(sockets[0]);rc=run(Role::Server,sockets[1],argv[1],argv[2]);_exit(rc);}
    close(sockets[1]);rc=run(Role::Client,sockets[0],argv[1],argv[2]);
  }catch(const std::exception& e){
    std::cerr<<e.what()<<'\n';shutdown(sockets[child==0?1:0],SHUT_RDWR);if(child==0)_exit(1);
  }
  int state=0;waitpid(child,&state,0);return rc||!WIFEXITED(state)||WEXITSTATUS(state);
}
