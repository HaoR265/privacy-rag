// Isolated production-mTLS failure diagnostics. No values are reconstructed.
#include "prag/secure_ops.hpp"
#include "prag/joint_matrix.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int party(Role role,int fd,const std::string& prefix,const std::string& cert,int mode) {
  std::array<unsigned char,16> sid{};sid[0]=109;sid[1]=mode;
  std::array<unsigned char,32> config{};config[0]=109;
  Channel io(fd,sid,config);
  const std::string own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  const std::string name=prefix+(role==Role::Client?".C":".S");
  SecureOps op(role,io,name+".ledger",false);
  const auto before=io.bytes();
  std::vector<std::vector<Word>> x(mode==2?17:1,std::vector<Word>(17,0));
  std::string reason;
  try {
    PrivateWeightBlock bad=[&](size_t,size_t,size_t,size_t)->std::vector<std::vector<Word>>{if(role!=Role::Server)throw std::runtime_error("C_READ_PRIVATE_WEIGHT");return {};};
    if(mode<3)op.private_matrix(x,mode==1&&role==Role::Server?2:1,12,12,bad);
    else if(mode==3)private_joint_matrix(op,x,role==Role::Client?std::vector<size_t>{1,2}:std::vector<size_t>{2,1},12,12,{bad,bad});
    else if(mode==4)op.private_token_lookup(0,role==Role::Client?151669:151936,1024,bad);
    else if(mode==5)op.private_token_lookup(0,151669,1024,[](size_t,size_t c,size_t,size_t n){return std::vector<std::vector<Word>>(n,std::vector<Word>(c,(Word(1)<<20)+1));});
    else if(mode==6){std::vector<Word> v(role==Role::Client?128:129);op.private_diagonal(v,role==Role::Server?v:std::vector<Word>{},12,12);}
    else if(mode==7)op.private_token_lookup(0,151669,17,bad);
    throw std::runtime_error("FAILURE_NOT_DETECTED");
  } catch(const std::exception& e) {reason=e.what();}
  const bool active=op.session_active();
  if(mode==2||mode==7) {
    if(reason!=(mode==2?"E_SHAPE":"E_TOKEN_SHAPE")||!active||io.bytes()!=before)throw std::runtime_error("PUBLIC_SHAPE_GATE_FAILED");
    op.close_session();
  } else {
    if(active)throw std::runtime_error("SESSION_REUSABLE_AFTER_FAILURE");
    const std::vector<std::string> errors={"E_BFV_WEIGHT_BLOCK_SHAPE","E_MATRIX_DESCRIPTOR","E_SHAPE","E_JOINT_DESCRIPTOR","E_TOKEN_DESCRIPTOR","E_TOKEN_WEIGHT_BOUND","E_DIAGONAL_DESCRIPTOR"};
    if(role==Role::Server && reason!=errors.at(mode))
      throw std::runtime_error("UNEXPECTED_SERVER_ERROR");
    bool rejected=false;
    try {op.private_matrix({{0}},1,12,12,{});}
    catch(const std::exception& e){rejected=std::string(e.what())=="E_SESSION_CLOSED";}
    if(!rejected)throw std::runtime_error("CLOSED_SESSION_REUSED");
  }
  std::ofstream out(name+".json");
  out<<"{\"mode\":"<<mode<<",\"error\":\""<<reason<<"\",\"session_active_after_error\":"
     <<(active?"true":"false")<<",\"network_bytes_after_entry\":"<<io.bytes()-before
     <<",\"native_TLS13_mTLS\":true,\"isolated_development\":false}\n";
  return 0;
}
int main(int argc,char**argv) {
  if(argc!=4)return 2;std::signal(SIGPIPE,SIG_IGN);
  int fds[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,fds))return 3;
  auto child=fork();if(child<0)return 3;int result=1;
  if(!child){close(fds[0]);try{result=party(Role::Server,fds[1],argv[1],argv[2],std::stoi(argv[3]));}
    catch(const std::exception&e){std::cerr<<e.what()<<'\n';}_exit(result);}
  close(fds[1]);try{result=party(Role::Client,fds[0],argv[1],argv[2],std::stoi(argv[3]));}
  catch(const std::exception&e){std::cerr<<e.what()<<'\n';}
  int status=0;waitpid(child,&status,0);return result||!WIFEXITED(status)||WEXITSTATUS(status);
}
