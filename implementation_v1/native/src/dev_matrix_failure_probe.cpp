// Isolated production-mTLS failure diagnostics. No values are reconstructed.
#include "prag/secure_ops.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int party(Role role,int fd,const std::string& prefix,const std::string& cert,int mode) {
  std::array<unsigned char,16> sid{};sid[0]=107;sid[1]=mode;
  std::array<unsigned char,32> config{};config[0]=107;
  Channel io(fd,sid,config);
  const std::string own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  const std::string name=prefix+(role==Role::Client?".C":".S");
  SecureOps op(role,io,name+".ledger",false);
  const auto before=io.bytes();
  std::vector<std::vector<Word>> x(mode==2?17:1,std::vector<Word>(17,0));
  std::string reason;
  try {
    op.private_matrix(x,mode==1&&role==Role::Server?2:1,12,12,
      [&](size_t,size_t,size_t,size_t)->std::vector<std::vector<Word>> {
        if(role!=Role::Server)throw std::runtime_error("C_READ_PRIVATE_WEIGHT");
        return {}; // Mode0: deliberately wrong private callback shape.
      });
    throw std::runtime_error("FAILURE_NOT_DETECTED");
  } catch(const std::exception& e) {reason=e.what();}
  const bool active=op.session_active();
  if(mode==2) {
    if(reason!="E_SHAPE"||!active||io.bytes()!=before)throw std::runtime_error("PUBLIC_SHAPE_GATE_FAILED");
    op.close_session();
  } else {
    if(active)throw std::runtime_error("SESSION_REUSABLE_AFTER_FAILURE");
    if(role==Role::Server && reason!=(mode==0?"E_BFV_WEIGHT_BLOCK_SHAPE":"E_MATRIX_DESCRIPTOR"))
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
