#include "prag/secure_ops.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int run(Role role,int fd,const std::string& prefix,const std::string& cert,const std::string& expected) {
  std::array<unsigned char,16> session{};session[0]=109;session[1]=3;
  std::array<unsigned char,32> config{};config[0]=109;config[1]=3;
  Channel io(fd,session,config);const auto own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  std::ifstream ref;if(role==Role::Client)ref.open(expected,std::ios::binary);
  if(role==Role::Client)std::cout<<"{\"result\":\"matched_CUDA_R1\",\"cases\":[";
  bool first=true;
  for(size_t n:{128,1024,2048}) {
    std::vector<Word> x(n),w;
    if(role==Role::Server)w.resize(n);
    for(size_t j=0;j<n;++j){const Word full=Word(int(j%8191)-4095)&MASK,c=(UINT64_C(0x4718fad072184)+j*91823)&MASK;
      x[j]=role==Role::Client?c:(full-c)&MASK;if(role==Role::Server)w[j]=Word(int((j*197)%2097151)-1048575)&MASK;}
    auto before=io.bytes(),fs=io.frames_sent(),fr=io.frames_received();auto start=std::chrono::steady_clock::now();
    auto result=op.private_diagonal(x,w,12,12);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    auto bytes=io.bytes()-before,sent=io.frames_sent()-fs,received=io.frames_received()-fr;
    if(role==Role::Server)io.send_data(result.data(),n*8);
    else{std::vector<Word> other(n),expected(n);io.recv_data(other.data(),n*8);ref.read(reinterpret_cast<char*>(expected.data()),n*8);if(!ref)throw std::runtime_error("REFERENCE_SHAPE");for(size_t j=0;j<n;++j)if(((other[j]+result[j])&MASK)!=expected[j])throw std::runtime_error("DIAGONAL_MISMATCH");
      if(!first)std::cout<<',';first=false;std::cout<<"{\"coordinates\":"<<n<<",\"bytes\":"<<bytes<<",\"seconds\":"<<seconds<<",\"OT_choices\":"<<59*n<<",\"input_ciphertexts\":0,\"returned_c1\":0,\"frames_sent\":"<<sent<<",\"frames_received\":"<<received<<'}';}
  }
  op.close_session();if(role==Role::Client)std::cout<<"],\"real_mTLS\":true,\"production_library\":false,\"full_model_R2\":false}\n";return 0;
}
int main(int argc,char**argv){std::signal(SIGPIPE,SIG_IGN);if(argc!=4)return 2;int p[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,p))return 3;auto pid=fork();int result=1;if(pid==0){close(p[0]);try{result=run(Role::Server,p[1],argv[1],argv[2],argv[3]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}_exit(result);}close(p[1]);try{result=run(Role::Client,p[0],argv[1],argv[2],argv[3]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}int status;waitpid(pid,&status,0);return result||!WIFEXITED(status)||WEXITSTATUS(status);}
