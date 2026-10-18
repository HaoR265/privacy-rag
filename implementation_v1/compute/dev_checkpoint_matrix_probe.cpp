// One real registered weight map or gamma diagonal; no Transformer is run.
// Reconstruction is confined to this diagnostic and is never linked to worker.
#include "artifacts.hpp"
#include <chrono>
#include <csignal>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <iostream>
using namespace prag;
using namespace prag::model;
static int run(Role role,int fd,const std::string& prefix,const std::string& cert,
               const std::string& weights,const std::string& reference,bool diagonal,
               const std::string& digest,Word row_l1) {
  const size_t dimension=diagonal?128:2048;
  std::unique_ptr<MappedWeights> matrix;
  if(role==Role::Server){
    matrix=std::make_unique<MappedWeights>(weights,dimension,diagonal?1:dimension,diagonal);
    // Same real mapped-byte and per-row bound checker used by worker.
    matrix->require_row_bound(digest.c_str(),row_l1,diagonal?Word(192<<12):numeric_profile::main_abs_f12);
  }
  std::array<unsigned char,16> session{};session[0]=107;session[1]=11+diagonal;
  std::array<unsigned char,32> config{};config[0]=107;
  Channel io(fd,session,config);
  const std::string own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  Row x(dimension);
  for(size_t j=0;j<dimension;++j){
    Word whole=Word((int((13*j+7)%255)-127)*int64_t(4096))&MASK;
    Word c=(UINT64_C(0x21fedcba987654)+j*125939)&MASK;
    x[j]=role==Role::Client?c:(whole-c)&MASK;
  }
  PrivateWeightBlock reader=[&](size_t begin,size_t actual,size_t first,size_t count){
    if(!matrix||role!=Role::Server)throw std::runtime_error("C_READ_PRIVATE_WEIGHT");
    Rows block(count,Row(actual,0));
    for(size_t o=0;o<count;++o)for(size_t j=0;j<actual;++j)
      if(!diagonal||first+o==begin+j)block[o][j]=matrix->at(first+o,diagonal?0:begin+j);
    return block;
  };
  auto start=std::chrono::steady_clock::now();auto before=io.bytes();
  auto y=op.private_matrix({x},dimension,12,12,reader).at(0);
  auto bytes=io.bytes()-before;
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  io.set_operation(1000000000);
  if(role==Role::Server)io.send_data(y.data(),int(y.size()*8));
  else{
    Row other(dimension),wanted(dimension);
    io.recv_data(other.data(),int(other.size()*8));
    std::ifstream f(reference,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(wanted.data()),wanted.size()*8)||f.peek()!=EOF)
      throw std::runtime_error("REFERENCE_SHAPE");
    for(size_t j=0;j<dimension;++j)if(((other[j]+y[j])&MASK)!=wanted[j])throw std::runtime_error("MATRIX_MISMATCH");
  }
  op.close_session();
  if(role==Role::Client)std::cout<<"{\"result\":\"matched_CUDA_R1\",\"application_bytes\":"<<bytes
    <<",\"elapsed_seconds\":"<<seconds<<",\"dimension\":"<<dimension
    <<",\"diagonal_gamma\":"<<(diagonal?"true":"false")
    <<",\"raw_fractional_bits\":24,\"native_TLS13_mTLS\":true,\"isolated_development\":false,\"model_executed\":false}\n";
  return 0;
}
int main(int argc,char**argv){
  if(argc!=8)return 2;std::signal(SIGPIPE,SIG_IGN);
  int fds[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,fds))return 3;
  auto child=fork();if(child<0)return 3;int result=1;
  auto call=[&](Role role,int fd){return run(role,fd,argv[1],argv[2],argv[3],argv[4],std::stoi(argv[5]),argv[6],std::stoull(argv[7]));};
  if(!child){close(fds[0]);try{result=call(Role::Server,fds[1]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}_exit(result);}
  close(fds[1]);try{result=call(Role::Client,fds[0]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}
  int status=0;waitpid(child,&status,0);return result||!WIFEXITED(status)||WEXITSTATUS(status);
}
