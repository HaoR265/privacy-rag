// Public synthetic matrix diagnostic. Reconstruction ONLY here, never library.
#include "prag/secure_ops.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static Word whole(size_t b,size_t j,int mode) {
  if(mode==1)return j==0 ? ((Word(1)<<55)-1) : (j==16?Word(-3)&MASK:0);
  return Word(int((j*13+b*19+7)%255)-127)&MASK;
}
static Word weight(size_t o,size_t j,int mode) {
  if(o==1)return 0;
  if(mode==1)return j==0 ? (o%2?Word(-1)&MASK:1) : (j==16?2:0);
  return Word(int((j*7+o*11+3)%63)-31)&MASK;
}
static int run(Role role,int fd,const std::string& prefix,const std::string& cert,
               size_t batch,size_t inner,size_t outputs,int mode,const std::string& reference) {
  std::array<unsigned char,16> session{};session[0]=108;
  std::array<unsigned char,32> config{};config[0]=108;
  Channel io(fd,session,config);
  const auto own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  std::vector<std::vector<Word>> x(batch,std::vector<Word>(inner));
  for(size_t b=0;b<batch;++b)for(size_t j=0;j<inner;++j) {
    const Word c=(UINT64_C(0x21fedcba987654)+b*797+j*125939)&MASK;
    x[b][j]=role==Role::Client?c:(whole(b,j,mode)-c)&MASK;
  }
  size_t callbacks=0;
  PrivateWeightBlock reader=[&](size_t begin,size_t actual,size_t first,size_t count) {
    if(role!=Role::Server)throw std::runtime_error("C_READ_PRIVATE_WEIGHT");
    ++callbacks;
    std::vector<std::vector<Word>> w(count,std::vector<Word>(actual));
    for(size_t o=0;o<count;++o)for(size_t j=0;j<actual;++j)w[o][j]=weight(first+o,begin+j,mode);
    return w;
  };
  const auto before=io.bytes(),fs=io.frames_sent(),fr=io.frames_received(),runs=io.direction_runs();
  const auto start=std::chrono::steady_clock::now();
  auto y=op.private_matrix(x,outputs,12,12,reader);
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  const auto bytes=io.bytes()-before, sent=io.frames_sent()-fs,received=io.frames_received()-fr,turns=io.direction_runs()-runs;
  rusage usage{};getrusage(RUSAGE_SELF,&usage);
  const uint64_t own_peak=usage.ru_maxrss*uint64_t(1024);
  // End measurement BEFORE separate public diagnostic reconstruction.
  io.set_operation(1000000000);
  uint64_t server_peak=0;
  if(role==Role::Server) {
    io.send_data(&own_peak,sizeof(own_peak));
    for(const auto& row:y)io.send_data(row.data(),int(row.size()*8));
  } else {
    io.recv_data(&server_peak,sizeof(server_peak));
    std::ifstream expected(reference,std::ios::binary);
    if(!expected)throw std::runtime_error("REFERENCE_MISSING");
    std::vector<Word> other(outputs),wanted(outputs);
    for(size_t b=0;b<batch;++b) {
      io.recv_data(other.data(),int(outputs*8));
      expected.read(reinterpret_cast<char*>(wanted.data()),outputs*8);
      if(!expected)throw std::runtime_error("REFERENCE_SHAPE");
      for(size_t o=0;o<outputs;++o)
        if(((y[b][o]+other[o])&MASK)!=wanted[o])throw std::runtime_error("MATRIX_MISMATCH");
    }
    if(expected.peek()!=EOF)throw std::runtime_error("REFERENCE_SHAPE");
  }
  op.close_session();
  if(role==Role::Client) {
    const size_t tile=512/batch,cts=(outputs+tile-1)/tile;
    std::cout<<"{\"native_version\":\"" PRAG_NATIVE_VERSION "\",\"shape_B_K_O\":["<<batch<<','<<inner<<','<<outputs
      <<"],\"result\":\"matched_CUDA_R1\",\"raw_fractional_bits\":24,\"input_ciphertexts\":"<<(inner+15)/16
      <<",\"returned_ciphertext_groups\":"<<cts<<",\"output_coordinates\":"<<batch*outputs
      <<",\"application_bytes\":"<<bytes<<",\"frames_sent\":"<<sent<<",\"frames_received\":"<<received
      <<",\"transcript_direction_runs\":"<<turns<<",\"direction_runs_are_not_RTT\":true,\"elapsed_seconds\":"<<seconds
      <<",\"client_process_peak_RSS_bytes\":"<<own_peak<<",\"server_process_peak_RSS_bytes\":"<<server_peak
      <<",\"native_TLS13_mTLS\":true,\"isolated_development\":false,\"protocol_GPU_used\":false,\"full_model_R2\":false}\n";
  }
  return 0;
}
int main(int argc,char**argv) {
  std::signal(SIGPIPE,SIG_IGN);
  if(argc!=8)return 2;
  const size_t batch=std::stoul(argv[3]),inner=std::stoul(argv[4]),outputs=std::stoul(argv[5]);
  const int mode=std::stoi(argv[6]);
  int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets))return 3;
  const auto child=fork();if(child<0)return 3;int result=1;
  if(child==0) {
    close(sockets[0]);try{result=run(Role::Server,sockets[1],argv[1],argv[2],batch,inner,outputs,mode,argv[7]);}
    catch(const std::exception&e){std::cerr<<e.what()<<'\n';}_exit(result);
  }
  close(sockets[1]);try{result=run(Role::Client,sockets[0],argv[1],argv[2],batch,inner,outputs,mode,argv[7]);}
  catch(const std::exception&e){std::cerr<<e.what()<<'\n';}
  int status=0;waitpid(child,&status,0);return result||!WIFEXITED(status)||WEXITSTATUS(status);
}
