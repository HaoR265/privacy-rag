#include "token_candidate.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int run(Role role,int fd,const std::string& prefix,const std::string& cert,
    size_t hidden,size_t id,const std::string& expected) {
  std::array<unsigned char,16> session{};session[0]=109;session[1]=2;
  std::array<unsigned char,32> config{};config[0]=109;config[1]=2;
  Channel io(fd,session,config);const auto own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  const size_t vocabulary=hidden==1024?151669:151936;size_t calls=0,visits=0;
  PrivateWeightBlock reader=[&](size_t begin,size_t count,size_t first,size_t rows){
    if(role!=Role::Server)throw std::runtime_error("C_WEIGHT_ACCESS");++calls;visits+=count*rows;
    std::vector<std::vector<Word>> w(rows,std::vector<Word>(count));
    for(size_t r=0;r<rows;++r)for(size_t j=0;j<count;++j)w[r][j]=begin+j<vocabulary?Word(int(((begin+j)*71+(first+r)*197)%65535)-32767)&MASK:0;
    return w;
  };
  const Word share=UINT64_C(0x47abc1243765123)&MASK,index=role==Role::Client?share:(id-share)&MASK;
  const auto before=io.bytes(),fs=io.frames_sent(),fr=io.frames_received(),dr=io.direction_runs();
  const auto start=std::chrono::steady_clock::now();TokenCandidateStats stats;
  auto result=token_candidate(op,index,vocabulary,hidden,reader,stats);
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  const auto bytes=io.bytes()-before,sent=io.frames_sent()-fs,received=io.frames_received()-fr,dirs=io.direction_runs()-dr;
  rusage usage{};getrusage(RUSAGE_SELF,&usage);uint64_t memory=usage.ru_maxrss*uint64_t(1024),server_memory=0;
  io.set_operation(1000000000);
  if(role==Role::Server){io.send_data(&memory,8);io.send_data(result.data(),hidden*8);}
  else{io.recv_data(&server_memory,8);std::vector<Word> other(hidden),want(hidden);io.recv_data(other.data(),hidden*8);std::ifstream f(expected,std::ios::binary);f.read(reinterpret_cast<char*>(want.data()),hidden*8);if(!f||f.peek()!=EOF)throw std::runtime_error("REFERENCE_SHAPE");for(size_t j=0;j<hidden;++j)if(((other[j]+result[j])&MASK)!=want[j])throw std::runtime_error("TOKEN_MISMATCH");}
  op.close_session();
  if(role==Role::Client)std::cout<<"{\"native_version\":\"1.0.9-candidate\",\"result\":\"matched_CUDA_R1\",\"hidden\":"<<hidden<<",\"public_diagnostic_id\":"<<id<<",\"input_ciphertexts\":19,\"returned_c1\":"<<hidden/512<<",\"GC_tiles\":"<<hidden/512<<",\"GC_ANDs\":"<<358*hidden<<",\"GC_OT_choices\":"<<61*hidden<<",\"switch_key_ciphertexts\":108,\"application_bytes\":"<<bytes<<",\"onehot_bytes\":"<<stats.onehot_bytes<<",\"onehot_seconds\":"<<stats.onehot_seconds<<",\"initial_key_bytes\":"<<stats.initial_key_bytes<<",\"switch_key_bytes\":"<<stats.switch_key_bytes<<",\"online_bfv_gc_bytes\":"<<stats.online_bfv_gc_bytes<<",\"elapsed_seconds\":"<<elapsed<<",\"frames_sent\":"<<sent<<",\"frames_received\":"<<received<<",\"direction_runs\":"<<dirs<<",\"client_peak_RSS\":"<<memory<<",\"server_peak_RSS\":"<<server_memory<<",\"TLS13_mTLS\":true,\"full_model_R2\":false,\"diagnostic_reconstruction_excluded\":true}\n";
  return 0;
}
int main(int argc,char**argv){std::signal(SIGPIPE,SIG_IGN);if(argc!=6)return 2;int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))return 3;auto pid=fork();int result=1;if(pid==0){close(pair[0]);try{result=run(Role::Server,pair[1],argv[1],argv[2],std::stoul(argv[3]),std::stoul(argv[4]),argv[5]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}_exit(result);}close(pair[1]);try{result=run(Role::Client,pair[0],argv[1],argv[2],std::stoul(argv[3]),std::stoul(argv[4]),argv[5]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}int status;waitpid(pid,&status,0);return result||!WIFEXITED(status)||WEXITSTATUS(status);}
