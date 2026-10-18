// Isolated final-opening diagnostic; never linked into a production worker.
#include "prag/batch.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static void read_exact(std::ifstream& in, void* dst, size_t bytes) {
  if(!in.read(static_cast<char*>(dst), bytes)) throw std::runtime_error("FIXTURE");
}
static int run(Role role, int fd, const std::string& prefix, const std::string& cert,
               const std::string& fixture, const std::string& session) {
  std::array<unsigned char,16> sid{};
  if(session.size()!=32)throw std::runtime_error("SESSION");
  for(size_t i=0;i<16;++i)sid[i]=std::stoul(session.substr(2*i,2),nullptr,16);
  std::array<unsigned char,32> cfg{};cfg[0]=112;cfg[1]=14;
  Channel io(fd,sid,cfg);auto own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  BatchOps batch(op);
  try {
    std::ifstream in(fixture,std::ios::binary);Word count;read_exact(in,&count,8);
    if(count!=8)throw std::runtime_error("FIXTURE_SHAPE");
    std::vector<Word> index(count),table(1280*128),wanted(count*128);
    std::vector<uint8_t> expected_flags(count);
    read_exact(in,index.data(),count*8);read_exact(in,table.data(),table.size()*8);
    read_exact(in,wanted.data(),wanted.size()*8);read_exact(in,expected_flags.data(),count);
    if(in.peek()!=EOF)throw std::runtime_error("FIXTURE_LENGTH");
    for(size_t j=0;j<count;++j){Word share=(0x438547a0000000ULL+151639*j)&MASK;index[j]=role==Role::Client?share:(index[j]-share)&MASK;}
    auto started=std::chrono::steady_clock::now();auto before=io.bytes();
    auto result=batch.lookup_rows(index,table,128);
    if(result.words.size()!=wanted.size()||result.bits.size()!=count)throw std::runtime_error("OUTPUT_SHAPE");
    auto bytes=io.bytes()-before;
    auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    op.close_session();
    // Diagnostic opening is explicitly after the cryptographic session closes.
    io.set_operation(1000000000);
    if(role==Role::Server){io.send_data(result.words.data(),result.words.size()*8);io.send_data(result.bits.data(),count);}
    else {
      std::vector<Word> peer(wanted.size());std::vector<uint8_t> peer_bits(count);
      io.recv_data(peer.data(),peer.size()*8);io.recv_data(peer_bits.data(),count);
      for(size_t j=0;j<wanted.size();++j)if(((peer[j]+result.words[j])&MASK)!=wanted[j])throw std::runtime_error("ORACLE_WORD_MISMATCH");
      for(size_t j=0;j<count;++j)if((peer_bits[j]^result.bits[j])!=expected_flags[j])throw std::runtime_error("ORACLE_FLAG_MISMATCH");
      std::cout<<"{\"result\":\"matched_CUDA_oracle\",\"rows\":8,\"columns\":128,\"output_words\":1024,\"flags\":8,\"seconds\":"<<seconds<<",\"application_bytes\":"<<bytes<<",\"model_executed\":false,\"production_enabled\":false}\n";
    }
    return 0;
  } catch(...) {op.abort_session();throw;}
}
int main(int argc,char**argv){
  if(argc!=5)return 2;std::signal(SIGPIPE,SIG_IGN);
  int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))return 3;
  auto child=fork();if(child<0)return 3;
  int status=1;
  if(!child){close(pair[0]);try{status=run(Role::Server,pair[1],argv[1],argv[2],argv[3],argv[4]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}close(pair[1]);_exit(status);}
  close(pair[1]);try{status=run(Role::Client,pair[0],argv[1],argv[2],argv[3],argv[4]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}close(pair[0]);
  int other=0;waitpid(child,&other,0);return status||!WIFEXITED(other)||WEXITSTATUS(other);
}
