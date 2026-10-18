#include "session.hpp"
#include "prag/heartbeat.hpp"
#include <csignal>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unistd.h>
#include <iostream>

namespace prag::model {
bool full_model_configuration_enabled() noexcept;
bool private_model_gpu_backend_available() noexcept;
}
namespace {
using namespace prag;
using namespace prag::model;
// Keep serialized output and returned local shares covered on every exit,
// including a broken local output pipe or a failed completion handshake.
struct OutputBytesGuard {
  void* data;std::size_t bytes;
  OutputBytesGuard(void* p,std::size_t n):data(p),bytes(n){}
  OutputBytesGuard(const OutputBytesGuard&)=delete;
  ~OutputBytesGuard(){if(bytes)OPENSSL_cleanse(data,bytes);}
};
void clear_words(Row& words){if(!words.empty())OPENSSL_cleanse(words.data(),words.size()*sizeof(Word));}
template<class T>void clear_optional(std::optional<T>& value){
  if(value){OPENSSL_cleanse(&*value,sizeof(T));value.reset();}
}
struct PreparedOutputGuard {
  PreparedLocal& value;
  ~PreparedOutputGuard(){clear_words(value.embedding.values);clear_words(value.eligible.values);clear_optional(value.client_failed);}
};
struct AnswerOutputGuard {
  ClientAnswer& value;
  ~AnswerOutputGuard(){
    clear_words(value.tokens);clear_optional(value.status);
    // vector<bool> exposes no contiguous writable storage. Clear its logical
    // contents without claiming secure erasure of its implementation storage.
    std::fill(value.validity.begin(),value.validity.end(),false);
  }
};
// NUM-004: changing this requires the corresponding R1/R2/range version and a
// complete scheduler integration. No CLI/environment escape hatch exists.
template<std::size_t N>std::array<unsigned char,N> hex(const std::string& text){
  if(text.size()!=N*2)throw std::invalid_argument("BAD_FORMAT");
  std::array<unsigned char,N> out{};
  auto digit=[](char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::invalid_argument("BAD_FORMAT");};
  for(std::size_t i=0;i<N;++i)out[i]=digit(text[2*i])*16+digit(text[2*i+1]);return out;
}
void read_exact(int fd,void* data,std::size_t n){auto* p=static_cast<unsigned char*>(data);while(n){auto z=read(fd,p,n);if(z<=0)throw std::runtime_error("DISCONNECTED");p+=z;n-=z;}}
void write_exact(int fd,const void* data,std::size_t n){auto* p=static_cast<const unsigned char*>(data);while(n){auto z=write(fd,p,n);if(z<=0)throw std::runtime_error("DISCONNECTED");p+=z;n-=z;}}
Word read_word(int fd){unsigned char b[8];read_exact(fd,b,8);return get_word(b);}
LocalSlots read_slots(int fd,std::size_t count){
  LocalSlots result{Row(count),std::vector<Bit>(count)};
  for(auto& v:result.tokens)v=read_word(fd);
  for(std::size_t i=0;i<count;++i){unsigned char v;read_exact(fd,&v,1);if(v>1)throw std::invalid_argument("BAD_FORMAT");result.validity[i]=v;}
  return result;
}
struct Inputs {std::vector<BoundRecord> records;LocalSlots embedding,body;std::array<unsigned char,41> public_binding;};
Inputs read_input(int fd,Role role,bool prepare,const std::array<unsigned char,16>& session,const std::array<unsigned char,32>& config,const std::array<unsigned char,32>& snapshot,const std::string& config_hex){
  // Input must be a sealed memory file from this party's scheduler, never a
  // potentially growing pathname or a file containing both parties' inputs.
  int seals=fcntl(fd,F_GET_SEALS);int expected=F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL;
  if(seals<0||(seals&expected)!=expected)throw std::invalid_argument("BAD_STATE");
  std::array<unsigned char,104> h{};read_exact(fd,h.data(),h.size());
  if(memcmp(h.data(),"PRAGIN01",8)||h[8]!=(prepare?1:2)||h[9]!=(role==Role::Client?0:1)||
    memcmp(h.data()+16,session.data(),16)||memcmp(h.data()+32,config.data(),32)||memcmp(h.data()+64,snapshot.data(),32))throw std::invalid_argument("VERSION_MISMATCH");
  for(unsigned i=10;i<16;++i)if(h[i])throw std::invalid_argument("BAD_FORMAT");
  Word count=get_word(h.data()+96);
  struct stat st{};if(fstat(fd,&st))throw std::runtime_error("BAD_FORMAT");
  std::size_t input_bytes=role==Role::Server?0:(prepare?257:288)*9;
  if(count>(SIZE_MAX-104-input_bytes)/12296||std::uint64_t(st.st_size)!=104+input_bytes+count*12296||(prepare?count!=0:count==0))throw std::invalid_argument("BAD_SHAPE");
  Inputs result;std::copy(snapshot.begin(),snapshot.end(),result.public_binding.begin());put_word(result.public_binding.data()+32,count);result.public_binding[40]=prepare?1:2;
  if(role==Role::Client){result.embedding=read_slots(fd,prepare?257:160);if(!prepare)result.body=read_slots(fd,128);}
  result.records.reserve(count);
  auto vector=[&](std::size_t size){Row x(size);for(auto& v:x)v=read_word(fd);return x;};
  for(std::size_t i=0;i<count;++i)result.records.push_back({{vector(256),role,config_hex},{vector(256),role,config_hex},{vector(1024),role,config_hex},{vector(1),role,config_hex}});
  return result;
}
void run(int argc,char** argv){
  if(argc!=22)throw std::invalid_argument("BAD_FORMAT");
  Role role=std::string(argv[1])=="C"?Role::Client:Role::Server;
  if(std::string(argv[1])!="C"&&std::string(argv[1])!="S")throw std::invalid_argument("BAD_FORMAT");
  bool prepare=std::string(argv[2])=="prepare";
  if(!prepare&&std::string(argv[2])!="query")throw std::invalid_argument("BAD_FORMAT");
  if(!private_model_gpu_backend_available()||!full_model_configuration_enabled())
    throw std::runtime_error("BACKEND_NOT_READY");
  require_compiled_numeric_profile(std::filesystem::path(argv[17]).parent_path()/"numeric_profile_v1_qk.json");
  auto session=hex<16>(argv[8]);auto config=hex<32>(argv[9]);auto snapshot=hex<32>(argv[10]);hex<32>(argv[11]);
  int input_fd=std::stoi(argv[6]),output_fd=std::stoi(argv[7]);
  auto input=read_input(input_fd,role,prepare,session,config,snapshot,argv[9]);close(input_fd);
  Channel compute(std::stoi(argv[3]),session,config),out(std::stoi(argv[4]),session,config),in(std::stoi(argv[5]),session,config);
  // Physical stream order is identical across peers. C.out pairs with S.in.
  std::array<Channel*,3> handshake_order=role==Role::Client?std::array<Channel*,3>{&compute,&out,&in}:std::array<Channel*,3>{&compute,&in,&out};
  for(auto* c:handshake_order){c->enable_tls(role,argv[12],argv[13],argv[14],argv[15]);if(c->peer_fingerprint_sha256()!=argv[11])throw std::runtime_error("NOT_AUTHORIZED");}
  std::array<unsigned char,41> peer{};
  compute.send_data(input.public_binding.data(),input.public_binding.size());compute.recv_data(peer.data(),peer.size());
  if(peer!=input.public_binding)throw std::invalid_argument("VERSION_MISMATCH");
  int event_fd=std::stoi(argv[20]),ack_fd=std::stoi(argv[21]);
  auto lifecycle=[&](unsigned char event){
    write_exact(event_fd,&event,1);unsigned char ack=0;read_exact(ack_fd,&ack,1);
    if(ack!=event)throw std::runtime_error("BAD_STATE");
  };
  // Locked SCI constructors defer base OT until first send/receive, so this
  // guard covers all network-dependent setup as well as model operations.
  SecureOps ops(role,compute,argv[19]);
  Heartbeat heartbeat(out,in,compute,[&ops]{ops.abort_session();});
  try{
    lifecycle(1); // same-party scheduler commits computing before first OT
    auto before_release=[&]{heartbeat.check();lifecycle(2);};
    SessionRunner runner(ops,argv[16],argv[17]);
    if(prepare){
      auto result=runner.prepare(role==Role::Client?&input.embedding:nullptr,before_release);
      PreparedOutputGuard result_guard{result};heartbeat.check();
      std::vector<unsigned char> encoded(8*(1024+1)+(role==Role::Client?1:0));
      OutputBytesGuard encoded_guard(encoded.data(),encoded.size());
      for(std::size_t i=0;i<1024;++i)put_word(encoded.data()+8*i,result.embedding.values[i]);put_word(encoded.data()+8192,result.eligible.values[0]);
      if(role==Role::Client)encoded[8200]=result.client_failed.value();
      write_exact(output_fd,encoded.data(),encoded.size());
    }else{
      auto result=runner.query(role==Role::Client?&input.embedding:nullptr,role==Role::Client?&input.body:nullptr,input.records,load_prompt_layout(argv[18]),before_release);
      AnswerOutputGuard result_guard{result};heartbeat.check();
      if(role==Role::Client){
        std::array<unsigned char,2312> encoded{};
        OutputBytesGuard encoded_guard(encoded.data(),encoded.size());
        for(std::size_t i=0;i<256;++i){put_word(encoded.data()+8*i,result.tokens[i]);encoded[2048+i]=result.validity[i];}
        put_word(encoded.data()+2304,result.status.value());write_exact(output_fd,encoded.data(),encoded.size());
      }
    }
    // Both fixed outputs have been written to the respective local pipes.
    // finish() authenticates peer completion before either heartbeat stream
    // closes; EOF alone must never turn an unfinished peer into success.
    heartbeat.finish();
    ops.close_session();close(output_fd);close(event_fd);close(ack_fd);
  }catch(...){ops.abort_session();heartbeat.close();throw;}
}
}
int main(int argc,char** argv){
  // The scheduler masks cancellation while creating/registering this child.
  // exec preserves that mask; restore cancellation before any gate or input.
  sigset_t cancellation;
  if(sigemptyset(&cancellation)||sigaddset(&cancellation,SIGINT)||sigaddset(&cancellation,SIGTERM)||sigprocmask(SIG_UNBLOCK,&cancellation,nullptr)){
    std::cerr<<"COMPUTE_ABORTED\n";return 2;
  }
  std::signal(SIGPIPE,SIG_IGN);struct rlimit no_core{0,0};setrlimit(RLIMIT_CORE,&no_core);prctl(PR_SET_DUMPABLE,0);
  try{run(argc,argv);return 0;}catch(const std::exception& e){
    // Fixed public codes only; never stream exception text or secret positions.
    if(std::string(e.what())=="BACKEND_NOT_READY")std::cerr<<"BACKEND_NOT_READY\n";
    else std::cerr<<"COMPUTE_ABORTED\n";
    return 2;
  }
}
