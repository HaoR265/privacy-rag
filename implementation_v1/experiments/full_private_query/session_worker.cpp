// Isolated development executable: the exact SessionRunner preparation/query
// pipeline, without changing the production worker's BACKEND_NOT_READY gate.
// No oracle, reconstruction of retrieval/intermediate values, shortened model,
// or production activation flag is accepted by this entry point.
#include "session.hpp"
#include "prag/heartbeat.hpp"
#include "meter.hpp"
#include <cerrno>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <set>

namespace {
using namespace prag;
using namespace prag::model;

struct OwnedFd {
  int value;
  explicit OwnedFd(int v):value(v){}
  ~OwnedFd(){if(value>=0)close(value);}
  OwnedFd(const OwnedFd&)=delete;
  void reset(){if(value>=0)close(value);value=-1;}
};
struct Buffer {
  std::vector<unsigned char> value;
  explicit Buffer(size_t size):value(size){}
  ~Buffer(){if(!value.empty())OPENSSL_cleanse(value.data(),value.size());}
};
void cleanse(Row& row){if(!row.empty())OPENSSL_cleanse(row.data(),row.size()*sizeof(Word));}
void cleanse(LocalSlots& slots){cleanse(slots.tokens);std::fill(slots.validity.begin(),slots.validity.end(),false);}
struct PreparedWiper {
  PreparedLocal& value;
  ~PreparedWiper(){cleanse(value.embedding.values);cleanse(value.eligible.values);value.client_failed.reset();}
};
struct AnswerWiper {
  ClientAnswer& value;
  ~AnswerWiper(){cleanse(value.tokens);std::fill(value.validity.begin(),value.validity.end(),false);value.status.reset();}
};
int descriptor(const char* text){
  size_t consumed=0;int value=std::stoi(text,&consumed);
  if(value<0||consumed!=std::string(text).size())throw std::invalid_argument("BAD_FORMAT");
  return value;
}
template<size_t N>std::array<unsigned char,N> hex(const std::string& text){
  if(text.size()!=2*N)throw std::invalid_argument("BAD_FORMAT");
  auto digit=[](char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::invalid_argument("BAD_FORMAT");};
  std::array<unsigned char,N> out{};
  for(size_t i=0;i<N;++i)out[i]=digit(text[2*i])*16+digit(text[2*i+1]);
  return out;
}
void read_exact(int fd,void* data,size_t n){
  auto* p=static_cast<unsigned char*>(data);
  while(n){auto count=read(fd,p,n);if(count<0&&errno==EINTR)continue;if(count<=0)throw std::runtime_error("DISCONNECTED");p+=count;n-=count;}
}
void write_exact(int fd,const void* data,size_t n){
  auto* p=static_cast<const unsigned char*>(data);
  while(n){auto count=write(fd,p,n);if(count<0&&errno==EINTR)continue;if(count<=0)throw std::runtime_error("DISCONNECTED");p+=count;n-=count;}
}
Word read_word(int fd){std::array<unsigned char,8>b{};read_exact(fd,b.data(),b.size());return get_word(b.data());}
LocalSlots read_slots(int fd,size_t count){
  LocalSlots slots{Row(count),std::vector<Bit>(count)};
  for(auto& v:slots.tokens)v=read_word(fd);
  for(size_t i=0;i<count;++i){unsigned char bit;read_exact(fd,&bit,1);if(bit>1)throw std::invalid_argument("BAD_FORMAT");slots.validity[i]=bit;}
  return slots;
}
struct Inputs {
  std::vector<BoundRecord> records;LocalSlots embedding,body;
  std::array<unsigned char,41> public_binding{};
  ~Inputs(){
    cleanse(embedding);cleanse(body);
    for(auto& record:records){cleanse(record.tokens.values);cleanse(record.validity.values);cleanse(record.embedding.values);cleanse(record.eligible.values);}
  }
};
// Same PRAGIN01 ABI and own-party payload as compute/worker_main.cpp. Each
// preparation has zero records; a nonempty query carries all N own records.
void read_input(Inputs& result,int fd,Role role,bool prepare,
    const std::array<unsigned char,16>& session,const std::array<unsigned char,32>& config,
    const std::array<unsigned char,32>& snapshot,const std::string& config_hex){
  const int seals=fcntl(fd,F_GET_SEALS),required=F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL;
  if(seals<0||(seals&required)!=required)throw std::invalid_argument("BAD_STATE");
  std::array<unsigned char,104> h{};read_exact(fd,h.data(),h.size());
  if(memcmp(h.data(),"PRAGIN01",8)||h[8]!=(prepare?1:2)||h[9]!=(role==Role::Client?0:1)||
      memcmp(h.data()+16,session.data(),16)||memcmp(h.data()+32,config.data(),32)||
      memcmp(h.data()+64,snapshot.data(),32))throw std::invalid_argument("VERSION_MISMATCH");
  for(unsigned i=10;i<16;++i)if(h[i])throw std::invalid_argument("BAD_FORMAT");
  const Word count=get_word(h.data()+96);
  struct stat st{};if(fstat(fd,&st)||st.st_size<0)throw std::runtime_error("BAD_FORMAT");
  const size_t local_bytes=role==Role::Server?0:(prepare?257:288)*9;
  if(count>(SIZE_MAX-104-local_bytes)/12296||uint64_t(st.st_size)!=104+local_bytes+count*12296||
      (prepare?count!=0:count==0))throw std::invalid_argument("BAD_SHAPE");
  std::copy(snapshot.begin(),snapshot.end(),result.public_binding.begin());
  put_word(result.public_binding.data()+32,count);result.public_binding[40]=prepare?1:2;
  if(role==Role::Client){result.embedding=read_slots(fd,prepare?257:160);if(!prepare)result.body=read_slots(fd,128);}
  result.records.reserve(count);
  auto vector=[&](size_t size){Row row(size);for(auto& word:row)word=read_word(fd);return row;};
  for(size_t i=0;i<count;++i)result.records.push_back({{vector(256),role,config_hex},
      {vector(256),role,config_hex},{vector(1024),role,config_hex},{vector(1),role,config_hex}});
  unsigned char extra;if(read(fd,&extra,1)!=0)throw std::invalid_argument("BAD_FORMAT");
}

// Explicit LIFO destruction preserves aborted nesting during exception unwind.
// Phase names and resource counters are public; no shares enter this meter.
struct PublicStages {
  PhaseMeter& meter;
  std::vector<std::pair<std::string,std::unique_ptr<PhaseMeter::Scope>>> active;
  explicit PublicStages(PhaseMeter& m):meter(m){}
  ~PublicStages(){while(!active.empty())active.pop_back();}
  void event(const char* name,bool begin){
    if(begin)active.emplace_back(name,std::make_unique<PhaseMeter::Scope>(meter,name));
    else{
      if(active.empty()||active.back().first!=name)throw std::runtime_error("PHASE_UNBALANCED");
      active.pop_back();
    }
  }
};

void run(int argc,char** argv){
  // argv 1..21 exactly match the production ABI; argv22 is a fresh trace prefix.
  if(argc!=23)throw std::invalid_argument("BAD_FORMAT");
  const std::string party=argv[1],mode=argv[2];
  if((party!="C"&&party!="S")||(mode!="prepare"&&mode!="query"))throw std::invalid_argument("BAD_FORMAT");
  const Role role=party=="C"?Role::Client:Role::Server;const bool prepare=mode=="prepare";
  const std::filesystem::path trace_prefix=argv[22];
  if(std::filesystem::path(argv[19])!=std::filesystem::path(trace_prefix.string()+".ledger"))throw std::invalid_argument("BAD_FORMAT");
  // O_EXCL reserves a new diagnostic file before PhaseMeter opens it. The
  // registered development controller must assign a fresh, private directory.
  {OwnedFd reservation(open((trace_prefix.string()+".trace.jsonl").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600));
   if(reservation.value<0)throw std::runtime_error("TRACE_EXISTS_OR_UNAVAILABLE");}
  require_compiled_numeric_profile(std::filesystem::path(argv[17]).parent_path()/"numeric_profile_v1_qk.json");
  const auto session=hex<16>(argv[8]);const auto config=hex<32>(argv[9]);const auto snapshot=hex<32>(argv[10]);hex<32>(argv[11]);
  std::set<int> fds;for(int index:{3,4,5,6,7,20,21})if(!fds.insert(descriptor(argv[index])).second)throw std::invalid_argument("BAD_FORMAT");
  OwnedFd input_fd(descriptor(argv[6])),output_fd(descriptor(argv[7])),event_fd(descriptor(argv[20])),ack_fd(descriptor(argv[21]));
  Inputs input;read_input(input,input_fd.value,role,prepare,session,config,snapshot,argv[9]);input_fd.reset();
  Channel compute(descriptor(argv[3]),session,config),out(descriptor(argv[4]),session,config),in(descriptor(argv[5]),session,config);
  const std::array<Channel*,3> order=role==Role::Client?std::array<Channel*,3>{&compute,&out,&in}:std::array<Channel*,3>{&compute,&in,&out};
  for(auto* lane:order){lane->enable_tls(role,argv[12],argv[13],argv[14],argv[15]);
    if(lane->peer_fingerprint_sha256()!=argv[11])throw std::runtime_error("NOT_AUTHORIZED");}
  std::array<unsigned char,41> peer{};compute.send_data(input.public_binding.data(),input.public_binding.size());compute.recv_data(peer.data(),peer.size());
  if(peer!=input.public_binding)throw std::invalid_argument("VERSION_MISMATCH");
  unsigned char last_event=0;
  auto lifecycle=[&](unsigned char event){
    if(event!=last_event+1||event>2)throw std::runtime_error("BAD_STATE");
    write_exact(event_fd.value,&event,1);unsigned char ack=0;read_exact(ack_fd.value,&ack,1);
    if(ack!=event)throw std::runtime_error("BAD_STATE");last_event=event;
  };
  SecureOps ops(role,compute,argv[19],false);
  Heartbeat heartbeat(out,in,compute,[&ops]{ops.abort_session();});
  try{
    lifecycle(1);
    const auto started=std::chrono::steady_clock::now();
    PhaseMeter meter(compute,trace_prefix.string());
    if(!meter.trace||!meter.ledger)throw std::runtime_error("TRACE_UNAVAILABLE");
    PublicStages stages(meter);
    SessionRunner runner(ops,argv[16],argv[17],[&](const char* name,bool begin){
      if(!std::uncaught_exceptions())heartbeat.check();stages.event(name,begin);
    });
    auto before_release=[&]{heartbeat.check();lifecycle(2);};
    if(prepare){
      auto result=runner.prepare(role==Role::Client?&input.embedding:nullptr,before_release);
      PreparedWiper wipe{result};
      heartbeat.check();Buffer encoded(8200+(role==Role::Client?1:0));
      for(size_t i=0;i<1024;++i)put_word(encoded.value.data()+8*i,result.embedding.values[i]);
      put_word(encoded.value.data()+8192,result.eligible.values[0]);
      if(role==Role::Client)encoded.value[8200]=result.client_failed.value();
      // Own preparation output is delivered only to this party's storage path.
      write_exact(output_fd.value,encoded.value.data(),encoded.value.size());
    }else{
      auto result=runner.query(role==Role::Client?&input.embedding:nullptr,
          role==Role::Client?&input.body:nullptr,input.records,load_prompt_layout(argv[18]),before_release);
      AnswerWiper wipe{result};
      heartbeat.check();
      if(role==Role::Client){
        Buffer encoded(2312);
        for(size_t i=0;i<256;++i){put_word(encoded.value.data()+8*i,result.tokens[i]);encoded.value[2048+i]=result.validity[i];}
        put_word(encoded.value.data()+2304,result.status.value());
        write_exact(output_fd.value,encoded.value.data(),encoded.value.size());
      }else if(!result.tokens.empty()||!result.validity.empty()||result.status.has_value())throw std::runtime_error("OUTPUT_BOUNDARY");
    }
    if(!stages.active.empty()||last_event!=2)throw std::runtime_error("PHASE_UNBALANCED");
    heartbeat.check();const auto stats=runner.gpu_stats();
    heartbeat.finish(); // Explicit peer completion; never treat HB EOF as success.
    const auto bytes=compute.bytes();
    const auto heartbeats=heartbeat.received();ops.close_session();
    output_fd.reset();event_fd.reset();ack_fd.reset();
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cout<<"{\"result\":\"completed_full_private_session\",\"mode\":\""<<mode
      <<"\",\"application_bytes\":"<<bytes<<",\"seconds\":"<<seconds
      <<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"gpu_live_owned_bytes\":"<<stats.live_bytes
      <<",\"gpu_peak_owned_bytes\":"<<stats.peak_bytes<<",\"gpu_kernels\":"<<stats.kernels
      <<",\"gpu_upload_bytes\":"<<stats.uploaded_bytes<<",\"gpu_download_bytes\":"<<stats.downloaded_bytes
      <<",\"heartbeats_received\":"<<heartbeats<<"}\n";
  }catch(...){ops.abort_session();heartbeat.close();throw;}
}
}
int main(int argc,char** argv){
  std::signal(SIGPIPE,SIG_IGN);rlimit no_core{0,0};setrlimit(RLIMIT_CORE,&no_core);prctl(PR_SET_DUMPABLE,0);
  try{run(argc,argv);return 0;}catch(const std::exception&){std::cerr<<"COMPUTE_ABORTED\n";return 2;}
}
