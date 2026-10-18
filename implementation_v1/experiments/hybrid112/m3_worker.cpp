// Isolated controller-facing diagnostic. Links the production protocol/model
// library, never provides a reconstruction API to either participant.
#include "session.hpp"
#include "prag/heartbeat.hpp"
#include "meter.hpp"
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <sys/prctl.h>
#include <sys/resource.h>
using namespace prag;using namespace prag::model;
template<size_t N>std::array<unsigned char,N> unhex(const std::string&s){if(s.size()!=2*N)throw std::runtime_error("FORMAT");std::array<unsigned char,N>a{};for(size_t i=0;i<N;++i){auto piece=s.substr(2*i,2);if(piece.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("FORMAT");a[i]=std::stoul(piece,nullptr,16);}return a;}
void exact(int fd,void*p,size_t n,bool output=false){auto*b=static_cast<unsigned char*>(p);while(n){auto r=output?write(fd,b,n):read(fd,b,n);if(r<=0)throw std::runtime_error("DISCONNECTED");b+=r;n-=r;}}
int main(int argc,char**argv){
 std::signal(SIGPIPE,SIG_IGN);rlimit no_core{0,0};setrlimit(RLIMIT_CORE,&no_core);prctl(PR_SET_DUMPABLE,0);
 try{
  if(argc!=19)throw std::runtime_error("ARGUMENTS");std::string role_text=argv[1];if(role_text!="C"&&role_text!="S")throw std::runtime_error("ROLE");auto role=role_text=="C"?Role::Client:Role::Server;
  auto sid=unhex<16>(argv[9]);auto cfg=unhex<32>(argv[10]);auto snap=unhex<32>(argv[11]);
  std::filesystem::path root=argv[12];std::string kind=argv[13];if(kind!="embedding"&&kind!="generation")throw std::runtime_error("MODEL");auto shape=kind=="embedding"?ModelShape::embedding():ModelShape::generation();
  std::string mode=argv[14];bool prepare=mode=="prepare";
  if((!prepare&&mode!="retrieve")||kind!="embedding")throw std::runtime_error("MODE");
  size_t rows=std::stoul(argv[15]),count=std::stoul(argv[16]);if((prepare&&(rows!=257||count!=0))||(!prepare&&(rows!=160||!count||count>2)))throw std::runtime_error("SHAPE");
  int input=std::stoi(argv[5]),output=std::stoi(argv[6]);int seals=F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL;int found_seals=fcntl(input,F_GET_SEALS);if(found_seals<0||(found_seals&seals)!=seals)throw std::runtime_error("UNSEALED");
  int lifecycle=std::stoi(argv[18]);if((role==Role::Server)!=(lifecycle>=0))throw std::runtime_error("LIFECYCLE_ROLE");
  std::array<unsigned char,96> header{};exact(input,header.data(),header.size());if(memcmp(header.data(),"PR112IN1",8)||header[8]!=(role==Role::Client?0:1)||memcmp(header.data()+16,sid.data(),16)||memcmp(header.data()+32,cfg.data(),32)||memcmp(header.data()+64,snap.data(),32))throw std::runtime_error("INPUT_BINDING");for(size_t i=9;i<16;++i)if(header[i])throw std::runtime_error("INPUT_FORMAT");
  LocalSlots owned;std::vector<BoundRecord> bound;
  if(role==Role::Client){owned.tokens.resize(rows);owned.validity.resize(rows);exact(input,owned.tokens.data(),rows*8);for(size_t i=0;i<rows;++i){uint8_t b;exact(input,&b,1);if(b>1)throw std::runtime_error("BIT");owned.validity[i]=b;}owned.validate(rows,151668);}
  auto words=[&](size_t n){Row values(n);exact(input,values.data(),n*8);for(auto x:values)if(x>MASK)throw std::runtime_error("WORD");return values;};
  for(size_t i=0;i<count;++i)bound.push_back({{words(256),role,argv[10]},{words(256),role,argv[10]},{words(1024),role,argv[10]},{words(1),role,argv[10]}});
  char extra;if(read(input,&extra,1)!=0)throw std::runtime_error("INPUT_LENGTH");close(input);
  Channel io(std::stoi(argv[2]),sid,cfg),out(std::stoi(argv[3]),sid,cfg),in(std::stoi(argv[4]),sid,cfg);std::string pki=argv[7],own=role==Role::Client?(prepare?"D":"U"):"S-compute";
  std::array<Channel*,3> order=role==Role::Client?std::array<Channel*,3>{&io,&out,&in}:std::array<Channel*,3>{&io,&in,&out};
  for(auto*c:order){c->enable_tls(role,pki+"/ca.pem",pki+"/"+own+".pem",pki+"/"+own+".key","localhost");if(c->peer_fingerprint_sha256()!=argv[8])throw std::runtime_error("PEER_BINDING");}
  std::array<unsigned char,32>other{};io.send_data(snap.data(),32);io.recv_data(other.data(),32);if(snap!=other)throw std::runtime_error("SNAPSHOT_BINDING");
  std::string record=std::string(argv[17])+"."+role_text;
  SecureOps op(role,io,record+".ledger",false);Heartbeat hb(out,in,io,[&]{op.abort_session();});
  try{
   auto gpu=make_rtx2060_shares();require_compiled_numeric_profile(root/"artifacts/constants/numeric_profile_v1_qk.json");Constants constants(root/"artifacts/constants/constants_v1.bin");CheckpointWeights weights(root/("artifacts/models/"+kind+"/quantized_f12"),shape,role);
   PhaseMeter meter(io,record);std::vector<std::unique_ptr<PhaseMeter::Scope>> stages;
   Numerics num(op,constants.exp,[&](const char*name,bool begin){hb.check();if(begin)stages.push_back(std::make_unique<PhaseMeter::Scope>(meter,name));else{if(stages.empty())throw std::runtime_error("PHASE_UNBALANCED");stages.pop_back();}},gpu.get());
   Transformer model(num,shape,weights,constants.cos,constants.sin,constants.attention_scale);
   auto start=std::chrono::steady_clock::now();
   auto shared=share_owned_slots(op,role==Role::Client?&owned:nullptr,rows,151668);
   auto position=Pipeline(num).positions(shared.validity);
   auto embedding=model.embedding(shared.tokens,shared.validity,position,op.public_bit(true));
   auto eligible=op.b2a_bit(embedding.second);Rows y;
   if(prepare){y.push_back(embedding.first);y.push_back(Row{eligible});}
   else{
     std::vector<Record> records;for(auto&b:bound)records.push_back(b.into(op));
     auto retrieved=Retriever(num).run(embedding.first,eligible,records);
     for(auto&e:retrieved.evidence){y.push_back(e.tokens);y.push_back(e.validity);y.push_back(Row{op.b2a_bit(e.valid)});}
     // I7 remains shared. Only the isolated final oracle receives own shares.
     for(auto bit:retrieved.private_keep_bits)y.push_back(Row{op.b2a_bit(bit)});
   }
   hb.check();
   if(role==Role::Server){char releasing='R',ack=0;exact(lifecycle,&releasing,1,true);exact(lifecycle,&ack,1);if(ack!='A')throw std::runtime_error("RELEASE_STATE");close(lifecycle);}
   uint8_t failure=num.failure(),other_failure=0,client_failure=0;
   if(prepare){if(role==Role::Server)io.send_data(&failure,1);else{io.recv_data(&other_failure,1);if(other_failure>1)throw std::runtime_error("FAILURE_BIT");client_failure=failure^other_failure;}}
   uint8_t ready=1,peer_ready=0;io.send_data(&ready,1);io.recv_data(&peer_ready,1);if(peer_ready!=ready)throw std::runtime_error("RELEASE_STATE");
   auto bytes=io.bytes();auto heartbeats=hb.received();op.close_session();hb.close();
   for(auto&r:y){exact(output,r.data(),r.size()*8,true);OPENSSL_cleanse(r.data(),r.size()*8);}
   if(!prepare)exact(output,&failure,1,true);else if(role==Role::Client)exact(output,&client_failure,1,true);
   close(output);OPENSSL_cleanse(owned.tokens.data(),owned.tokens.size()*8);std::fill(owned.validity.begin(),owned.validity.end(),false);
   auto stats=gpu->stats();rusage use{};getrusage(RUSAGE_SELF,&use);
   std::cout<<"{\"result\":\"completed_shared_M3_stage\",\"application_bytes\":"<<bytes<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"peak_rss_kib\":"<<use.ru_maxrss<<",\"gpu_peak_owned_bytes\":"<<stats.peak_bytes<<",\"gpu_kernels\":"<<stats.kernels<<",\"gpu_upload_bytes\":"<<stats.uploaded_bytes<<",\"gpu_download_bytes\":"<<stats.downloaded_bytes<<",\"heartbeats_received\":"<<heartbeats<<"}\n";
  }catch(...){op.abort_session();hb.close();throw;}
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
