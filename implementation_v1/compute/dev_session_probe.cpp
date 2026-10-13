// Isolated public synthetic diagnostics. NEVER linked into production.
// Any reconstruction below is explicitly for known public probe values.
#include "session.hpp"
#include <openssl/evp.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace prag;
using namespace prag::model;

static void require(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
static void check_artifacts(const std::filesystem::path& root,const std::string& fixture){
  std::ifstream expected(fixture);std::string magic,digest;std::size_t count=0;
  require(bool(expected>>magic>>digest>>count)&&magic=="PRDEVART1"&&count==1860,"E_PROBE_FIXTURE");
  Constants c(root/"constants/constants_v1.bin");
  EVP_MD_CTX* raw=EVP_MD_CTX_new();require(raw!=nullptr,"E_PROBE_HASH");
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(raw,EVP_MD_CTX_free);
  require(EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"E_PROBE_HASH");
  std::size_t words=0;
  auto hash_word=[&](Word v){unsigned char data[8];put_word(data,v);require(EVP_DigestUpdate(ctx.get(),data,8)==1,"E_PROBE_HASH");++words;};
  hash_word(c.attention_scale);
  for(const auto& t:c.exp)for(auto v:t)hash_word(v);
  for(const auto* table:{&c.cos,&c.sin})for(const auto& row:*table)for(auto v:row)hash_word(v);
  unsigned char bytes[32];unsigned length=0;require(EVP_DigestFinal_ex(ctx.get(),bytes,&length)==1&&length==32,"E_PROBE_HASH");
  std::ostringstream hex;hex<<std::hex<<std::setfill('0');for(auto b:bytes)hex<<std::setw(2)<<unsigned(b);
  require(words==164609&&hex.str()==digest&&c.attention_scale==92681,"E_PROBE_CONSTANT_MISMATCH");
  // Only this S-local developer branch maps real public checkpoint weights.
  // C never receives any checkpoint value or matrix.
  for(const auto& kind:{std::string("embedding"),std::string("generation")}){
    auto dimensions=kind=="embedding"?ModelShape::embedding():ModelShape::generation();
    CheckpointWeights weights(root/"models"/kind/"quantized_f12",dimensions,Role::Server);
    for(int layer=-2;layer<28;++layer){
      std::vector<PrivateMatrix> matrices;
      if(layer==-2)matrices.push_back(weights.token_table());
      else if(layer==-1)matrices.push_back(weights.final_norm());
      else{auto l=weights.layer(layer);matrices={l.in_norm,l.q,l.k,l.v,l.q_norm,l.k_norm,l.o,l.ffn_norm,l.gate,l.up,l.down};}
      for(std::size_t field=0;field<matrices.size();++field){
        auto& matrix=matrices[field];require(matrix.owned(),"E_PROBE_WEIGHT_OWNERSHIP");
        for(unsigned sample=0;sample<3;++sample){
          std::string expected_kind;int expected_layer;std::size_t expected_field,row,col;int64_t signed_value;
          require(bool(expected>>expected_kind>>expected_layer>>expected_field>>row>>col>>signed_value),"E_PROBE_FIXTURE");
          require(expected_kind==kind&&expected_layer==layer&&expected_field==field,"E_PROBE_FIXTURE_ORDER");
          const auto expected_row=sample==0?0:sample==1?matrix.rows/2:matrix.rows-1;
          const auto expected_col=sample==0?0:sample==1?matrix.cols/2:matrix.cols-1;
          require(row==expected_row&&col==expected_col,"E_PROBE_FIXTURE_POSITION");
          require(matrix.at(row,col)==(static_cast<Word>(signed_value)&MASK),"E_PROBE_WEIGHT_MISMATCH");
          --count;
        }
      }
    }
  }
  std::string extra;require(count==0&&!(expected>>extra),"E_PROBE_FIXTURE_TRAILING");
}
static void check_client_weights_absent(){
  // Nonexistent directory demonstrates that the C provider uses dimensions only.
  for(auto shape:{ModelShape::embedding(),ModelShape::generation()}){
    CheckpointWeights weights("/nonexistent-public-shape-only",shape,Role::Client);
    require(!weights.token_table().owned()&&!weights.final_norm().owned(),"E_PROBE_CLIENT_WEIGHT");
    for(std::size_t layer=0;layer<28;++layer){
      auto l=weights.layer(layer);
      for(auto* w:{&l.in_norm,&l.q,&l.k,&l.v,&l.q_norm,&l.k_norm,&l.o,&l.ffn_norm,&l.gate,&l.up,&l.down})
        require(!w->owned(),"E_PROBE_CLIENT_WEIGHT");
    }
  }
}
static int run(Role role,int fd,const std::string& prefix,const std::filesystem::path& artifacts,const std::string& fixture){
  if(role==Role::Server)check_artifacts(artifacts,fixture);else check_client_weights_absent();
  std::array<unsigned char,16> session{};session[0]=51;
  std::array<unsigned char,32> configuration{};configuration[0]=7;
  Channel io(fd,session,configuration);
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),true);
  for(std::size_t count:{128,160,257}){
    LocalSlots public_input{Row(count),std::vector<Bit>(count)};
    for(std::size_t i=0;i<count;++i){public_input.tokens[i]=(i*613+count)%151643;public_input.validity[i]=(i%3)!=1;}
    auto before=io.bytes();
    auto shared=share_owned_slots(op,role==Role::Client?&public_input:nullptr,count,151642);
    require(io.bytes()-before==96+9*count,"E_PROBE_INPUT_FRAME");
    // Developer-only opening of synthetic public input. Not a SessionRunner path.
    std::vector<unsigned char> wire(count*9);
    if(role==Role::Server){
      for(std::size_t i=0;i<count;++i){put_word(wire.data()+i*8,shared.tokens[i]);wire[count*8+i]=shared.validity[i];}
      io.send_data(wire.data(),wire.size());
    }else{
      io.recv_data(wire.data(),wire.size());
      for(std::size_t i=0;i<count;++i){
        require(op.add(shared.tokens[i],get_word(wire.data()+i*8))==public_input.tokens[i],"E_PROBE_INPUT_TOKEN");
        require((shared.validity[i]!=(wire[count*8+i]!=0))==public_input.validity[i],"E_PROBE_INPUT_VALIDITY");
      }
    }
  }
  // One real OT-backed multiplication checks this is the actual SecureOps backend.
  auto split=[&](Word value){return role==Role::Client?912837ULL:(value-912837ULL)&MASK;};
  Word product=op.mul(split(7),split(9));
  if(role==Role::Server)io.send_data(&product,8);
  else{Word other;io.recv_data(&other,8);require(op.add(product,other)==63,"E_PROBE_REAL_OT");}
  for(Word status=0;status<4;++status){
    AnswerShares local{Row(256),std::vector<Bit>(256),split(status)};
    for(std::size_t i=0;i<256;++i){
      Word token=(i*419+status)%151643;local.tokens[i]=split(token);
      bool value=i<213-status,client_share=(i%2)!=0;
      local.validity[i]=role==Role::Client?client_share:value!=client_share;
    }
    auto before=io.bytes();auto answer=release_answer_to_client(op,local);
    // One fixed 2312-byte release and one fixed PRAGACK1 8-byte receipt,
    // each with a 96-byte transport header. Server validates receipt contents.
    require(io.bytes()-before==2512,"E_PROBE_RELEASE_FRAME");
    if(role==Role::Client){
      require(answer.tokens.size()==256&&answer.validity.size()==256&&answer.status==status,"E_PROBE_ANSWER_SHAPE");
      for(std::size_t i=0;i<256;++i){
        require(answer.tokens[i]==(i*419+status)%151643,"E_PROBE_ANSWER_TOKEN");
        require(answer.validity[i]==(i<213-status),"E_PROBE_ANSWER_VALIDITY");
      }
    }else require(answer.tokens.empty()&&answer.validity.empty(),"E_PROBE_SERVER_ANSWER");
  }
  if(role==Role::Client)
    std::cout<<"{\"entry\":\"isolated_public_session_artifact_probe\",\"two_processes\":true,\"owner_slots_128_160_257\":\"matched\",\"real_OT_multiplication\":\"matched\",\"client_answer_256_slots_status_0_to_3\":\"matched\",\"server_answer_vectors_empty\":true,\"release_and_fixed_ack_bytes_per_case\":2512,\"server_checkpoint_samples\":1860,\"constant_words_checked\":164609,\"client_provider_has_no_weights\":true,\"client_channel_bytes\":"<<io.bytes()<<",\"production_TLS_not_run\":true,\"full_model_not_run\":true}\n";
  return 0;
}
int main(int argc,char** argv){
  if(argc!=4){std::cerr<<"usage: prag_dev_session_probe FRESH_LEDGER_PREFIX ARTIFACTS_DIR PUBLIC_EXPECTED_FIXTURE\n";return 2;}
  int sockets[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sockets))return 3;
  pid_t child=fork();if(child<0){close(sockets[0]);close(sockets[1]);return 3;}
  int rc=1;
  try{
    if(child==0){close(sockets[0]);rc=run(Role::Server,sockets[1],argv[1],argv[2],argv[3]);_exit(rc);}
    close(sockets[1]);rc=run(Role::Client,sockets[0],argv[1],argv[2],argv[3]);
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(child==0)_exit(1);shutdown(sockets[0],SHUT_RDWR);}
  int state=0;waitpid(child,&state,0);return rc||!WIFEXITED(state)||WEXITSTATUS(state);
}
