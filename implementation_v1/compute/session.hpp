#pragma once
#include "artifacts.hpp"
#include <openssl/rand.h>
#include <openssl/crypto.h>
#include <optional>

namespace prag::model {
// Typed outer boundary; internal kernels have scales fixed by their signatures.
// A session owns exactly one local party and one frozen model/numeric digest.
template<unsigned Fraction, std::size_t Width> struct SharedVector {
  Row values;
  Role party;
  std::string configuration;
  void require(const SecureOps& s)const {
    if(values.size()!=Width||party!=s.role()||configuration.size()!=64)
      throw std::invalid_argument("BAD_SHAPE");
    for(auto v:values)if(v>mask)throw std::invalid_argument("BAD_FORMAT");
  }
};
struct BoundRecord {
  SharedVector<0,256> tokens,validity;
  SharedVector<20,1024> embedding;
  SharedVector<0,1> eligible;
  Record into(SecureOps& s)const {
    tokens.require(s);validity.require(s);embedding.require(s);eligible.require(s);
    const auto expected=s.channel().configuration_hex();
    for(const auto* c:{&tokens.configuration,&validity.configuration,&embedding.configuration,&eligible.configuration})
      if(*c!=expected)throw std::invalid_argument("VERSION_MISMATCH");
    return {tokens.values,validity.values,embedding.values,eligible.values[0]};
  }
};
struct LocalSlots {
  Row tokens;
  std::vector<Bit> validity;
  // Exists only at C, whose role already authorizes knowledge of these inputs.
  void validate(std::size_t count,Word largest_token)const {
    if(tokens.size()!=count||validity.size()!=count)throw std::invalid_argument("BAD_SHAPE");
    for(auto id:tokens)if(id>largest_token)throw std::invalid_argument("BAD_FORMAT");
  }
};
struct SharedSlots {Row tokens;std::vector<Bit> validity;};
inline Word fresh_word(){Word x;if(RAND_priv_bytes(reinterpret_cast<unsigned char*>(&x),sizeof(x))!=1)throw std::runtime_error("E_RANDOM");return x&mask;}
inline void put_word(unsigned char* p,Word x){for(unsigned j=0;j<8;++j){p[j]=x&255;x>>=8;}}
inline Word get_word(const unsigned char* p){Word x=0;for(int j=7;j>=0;--j)x=(x<<8)|p[j];if(x>mask)throw std::invalid_argument("BAD_FORMAT");return x;}

// Only the owner supplies LocalSlots. S receives independently randomized shares,
// not IDs or validity bits. Public shape fixes the full payload size.
inline SharedSlots share_owned_slots(SecureOps& s,const LocalSlots* owned,std::size_t count,Word maximum){
  if((s.role()==Role::Client)!=(owned!=nullptr))throw std::invalid_argument("BAD_STATE");
  if(count!=128&&count!=160&&count!=257)throw std::invalid_argument("BAD_SHAPE");
  if(owned)owned->validate(count,maximum);
  s.reserve_material("owner_input_masks",count*2);
  SharedSlots result{Row(count),std::vector<Bit>(count)};
  std::vector<unsigned char> wire(count*9);
  if(owned){
    for(std::size_t i=0;i<count;++i){
      result.tokens[i]=fresh_word();result.validity[i]=(fresh_word()&1)!=0;
      put_word(wire.data()+i*8,s.sub(owned->tokens[i],result.tokens[i]));
      wire[count*8+i]=owned->validity[i]!=result.validity[i];
    }
    s.channel().send_data(wire.data(),wire.size());
  }else{
    s.channel().recv_data(wire.data(),wire.size());
    for(std::size_t i=0;i<count;++i){result.tokens[i]=get_word(wire.data()+i*8);if(wire[count*8+i]>1)throw std::invalid_argument("BAD_FORMAT");result.validity[i]=wire[count*8+i];}
  }
  OPENSSL_cleanse(wire.data(),wire.size());
  return result;
}

struct ClientAnswer {Row tokens;std::vector<Bit> validity;std::optional<Word> status;};
// This is the ONLY normal-answer reconstruction. It can run after the whole
// fixed schedule only. No record, score, index, or activation is accepted here.
inline ClientAnswer release_answer_to_client(SecureOps& s,const AnswerShares& local){
  if(local.tokens.size()!=256||local.validity.size()!=256)throw std::invalid_argument("BAD_SHAPE");
  s.channel().set_stage(90);s.channel().set_operation(0);
  std::array<unsigned char,2312> wire{};
  ClientAnswer answer;
  if(s.role()==Role::Server){
    for(std::size_t i=0;i<256;++i){put_word(wire.data()+8*i,local.tokens[i]);wire[2048+i]=local.validity[i];}
    put_word(wire.data()+2304,local.status);
    s.channel().send_data(wire.data(),wire.size());
  }else{
    s.channel().recv_data(wire.data(),wire.size());answer.tokens.resize(256);answer.validity.resize(256);
    for(std::size_t i=0;i<256;++i){
      if(wire[2048+i]>1)throw std::invalid_argument("BAD_FORMAT");
      answer.tokens[i]=s.add(local.tokens[i],get_word(wire.data()+8*i));answer.validity[i]=local.validity[i]!=(wire[2048+i]!=0);
      if(answer.tokens[i]>151642)throw std::invalid_argument("BAD_FORMAT");
    }
    answer.status=s.add(local.status,get_word(wire.data()+2304));
    if(answer.status>3)throw std::invalid_argument("BAD_FORMAT");
  }
  // Receipt contains no status or EOS information. Lost receipt aborts release;
  // neither side retries or reuses this session's cryptographic materials.
  std::array<unsigned char,8> ack={'P','R','A','G','A','C','K','1'},received{};
  if(s.role()==Role::Client)s.channel().send_data(ack.data(),ack.size());
  else{s.channel().recv_data(received.data(),received.size());if(received!=ack)throw std::runtime_error("REPLAY_OR_ORDER");}
  OPENSSL_cleanse(wire.data(),wire.size());return answer;
}
struct PreparedLocal {SharedVector<20,1024> embedding;SharedVector<0,1> eligible;std::optional<bool> client_failed;};
class SessionRunner {
  SecureOps& s;Constants constants;CheckpointWeights embedding_weights,generation_weights;
  std::unique_ptr<LocalShares> local=make_rtx2060_shares();
  Numerics numbers;
 public:
  SessionRunner(SecureOps& ops,const std::filesystem::path& models,const std::filesystem::path& constants_path,NumericPhase public_phase={})
   :s(ops),constants(constants_path),embedding_weights(models/"embedding/quantized_f12",ModelShape::embedding(),ops.role()),
    generation_weights(models/"generation/quantized_f12",ModelShape::generation(),ops.role()),numbers(ops,constants.exp,std::move(public_phase),local.get()){}
  LocalShareStats gpu_stats()const{return local->stats();}
  PreparedLocal prepare(const LocalSlots* owner_document,const std::function<void()>& before_release={}){
    s.channel().set_stage(10);
    auto input=share_owned_slots(s,owner_document,257,151668);
    Transformer embedder(numbers,ModelShape::embedding(),embedding_weights,constants.cos,constants.sin,constants.attention_scale);
    Pipeline pipeline(numbers);s.channel().set_stage(20);
    auto [z,e]=embedder.embedding(input.tokens,input.validity,pipeline.positions(input.validity),s.public_bit(true));
    Word arithmetic_e=s.b2a_bit(e);
    if(before_release)before_release();
    // Preparation keeps z/e shared. Only the secret failure status is opened D.
    s.channel().set_stage(90);s.channel().set_operation(0);
    unsigned char peer=0,own=numbers.failure();bool failed=false;
    if(s.role()==Role::Server)s.channel().send_data(&own,1);
    else{s.channel().recv_data(&peer,1);if(peer>1)throw std::invalid_argument("BAD_FORMAT");failed=(peer!=0)!=numbers.failure();}
    unsigned char ack=1;
    if(s.role()==Role::Client)s.channel().send_data(&ack,1);
    else{s.channel().recv_data(&ack,1);if(ack!=1)throw std::runtime_error("REPLAY_OR_ORDER");}
    return {{std::move(z),s.role(),s.channel().configuration_hex()},{{arithmetic_e},s.role(),s.channel().configuration_hex()},s.role()==Role::Client?std::optional<bool>(failed):std::nullopt};
  }
  ClientAnswer query(const LocalSlots* owner_embedding,const LocalSlots* owner_body,const std::vector<BoundRecord>& snapshot,const PromptLayout& layout,const std::function<void()>& before_release={}){
    if(snapshot.empty())throw std::invalid_argument("PUBLIC_N_ZERO_HANDLED_BEFORE_COMPUTE");
    std::vector<Record> records;records.reserve(snapshot.size());for(const auto& r:snapshot)records.push_back(r.into(s));
    s.channel().set_stage(10);auto input=share_owned_slots(s,owner_embedding,160,151668);auto question=share_owned_slots(s,owner_body,128,151642);
    Transformer embedder(numbers,ModelShape::embedding(),embedding_weights,constants.cos,constants.sin,constants.attention_scale);
    Transformer generator(numbers,ModelShape::generation(),generation_weights,constants.cos,constants.sin,constants.attention_scale);
    Pipeline pipeline(numbers);s.channel().set_stage(20);
    auto answer=pipeline.query(embedder,generator,input.tokens,input.validity,question.tokens,question.validity,records,layout);
    if(before_release)before_release();
    return release_answer_to_client(s,answer);
  }
};
}
