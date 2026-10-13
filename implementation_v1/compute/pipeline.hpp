#pragma once
#include "model.hpp"

namespace prag::model {
struct PromptLayout {
  std::vector<std::pair<std::size_t,Row>> public_segments;
  std::size_t question_offset,evidence1_offset,evidence2_offset;
  Word dummy_token=0;
  void validate()const{
    std::array<bool,1024> used{};
    auto mark=[&](std::size_t offset,std::size_t length){
      if(offset>1024||length>1024-offset)throw std::invalid_argument("BAD_SHAPE");
      for(std::size_t i=offset;i<offset+length;++i){if(used[i])throw std::invalid_argument("BAD_SHAPE");used[i]=true;}
    };
    mark(question_offset,128);mark(evidence1_offset,256);mark(evidence2_offset,256);
    for(const auto& segment:public_segments)mark(segment.first,segment.second.size());
    if(dummy_token>151642)throw std::invalid_argument("BAD_FORMAT");
  }
};
struct Prompt {Row tokens,positions;std::vector<Bit> validity;Word effective_length;};
struct AnswerShares {Row tokens;std::vector<Bit> validity;Word status;};
// Public status codes are recovered ONLY at the final permitted client release.
// 0 normal, 1 insufficient evidence, 2 numeric configuration failure, 3 length cap.
class Pipeline {
  Numerics& n;SecureOps& s;
 public:
  explicit Pipeline(Numerics& num):n(num),s(num.ops()){}
  Row positions(const std::vector<Bit>& valid){
    Word count=n.constant(0);Row p; p.reserve(valid.size());
    for(Bit bit:valid){count=s.add(count,s.b2a_bit(bit));p.push_back(s.select(bit,s.sub(count,n.constant(1)),n.constant(0)));}
    return p;
  }
  Prompt assemble(const Row& question,const std::vector<Bit>& question_valid,const std::vector<Evidence>& evidence,const PromptLayout& layout){
    layout.validate();
    if(question.size()!=128||question_valid.size()!=128||evidence.size()!=2)throw std::invalid_argument("BAD_SHAPE");
    Prompt p{Row(1024,n.constant(layout.dummy_token)),{},std::vector<Bit>(1024,s.public_bit(false)),n.constant(0)};
    for(const auto& segment:layout.public_segments)for(std::size_t i=0;i<segment.second.size();++i){p.tokens[segment.first+i]=n.constant(segment.second[i]);p.validity[segment.first+i]=s.public_bit(true);}
    for(std::size_t i=0;i<128;++i){p.tokens[layout.question_offset+i]=question[i];p.validity[layout.question_offset+i]=question_valid[i];}
    for(std::size_t slot=0;slot<2;++slot){
      if(evidence[slot].tokens.size()!=256||evidence[slot].validity.size()!=256)throw std::invalid_argument("BAD_SHAPE");
      auto offset=slot==0?layout.evidence1_offset:layout.evidence2_offset;
      for(std::size_t i=0;i<256;++i){p.tokens[offset+i]=evidence[slot].tokens[i];p.validity[offset+i]=n.nonzero(evidence[slot].validity[i]);}
    }
    p.positions=positions(p.validity);
    for(Bit bit:p.validity)p.effective_length=s.add(p.effective_length,s.b2a_bit(bit));
    return p;
  }
  AnswerShares generate(Transformer& model,const Prompt& prompt,Bit any_evidence,Word dummy=0){
    if(prompt.tokens.size()!=1024)throw std::invalid_argument("BAD_SHAPE");
    Cache cache(s,n.local());
    auto [hidden,has_prompt]=model.last_hidden(model.forward(prompt.tokens,prompt.validity,prompt.positions,&cache,0));
    n.fail(s.bnot(has_prompt));
    Word token=model.greedy(hidden);Bit alive=s.public_bit(true);
    AnswerShares result{Row(256),std::vector<Bit>(256),n.constant(0)};
    for(std::size_t step=0;step<256;++step){
      Bit eos=s.bor(s.equal(token,n.constant(151643)),s.equal(token,n.constant(151645)));
      alive=s.band(alive,s.bnot(eos));
      result.validity[step]=alive;result.tokens[step]=s.select(alive,token,n.constant(dummy));
      if(step<255){ // fixed public schedule, never EOS-dependent
        Word feedback=s.select(alive,token,n.constant(dummy));
        Word position=s.select(alive,s.add(prompt.effective_length,n.constant(step)),n.constant(0));
        auto y=model.forward({feedback},{alive},{position},&cache,1024+step);
        token=model.greedy(y.hidden.at(0));
      }
    }
    result.status=s.select(alive,n.constant(3),n.constant(0));
    result.status=s.select(any_evidence,result.status,n.constant(1));
    result.status=s.select(n.failure(),n.constant(2),result.status);
    Bit permitted=s.band(any_evidence,s.bnot(n.failure()));
    for(std::size_t i=0;i<256;++i){result.validity[i]=s.band(permitted,result.validity[i]);result.tokens[i]=s.select(result.validity[i],result.tokens[i],n.constant(dummy));}
    return result;
  }
  AnswerShares query(Transformer& embedder,Transformer& generator,const Row& embedding_tokens,const std::vector<Bit>& embedding_validity,const Row& question,const std::vector<Bit>& question_valid,const std::vector<Record>& records,const PromptLayout& layout){
    if(records.empty())throw std::invalid_argument("PUBLIC_N_ZERO_HANDLED_BEFORE_COMPUTE");
    auto [q,qvalid]=embedder.embedding(embedding_tokens,embedding_validity,positions(embedding_validity),s.public_bit(true));
    Retriever retrieval(n);auto selected=retrieval.run(q,s.b2a_bit(qvalid),records,layout.dummy_token);
    auto prompt=assemble(question,question_valid,selected.evidence,layout);
    return generate(generator,prompt,s.bor(selected.evidence[0].valid,selected.evidence[1].valid),layout.dummy_token);
  }
};
}
