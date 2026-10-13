#pragma once
#include "numerics.hpp"

namespace prag::model {
class Retriever {
  Numerics& n; SecureOps& s;
  Candidate winner(const Candidate& a,const Candidate& b){
    Bit validity_equal=s.bnot(s.bxor(a.valid,b.valid));
    Bit score_equal=s.equal(a.score,b.score);
    Bit earlier=s.gt(b.index,a.index,false);
    Bit choose=s.bor(s.band(a.valid,s.bnot(b.valid)),s.band(validity_equal,s.bor(s.gt(a.score,b.score,true),s.band(score_equal,earlier))));
    return {s.select(choose,a.score,b.score),s.select(choose,a.index,b.index),s.bxor(b.valid,s.band(choose,s.bxor(a.valid,b.valid)))};
  }
 public:
  explicit Retriever(Numerics& num):n(num),s(num.ops()){}
  std::array<Candidate,2> top2(const std::vector<Candidate>& candidates){
    if(candidates.empty())throw std::invalid_argument("BAD_SHAPE");
    auto input=candidates; std::array<Candidate,2> out;
    for(unsigned round=0;round<2;++round){
      auto tree=input;
      while(tree.size()>1){
        std::vector<Candidate> next;
        for(std::size_t i=0;i<tree.size();i+=2)
          next.push_back(i+1<tree.size()?winner(tree[i],tree[i+1]):tree[i]);
        tree=std::move(next);
      }
      out[round]=tree[0];
      for(auto& c:input)c.valid=s.band(c.valid,s.bnot(s.band(out[round].valid,s.equal(c.index,out[round].index))));
    }
    return out;
  }
  Retrieval run(std::vector<Word> query,Word query_valid,const std::vector<Record>& records,Word dummy_token=0){
    if(query.size()!=1024||records.empty())throw std::invalid_argument("BAD_SHAPE");
    for(const auto& r:records)if(r.tokens.size()!=256||r.validity.size()!=256||r.embedding.size()!=1024)throw std::invalid_argument("BAD_SHAPE");
    const Word B=1024ULL*(2ULL*(1ULL<<21)*4095+4095ULL*4095);
    std::vector<Candidate> exact,lower;
    std::vector<Word> upper;
    query=n.checked_many(query,2<<20);
    auto coarse_query=n.floor_many(query,12);
    Bit qvalid=n.nonzero(query_valid);
    for(std::size_t i=0;i<records.size();++i){
      auto z=n.checked_many(records[i].embedding,2<<20);
      auto fine=n.local().sum_rows(n.multiply_many(query,z),1024)[0];
      auto coarse=n.local().sum_rows(n.multiply_many(coarse_query,n.floor_many(z,12)),1024)[0];
      Word approx=s.mul_public(coarse,Word{1}<<24);
      Bit valid=s.band(qvalid,n.nonzero(records[i].embedding_valid));
      exact.push_back({fine,n.constant(i),valid});
      lower.push_back({s.sub(approx,n.constant(B)),n.constant(i),valid});
      upper.push_back(s.add(approx,n.constant(B)));
    }
    auto threshold=top2(lower)[1];
    std::vector<Bit> keep;
    for(std::size_t i=0;i<exact.size();++i)
      keep.push_back(s.band(exact[i].valid,s.bor(s.bnot(threshold.valid),s.bnot(s.gt(threshold.score,upper[i],true)))));
    // Deliberately no compaction or skipping. I7 keep bits stay secret.
    auto selected=top2(exact);
    std::vector<Evidence> evidence;
    for(const auto& selected_row:selected){
      Evidence e{std::vector<Word>(256,n.constant(0)),std::vector<Word>(256,n.constant(0)),selected_row.valid};
      for(std::size_t i=0;i<records.size();++i){
        Bit choose=s.band(selected_row.valid,s.band(exact[i].valid,s.equal(selected_row.index,n.constant(i))));
        std::vector<uint8_t> selected(256,choose);
        e.tokens=n.select_many(selected,records[i].tokens,e.tokens);
        e.validity=n.select_many(selected,records[i].validity,e.validity);
      }
      // Records bind m to 0/1 before delivery; equality-to-zero is retained,
      // using both unsigned comparisons for a general canonical ring share.
      auto present=n.compare_many(e.validity,std::vector<Word>(256,n.constant(0)),false);
      e.tokens=n.select_many(present,e.tokens,std::vector<Word>(256,n.constant(dummy_token)));
      evidence.push_back(std::move(e));
    }
    return {std::move(evidence),std::move(keep)};
  }
};
}
