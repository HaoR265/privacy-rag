#pragma once
#include "secure_tensor.hpp"
#include "meter.hpp"
#include "numeric_profile.hpp"
#include <array>
#include <algorithm>

namespace prag::model {
// All data-dependent choices are calls to SecureOps, never host-language branches.
// Loops depend only on frozen dimensions. failure is a local XOR share.
class ProfileNumerics {
  SecureOps& s;
  Bit failed;
  std::array<std::vector<Word>,3> exponent_tables;
 public:
  ProfileNumerics(SecureOps& ops,std::array<std::vector<Word>,3> tables):s(ops),failed(s.public_bit(false)),exponent_tables(std::move(tables)) {
    for(const auto& t:exponent_tables) if(t.size()!=256) throw std::invalid_argument("BAD_SHAPE");
  }
  SecureOps& ops(){return s;}
  Bit failure() const {return failed;}
  void fail(Bit b){failed=s.bor(failed,b);}
  Word constant(Word x){return s.public_word(x&mask);}
  Bit nonzero(Word x){return s.bnot(s.equal(x,constant(0)));}
  Bit less(Word x,Word y){return s.gt(y,x,true);}
  Word checked_encoded(Word x,Word lim,Word dummy=0) {
    PhaseMeter::Scope meter(*current_meter,"output_or_state_range");
    if(lim>=(Word{1}<<56))throw std::invalid_argument("BAD_FORMAT");
    Bit bad=s.bor(s.gt(x,constant(lim),true),less(x,constant((-lim)&mask)));
    fail(bad); return s.select(bad,constant(dummy),x);
  }
  Word checked(Word x,Word magnitude,unsigned f,Word dummy=0) {
    if(f>=56||magnitude>=((Word{1}<<56)>>f))throw std::invalid_argument("BAD_FORMAT");
    return checked_encoded(x,magnitude<<f,dummy);
  }
  Word state(Word x) {
    PhaseMeter::Scope meter(*current_meter,"NR_state_guard");
    Bit bad=s.bor(less(x,constant(0)),s.gt(x,constant(2*one24),true));
    fail(bad); return s.select(bad,constant(one24/2),x);
  }
  Word product24(Word a,Word b){PhaseMeter::Scope meter(*current_meter,"NR_product_floor24");return s.floor(s.mul(a,b),24);}
  Word inverse_sqrt(Word mantissa) {
    PhaseMeter::Scope meter(*current_meter,"six_NR");
    Word y=constant(one24/2);
    for(unsigned r=0;r<6;++r){
      y=state(y);
      Word u=product24(y,y);
      Word v=product24(mantissa,u);
      Word w=product24(y,s.sub(constant(3*one24),v));
      y=s.floor(w,1);
    }
    return state(y);
  }
  // Derive secret exponent from a fixed scan of the 59-bit word.  Positive dummy
  // is selected before the NR arithmetic; no local-share range interpretation.
  Word exponent(Word& a,int low,int high,bool sqrt_mode) {
    PhaseMeter::Scope meter(*current_meter,"exponent");
    auto bits=s.a2b(a); Bit found=s.public_bit(false); Word result=constant(0);
    for(int j=58;j>=0;--j) {
      Bit top=s.band(bits[j],s.bnot(found));
      int k=j-24;
      if(sqrt_mode) k=k>=0?k/2:-((-k+1)/2); // public integer floor
      result=s.select(top,constant(static_cast<Word>(k)),result);
      found=s.bor(found,bits[j]);
    }
    Bit invalid=s.bor(s.bor(s.bnot(found),bits[58]),s.bor(less(result,constant(static_cast<Word>(low))),s.gt(result,constant(static_cast<Word>(high)),true)));
    fail(invalid);
    a=s.select(invalid,constant(one24),a);
    return s.select(invalid,constant(0),result);
  }
  // Exact signed barrel shift using 128 secret wires. A left shift never passes
  // through the 59-bit ring before checking the high wires (NUM-004).
  Word scale_shift(Word x,Word exponent,int base) {
    PhaseMeter::Scope meter(*current_meter,"scale_shift");
    Word shift=s.sub(constant(static_cast<Word>(base)),exponent);
    Bit right=less(shift,constant(0));
    Word amount=s.select(right,s.sub(constant(0),shift),shift);
    Bit out=s.gt(amount,constant(32),false); fail(out);
    amount=s.select(out,constant(0),amount);
    auto ab=s.a2b(amount), xb=s.a2b(x);
    std::array<Bit,128> l{},r{};
    for(std::size_t j=0;j<128;++j) l[j]=r[j]=xb[std::min<std::size_t>(j,58)];
    auto mux=[&](Bit c,Bit a,Bit b){return s.bxor(b,s.band(c,s.bxor(a,b)));};
    for(unsigned stage=0;stage<6;++stage){
      std::size_t distance=std::size_t{1}<<stage;
      auto ll=l,rr=r;
      for(std::size_t j=0;j<128;++j){
        l[j]=mux(ab[stage],j>=distance?ll[j-distance]:s.public_bit(false),ll[j]);
        r[j]=mux(ab[stage],j+distance<128?rr[j+distance]:rr[127],rr[j]);
      }
    }
    std::array<Bit,128> z{};
    for(std::size_t j=0;j<128;++j) z[j]=mux(right,r[j],l[j]);
    Bit bad=s.public_bit(false);
    // Conservative engineering domain |integer| < 2^56 (also covers sign).
    for(std::size_t j=57;j<128;++j) bad=s.bor(bad,s.bxor(z[j],z[56]));
    Bit low_nonzero=s.public_bit(false);
    for(std::size_t j=0;j<56;++j)low_nonzero=s.bor(low_nonzero,z[j]);
    bad=s.bor(bad,s.band(z[56],s.bnot(low_nonzero))); // exclude exactly -2^56
    fail(bad);
    std::array<Bit,59> result{};
    for(std::size_t j=0;j<59;++j) result[j]=s.band(s.bnot(bad),z[j]);
    return s.b2a(result);
  }
  Word mantissa(Word a,Word k,bool sqrt_mode) {
    Word exponent_arg=sqrt_mode?s.mul_public(k,2):k;
    return scale_shift(a,exponent_arg,0);
  }
  Word reciprocal(Word a) {
    Bit bad=s.bor(less(a,constant(one24)),s.gt(a,constant(1280*one24),true));
    fail(bad); a=s.select(bad,constant(one24),a);
    Word k=exponent(a,0,10,false),m=mantissa(a,k,false),r=constant(one24/2);
    for(unsigned i=0;i<6;++i){r=state(r);Word v=product24(m,r);r=product24(r,s.sub(constant(2*one24),v));}
    return scale_shift(state(r),k,0);
  }
  Word exp_negative(Word a) {
    Bit tail=s.gt(a,constant(32<<12),true);
    Bit negative=less(a,constant(0)); fail(negative);
    Word safe=s.select(s.bor(tail,negative),constant(0),a);
    auto bits=s.a2b(safe);
    std::array<Word,3> e{};
    for(unsigned digit=0;digit<3;++digit){
      std::array<Bit,59> word{}; word.fill(s.public_bit(false));
      for(unsigned j=0;j<8;++j) word[j]=bits[digit*8+j];
      e[digit]=s.lookup(s.b2a(word),exponent_tables[digit]);
    }
    Word result=product24(product24(e[0],e[1]),e[2]);
    return s.select(s.bor(tail,negative),constant(0),result);
  }
  Word silu(Word x){
    x=checked_encoded(x,numeric_profile::main_abs_f12);
    Bit neg=less(x,constant(0)); Word a=s.select(neg,s.sub(constant(0),x),x);
    Word e=exp_negative(a),r=reciprocal(s.add(constant(one24),e));
    Word sigmoid=s.select(neg,product24(e,r),r);
    return checked_encoded(s.floor(s.mul(x,sigmoid),24),numeric_profile::main_abs_f12);
  }
  struct GuardedSquares {std::vector<Word> values;Word sum;Bit row_failed;};
  GuardedSquares guarded_square_sum(const std::vector<Word>& input,Word coordinate_cap,Word sum_limit){
    if(input.empty()||!coordinate_cap||!sum_limit||sum_limit>=(Word{1}<<56)||coordinate_cap>sum_limit/coordinate_cap)
      throw std::invalid_argument("BAD_FORMAT");
    GuardedSquares out{input,constant(0),s.public_bit(false)};
    for(auto& v:out.values){
      Bit bad;{PhaseMeter::Scope meter(*current_meter,"coordinate_range");bad=s.bor(s.gt(v,constant(coordinate_cap),true),less(v,constant((-coordinate_cap)&mask)));}
      out.row_failed=s.bor(out.row_failed,bad);
      {PhaseMeter::Scope meter(*current_meter,"coordinate_substitute");v=s.select(bad,constant(0),v);} // bound before squaring
      Word term;{PhaseMeter::Scope meter(*current_meter,"square");term=s.mul(v,v);}Word room=s.sub(constant(sum_limit),out.sum);
      PhaseMeter::Scope meter(*current_meter,"guarded_sum_node");Bit over=s.gt(term,room,true);
      out.row_failed=s.bor(out.row_failed,over);
      out.sum=s.add(out.sum,s.select(over,constant(0),term)); // bound before adding
    }
    // A rejected row must not pair a partial sum with the original vector.
    for(auto& v:out.values)v=s.select(out.row_failed,constant(0),v);
    out.sum=s.select(out.row_failed,constant(0),out.sum);
    fail(out.row_failed);
    return out;
  }
  std::vector<Word> rms(const std::vector<Word>& input,const PrivateMatrix& gamma,Word output_abs_f12=numeric_profile::main_abs_f12,Word input_abs_f12=0) {
    auto x=input; std::size_t d=x.size();
    if((d!=128 && d!=1024 && d!=2048)||gamma.rows!=d||gamma.cols!=1||gamma.fractional_bits!=12) throw std::invalid_argument("BAD_SHAPE");
    if(!input_abs_f12)input_abs_f12=d==128?std::min(numeric_profile::main_abs_f12,numeric_profile::rms_coordinate_cap(d)):numeric_profile::rms_coordinate_cap(d);
    auto guarded=guarded_square_sum(x,input_abs_f12,numeric_profile::rms_sum_limit(d));x=std::move(guarded.values);
    unsigned logd=0;for(auto v=d;v>1;v>>=1)++logd;
    Word a=s.select(guarded.row_failed,constant(one24),s.add(s.floor(guarded.sum,logd),constant(16)));
    Word k=exponent(a,-10,8,true),y=inverse_sqrt(mantissa(a,k,true));
    std::vector<Word> normalized(d);
    for(std::size_t i=0;i<d;++i){
      Word scaled=scale_shift(x[i],k,12);
      scaled=checked(scaled,96,24);
      normalized[i]=s.floor(product24(scaled,y),12);
    }
    // Gamma is private to S. One asymmetric ring-OT cross term per coordinate
    // plus S's local product avoids expanding a diagonal into dense BFV work.
    if(gamma.owned()!=(s.role()==Role::Server))throw std::invalid_argument("BAD_STATE");
    std::vector<Word> weights;
    if(gamma.owned()){weights.resize(d);for(std::size_t i=0;i<d;++i)weights[i]=gamma.at(i,0);}
    std::vector<Word> result;{PhaseMeter::Scope meter(*current_meter,"gamma");result=s.private_diagonal(normalized,weights,12,12);}
    for(auto& v:result)v=checked_encoded(s.floor(v,12),output_abs_f12);
    return result;
  }
  std::pair<std::vector<Word>,Bit> l2(std::vector<Word> x,Bit record_valid,Bit last_valid){
    if(x.size()!=1024) throw std::invalid_argument("BAD_SHAPE");
    auto guarded=guarded_square_sum(x,numeric_profile::l2_coordinate_cap,numeric_profile::l2_sum_limit);x=std::move(guarded.values);
    Bit valid=s.band(s.band(record_valid,last_valid),s.band(s.bnot(guarded.row_failed),s.gt(guarded.sum,constant(0),true)));
    Word a=s.select(valid,guarded.sum,constant(one24));
    Word k=exponent(a,-12,13,true),y=inverse_sqrt(mantissa(a,k,true));
    for(auto& v:x){
      // Invalid/zero branch uses zero vector before normalization.
      v=s.select(valid,v,constant(0));
      Word scaled=scale_shift(v,k,12);
      scaled=checked(scaled,2,24);
      v=checked(s.floor(product24(scaled,y),4),2,20);
    }
    valid=s.band(valid,s.bnot(failed));
    for(auto& v:x)v=s.select(valid,v,constant(0));
    return {x,valid};
  }
  std::vector<Word> softmax(const std::vector<Word>& scores,const std::vector<Bit>& allowed,Bit query_valid){
    if(scores.empty()||scores.size()!=allowed.size()||scores.size()>1280)throw std::invalid_argument("BAD_SHAPE");
    auto x=scores;auto m=allowed;Bit seen=s.public_bit(false);Word maximum=constant(0);
    for(std::size_t i=0;i<x.size();++i){
      m[i]=s.bor(s.band(query_valid,m[i]),s.band(s.bnot(query_valid),s.public_bit(i==0)));
      x[i]=s.select(query_valid,x[i],constant(0));
      Bit take=s.band(m[i],s.bor(s.bnot(seen),s.gt(x[i],maximum,true)));
      maximum=s.select(take,x[i],maximum);seen=s.bor(seen,m[i]);
    }
    // Valid self-attention must contain a valid causal key; keep failure secret.
    fail(s.bnot(seen));
    m[0]=s.bor(m[0],s.bnot(seen));
    x[0]=s.select(seen,x[0],maximum);
    Word denom=constant(0);std::vector<Word> e(x.size());
    for(std::size_t i=0;i<x.size();++i){
      Word a=s.select(m[i],s.sub(maximum,x[i]),constant(0));
      e[i]=s.select(m[i],exp_negative(a),constant(0));denom=s.add(denom,e[i]);
    }
    Word inv=reciprocal(denom);
    for(auto& v:e){v=product24(v,inv);Bit bad=s.bor(less(v,constant(0)),s.gt(v,constant(one24),true));fail(bad);{PhaseMeter::Scope meter(*current_meter,"coordinate_substitute");v=s.select(bad,constant(0),v);}}
    return e; // remains f24; query mask is applied AFTER PV
  }
};
}
