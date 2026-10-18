#pragma once
#include "secure_tensor.hpp"
#include "numeric_profile.hpp"
#include "batch_numerics.hpp"
#include "local_shares.hpp"
#include <array>
#include <algorithm>
#include <openssl/crypto.h>

namespace prag::model {
// All data-dependent choices are calls to SecureOps, never host-language branches.
// Loops depend only on frozen dimensions. failure is a local XOR share.
class Numerics {
  SecureOps& s;
  Bit failed;
  std::array<std::vector<Word>,3> exponent_tables;
  std::unique_ptr<BatchNumerics> batch_;
  NumericPhase phase_;
  LocalShares* local_;
 public:
  Numerics(SecureOps& ops,std::array<std::vector<Word>,3> tables,NumericPhase phase={},LocalShares* local=nullptr):s(ops),failed(s.public_bit(false)),exponent_tables(std::move(tables)),phase_(std::move(phase)),local_(local) {
    for(const auto& t:exponent_tables) if(t.size()!=256) throw std::invalid_argument("BAD_SHAPE");
  }
  SecureOps& ops(){return s;}
  BatchNumerics& batches(){if(!batch_)batch_=std::make_unique<BatchNumerics>(s,phase_,local_);return *batch_;}
  std::vector<Word> attention(const std::vector<Word>&q,const std::vector<Word>&k,const std::vector<Word>&v,size_t rows,size_t keys,const std::vector<uint8_t>&allowed,const std::vector<uint8_t>&valid,numeric_profile::QKBounds bounds,Word attention_scale){auto out=batches().attention(q,k,v,rows,keys,allowed,valid,bounds,attention_scale,exponent_tables);fail(batches().take_failure());return out;}
  std::vector<Word> attention_logical(const std::vector<Word>&q,const std::vector<Word>&k,const std::vector<Word>&v,size_t rows,size_t keys,const std::vector<uint8_t>&allowed,const std::vector<uint8_t>&valid,numeric_profile::QKBounds bounds,Word attention_scale,Word layer,Word head,Word step){auto out=batches().attention_logical(q,k,v,rows,keys,allowed,valid,bounds,attention_scale,exponent_tables,layer,head,step);fail(batches().take_failure());return out;}
  LocalShares& local(){if(!local_)throw std::runtime_error("E_HYBRID_GPU_REQUIRED");return *local_;}
  void stage(const char* name,bool start){if(phase_)phase_(name,start);}
  template<class F> std::vector<Word> chunks(const std::vector<Word>& x,F f){
    if(x.empty())throw std::invalid_argument("BAD_SHAPE");
    std::vector<Word> result;result.reserve(x.size());
    for(size_t first=0;first<x.size();first+=4096){
      size_t count=std::min(size_t(4096),x.size()-first);
      auto y=f(std::vector<Word>(x.begin()+first,x.begin()+first+count),first);
      result.insert(result.end(),y.begin(),y.end());
    }fail(batches().take_failure());return result;
  }
  std::vector<Word> checked_many(const std::vector<Word>&x,Word cap){
    if(!cap||cap>=(Word{1}<<56))throw std::invalid_argument("BAD_FORMAT");
    return chunks(x,[&](auto block,size_t){return batches().checked(block,(-cap)&mask,cap);});
  }
  std::vector<Word> floor_many(const std::vector<Word>&x,unsigned shift){
    return chunks(x,[&](auto block,size_t){return batches().floor(block,shift);});
  }
  std::vector<Word> multiply_many(const std::vector<Word>&x,const std::vector<Word>&y){
    if(x.size()!=y.size())throw std::invalid_argument("BAD_SHAPE");
    return chunks(x,[&](auto block,size_t first){return batches().ops().multiply(block,std::vector<Word>(y.begin()+first,y.begin()+first+block.size()));});
  }
  std::vector<uint8_t> compare_many(const std::vector<Word>&x,const std::vector<Word>&y,bool sign=true){
    if(x.empty()||x.size()!=y.size())throw std::invalid_argument("BAD_SHAPE");
    std::vector<uint8_t> result;
    for(size_t first=0;first<x.size();first+=4096){size_t count=std::min(size_t(4096),x.size()-first);std::vector<Word> both(x.begin()+first,x.begin()+first+count);both.insert(both.end(),y.begin()+first,y.begin()+first+count);auto bits=batches().ops().run({BatchCode::Compare,1,count,{Word(sign),0,0}},both).bits;result.insert(result.end(),bits.begin(),bits.end());}return result;
  }
  std::vector<Word> select_many(const std::vector<uint8_t>&b,const std::vector<Word>&x,const std::vector<Word>&y){
    if(b.size()!=x.size()||x.size()!=y.size())throw std::invalid_argument("BAD_SHAPE");
    return chunks(x,[&](auto block,size_t first){return batches().select(std::vector<uint8_t>(b.begin()+first,b.begin()+first+block.size()),block,std::vector<Word>(y.begin()+first,y.begin()+first+block.size()));});
  }
  std::vector<Word> swiglu_many(const std::vector<Word>&gate,const std::vector<Word>&up){
    if(gate.size()!=up.size())throw std::invalid_argument("BAD_SHAPE");
    return chunks(gate,[&](auto block,size_t first){auto silu=batches().silu(block,exponent_tables);auto raw=batches().ops().multiply(silu,std::vector<Word>(up.begin()+first,up.begin()+first+block.size()));return batches().floor12_range(raw,(-numeric_profile::swiglu_joint_abs_f12)&mask,numeric_profile::swiglu_joint_abs_f12);});
  }
  // Public row windows, never across layers or Decode forward calls.
  // Upstream projection supplies guarded gate/up values; retain SiLU guard.
  std::vector<std::vector<Word>> swiglu_rows(
      const std::vector<std::vector<Word>>& gate,
      const std::vector<std::vector<Word>>& up) {
    if(gate.empty()||gate.size()>1024||gate.size()!=up.size())
      throw std::invalid_argument("BAD_SHAPE");
    const size_t width=gate.front().size();
    if(width!=3072&&width!=6144)throw std::invalid_argument("BAD_SHAPE");
    for(size_t row=0;row<gate.size();++row)
      if(gate[row].size()!=width||up[row].size()!=width)
        throw std::invalid_argument("BAD_SHAPE");
    std::vector<std::vector<Word>> out;
    struct OutputGuard {
      std::vector<std::vector<Word>>& rows;SecureOps& op;bool success=false;
      ~OutputGuard(){if(!success){for(auto& row:rows)if(!row.empty())OPENSSL_cleanse(row.data(),row.size()*sizeof(Word));op.abort_session();}}
    } guard{out,s};
    out.reserve(gate.size());
    for(size_t first=0;first<gate.size();first+=16) {
      size_t rows=std::min(size_t(16),gate.size()-first);
      std::vector<Word> g,u,joint;
      // All host staging holds this party's shares only. Clear on success/failure.
      struct Clear { std::vector<Word>&v;~Clear(){if(!v.empty())OPENSSL_cleanse(v.data(),v.size()*sizeof(Word));} } cg{g},cu{u},cj{joint};
      g.reserve(rows*width);u.reserve(rows*width);
      for(size_t row=first;row<first+rows;++row){
        g.insert(g.end(),gate[row].begin(),gate[row].end());
        u.insert(u.end(),up[row].begin(),up[row].end());
      }
      joint=swiglu_many(g,u); // <=4096 physical blocks and one secret OR per window.
      for(size_t row=0;row<rows;++row)
        out.emplace_back(joint.begin()+row*width,joint.begin()+(row+1)*width);
    }
    guard.success=true;return out;
  }
  std::vector<Word> rope_coefficients(const std::vector<Word>&positions,const std::vector<Word>&table){
    std::vector<Word> result;
    for(size_t i=0;i<positions.size();i+=16){auto count=std::min(size_t(16),positions.size()-i);auto y=batches().rope_coefficients(std::vector<Word>(positions.begin()+i,positions.begin()+i+count),table);result.insert(result.end(),y.begin(),y.end());fail(batches().take_failure());}return result;
  }
  std::vector<std::vector<Word>> rope_many(const std::vector<std::vector<Word>>&heads,const std::vector<Word>&coeff,size_t count,Word cap){
    if(heads.empty()||heads.size()%count||coeff.size()!=heads.size()/count*128)throw std::invalid_argument("BAD_SHAPE");
    std::vector<Word> a,b,c,t;
    for(size_t h=0;h<heads.size();++h){if(heads[h].size()!=128)throw std::invalid_argument("BAD_SHAPE");for(size_t j=0;j<64;++j){a.push_back(heads[h][j]);b.push_back(heads[h][j+64]);c.push_back(coeff[(h/count)*128+j]);t.push_back(coeff[(h/count)*128+j+64]);}}
    auto first=local().add(multiply_many(a,c),local().scale(multiply_many(b,t),mask));
    auto second=local().add(multiply_many(b,c),multiply_many(a,t));
    first=checked_many(floor_many(first,20),cap);second=checked_many(floor_many(second,20),cap);
    std::vector<std::vector<Word>> out(heads.size(),std::vector<Word>(128));
    for(size_t h=0;h<heads.size();++h){std::copy_n(first.begin()+h*64,64,out[h].begin());std::copy_n(second.begin()+h*64,64,out[h].begin()+64);}return out;
  }
  Bit failure() const {return failed;}
  void fail(Bit b){failed=s.bor(failed,b);}
  Word constant(Word x){return s.public_word(x&mask);}
  Bit nonzero(Word x){return s.bnot(s.equal(x,constant(0)));}
  Bit less(Word x,Word y){return s.gt(y,x,true);}
  Word checked_encoded(Word x,Word lim,Word dummy=0) {
    if(lim>=(Word{1}<<56))throw std::invalid_argument("BAD_FORMAT");
    Bit bad=s.bor(s.gt(x,constant(lim),true),less(x,constant((-lim)&mask)));
    fail(bad); return s.select(bad,constant(dummy),x);
  }
  Word checked(Word x,Word magnitude,unsigned f,Word dummy=0) {
    if(f>=56||magnitude>=((Word{1}<<56)>>f))throw std::invalid_argument("BAD_FORMAT");
    return checked_encoded(x,magnitude<<f,dummy);
  }
  Word state(Word x) {
    Bit bad=s.bor(less(x,constant(0)),s.gt(x,constant(2*one24),true));
    fail(bad); return s.select(bad,constant(one24/2),x);
  }
  Word product24(Word a,Word b){return s.floor(s.mul(a,b),24);}
  Word inverse_sqrt(Word mantissa) {
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
      Bit bad=s.bor(s.gt(v,constant(coordinate_cap),true),less(v,constant((-coordinate_cap)&mask)));
      out.row_failed=s.bor(out.row_failed,bad);
      v=s.select(bad,constant(0),v); // bound before squaring
      Word term=s.mul(v,v),room=s.sub(constant(sum_limit),out.sum);
      Bit over=s.gt(term,room,true);
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
    auto result=batches().rms(input,1,input.size(),gamma,output_abs_f12,input_abs_f12);
    fail(batches().take_failure());return result;
  }
  std::vector<std::vector<Word>> rms_many(const std::vector<std::vector<Word>>& input,const PrivateMatrix& gamma,Word output_abs_f12=numeric_profile::main_abs_f12,Word input_abs_f12=0){
    if(input.empty())throw std::invalid_argument("BAD_SHAPE");
    const auto d=input.front().size();for(const auto& row:input)if(row.size()!=d)throw std::invalid_argument("BAD_SHAPE");
    std::vector<std::vector<Word>> result;result.reserve(input.size());
    for(size_t first=0;first<input.size();first+=16){
      const auto count=std::min(size_t(16),input.size()-first);std::vector<Word> flat;flat.reserve(count*d);
      for(size_t i=0;i<count;++i)flat.insert(flat.end(),input[first+i].begin(),input[first+i].end());
      auto out=batches().rms(flat,count,d,gamma,output_abs_f12,input_abs_f12);
      for(size_t i=0;i<count;++i)result.emplace_back(out.begin()+i*d,out.begin()+(i+1)*d);
      fail(batches().take_failure());
    }return result;
  }
  std::pair<std::vector<Word>,Bit> l2(std::vector<Word> x,Bit record_valid,Bit last_valid){
    if(x.size()!=1024)throw std::invalid_argument("BAD_SHAPE");
    auto& batch=batches();Word cap=numeric_profile::l2_coordinate_cap;
    auto guarded=batch.ops().run({BatchCode::Range,1,1024,{(-cap)&mask,cap,0}},x);
    auto squared=batch.ops().multiply(guarded.words,guarded.words);
    auto sum=batch.ops().run({BatchCode::GuardedSum,1,1024,{numeric_profile::l2_sum_limit,0,0}},squared,guarded.bits);
    fail(sum.bits[0]);
    Bit valid=s.band(s.band(record_valid,last_valid),s.band(s.bnot(sum.bits[0]),s.gt(sum.words[0],constant(0),true)));
    std::vector<Word> a{s.select(valid,sum.words[0],constant(one24))};
    auto split=batch.ops().run({BatchCode::Exponent,1,1,{Word(-12)&mask,13,1}},a);fail(split.bits[0]);
    std::vector<Word> k{split.words[1]},twice{s.mul_public(k[0],2)};
    auto y=batch.inverse_sqrt(batch.shift({split.words[0]},twice,1,0));
    x=batch.select(std::vector<uint8_t>(1024,valid),guarded.words,std::vector<Word>(1024,constant(0)));
    x=batch.shift(x,k,1024,12);x=batch.checked(x,(-2*one24)&mask,2*one24);
    x=batch.floor(batch.product24(x,std::vector<Word>(1024,y[0])),4);
    x=batch.checked(x,Word(-int64_t(2<<20))&mask,2<<20);
    fail(batch.take_failure());valid=s.band(valid,s.bnot(failed));
    x=batch.select(std::vector<uint8_t>(1024,valid),x,std::vector<Word>(1024,constant(0)));
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
    for(auto& v:e){v=product24(v,inv);Bit bad=s.bor(less(v,constant(0)),s.gt(v,constant(one24),true));fail(bad);v=s.select(bad,constant(0),v);}
    return e; // remains f24; query mask is applied AFTER PV
  }
};
}
