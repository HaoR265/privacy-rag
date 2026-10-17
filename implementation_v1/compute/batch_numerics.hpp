#pragma once
#include "numeric_profile.hpp"
#include "prag/batch.hpp"
#include "secure_tensor.hpp"
#include "local_shares.hpp"
#include <algorithm>
#include <exception>
#include <openssl/crypto.h>
namespace prag::model {
// Public stage callback contains no data-dependent metadata. Production may
// leave it empty; diagnostic callers measure authenticated transcript counters.
// Only new logical-attention staging uses this owner. Move replacement erases
// the previous allocation before releasing it; failed stages erase on unwind.
template<class T> class LogicalBuffer : public std::vector<T> {
  using Base=std::vector<T>;
  void wipe() noexcept { if(!this->empty())OPENSSL_cleanse(this->data(),this->size()*sizeof(T)); }
 public:
  using Base::Base;
  LogicalBuffer()=default;
  LogicalBuffer(Base&& x):Base(std::move(x)){}
  LogicalBuffer(const LogicalBuffer&)=delete;
  LogicalBuffer& operator=(const LogicalBuffer&)=delete;
  LogicalBuffer& operator=(Base&& x){wipe();Base::operator=(std::move(x));return *this;}
  ~LogicalBuffer(){wipe();}
  Base release(){return std::move(static_cast<Base&>(*this));}
};
using NumericPhase = std::function<void(const char *, bool)>;
class BatchNumerics {
  SecureOps &s;
  BatchOps batch;
  NumericPhase phase;
  Word logical_matrix_sequence = 0;
  LocalShares* local;
  std::vector<uint8_t> pending_failures;
  struct Stage {
    NumericPhase &cb;
    const char *name;
    Stage(NumericPhase &c, const char *n) : cb(c), name(n) {
      if (cb)
        cb(name, true);
    }
    ~Stage() noexcept(false) {
      if(!cb)return;
      if(std::uncaught_exceptions()){try{cb(name,false);}catch(...){}}
      else cb(name,false);
    }
  };
  void note(const std::vector<uint8_t> &bad) {
    if (bad.empty())
      return;
    // Compress each fixed public stage to one XOR share. No secret early exit.
    auto flag = batch.run({BatchCode::Or, 1, bad.size(), {}}, {}, bad);
    pending_failures.push_back(flag.bits[0]);
  }
  std::vector<Word> broadcast(const std::vector<Word> &x, size_t d) {
    std::vector<Word> r;
    for (auto v : x)
      r.insert(r.end(), d, v);
    return r;
  }

  // Only called by silu after exp_negative: not a general reciprocal API.
  // For every canonical input share, silu checks X before forming |X|.
  // The guarded input and bounded public tables prove D in [1,2] at f24.
  std::vector<Word> reciprocal_silu(const std::vector<Word>& input) {
    auto prepared=batch.run({BatchCode::ReciprocalSiLUPrepare,1,input.size(),{}},input);
    note(prepared.bits);
    const size_t count=input.size();
    std::vector<Word> m(prepared.words.begin(),prepared.words.begin()+count),
        k(prepared.words.begin()+count,prepared.words.end()),
        r(count,constant(one24/2));
    for(unsigned round=0;round<6;++round) {
      r=state(r); // Keep every original pre-multiplication state guard.
      if(round==0) {
        // R0=2^23 is public. These are exactly the two original Tr24 nodes:
        // Tr24(M*2^23)=floor(M/2), Tr24(2^23*(2*one24-V))=floor((2*one24-V)/2).
        auto v=floor(m,1);
        for(auto& x:v)x=s.sub(constant(2*one24),x);
        r=floor(v,1);
      } else {
        auto v=product24(m,r);
        for(auto& x:v)x=s.sub(constant(2*one24),x);
        r=product24(r,v);
      }
    }
    r.insert(r.end(),k.begin(),k.end());
    auto restored=batch.run({BatchCode::ReciprocalSiLURestore,1,count,{}},r);
    note(restored.bits); // Original final state and exact scale-restoration guard.
    return restored.words;
  }

public:
  explicit BatchNumerics(SecureOps &ops, NumericPhase callback = {}, LocalShares* gpu=nullptr)
      : s(ops), batch(ops), phase(std::move(callback)),local(gpu) {}
  BatchOps &ops() { return batch; }
  Word constant(Word x) { return s.public_word(x & MASK); }
  Bit take_failure() {
    if (pending_failures.empty())
      return s.public_bit(false);
    auto f = batch
                 .run({BatchCode::Or, 1, pending_failures.size(), {}}, {},
                      pending_failures)
                 .bits[0];
    pending_failures.clear();
    return bool(f);
  }
  std::vector<Word> checked(const std::vector<Word> &x, Word lower, Word upper,
                            Word dummy = 0) {
    auto out = batch.run(
        {BatchCode::Range, 1, x.size(), {lower & MASK, upper, dummy & MASK}},
        x);
    note(out.bits);
    return out.words;
  }
  // Exact signed ring59 floor12 followed by the original range guard.
  // One input conversion and one fresh output mask; failure remains secret.
  std::vector<Word> floor12_range(const std::vector<Word>& x, Word lower,
                                  Word upper, Word dummy=0) {
    auto out=batch.run({BatchCode::Floor12Range,1,x.size(),
                       {lower & MASK,upper,dummy & MASK}},x);
    note(out.bits); return out.words;
  }
  std::vector<Word> floor(const std::vector<Word> &x, unsigned shift) {
    return batch.run({BatchCode::Floor, 1, x.size(), {shift, 0, 0}}, x).words;
  }
  std::vector<Word> select(const std::vector<uint8_t> &b,
                           const std::vector<Word> &x,
                           const std::vector<Word> &y) {
    auto both = x;
    both.insert(both.end(), y.begin(), y.end());
    return batch.run({BatchCode::Select, 1, b.size(), {}}, both, b).words;
  }
  std::vector<Word> shift(const std::vector<Word> &x,
                          const std::vector<Word> &exponent, size_t columns,
                          int base) {
    auto all = x;
    all.insert(all.end(), exponent.begin(), exponent.end());
    auto result = batch.run(
        {BatchCode::Shift, exponent.size(), columns, {Word(base) & MASK, 0, 0}},
        all);
    note(result.bits);
    return result.words;
  }
  std::vector<Word> state(const std::vector<Word> &x) {
    return checked(x, 0, 2 * one24, one24 / 2);
  }
  std::vector<Word> product24(const std::vector<Word> &x,
                              const std::vector<Word> &y) {
    return floor(batch.multiply(x, y), 24);
  }
  std::vector<Word> inverse_sqrt(const std::vector<Word> &m) {
    auto y = std::vector<Word>(m.size(), constant(one24 / 2));
    for (unsigned round = 0; round < 6; ++round) {
      Stage step(phase, "NR_round");
      {
        Stage guard(phase, "NR_state");
        y = state(y);
      }
      std::vector<Word> u, v, w;
      {
        Stage p(phase, "NR_y_square_floor24");
        u = product24(y, y);
      }
      {
        Stage p(phase, "NR_multiply_m_floor24");
        v = product24(m, u);
      }
      for (auto &x : v)
        x = s.sub(constant(3 * one24), x);
      {
        Stage p(phase, "NR_correction_floor24");
        w = product24(y, v);
      }
      {
        Stage p(phase, "NR_half_floor1");
        y = floor(w, 1);
      }
    }
    Stage p(phase, "NR_final_state");
    return state(y);
  }
  std::vector<Word> reciprocal(const std::vector<Word> &input) {
    auto a = checked(input, one24, 1280 * one24, one24);
    auto split = batch.run({BatchCode::Exponent, 1, a.size(), {0, 10, 0}}, a);
    note(split.bits);
    const size_t count = a.size();
    a.assign(split.words.begin(), split.words.begin() + count);
    std::vector<Word> k(split.words.begin() + count, split.words.end());
    auto m = shift(a, k, 1, 0),
         r = std::vector<Word>(count, constant(one24 / 2));
    for (unsigned round = 0; round < 6; ++round) {
      r = state(r);
      auto v = product24(m, r);
      for (auto &x : v)
        x = s.sub(constant(2 * one24), x);
      r = product24(r, v);
    }
    return shift(state(r), k, 1, 0);
  }
  std::vector<Word>
  exp_negative(const std::vector<Word> &x,
               const std::array<std::vector<Word>, 3> &tables) {
    for (const auto &table : tables) {
      if (table.size() != 256)
        throw std::runtime_error("E_EXP_TABLE_SHAPE");
      for (auto v : table)
        if (v > one24)
          throw std::runtime_error("E_EXP_TABLE_RANGE");
    }
    auto prep = batch.exp_lookup(x,tables);
    const size_t count = x.size();
    note(std::vector<uint8_t>(prep.bits.begin() + count, prep.bits.end()));
    std::array<std::vector<Word>, 3> values;
    for (size_t digit = 0; digit < 3; ++digit)
      values[digit] = std::vector<Word>(prep.words.begin() + digit * count,
                                      prep.words.begin() + (digit + 1) * count);
    auto out = product24(product24(values[0], values[1]), values[2]);
    return select(
        std::vector<uint8_t>(prep.bits.begin(), prep.bits.begin() + count),
        std::vector<Word>(count, constant(0)), out);
  }
  std::vector<Word> silu(const std::vector<Word>& input,
                         const std::array<std::vector<Word>,3>& tables) {
    Stage stage(phase,"SiLU");
    static_assert(numeric_profile::main_abs_f12==8388608);
    auto result=batch.silu_composite(input,tables);note(result.bits);return result.words;
  }

  std::vector<Word> rope_coefficients(const std::vector<Word>& positions,
                                      const std::vector<Word>& table) {
    Stage stage(phase,"RoPE_shared_public_coefficients");
    auto out=batch.lookup_rows(positions,table,128);note(out.bits);return out.words;
  }
  std::vector<Word> softmax(const std::vector<Word> &scores, size_t rows,
                            size_t columns, const std::vector<uint8_t> &allowed,
                            const std::vector<uint8_t> &query_valid,
                            const std::array<std::vector<Word>, 3> &tables) {
    try {
      if (!rows || rows > 16 || !columns || columns > 1280 ||
          scores.size() != rows * columns || allowed.size() != scores.size() ||
          query_valid.size() != rows)
        throw std::runtime_error("E_SOFTMAX_SHAPE");
      auto bits = allowed;
      bits.insert(bits.end(), query_valid.begin(), query_valid.end());
      BatchResult prepared;
      {
        Stage p(phase, "softmax_row_max_mask");
        prepared = batch.run({BatchCode::SoftmaxPrepare, rows, columns, {}},
                             scores, bits);
        note(std::vector<uint8_t>(prepared.bits.begin() + scores.size(),
                                  prepared.bits.end()));
      }
      std::vector<Word> e;
      {
        Stage p(phase, "softmax_exp_LUT");
        e = exp_negative(prepared.words, tables);
        prepared.bits.resize(scores.size());
        e = select(prepared.bits, e, std::vector<Word>(e.size(), constant(0)));
      }
      std::vector<Word> denom(rows,constant(0));
      if(local)denom=local->sum_rows(e,columns);
      else for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col)denom[row]=s.add(denom[row],e[row*columns+col]);
      std::vector<Word> inv;
      {
        Stage p(phase, "softmax_reciprocal_six_NR");
        inv = reciprocal(denom);
      }
      {
        Stage p(phase, "softmax_multiply_floor24");
        e = product24(e, broadcast(inv, columns));
      }
      {
        Stage p(phase, "softmax_probability_range");
        e = checked(e, 0, one24);
      }
      return e;
    } catch (...) {
      s.abort_session();
      throw;
    }
  }
  std::vector<Word> attention(const std::vector<Word> &q,
                              const std::vector<Word> &k,
                              const std::vector<Word> &v, size_t rows,
                              size_t keys, const std::vector<uint8_t> &allowed,
                              const std::vector<uint8_t> &valid,
                              numeric_profile::QKBounds bounds,
                              Word attention_scale,
                              const std::array<std::vector<Word>, 3> &tables) {
    try {
      if (!rows || rows > 16 || !keys || keys > 1280 ||
          q.size() != rows * 128 || k.size() != keys * 128 ||
          v.size() != keys * 128 || valid.size() != rows ||
          allowed.size() != rows * keys)
        throw std::runtime_error("E_ATTENTION_SHAPE");
      if (bounds.q_rope > 1548293 || bounds.k_rope > 12648476 ||
          attention_scale != 92681)
        throw std::runtime_error("E_ATTENTION_PROFILE");
      std::vector<Word> kt(128 * keys);
      if(local){std::vector<size_t> map(kt.size());for(size_t c=0;c<128;++c)for(size_t j=0;j<keys;++j)map[c*keys+j]=j*128+c;kt=local->layout(k,map);}
      else for(size_t j=0;j<keys;++j)for(size_t c=0;c<128;++c)kt[c*keys+j]=k[j*128+c];
      std::vector<Word> scores;
      {
        Stage p(phase, "shared_QK");
        scores = batch.matrix(q, kt, rows, 128, keys, 12, 12,
                              128 * bounds.q_rope * bounds.k_rope, phase);
      }
      {
        Stage p(phase, "QK_floor12_scale_floor20");
        scores = floor(scores, 12);
        if(local)scores=local->scale(scores,attention_scale);
        else for(auto&x:scores)x=s.mul_public(x,attention_scale);
        scores = floor(scores, 20);
      }
      auto probabilities = softmax(scores, rows, keys, allowed, valid, tables);
      std::vector<Word> out;
      {
        Stage p(phase, "shared_PV");
        out = batch.matrix(probabilities, v, rows, keys, 128, 24, 12,
                           4 * one24 * numeric_profile::main_abs_f12, phase);
      }
      {
        Stage p(phase, "PV_floor24_query_mask_range");
        out = floor(out, 24);
        std::vector<uint8_t> row_valid;
        for (auto b : valid)
          row_valid.insert(row_valid.end(), 128, b);
        out =
            select(row_valid, out, std::vector<Word>(out.size(), constant(0)));
        out = checked(out, (-numeric_profile::main_abs_f12) & MASK,
                      numeric_profile::main_abs_f12);
      }
      return out;
    } catch (...) {
      s.abort_session();
      throw;
    }
  }
  std::vector<Word> attention_logical(const std::vector<Word> &q,
                              const std::vector<Word> &k,
                              const std::vector<Word> &v, size_t rows,
                              size_t keys, const std::vector<uint8_t> &allowed,
                              const std::vector<uint8_t> &valid,
                              numeric_profile::QKBounds bounds,
                              Word attention_scale,
                              const std::array<std::vector<Word>, 3> &tables,
                              Word layer, Word head, Word step) {
    try {
      if(rows==1)return attention(q,k,v,rows,keys,allowed,valid,bounds,attention_scale,tables);
      if (!rows || rows > 1024 || !keys || keys > 1280 ||
          q.size() != rows * 128 || k.size() != keys * 128 ||
          v.size() != keys * 128 || valid.size() != rows ||
          allowed.size() != rows * keys)
        throw std::runtime_error("E_ATTENTION_SHAPE");
      if (bounds.q_rope > 1548293 || bounds.k_rope > 12648476 ||
          attention_scale != 92681)
        throw std::runtime_error("E_ATTENTION_PROFILE");
      LogicalBuffer<Word> kt(128 * keys);
      if(local){std::vector<size_t> map(kt.size());for(size_t c=0;c<128;++c)for(size_t j=0;j<keys;++j)map[c*keys+j]=j*128+c;kt=local->layout(k,map);}
      else for(size_t j=0;j<keys;++j)for(size_t c=0;c<128;++c)kt[c*keys+j]=k[j*128+c];
      LogicalBuffer<Word> scores;
      {
        Stage p(phase, "shared_QK");
        scores = batch.logical_matrix(q, kt, rows, 128, keys, 12, 12,
                              128 * bounds.q_rope * bounds.k_rope,
                              {++logical_matrix_sequence,layer,head,step,1},phase);
      }
      LogicalBuffer<Word> probabilities; probabilities.reserve(rows*keys);
      // Numerical circuit bounds are unchanged: at most 16 rows per call.
      for(size_t first=0;first<rows;first+=16) {
        const size_t count=std::min(size_t(16),rows-first);
        LogicalBuffer<Word> block(scores.begin()+first*keys,scores.begin()+(first+count)*keys);
        {
          Stage p(phase,"QK_floor12_scale_floor20");
          block=floor(block,12);
          if(local)block=local->scale(block,attention_scale);
          else for(auto& x:block)x=s.mul_public(x,attention_scale);
          block=floor(block,20);
        }
        LogicalBuffer<uint8_t> block_allowed(allowed.begin()+first*keys,allowed.begin()+(first+count)*keys);
        LogicalBuffer<uint8_t> block_valid(valid.begin()+first,valid.begin()+first+count);
        LogicalBuffer<Word> prob=softmax(block,count,keys,block_allowed,block_valid,tables);
        probabilities.insert(probabilities.end(),prob.begin(),prob.end());
      }
      LogicalBuffer<Word> out;
      {
        Stage p(phase, "shared_PV");
        out = batch.logical_matrix(probabilities, v, rows, keys, 128, 24, 12,
                           4 * one24 * numeric_profile::main_abs_f12,
                           {++logical_matrix_sequence,layer,head,step,2},phase);
      }
      for(size_t first=0;first<rows;first+=16) {
        const size_t count=std::min(size_t(16),rows-first);
        Stage p(phase,"PV_floor24_query_mask_range");
        LogicalBuffer<Word> block(out.begin()+first*128,out.begin()+(first+count)*128);
        block=floor(block,24);
        LogicalBuffer<uint8_t> row_valid;row_valid.reserve(count*128);
        for(size_t r=first;r<first+count;++r)row_valid.insert(row_valid.end(),128,valid[r]);
        block=select(row_valid,block,std::vector<Word>(block.size(),constant(0)));
        block=checked(block,(-numeric_profile::main_abs_f12)&MASK,numeric_profile::main_abs_f12);
        std::copy(block.begin(),block.end(),out.begin()+first*128);
      }
      return out.release();
    } catch (...) {
      s.abort_session();
      throw;
    }
  }
  std::vector<Word> rms(const std::vector<Word> &input, size_t rows, size_t d,
                        const PrivateMatrix &gamma,
                        Word output_cap = numeric_profile::main_abs_f12,
                        Word input_cap = 0) {
    try {
      if (!rows || rows > 16 || (d != 128 && d != 1024 && d != 2048) ||
          input.size() != rows * d || gamma.rows != d || gamma.cols != 1 ||
          gamma.fractional_bits != 12 ||
          gamma.owned() != (s.role() == Role::Server) || !output_cap ||
          output_cap >= (Word{1} << 56))
        throw std::runtime_error("E_RMS_SHAPE");
      if (!input_cap)
        input_cap = d == 128 ? std::min(numeric_profile::main_abs_f12,
                                        numeric_profile::rms_coordinate_cap(d))
                             : numeric_profile::rms_coordinate_cap(d);
      auto limit = numeric_profile::rms_sum_limit(d);
      if (!input_cap || input_cap > limit / input_cap)
        throw std::runtime_error("E_RMS_COORDINATE_CAP");
      BatchResult guard;
      std::vector<Word> x, squares;
      {
        Stage p(phase, "coordinate_range");
        guard = batch.run(
            {BatchCode::Range, rows, d, {(-input_cap) & MASK, input_cap, 0}},
            input);
        x = std::move(guard.words);
      }
      {
        Stage p(phase, "square");
        squares = batch.multiply(x, x);
      }
      BatchResult sums;
      {
        Stage p(phase, "guarded_row_sum");
        sums = batch.run({BatchCode::GuardedSum, rows, d, {limit, 0, 0}},
                         squares, guard.bits);
        note(sums.bits);
      }
      {
        Stage p(phase, "whole_row_substitute");
        std::vector<uint8_t> rowbad;
        for (auto b : sums.bits)
          rowbad.insert(rowbad.end(), d, b);
        x = select(rowbad, std::vector<Word>(x.size(), constant(0)), x);
      }
      std::vector<Word> a;
      {
        Stage p(phase, "mean_floor_epsilon");
        unsigned logd = 0;
        for (auto v = d; v > 1; v >>= 1)
          ++logd;
        a = floor(sums.words, logd);
        for (auto &v : a)
          v = s.add(v, constant(16));
        a = select(sums.bits, std::vector<Word>(rows, constant(one24)), a);
      }
      std::vector<Word> k, m;
      {
        Stage p(phase, "rsqrt_exponent");
        auto result = batch.run(
            {BatchCode::Exponent, 1, rows, {Word(-10) & MASK, 8, 1}}, a);
        note(result.bits);
        a.assign(result.words.begin(), result.words.begin() + rows);
        k.assign(result.words.begin() + rows, result.words.end());
      }
      {
        Stage p(phase, "rsqrt_mantissa");
        auto twice = k;
        for (auto &v : twice)
          v = s.mul_public(v, 2);
        m = shift(a, twice, 1, 0);
      }
      std::vector<Word> y;
      {
        Stage p(phase, "six_NR");
        y = inverse_sqrt(m);
      }
      {
        Stage p(phase, "input_scale");
        x = shift(x, k, d, 12);
      }
      {
        Stage p(phase, "scaled_range");
        x = checked(x, Word(-int64_t(96 * one24)) & MASK, 96 * one24);
      }
      {
        Stage p(phase, "normalize_multiply_floor24");
        x = product24(x, broadcast(y, d));
      }
      {
        Stage p(phase, "normalize_floor12");
        x = floor(x, 12);
      }
      {
        Stage p(phase, "private_gamma");
        std::vector<Word> out;
        out.reserve(x.size());
        for (size_t first = 0; first < x.size(); first += 8192) {
          size_t count = std::min(size_t(8192), x.size() - first);
          std::vector<Word> block(x.begin() + first, x.begin() + first + count),
              weights;
          if (gamma.owned()) {
            weights.resize(count);
            for (size_t j = 0; j < count; ++j)
              weights[j] = gamma.at((first + j) % d, 0);
          }
          auto result = s.private_diagonal(block, weights, 12, 12);
          out.insert(out.end(), result.begin(), result.end());
        }
        x = std::move(out);
      }
      {
        Stage p(phase, "gamma_floor12");
        x = floor(x, 12);
      }
      {
        Stage p(phase, "output_range");
        x = checked(x, (-output_cap) & MASK, output_cap);
      }
      return x;
    } catch (...) {
      s.abort_session();
      throw;
    }
  }
};
} // namespace prag::model
