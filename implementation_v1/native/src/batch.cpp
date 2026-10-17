#include "prag/silu_graph.hpp"
#include "topology_count_cache.hpp"
#include "prag/batch.hpp"
#include "GC/halfgate_eva.h"
#include "GC/halfgate_gen.h"
#include "OT/iknp.h"
#include "prag/gc_batch_channel.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <openssl/crypto.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <stdexcept>
#include <type_traits>
namespace prag {
namespace {
using Label = sci::block128;
using Wires = std::vector<Label>;
// Public metadata exists only in the Exponent instantiation. Other circuits
// retain the original 16-byte raw labels and do not convert their input arrays.
struct PublicWire {
  Label label = _mm_setzero_si128();
  int8_t public_value = -1;
  PublicWire() = default;
  explicit PublicWire(Label v) : label(v) {}
  PublicWire(Label v, int8_t bit) : label(v), public_value(bit) {}
};
constexpr size_t max_words = 32768;
struct Layout {
  size_t iw, ib, ow, ob;
};
int64_t signed_word(Word x) {
  x &= MASK;
  return (x & (Word{1} << 58)) ? int64_t(x) - int64_t(Word{1} << 59)
                               : int64_t(x);
}
Layout layout(const BatchSpec &d) {
  if (!d.rows || !d.cols || d.rows > max_words || d.cols > max_words ||
      d.rows > max_words / d.cols)
    throw std::runtime_error("E_BATCH_SHAPE");
  size_t n = d.rows * d.cols;
  auto p = d.parameters;
  for (auto x : p)
    if (x > MASK)
      throw std::runtime_error("E_BATCH_FORMAT");
  switch (d.code) {
  case BatchCode::ReciprocalSiLUPrepare:
    if (p != std::array<Word,3>{}) throw std::runtime_error("E_BATCH_FORMAT");
    return {n, 0, 2*n, n};
  case BatchCode::ReciprocalSiLURestore:
    if (p != std::array<Word,3>{}) throw std::runtime_error("E_BATCH_FORMAT");
    return {2*n, 0, n, n};
  case BatchCode::Floor12Range:
  case BatchCode::Floor24Range:
  case BatchCode::Range:
    if (signed_word(p[0]) > signed_word(p[1]) ||
        signed_word(p[2]) < signed_word(p[0]) ||
        signed_word(p[2]) > signed_word(p[1]))
      throw std::runtime_error("E_BATCH_RANGE");
    return {n, 0, n, n};
  case BatchCode::Floor:
    if (p[0] > 58 || p[1] || p[2])
      throw std::runtime_error("E_BATCH_SHIFT");
    return {n, 0, n, 0};
  case BatchCode::Compare:
    if (p[0] > 1 || p[1] || p[2])
      throw std::runtime_error("E_BATCH_FORMAT");
    return {2 * n, 0, 0, n};
  case BatchCode::Select:
    if (p != std::array<Word, 3>{})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {2 * n, n, n, 0};
  case BatchCode::A2B:
    if (p != std::array<Word, 3>{})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n, 0, 0, 59 * n};
  case BatchCode::B2A:
    if (p != std::array<Word, 3>{})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {0, 59 * n, n, 0};
  case BatchCode::Or:
    if (p != std::array<Word, 3>{})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {0, n, 0, d.rows};
  case BatchCode::Exponent:
    if (signed_word(p[0]) < -58 || signed_word(p[1]) > 58 ||
        signed_word(p[0]) > signed_word(p[1]) || p[2] > 1)
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n, 0, 2 * n, n};
  case BatchCode::Shift:
    if (signed_word(p[0]) < -64 || signed_word(p[0]) > 64 || p[1] || p[2])
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n + d.rows, 0, n, n};
  case BatchCode::GuardedSum:
    if (d.rows > 16 || d.cols > 2048 || !p[0] || p[0] >= (Word{1} << 55) ||
        p[1] || p[2])
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n, n, d.rows, d.rows};
  case BatchCode::PublicLookup:
    if (p != std::array<Word, 3>{256, 0, 0})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n, 0, n, 0};
  case BatchCode::PublicLookupRows:
    if (d.rows > 16 || d.cols != 128 || p != std::array<Word,3>{1280,0,0})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {d.rows, 0, n, d.rows};
  case BatchCode::SoftmaxPrepare:
    if (d.rows > 16 || d.cols > 1280 || p != std::array<Word, 3>{})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n, n + d.rows, n, n + d.rows};
  case BatchCode::SiLUComposite:
    if(p!=std::array<Word,3>{8388608,65536,1}) throw std::runtime_error("E_BATCH_FORMAT");
    return {n,0,n,n};
  case BatchCode::SiLUInput:
    if (!p[0] || p[0] >= (Word{1}<<58) || p[1] || p[2])
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n, 0, 4*n, 4*n};
  case BatchCode::ExpLookup:
  case BatchCode::ExpPrepare:
    if (p != std::array<Word, 3>{})
      throw std::runtime_error("E_BATCH_FORMAT");
    return {n, 0, 3 * n, 2 * n};
  }
  throw std::runtime_error("E_BATCH_OPCODE");
}
template<bool Fold> struct Circuit {
  using Wire = std::conditional_t<Fold, PublicWire, Label>;
  using Wires = std::vector<Wire>;
  static Wire secret(Label value) {
    if constexpr (Fold) return PublicWire(value);
    else return value;
  }
  static Label label(Wire value) {
    if constexpr (Fold) return value.label;
    else return value;
  }
  sci::CircuitExecution *c = nullptr;
  uint64_t gates = 0;
  Wire zero() {
    Label v = c ? c->public_label(false) : _mm_setzero_si128();
    if constexpr (Fold) return {v, 0}; else return v;
  }
  Wire one() {
    Label v = c ? c->public_label(true) : _mm_setzero_si128();
    if constexpr (Fold) return {v, 1}; else return v;
  }
  Wire X(Wire a, Wire b) {
    if constexpr (Fold) {
      if (a.public_value >= 0) return a.public_value ? N(b) : b;
      if (b.public_value >= 0) return b.public_value ? N(a) : a;
    }
    return secret(c ? c->xor_gate(label(a), label(b)) : _mm_setzero_si128());
  }
  Wire N(Wire a) {
    if constexpr (Fold) {
      if (a.public_value >= 0) return a.public_value ? zero() : one();
    }
    return secret(c ? c->not_gate(label(a)) : _mm_setzero_si128());
  }
  Wire A(Wire a, Wire b) {
    if constexpr (Fold) {
      if (a.public_value >= 0) return a.public_value ? b : zero();
      if (b.public_value >= 0) return b.public_value ? a : zero();
    }
    ++gates;
    return secret(c ? c->and_gate(label(a), label(b)) : _mm_setzero_si128());
  }
  Wire O(Wire a, Wire b) { return X(X(a, b), A(a, b)); }
  Wire M(Wire s, Wire x, Wire y) {
    if constexpr (Fold) {
      if (s.public_value >= 0) return s.public_value ? x : y;
    }
    return X(y, A(s, X(x, y)));
  }
  Wires constant(Word x) {
    Wires r(59);
    for (size_t j = 0; j < 59; ++j)
      r[j] = ((x >> j) & 1) ? one() : zero();
    return r;
  }
  Wires add(const Wires &a, const Wires &b) {
    Wires out(59);
    auto carry = zero();
    for (size_t j = 0; j < 59; ++j) {
      auto q = X(a[j], b[j]);
      out[j] = X(q, carry);
      // The final carry is discarded in the 59-bit ring. Majority uses one AND.
      if (j + 1 < 59)
        carry = X(a[j], A(q, X(a[j], carry)));
    }
    return out;
  }
  Wires sub(const Wires &a, const Wires &b) {
    Wires out(59);
    auto carry = one();
    for (size_t j = 0; j < 59; ++j) {
      auto q = X(a[j], N(b[j]));
      out[j] = X(q, carry);
      // a + ~b + 1, with the same one-AND majority and discarded final carry.
      if (j + 1 < 59)
        carry = X(a[j], A(q, X(a[j], carry)));
    }
    return out;
  }
  // Carry-only a-b-1: final carry means a>b. One AND/bit via majority.
  Wire gt(const Wires &a, const Wires &b, bool sign = true) {
    auto carry = zero();
    for (size_t j = 0; j < 59; ++j) {
      auto x = a[j], y = N(b[j]);
      if (sign && j == 58) {
        x = N(x);
        y = N(y);
      }
      carry = X(x, A(X(x, y), X(x, carry)));
    }
    return carry;
  }
  Wires mux(Wire s, const Wires &x, const Wires &y) {
    Wires r(59);
    for (size_t j = 0; j < 59; ++j)
      r[j] = M(s, x[j], y[j]);
    return r;
  }
  Wires floor(const Wires &x, size_t shift) {
    Wires r(59);
    for (size_t j = 0; j < 59; ++j)
      r[j] = x[std::min(size_t(58), j + shift)];
    return r;
  }
  Wires program(const BatchSpec &d, const Layout &l, const Wires &ci,
                const Wires &si, const std::vector<Word> &table = {}) {
    const size_t n = d.rows * d.cols;
    std::vector<Wires> words(l.iw), out;
    Wires bits(l.ib), flags;
    for (size_t i = 0; i < l.iw; ++i) {
      Wires a(ci.begin() + 59 * i, ci.begin() + 59 * (i + 1)),
          b(si.begin() + 59 * i, si.begin() + 59 * (i + 1));
      words[i] = add(a, b);
    }
    for (size_t i = 0; i < l.ib; ++i)
      bits[i] = X(ci[59 * l.iw + i], si[59 * l.iw + i]);
    auto p = d.parameters;
    switch (d.code) {
    case BatchCode::ReciprocalSiLUPrepare: {
      // Private SiLU-only contract: E is derived from bounded exp tables and
      // the two specified floors, so D=1+E lies in [1,2] at f24.
      // Keep the original denominator guard; guard the narrower contract too.
      const auto one24 = Word{1} << 24;
      std::vector<Wires> exponents;
      for (auto &x : words) {
        auto original_bad = O(gt(constant(one24), x),
                              gt(x, constant(1280*one24)));
        auto a = mux(original_bad, constant(one24), x);
        auto domain_bad = gt(a, constant(2*one24));
        a = mux(domain_bad, constant(one24), a);
        // At D=2 exactly, k=1 and M=D>>1=1; all smaller valid D use k=0.
        auto kbit = a[25];
        auto k = constant(0); k[0] = kbit;
        out.push_back(mux(kbit, floor(a,1), a));
        exponents.push_back(std::move(k));
        flags.push_back(O(original_bad, domain_bad));
      }
      out.insert(out.end(), exponents.begin(), exponents.end());
      break;
    }
    case BatchCode::ReciprocalSiLURestore: {
      const auto one24 = Word{1} << 24;
      for(size_t i=0;i<n;++i) {
        auto bad_state=O(gt(constant(0),words[i]),gt(words[i],constant(2*one24)));
        auto r=mux(bad_state,constant(one24/2),words[i]);
        auto bad_k=zero();
        for(size_t j=1;j<59;++j)bad_k=O(bad_k,words[n+i][j]);
        auto kbit=A(N(bad_k),words[n+i][0]);
        out.push_back(mux(kbit,floor(r,1),r));
        flags.push_back(O(bad_state,bad_k));
      }
      break;
    }
    case BatchCode::SiLUComposite: {
      const auto bound=Word{1}<<23;
      for(const auto& raw:words) {
        auto invalid=O(gt(constant((-bound)&MASK),raw),gt(raw,constant(bound)));
        auto x=mux(invalid,constant(0),raw),neg=x[58];
        auto m=mux(neg,sub(constant(0),x),x);
        Wires values;values.reserve(silu_graph::nodes.size()+1);values.push_back(zero());
        auto ref=[&](uint32_t r){auto w=values[r>>1];return (r&1)?N(w):w;};
        for(const auto& node:silu_graph::nodes) {
          // Node opcodes and references are public, frozen graph topology.
          // Secret bits remain circuit wires; no host index or reconstruction.
          if(node[0]==0)values.push_back(m[node[1]]);
          else if(node[0]==1)values.push_back(A(ref(node[1]),ref(node[2])));
          else if(node[0]==2)values.push_back(X(ref(node[1]),ref(node[2])));
          else throw std::runtime_error("E_SILU_GRAPH_OPCODE");
        }
        auto correction=constant(0),delta=constant(0);
        for(size_t bit=0;bit<11;++bit)correction[bit]=ref(silu_graph::roots[bit]);
        delta[0]=ref(silu_graph::roots[11]);
        auto negative=sub(constant(0),sub(correction,delta));
        auto positive=sub(m,correction);
        auto tail=gt(m,constant(65535));
        positive=mux(tail,sub(m,constant(1)),positive);
        negative=mux(tail,constant(0),negative);
        auto result=mux(neg,negative,positive);
        auto out_bad=O(gt(constant((-bound)&MASK),result),gt(result,constant(bound)));
        out.push_back(mux(out_bad,constant(0),result));flags.push_back(O(invalid,out_bad));
      }
      break;
    }
    case BatchCode::Floor12Range:
    case BatchCode::Floor24Range: {
      auto low=constant(p[0]),high=constant(p[1]),dummy=constant(p[2]);
      for(auto& raw:words) {
        auto x=floor(raw,d.code==BatchCode::Floor12Range?12:24);
        auto bad=O(gt(low,x),gt(x,high));
        out.push_back(mux(bad,dummy,x));
        flags.push_back(bad);
      }
      break;
    }
    case BatchCode::Range: {
      auto low = constant(p[0]), high = constant(p[1]), dummy = constant(p[2]);
      for (auto &x : words) {
        auto bad = O(gt(low, x), gt(x, high));
        out.push_back(mux(bad, dummy, x));
        flags.push_back(bad);
      }
      break;
    }
    case BatchCode::Floor:
      for (auto &x : words)
        out.push_back(floor(x, p[0]));
      break;
    case BatchCode::Compare:
      for (size_t i = 0; i < n; ++i)
        flags.push_back(gt(words[i], words[n + i], p[0]));
      break;
    case BatchCode::Select:
      for (size_t i = 0; i < n; ++i)
        out.push_back(mux(bits[i], words[i], words[n + i]));
      break;
    case BatchCode::A2B:
      for (auto &x : words)
        flags.insert(flags.end(), x.begin(), x.end());
      break;
    case BatchCode::B2A:
      for (size_t i = 0; i < n; ++i)
        out.emplace_back(bits.begin() + 59 * i, bits.begin() + 59 * (i + 1));
      break;
    case BatchCode::Or:
      for (size_t i = 0; i < d.rows; ++i) {
        auto v = zero();
        for (size_t j = 0; j < d.cols; ++j)
          v = O(v, bits[i * d.cols + j]);
        flags.push_back(v);
      }
      break;
    case BatchCode::Exponent: {
      std::vector<Wires> exponents;
      for (auto &x : words) {
        auto found = zero();
        auto k = constant(0);
        for (int j = 58; j >= 0; --j) {
          auto top = A(x[j], N(found));
          int e = j - 24;
          if (p[2])
            e = e >= 0 ? e / 2 : -((-e + 1) / 2);
          k = mux(top, constant(Word(e) & MASK), k);
          found = O(found, x[j]);
        }
        auto bad = O(O(N(found), x[58]),
                     O(gt(constant(p[0]), k), gt(k, constant(p[1]))));
        out.push_back(mux(bad, constant(Word{1} << 24), x));
        exponents.push_back(mux(bad, constant(0), k));
        flags.push_back(bad);
      }
      out.insert(out.end(), exponents.begin(), exponents.end());
      break;
    }
    case BatchCode::Shift: {
      for (size_t row = 0; row < d.rows; ++row) {
        auto shift = sub(constant(p[0]), words[n + row]);
        auto right = shift[58];
        auto amount = mux(right, sub(constant(0), shift), shift);
        auto invalid = gt(amount, constant(32), false);
        amount = mux(invalid, constant(0), amount);
        for (size_t col = 0; col < d.cols; ++col) {
          auto &x = words[row * d.cols + col];
          // 59 signed input bits + the maximum 32-bit left shift suffices.
          // Wires above 90 in the previous 128-bit circuit are identical sign
          // copies; their repeated equality tests add no information.
          constexpr size_t width = 91;
          Wires left(width), r(width);
          for (size_t j = 0; j < width; ++j)
            left[j] = r[j] = x[std::min(j, size_t(58))];
          for (size_t stage = 0; stage < 6; ++stage) {
            size_t dist = size_t(1) << stage;
            auto ll = left, rr = r;
            for (size_t j = 0; j < width; ++j) {
              left[j] =
                  M(amount[stage], j >= dist ? ll[j - dist] : zero(), ll[j]);
              r[j] = M(amount[stage], rr[std::min(j + dist, width - 1)], rr[j]);
            }
          }
          Wires z(width);
          for (size_t j = 0; j < width; ++j)
            z[j] = M(right, r[j], left[j]);
          auto bad = zero();
          for (size_t j = 57; j < width; ++j)
            bad = O(bad, X(z[j], z[56]));
          auto nz = zero();
          for (size_t j = 0; j < 56; ++j)
            nz = O(nz, z[j]);
          bad = O(bad, A(z[56], N(nz)));
          Wires result(59);
          for (size_t j = 0; j < 59; ++j)
            result[j] = A(N(bad), z[j]);
          out.push_back(std::move(result));
          flags.push_back(O(invalid, bad));
        }
      }
      break;
    }
    case BatchCode::GuardedSum: {
      auto limit = constant(p[0]);
      for (size_t row = 0; row < d.rows; ++row) {
        std::vector<Wires> partial;
        auto bad = zero();
        for (size_t col = 0; col < d.cols; ++col) {
          size_t i = row * d.cols + col;
          auto over = gt(words[i], limit, false);
          bad = O(bad, O(bits[i], over));
          partial.push_back(mux(over, constant(0), words[i]));
        }
        while (partial.size() > 1) {
          std::vector<Wires> next;
          for (size_t j = 0; j < partial.size(); j += 2) {
            if (j + 1 == partial.size()) {
              next.push_back(partial[j]);
              continue;
            }
            auto room = sub(limit, partial[j]);
            auto over = gt(partial[j + 1], room, false);
            bad = O(bad, over);
            auto safe = mux(over, constant(0), partial[j + 1]);
            next.push_back(add(partial[j], safe));
          }
          partial = std::move(next);
        }
        out.push_back(mux(bad, constant(0), partial[0]));
        flags.push_back(bad);
      }
      break;
    }
    case BatchCode::PublicLookup: {
      if (table.size() != 256)
        throw std::runtime_error("E_BATCH_LOOKUP_TABLE");
      for (auto &index : words) {
        auto high = zero();
        for (size_t j = 8; j < 59; ++j)
          high = O(high, index[j]);
        Wires nodes{N(high)};
        for (int j = 7; j >= 0; --j) {
          Wires next;
          next.reserve(nodes.size() * 2);
          for (auto v : nodes) {
            auto right = A(v, index[j]);
            next.push_back(X(v, right));
            next.push_back(right);
          }
          nodes = std::move(next);
        }
        Wires selected(59, zero());
        for (size_t j = 0; j < 256; ++j)
          for (size_t bit = 0; bit < 59; ++bit)
            if ((table[j] >> bit) & 1)
              selected[bit] = X(selected[bit], nodes[j]);
        out.push_back(std::move(selected));
      }
      break;
    }
    case BatchCode::PublicLookupRows: {
      // A single secret 11-bit decoder per position is reused for all 128
      // public coefficients. Out-of-domain positions return zero and a flag.
      for (auto &index : words) {
        auto bad = gt(index, constant(1279), false);
        Wires nodes{N(bad)};
        for (int j=10;j>=0;--j) {
          Wires next; next.reserve(nodes.size()*2);
          for(auto v:nodes){auto right=A(v,index[j]);next.push_back(X(v,right));next.push_back(right);}
          nodes=std::move(next);
        }
        for(size_t col=0;col<d.cols;++col){
          Wires selected(59,zero());
          for(size_t j=0;j<1280;++j)
            for(size_t bit=0;bit<59;++bit)
              if((table[j*d.cols+col]>>bit)&1)selected[bit]=X(selected[bit],nodes[j]);
          out.push_back(std::move(selected));
        }
        flags.push_back(bad);
      }
      break;
    }
    case BatchCode::SoftmaxPrepare: {
      Wires bad_rows;
      for (size_t row = 0; row < d.rows; ++row) {
        auto valid = bits[n + row], seen = zero();
        auto maximum = constant(0);
        std::vector<Wires> xs;
        Wires allowed;
        for (size_t col = 0; col < d.cols; ++col) {
          auto index = row * d.cols + col;
          auto allow =
              O(A(valid, bits[index]), A(N(valid), col == 0 ? one() : zero()));
          auto x = mux(valid, words[index], constant(0));
          auto take = A(allow, O(N(seen), gt(x, maximum)));
          maximum = mux(take, x, maximum);
          seen = O(seen, allow);
          xs.push_back(std::move(x));
          allowed.push_back(allow);
        }
        auto missing = N(seen);
        allowed[0] = O(allowed[0], missing);
        xs[0] = mux(seen, xs[0], maximum);
        for (size_t col = 0; col < d.cols; ++col) {
          out.push_back(mux(allowed[col], sub(maximum, xs[col]), constant(0)));
          flags.push_back(allowed[col]);
        }
        bad_rows.push_back(missing);
      }
      flags.insert(flags.end(), bad_rows.begin(), bad_rows.end());
      break;
    }
    case BatchCode::SiLUInput:
    case BatchCode::ExpLookup: {
      // The original guard makes safe a value in [0, 2^17]. Thus its
      // high byte is exactly 0, 1, or 2; no assumption about input shares.
      std::vector<Wires> second, third;
      Wires negatives, signs, range_bad;
      std::vector<Wires> safe_inputs;
      for (auto &input : words) {
        auto x=input;
        if(d.code==BatchCode::SiLUInput) {
          auto bad=O(gt(constant((Word(0)-p[0])&MASK),x),gt(x,constant(p[0])));
          x=mux(bad,constant(0),x);
          safe_inputs.push_back(x);range_bad.push_back(bad);
          auto sign=gt(constant(0),x,true);signs.push_back(sign);
          x=mux(sign,sub(constant(0),x),x);
        }
        auto negative=x[58], off=O(negative,gt(x,constant(32 << 12)));
        auto safe=mux(off,constant(0),x);
        std::array<Wires,3> selected;
        for (size_t digit=0;digit<3;++digit) {
          Wires nodes;
          if (digit<2) {
            nodes={one()};
            for (int j=7;j>=0;--j) {
              Wires next;next.reserve(nodes.size()*2);
              for (auto v:nodes) {
                auto right=A(v,safe[8*digit+j]);
                next.push_back(X(v,right));next.push_back(right);
              }
              nodes=std::move(next);
            }
          } else {
            nodes={N(O(safe[16],safe[17])),safe[16],safe[17]};
          }
          selected[digit]=Wires(59,zero());
          for(size_t j=0;j<nodes.size();++j)
            for(size_t bit=0;bit<59;++bit)
              if((table[256*digit+j]>>bit)&1)
                selected[digit][bit]=X(selected[digit][bit],nodes[j]);
        }
        out.push_back(std::move(selected[0]));
        second.push_back(std::move(selected[1]));
        third.push_back(std::move(selected[2]));
        flags.push_back(off);negatives.push_back(negative);
      }
      out.insert(out.end(),second.begin(),second.end());
      out.insert(out.end(),third.begin(),third.end());
      flags.insert(flags.end(),negatives.begin(),negatives.end());
      if(d.code==BatchCode::SiLUInput) {
        out.insert(out.end(),safe_inputs.begin(),safe_inputs.end());
        flags.insert(flags.end(),signs.begin(),signs.end());
        flags.insert(flags.end(),range_bad.begin(),range_bad.end());
      }
      break;
    }
    case BatchCode::ExpPrepare: {
      std::vector<Wires> second, third;
      Wires negatives;
      for (auto &x : words) {
        auto negative = x[58], off = O(negative, gt(x, constant(32 << 12)));
        auto safe = mux(off, constant(0), x);
        std::array<Wires, 3> digits{constant(0), constant(0), constant(0)};
        for (size_t digit = 0; digit < 3; ++digit)
          for (size_t j = 0; j < 8; ++j)
            digits[digit][j] = safe[8 * digit + j];
        out.push_back(std::move(digits[0]));
        second.push_back(std::move(digits[1]));
        third.push_back(std::move(digits[2]));
        flags.push_back(off);
        negatives.push_back(negative);
      }
      out.insert(out.end(), second.begin(), second.end());
      out.insert(out.end(), third.begin(), third.end());
      flags.insert(flags.end(), negatives.begin(), negatives.end());
      break;
    }
    }
    if (out.size() != l.ow || flags.size() != l.ob)
      throw std::runtime_error("E_BATCH_CIRCUIT_SHAPE");
    Wires output;
    output.reserve(59 * l.ow + l.ob);
    size_t pos = 59 * l.iw + l.ib;
    for (auto &x : out) {
      Wires rho(si.begin() + pos, si.begin() + pos + 59);
      pos += 59;
      auto y = sub(x, rho);
      output.insert(output.end(), y.begin(), y.end());
    }
    for (auto b : flags)
      output.push_back(X(b, si[pos++]));
    return output;
  }
};
struct CircuitResult { Wires output; uint64_t gates; };
template<bool Fold>
CircuitResult execute_circuit(sci::CircuitExecution *backend, const BatchSpec &d,
    const Layout &l, const Wires &ci, const Wires &si,
    const std::vector<Word> &table) {
  Circuit<Fold> circuit{backend};
  if constexpr (Fold) {
    typename Circuit<Fold>::Wires cinput, sinput;
    cinput.reserve(ci.size()); sinput.reserve(si.size());
    for (auto value : ci) cinput.emplace_back(value);
    for (auto value : si) sinput.emplace_back(value);
    auto internal = circuit.program(d, l, cinput, sinput, table);
    Wires output; output.reserve(internal.size());
    for (auto value : internal) output.push_back(value.label);
    return {std::move(output), circuit.gates};
  } else {
    auto output = circuit.program(d, l, ci, si, table);
    return {std::move(output), circuit.gates};
  }
}
CircuitResult execute_circuit(sci::CircuitExecution *backend, const BatchSpec &d,
    const Layout &l, const Wires &ci, const Wires &si,
    const std::vector<Word> &table) {
  if (d.code == BatchCode::Exponent)
    return execute_circuit<true>(backend, d, l, ci, si, table);
  return execute_circuit<false>(backend, d, l, ci, si, table);
}
void append(std::vector<uint8_t> &bits, const std::vector<Word> &words) {
  for (auto v : words)
    for (size_t j = 0; j < 59; ++j)
      bits.push_back((v >> j) & 1);
}
} // namespace
struct BatchOps::Impl {
  SecureOps &op;
  Channel &io;
  Role role;
  sci::IKNP<Channel> ot;
  GcBatchChannel tables;
  BatchStats counters;
  TopologyCountCache topology_counts;
  Word last_logical_matrix_sequence=0;
  std::unique_ptr<sci::CircuitExecution> circuit;
  sci::HalfGateGen<GcBatchChannel> *generator = nullptr;
  explicit Impl(SecureOps &o)
      : op(o), io(o.channel()), role(o.role()), ot(&io),
        tables(io, role == Role::Server) {
    if (!op.session_active() || !io.authenticated() ||
        op.isolated_development())
      throw std::runtime_error("E_BATCH_MTLS_REQUIRED");
  }
  void descriptor(uint64_t code, size_t rows, size_t cols,
                  std::array<Word, 3> params) {
    if (!op.session_active())
      throw std::runtime_error("E_SESSION_CLOSED");
    std::array<unsigned char, 64> own{}, other{};
    std::memcpy(own.data(), "PRAGBAT1", 8);
    std::array<Word, 7> fields{1,         code,      rows,     cols,
                               params[0], params[1], params[2]};
    for (size_t i = 0; i < fields.size(); ++i)
      for (size_t j = 0; j < 8; ++j)
        own[8 + 8 * i + j] = (fields[i] >> (56 - 8 * j)) & 255;
    op.reserve_material("batch_descriptor", 1);
    if (role == Role::Client) {
      io.send_data(own.data(), own.size());
      io.recv_data(other.data(), other.size());
    } else {
      io.recv_data(other.data(), other.size());
      io.send_data(own.data(), own.size());
    }
    if (own != other)
      throw std::runtime_error("E_BATCH_DESCRIPTOR_MISMATCH");
  }
  Wires input(const std::vector<uint8_t> &values, Role owner) {
    size_t n = values.size();
    Wires w(n);
    if (!n)
      return w;
    if (role == Role::Server) {
      Wires alternate(n), selected(n);
      sci::PRG128 rng;
      rng.random_block(w.data(), int(n));
      for (size_t j = 0; j < n; ++j) {
        alternate[j] = w[j] ^ generator->delta;
        selected[j] = values[j] ? alternate[j] : w[j];
      }
      if (owner == Role::Server)
        io.send_block(selected.data(), int(n));
      else
        ot.send(w.data(), alternate.data(), int(n));
    } else if (owner == Role::Server)
      io.recv_block(w.data(), int(n));
    else {
      std::unique_ptr<bool[]> choices(new bool[n]);
      for (size_t j = 0; j < n; ++j)
        choices[j] = values[j];
      ot.recv_pre(choices.get(), int(n));
      constexpr size_t post = sci::AES_BATCH_SIZE / 2;
      std::vector<Label> encoded(2 * post);
      for (size_t first = 0; first < n; first += post) {
        size_t count = std::min(post, n - first);
        io.recv_data(encoded.data(), int(2 * count * sizeof(Label)));
        if (count == post)
          ot.crh.H<post>(ot.tT + first, ot.tT + first);
        else
          ot.crh.Hn(ot.tT + first, ot.tT + first, int(count));
        for (size_t j = 0; j < count; ++j)
          w[first + j] = encoded[2 * j + choices[first + j]] ^ ot.tT[first + j];
      }
      delete[] ot.tT;
      ot.tT = nullptr;
    }
    return w;
  }
  BatchResult run(const BatchSpec &d, const std::vector<Word> &words,
                  const std::vector<uint8_t> &bits,
                  const std::vector<Word> &table = {}) {
    const auto l = layout(d);
    if (words.size() != l.iw || bits.size() != l.ib)
      throw std::runtime_error("E_BATCH_INPUT_SHAPE");
    for (auto v : words)
      if (v > MASK)
        throw std::runtime_error("E_BATCH_WORD");
    for (auto b : bits)
      if (b > 1)
        throw std::runtime_error("E_BATCH_BIT");
    const size_t expected_table = d.code == BatchCode::PublicLookup ? 256 :
        d.code == BatchCode::PublicLookupRows ? 1280*d.cols :
        (d.code == BatchCode::ExpLookup || d.code == BatchCode::SiLUInput || d.code == BatchCode::SiLUComposite) ? 768 : 0;
    if (table.size() != expected_table)
      throw std::runtime_error("E_BATCH_LOOKUP_TABLE");
    for (auto x : table)
      if (x > MASK)
        throw std::runtime_error("E_BATCH_LOOKUP_TABLE");
    if(d.code==BatchCode::ExpLookup || d.code==BatchCode::SiLUInput)
      for(auto x:table) if(x>(Word{1}<<24))
        throw std::runtime_error("E_EXP_TABLE_RANGE");
    if(d.code==BatchCode::SiLUComposite && !std::equal(table.begin(),table.end(),silu_graph::tables.begin()))
      throw std::runtime_error("E_SILU_COMPOSITE_TABLE_BINDING");
    descriptor(uint64_t(d.code), d.rows, d.cols, d.parameters);
    if(d.code==BatchCode::SiLUComposite) {
      std::array<unsigned char,32> peer{};
      if(role==Role::Client){io.send_data(silu_graph::sha256.data(),32);io.recv_data(peer.data(),32);}
      else {io.recv_data(peer.data(),32);io.send_data(silu_graph::sha256.data(),32);}
      if(peer!=silu_graph::sha256)throw std::runtime_error("E_SILU_COMPOSITE_GRAPH_BINDING");
    }

    if (!table.empty()) {
      std::vector<unsigned char> encoded(table.size()*8);
      for (size_t i = 0; i < table.size(); ++i)
        for (size_t j = 0; j < 8; ++j)
          encoded[8 * i + j] = (table[i] >> (8 * j)) & 255;
      std::array<unsigned char, 32> digest{}, other{};
      SHA256(encoded.data(), encoded.size(), digest.data());
      if (role == Role::Client) {
        io.send_data(digest.data(), 32);
        io.recv_data(other.data(), 32);
      } else {
        io.recv_data(other.data(), 32);
        io.send_data(digest.data(), 32);
      }
      if (digest != other)
        throw std::runtime_error("E_BATCH_PUBLIC_TABLE_MISMATCH");
    }
    const size_t inbits = 59 * l.iw + l.ib, outbits = 59 * l.ow + l.ob;
    // Topology counting only: no operand values, cryptography, or diagnostic
    // outputs. Both parties run the same public program before material use.
    uint64_t count = 0;
    if (topology_counts.find(d, table, count)) {
      ++counters.topology_cache_hits;
    } else {
      ++counters.topology_cache_misses;
      count = execute_circuit(nullptr, d, l, Wires(inbits), Wires(inbits + outbits), table).gates;
      topology_counts.insert(d, table, count);
    }
    op.reserve_material("batch_GC_AND", count);
    op.reserve_material("batch_input_OT", inbits);
    op.reserve_material("batch_mask_words", l.ow);
    op.reserve_material("batch_mask_bits", l.ob);
    BatchResult result{std::vector<Word>(l.ow), std::vector<uint8_t>(l.ob)};
    if (role == Role::Server) {
      if (l.ow && RAND_priv_bytes(
                      reinterpret_cast<unsigned char *>(result.words.data()),
                      int(l.ow * 8)) != 1)
        throw std::runtime_error("E_RNG");
      for (auto &x : result.words)
        x &= MASK;
      if (l.ob && RAND_priv_bytes(result.bits.data(), int(l.ob)) != 1)
        throw std::runtime_error("E_RNG");
      for (auto &b : result.bits)
        b &= 1;
    }
    circuit.reset();
    generator = nullptr;
    if (role == Role::Server) {
      generator = new sci::HalfGateGen<GcBatchChannel>(&tables);
      circuit.reset(generator);
    } else
      circuit.reset(new sci::HalfGateEva<GcBatchChannel>(&tables));
    std::vector<uint8_t> own;
    own.reserve(inbits + outbits);
    append(own, words);
    own.insert(own.end(), bits.begin(), bits.end());
    auto cinput = input(own, Role::Client);
    append(own, result.words);
    own.insert(own.end(), result.bits.begin(), result.bits.end());
    auto sinput = input(own, Role::Server);
    tables.begin(count);
    auto real = execute_circuit(circuit.get(), d, l, cinput, sinput, table);
    auto &output = real.output;
    tables.end();
    if (real.gates != count)
      throw std::runtime_error("E_BATCH_GATES");
    std::vector<uint8_t> permutation(outbits);
    if (role == Role::Server) {
      for (size_t i = 0; i < outbits; ++i)
        permutation[i] = sci::getLSB(output[i]);
      io.send_data(permutation.data(), int(outbits));
    } else {
      io.recv_data(permutation.data(), int(outbits));
      for (size_t i = 0; i < outbits; ++i) {
        auto b = uint8_t(sci::getLSB(output[i])) ^ permutation[i];
        if (b > 1)
          throw std::runtime_error("E_BATCH_OUTPUT_BIT");
        if (i < 59 * l.ow)
          result.words[i / 59] |= Word(b) << (i % 59);
        else
          result.bits[i - 59 * l.ow] = b;
      }
    }
    OPENSSL_cleanse(own.data(), own.size());
    counters.calls++;
    counters.and_gates += count;
    counters.input_ot_choices += inbits;
    counters.masked_words += l.ow;
    counters.masked_bits += l.ob;
    return result;
  }
};
BatchOps::BatchOps(SecureOps &o) : p_(std::make_unique<Impl>(o)) {}
BatchOps::~BatchOps() = default;
BatchResult BatchOps::run(const BatchSpec &d, const std::vector<Word> &w,
                          const std::vector<uint8_t> &b) {
  try {
    return p_->run(d, w, b);
  } catch (...) {
    p_->op.abort_session();
    throw;
  }
}
std::vector<Word> BatchOps::multiply(const std::vector<Word> &a,
                                     const std::vector<Word> &b) {
  try {
    if (a.empty() || a.size() != b.size() || a.size() > max_words)
      throw std::runtime_error("E_BATCH_SHAPE");
    for (auto v : a)
      if (v > MASK)
        throw std::runtime_error("E_BATCH_WORD");
    for (auto v : b)
      if (v > MASK)
        throw std::runtime_error("E_BATCH_WORD");
    p_->descriptor(100, 1, a.size(), {});
    return p_->op.mul_many(a, b);
  } catch (...) {
    p_->op.abort_session();
    throw;
  }
}
std::vector<Word> BatchOps::lookup(const std::vector<Word> &x,
                                   const std::vector<Word> &table) {
  try {
    return p_
        ->run({BatchCode::PublicLookup, 1, x.size(), {256, 0, 0}}, x, {}, table)
        .words;
  } catch (...) {
    p_->op.abort_session();
    throw;
  }
}
BatchResult BatchOps::exp_lookup(const std::vector<Word>& x,
                                  const std::array<std::vector<Word>,3>& tables) {
  try {
    std::vector<Word> flat;flat.reserve(768);
    for(const auto& table:tables) {
      if(table.size()!=256)throw std::runtime_error("E_EXP_TABLE_SHAPE");
      flat.insert(flat.end(),table.begin(),table.end());
    }
    return p_->run({BatchCode::ExpLookup,1,x.size(),{}},x,{},flat);
  } catch (...) {p_->op.abort_session();throw;}
}
BatchResult BatchOps::silu_composite(const std::vector<Word>& x,
    const std::array<std::vector<Word>,3>& tables) {
  try {
    std::vector<Word> flat;flat.reserve(768);
    for(const auto&t:tables){if(t.size()!=256)throw std::runtime_error("E_EXP_TABLE_SHAPE");flat.insert(flat.end(),t.begin(),t.end());}
    return p_->run({BatchCode::SiLUComposite,1,x.size(),{8388608,65536,1}},x,{},flat);
  }catch(...){p_->op.abort_session();throw;}
}
BatchResult BatchOps::silu_input(const std::vector<Word>& x,
    const std::array<std::vector<Word>,3>& tables, Word bound) {
  try {
    std::vector<Word> flat;flat.reserve(768);
    for(const auto& table:tables) {
      if(table.size()!=256)throw std::runtime_error("E_EXP_TABLE_SHAPE");
      flat.insert(flat.end(),table.begin(),table.end());
    }
    return p_->run({BatchCode::SiLUInput,1,x.size(),{bound,0,0}},x,{},flat);
  } catch(...) {p_->op.abort_session();throw;}
}
BatchResult BatchOps::lookup_rows(const std::vector<Word>& x,
                                   const std::vector<Word>& table, size_t columns) {
  try { return p_->run({BatchCode::PublicLookupRows,x.size(),columns,{1280,0,0}},x,{},table); }
  catch (...) {p_->op.abort_session();throw;}
}
#include "batch_matrix.inc"
#include "logical_matrix.inc"
BatchStats BatchOps::stats() const { return p_->counters; }
} // namespace prag
