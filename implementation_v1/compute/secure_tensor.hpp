#pragma once
#include "prag/secure_ops.hpp"
#include <cstddef>
#include <functional>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace prag::model {
using Word=std::uint64_t;
using Bit=bool;
constexpr Word mask=(Word{1}<<59)-1;
constexpr Word one24=Word{1}<<24;
struct Tensor {
  std::vector<Word> values; // one party only, never a clear tensor
  std::vector<std::size_t> shape;
  unsigned fractional_bits;
  std::string configuration;
  Tensor(std::vector<Word> x,std::vector<std::size_t> dims,unsigned f,std::string config)
      :values(std::move(x)),shape(std::move(dims)),fractional_bits(f),configuration(std::move(config)) {
    std::size_t n=1;
    for(auto d:shape) { if(d && n>SIZE_MAX/d) throw std::invalid_argument("BAD_SHAPE"); n*=d; }
    if(n!=values.size() || f>58 || configuration.empty()) throw std::invalid_argument("BAD_SHAPE");
    for(auto v:values) if(v>mask) throw std::invalid_argument("BAD_FORMAT");
  }
};
struct PrivateMatrix {
  std::size_t rows, cols;
  unsigned fractional_bits=12;
  std::vector<std::vector<Word>> server_values; // empty at C, shape is public
  std::function<Word(std::size_t,std::size_t)> server_reader;
  Word maximum_row_l1=0; // public bound pinned by the numeric profile
  Word input_abs_f12=0; // precondition for ordinary/down full raw dot products
  bool owned()const{return bool(server_reader)||!server_values.empty();}
  Word at(std::size_t r,std::size_t c)const{return server_reader?server_reader(r,c):server_values.at(r).at(c);}
};
struct Record {
  std::vector<Word> tokens,validity,embedding;
  Word embedding_valid;
};
struct Candidate { Word score,index; Bit valid; };
struct Evidence { std::vector<Word> tokens,validity; Bit valid; };
struct Retrieval { std::vector<Evidence> evidence; std::vector<Bit> private_keep_bits; };
}
