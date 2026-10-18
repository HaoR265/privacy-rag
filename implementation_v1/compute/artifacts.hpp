#pragma once
#include "pipeline.hpp"
#include "vector_identities.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <filesystem>
#include <regex>
#include <sstream>
#include <cstring>
#include <unordered_map>
#include <openssl/evp.h>

namespace prag::model {
inline void require_compiled_numeric_profile(const std::filesystem::path& path){
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("VERSION_MISMATCH");
  std::string data((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
  unsigned char hash[32];unsigned int size=0;
  if(EVP_Digest(data.data(),data.size(),hash,&size,EVP_sha256(),nullptr)!=1||size!=32)throw std::runtime_error("VERSION_MISMATCH");
  std::string encoded;constexpr char digits[]="0123456789abcdef";
  for(auto b:hash){encoded+=digits[b>>4];encoded+=digits[b&15];}
  if(encoded!=numeric_profile::sha256)throw std::runtime_error("VERSION_MISMATCH");
}
class MappedWeights {
  void* memory=MAP_FAILED;std::size_t length=0,offset=0,rows=0,cols=0;
 public:
  MappedWeights(const MappedWeights&)=delete;
  MappedWeights& operator=(const MappedWeights&)=delete;
  MappedWeights(const std::filesystem::path& path,std::size_t r,std::size_t c,bool vector):rows(r),cols(c){
    if(r==0||c==0)throw std::invalid_argument("BAD_SHAPE");
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)throw std::runtime_error("E_WEIGHT_FILE");
    struct stat st{};if(fstat(fd,&st)||!S_ISREG(st.st_mode)){close(fd);throw std::runtime_error("E_WEIGHT_FILE");}
    length=st.st_size;memory=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(memory==MAP_FAILED)throw std::runtime_error("E_WEIGHT_MAP");
    try{
      auto* p=static_cast<const unsigned char*>(memory);
      if(length<12||memcmp(p,"\x93NUMPY",6)||(p[6]!=1&&p[6]!=2)||p[7]!=0)throw std::runtime_error("E_NPY_FORMAT");
      std::size_t n=p[8]|(std::size_t(p[9])<<8);offset=10;
      if(p[6]==2){n|=(std::size_t(p[10])<<16)|(std::size_t(p[11])<<24);offset=12;}
      if(n>65536||offset+n>length)throw std::runtime_error("E_NPY_FORMAT");
      std::string h(reinterpret_cast<const char*>(p+offset),n);offset+=n;
      if(h.find("'descr': '<i4'")==std::string::npos||h.find("'fortran_order': False")==std::string::npos)throw std::runtime_error("E_NPY_FORMAT");
      std::smatch m;std::regex shape("'shape': \\(([^)]*)\\)");if(!std::regex_search(h,m,shape))throw std::runtime_error("E_NPY_FORMAT");
      std::vector<std::size_t> dims;std::stringstream stream(m[1].str());std::string piece;
      while(std::getline(stream,piece,',')){if(piece.find_first_not_of(" ")==std::string::npos)continue;dims.push_back(std::stoull(piece));}
      if(dims!=(vector?std::vector<std::size_t>{r}:std::vector<std::size_t>{r,c})||r>SIZE_MAX/c||r*c>(SIZE_MAX-offset)/4||offset+r*c*4!=length)throw std::runtime_error("E_NPY_SHAPE");
    }catch(...){munmap(memory,length);memory=MAP_FAILED;throw;}
  }
  ~MappedWeights(){if(memory!=MAP_FAILED)munmap(memory,length);}
  void require_vector_identity(const char* expected_hash)const{
    unsigned char hash[32];unsigned int size=0;
    if(EVP_Digest(memory,length,hash,&size,EVP_sha256(),nullptr)!=1||size!=32)throw std::runtime_error("VERSION_MISMATCH");
    std::string hex;constexpr char digits[]="0123456789abcdef";for(auto b:hash){hex+=digits[b>>4];hex+=digits[b&15];}
    if(hex!=expected_hash)throw std::runtime_error("VERSION_MISMATCH");
    for(size_t r=0;r<rows;++r){auto x=at(r,0);auto magnitude=x&(Word{1}<<58)?(-x)&mask:x;if(magnitude>numeric_profile::weight_abs_f12)throw std::runtime_error("E_WEIGHT_BOUND");}
  }
  void require_row_bound(const char* expected_hash,Word row_l1,Word input_cap)const{
    // Verify the exact mapped bytes before using a profile-specific input.
    // This is S-local registration, not an operation on secret activations.
    unsigned char hash[32];unsigned int size=0;
    if(EVP_Digest(memory,length,hash,&size,EVP_sha256(),nullptr)!=1||size!=32)throw std::runtime_error("VERSION_MISMATCH");
    std::string hex;constexpr char digits[]="0123456789abcdef";
    for(auto b:hash){hex+=digits[b>>4];hex+=digits[b&15];}
    if(hex!=expected_hash||!input_cap||!row_l1||row_l1>((Word{1}<<56)-1)/input_cap)throw std::runtime_error("VERSION_MISMATCH");
    Word maximum=0;
    for(std::size_t r=0;r<rows;++r){
      Word sum=0;
      for(std::size_t c=0;c<cols;++c){
        auto value=at(r,c);auto magnitude=(value&(Word{1}<<58))?((~value+1)&mask):value;
        if(magnitude>numeric_profile::weight_abs_f12||magnitude>row_l1-sum)throw std::runtime_error("E_WEIGHT_BOUND");
        sum+=magnitude;
      }
      maximum=std::max(maximum,sum);
    }
    if(maximum!=row_l1)throw std::runtime_error("E_WEIGHT_BOUND");
  }
  void require_linear_bound(const numeric_profile::LinearBound& bound)const{
    if(bound.rows!=rows||bound.cols!=cols)throw std::runtime_error("E_WEIGHT_BOUND");
    require_row_bound(bound.sha256,bound.row_l1,bound.input_abs_f12);
  }
  void require_down_bound(const numeric_profile::DownBound& bound)const{
    require_row_bound(bound.sha256,bound.row_l1,numeric_profile::swiglu_joint_abs_f12);
  }
  Word at(std::size_t r,std::size_t c)const{
    if(r>=rows||c>=cols)throw std::invalid_argument("BAD_SHAPE");
    const auto* p=static_cast<const unsigned char*>(memory)+offset+4*(r*cols+c);
    uint32_t u=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
    int64_t signed_value=u<(uint32_t{1}<<31)?int64_t(u):int64_t(u)-(int64_t{1}<<32);
    return static_cast<Word>(signed_value)&mask;
  }
};
class CheckpointWeights final:public Weights {
  std::filesystem::path root;ModelShape shape;bool server;PrivateMatrix embedding;
  std::unordered_map<std::string,std::shared_ptr<MappedWeights>> mappings;
  PrivateMatrix get(const std::string& name,std::size_t rows,std::size_t cols,bool vector=false){
    PrivateMatrix out{rows,cols,12,{},{}};
    const numeric_profile::LinearBound* bound=nullptr;
    if(!vector){
      const auto& bounds=shape.hidden==1024?numeric_profile::embedding_linear:numeric_profile::generation_linear;
      for(const auto& candidate:bounds)if(name==candidate.tensor){bound=&candidate;break;}
      if(!bound||bound->rows!=rows||bound->cols!=cols)throw std::runtime_error("VERSION_MISMATCH");
      out.maximum_row_l1=bound->row_l1;out.input_abs_f12=bound->input_abs_f12;
    }
    if(server){
      auto found=mappings.find(name);
      if(found==mappings.end()){
        auto mapped=std::make_shared<MappedWeights>(root/(name+".npy"),rows,cols,vector);
        if(bound)mapped->require_linear_bound(*bound);
        else {
          const auto& identities=shape.hidden==1024?embedding_vector_identities:generation_vector_identities;
          const VectorIdentity* identity=nullptr;for(const auto& item:identities)if(name==item.name){identity=&item;break;}
          if(!identity||identity->rows!=rows)throw std::runtime_error("VERSION_MISMATCH");
          mapped->require_vector_identity(identity->sha256);
        }
        found=mappings.emplace(name,std::move(mapped)).first;
      }
      auto mapped=found->second;
      out.server_reader_worker_safe=true;
      out.server_reader=[mapped](std::size_t r,std::size_t c){return mapped->at(r,c);};
    }return out;
  }
  PrivateMatrix down(std::size_t layer){
    if(layer>=28)throw std::invalid_argument("BAD_SHAPE");
    return get("model.layers."+std::to_string(layer)+".mlp.down_proj.weight",shape.hidden,shape.ffn);
  }
 public:
  CheckpointWeights(const std::filesystem::path& directory,ModelShape dimensions,Role role):root(directory),shape(dimensions),server(role==Role::Server){shape.validate();embedding=get("model.embed_tokens.weight",shape.vocabulary,shape.hidden);}
  const PrivateMatrix& token_table()override{return embedding;}
  PrivateMatrix final_norm()override{return get("model.norm.weight",shape.hidden,1,true);}
  LayerWeights layer(std::size_t l)override{
    if(l>=28)throw std::invalid_argument("BAD_SHAPE");
    auto p="model.layers."+std::to_string(l)+".";
    // Keep only the public current layer and tied embedding mapping. Captured
    // readers retain their mapping until an in-flight layer releases it.
    for(auto it=mappings.begin();it!=mappings.end();){if(it->first!="model.embed_tokens.weight"&&it->first.rfind(p,0)!=0)it=mappings.erase(it);else ++it;}
    return {get(p+"input_layernorm.weight",shape.hidden,1,true),get(p+"self_attn.q_proj.weight",2048,shape.hidden),
      get(p+"self_attn.k_proj.weight",1024,shape.hidden),get(p+"self_attn.v_proj.weight",1024,shape.hidden),
      get(p+"self_attn.q_norm.weight",128,1,true),get(p+"self_attn.k_norm.weight",128,1,true),
      get(p+"self_attn.o_proj.weight",shape.hidden,2048),get(p+"post_attention_layernorm.weight",shape.hidden,1,true),
      get(p+"mlp.gate_proj.weight",shape.ffn,shape.hidden),get(p+"mlp.up_proj.weight",shape.ffn,shape.hidden),down(l)};
  }
};
struct Constants {
  std::array<Row,3> exp;std::vector<Row> cos,sin;Word attention_scale;
  explicit Constants(const std::filesystem::path& path){
    std::ifstream f(path,std::ios::binary);char magic[8];if(!f.read(magic,8)||memcmp(magic,"PRCONST1",8))throw std::runtime_error("E_CONSTANT_FORMAT");
    auto read=[&](){unsigned char v[8];if(!f.read(reinterpret_cast<char*>(v),8))throw std::runtime_error("E_CONSTANT_FORMAT");Word z=0;for(int i=7;i>=0;--i)z=(z<<8)|v[i];if(z>mask)throw std::runtime_error("E_CONSTANT_FORMAT");return z;};
    attention_scale=read();for(auto& t:exp){t.resize(256);for(auto& x:t)x=read();}
    cos.assign(64,Row(1280));sin=cos;
    for(auto* table:{&cos,&sin})for(auto& feature:*table)for(auto& value:feature)value=read();
    if(f.peek()!=EOF)throw std::runtime_error("E_CONSTANT_FORMAT");
  }
};
inline PromptLayout load_prompt_layout(const std::filesystem::path& path){
  std::ifstream f(path,std::ios::binary);char magic[8];
  if(!f.read(magic,8)||memcmp(magic,"PRTMPL01",8))throw std::runtime_error("E_TEMPLATE_FORMAT");
  auto read=[&](){unsigned char v[8];if(!f.read(reinterpret_cast<char*>(v),8))throw std::runtime_error("E_TEMPLATE_FORMAT");Word z=0;for(int i=7;i>=0;--i)z=(z<<8)|v[i];return z;};
  PromptLayout layout;layout.question_offset=read();layout.evidence1_offset=read();layout.evidence2_offset=read();layout.dummy_token=read();
  Row tokens(1024);for(auto& t:tokens){t=read();if(t>151935)throw std::runtime_error("E_TEMPLATE_FORMAT");}
  std::array<unsigned char,1024> valid{};if(!f.read(reinterpret_cast<char*>(valid.data()),valid.size())||f.peek()!=EOF)throw std::runtime_error("E_TEMPLATE_FORMAT");
  for(std::size_t i=0;i<1024;){
    if(valid[i]>1)throw std::runtime_error("E_TEMPLATE_FORMAT");
    if(!valid[i]){++i;continue;}
    auto start=i;Row ids;
    while(i<1024&&valid[i]==1)ids.push_back(tokens[i++]);
    layout.public_segments.emplace_back(start,std::move(ids));
  }
  layout.validate();return layout;
}
}
