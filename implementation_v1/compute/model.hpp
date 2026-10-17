#pragma once
#include "retrieval.hpp"
#include "numeric_profile.hpp"
#include "prag/joint_matrix.hpp"
#include <functional>
#include <memory>

namespace prag::model {
struct ModelShape {
  std::size_t hidden,ffn,vocabulary;
  static ModelShape embedding(){return {1024,3072,151669};}
  static ModelShape generation(){return {2048,6144,151936};}
  numeric_profile::QKBounds qk_bounds()const{return hidden==1024?numeric_profile::embedding:numeric_profile::generation;}
  Word residual_abs_f12()const{return numeric_profile::rms_coordinate_cap(hidden);}
  void validate()const {
    if(!((hidden==1024&&ffn==3072&&vocabulary==151669)||(hidden==2048&&ffn==6144&&vocabulary==151936)))throw std::invalid_argument("BAD_SHAPE");
  }
};
struct LayerWeights {PrivateMatrix in_norm,q,k,v,q_norm,k_norm,o,ffn_norm,gate,up,down;};
// A provider loads only server-owned weights; C receives only public dimensions.
struct Weights {
  virtual ~Weights()=default;
  virtual const PrivateMatrix& token_table()=0;
  virtual LayerWeights layer(std::size_t)=0;
  virtual PrivateMatrix final_norm()=0;
};
using Row=std::vector<Word>;
using Rows=std::vector<Row>;
struct Cache {
  static constexpr size_t slots=1280, width=128, heads=8, depth=28;
  std::unique_ptr<ShareStorage> storage;
  std::vector<Bit> valid;
  SecureOps* const owner;
  std::size_t next_write=0;
  bool usable=true;
  explicit Cache(SecureOps& s,LocalShares& gpu):storage(gpu.allocate(2*depth*heads*slots*width)),valid(slots,s.public_bit(false)),owner(&s){}
  Cache(const Cache&)=delete;Cache& operator=(const Cache&)=delete;
  ~Cache(){clear();}
  void clear()noexcept{usable=false;if(storage)storage->clear();std::fill(valid.begin(),valid.end(),false);next_write=0;}
  size_t offset(bool value,size_t layer,size_t head,size_t row)const {
    if(layer>=depth||head>=heads||row>=slots)throw std::invalid_argument("BAD_SHAPE");
    return (((size_t(value)*depth+layer)*heads+head)*slots+row)*width;
  }
  void write(bool value,size_t layer,size_t head,size_t row,const Row& x){
    if(x.size()!=width)throw std::invalid_argument("BAD_SHAPE");storage->write(offset(value,layer,head,row),x);
  }
  Row read(bool value,size_t layer,size_t head,size_t count){
    if(!count||count>slots)throw std::invalid_argument("BAD_SHAPE");return storage->read(offset(value,layer,head,0),count*width);
  }
  void validate_storage()const{if(!storage||valid.size()!=slots||next_write>1279)throw std::invalid_argument("BAD_SHAPE");}
};
struct Forward {Rows hidden;std::vector<Bit> valid;};
class Transformer {
  Numerics& n;SecureOps& s;ModelShape shape;Weights& weights;
  std::vector<std::vector<Word>> cos_table,sin_table;Word attention_scale;
  void require_matrix(const PrivateMatrix& w,std::size_t rows,std::size_t cols)const {
    if(w.rows!=rows||w.cols!=cols||w.fractional_bits!=12)throw std::invalid_argument("BAD_SHAPE");
    if(w.owned()!=(s.role()==Role::Server))throw std::invalid_argument("BAD_STATE");
  }
  void require_layer(const LayerWeights& w)const {
    require_matrix(w.in_norm,shape.hidden,1);require_matrix(w.ffn_norm,shape.hidden,1);
    require_matrix(w.q_norm,128,1);require_matrix(w.k_norm,128,1);
    require_matrix(w.q,2048,shape.hidden);require_matrix(w.k,1024,shape.hidden);require_matrix(w.v,1024,shape.hidden);
    require_matrix(w.o,shape.hidden,2048);require_matrix(w.gate,shape.ffn,shape.hidden);require_matrix(w.up,shape.ffn,shape.hidden);require_matrix(w.down,shape.hidden,shape.ffn);
  }
  Rows project_many(const Rows& x,const PrivateMatrix& w,bool rescale=true,
                    Word input_cap=numeric_profile::main_abs_f12,Word output_cap=numeric_profile::main_abs_f12){
    if(x.empty()||w.fractional_bits!=12||w.owned()!=(s.role()==Role::Server))throw std::invalid_argument("BAD_SHAPE");
    for(const auto& row:x)if(w.cols!=row.size())throw std::invalid_argument("BAD_SHAPE");
    // Exact registered row L1 bounds every full and arbitrary partial sum.
    if(!input_cap||w.input_abs_f12!=input_cap||!w.maximum_row_l1||
       w.maximum_row_l1>((Word{1}<<56)-1)/input_cap)throw std::invalid_argument("E_WEIGHT_BOUND");
    auto reader=[&](std::size_t begin,std::size_t columns,std::size_t first,std::size_t count){
      if(!w.owned()||begin>w.cols||columns>w.cols-begin||first>w.rows||count>w.rows-first)
        throw std::invalid_argument("BAD_SHAPE");
      Rows block(count,Row(columns));
      for(std::size_t r=0;r<count;++r)for(std::size_t c=0;c<columns;++c)block[r][c]=w.at(first+r,begin+c);
      return block;
    };
    Rows result;result.reserve(x.size());
    // Public batch rows share each encrypted input polynomial. The native
    // matrix instance owns all output tiles and reuses those input ciphertexts.
    for(std::size_t start=0;start<x.size();start+=16){
      const auto count=std::min<std::size_t>(16,x.size()-start);
      Rows input(x.begin()+start,x.begin()+start+count);
      auto tile=s.private_matrix(input,w.rows,12,12,reader);
      if(tile.size()!=count)throw std::invalid_argument("BAD_SHAPE");
      for(auto& row:tile){
        if(row.size()!=w.rows)throw std::invalid_argument("BAD_SHAPE");
        if(rescale)row=n.checked_many(n.floor_many(row,12),output_cap);
        result.push_back(std::move(row));
      }
    }
    return result;
  }
  std::vector<Rows> project_joint_many(const Rows& x,const std::vector<PrivateMatrix>& maps){
    if(x.empty()||maps.size()<2||maps.size()>3)throw std::invalid_argument("BAD_SHAPE");
    std::vector<std::size_t> sizes;std::vector<PrivateWeightBlock> readers;
    for(const auto& w:maps){
      if(w.fractional_bits!=12||w.owned()!=(s.role()==Role::Server))throw std::invalid_argument("BAD_SHAPE");
      for(const auto& row:x)if(row.size()!=w.cols)throw std::invalid_argument("BAD_SHAPE");
      if(w.input_abs_f12!=numeric_profile::main_abs_f12||!w.maximum_row_l1||
         w.maximum_row_l1>((Word{1}<<56)-1)/numeric_profile::main_abs_f12)throw std::invalid_argument("E_WEIGHT_BOUND");
      sizes.push_back(w.rows);
      readers.push_back([w](std::size_t begin,std::size_t columns,std::size_t first,std::size_t count){
        if(!w.owned()||begin>w.cols||columns>w.cols-begin||first>w.rows||count>w.rows-first)throw std::invalid_argument("BAD_SHAPE");
        Rows result(count,Row(columns));
        for(std::size_t r=0;r<count;++r)for(std::size_t c=0;c<columns;++c)result[r][c]=w.at(first+r,begin+c);
        return result;
      });
    }
    std::vector<Rows> result(maps.size());
    for(std::size_t start=0;start<x.size();start+=16){
      const auto count=std::min<std::size_t>(16,x.size()-start);
      auto tile=private_joint_matrix(s,Rows(x.begin()+start,x.begin()+start+count),sizes,12,12,readers);
      for(auto& row:tile){
        std::size_t offset=0;
        for(std::size_t g=0;g<maps.size();++g){
          Row output(row.begin()+offset,row.begin()+offset+sizes[g]);offset+=sizes[g];
          output=n.checked_many(n.floor_many(output,12),numeric_profile::main_abs_f12);
          result[g].push_back(std::move(output));
        }
      }
    }
    return result;
  }
  Row project(const Row& x,const PrivateMatrix& w,bool rescale=true,
              Word input_cap=numeric_profile::main_abs_f12,Word output_cap=numeric_profile::main_abs_f12){
    return project_many(Rows{x},w,rescale,input_cap,output_cap).at(0);
  }
  Rows tokens(Row token_ids){
    if(token_ids.empty()||token_ids.size()>4)throw std::invalid_argument("BAD_SHAPE");
    const auto& table=weights.token_table();
    require_matrix(table,shape.vocabulary,shape.hidden);
    for(auto& token_id:token_ids){
      Bit valid=s.bnot(s.gt(token_id,n.constant(shape.vocabulary-1),false));n.fail(s.bnot(valid));
      token_id=s.select(valid,token_id,n.constant(0));
    }
    // Native lookup builds a fixed shared one-hot vector, scans all 155648
    // entries and returns 512 distinct targets per ciphertext under fresh masks.
    // Only S invokes this reader. Public row/block coordinates never select a
    // secret Token; no H-by-vocabulary transposed matrix is materialized.
    PrivateWeightBlock reader=[&](std::size_t begin,std::size_t columns,
                    std::size_t first,std::size_t rows){
      if(!table.owned()||begin>155648||columns>155648-begin||
         first>shape.hidden||rows>shape.hidden-first)
        throw std::invalid_argument("BAD_SHAPE");
      std::vector<Row> w(rows,Row(columns,0));
      for(std::size_t j=0;j<columns && begin+j<shape.vocabulary;++j)
        for(std::size_t r=0;r<rows;++r)w[r][j]=table.at(begin+j,first+r);
      return w;
    };
    if(table.server_reader_worker_safe)reader=worker_safe_private_reader(std::move(reader));
    Rows result;
    if(token_ids.size()==1)result.push_back(s.private_token_lookup(token_ids[0],shape.vocabulary,shape.hidden,reader));
    else result=s.private_token_lookup_many(token_ids,shape.vocabulary,shape.hidden,reader);
    if(result.size()!=token_ids.size())throw std::invalid_argument("BAD_SHAPE");
    // f0 one-hot times f12 weights already yields raw f12: no floor12 here.
    for(auto& row:result){
      if(row.size()!=shape.hidden)throw std::invalid_argument("BAD_SHAPE");
      row=n.checked_many(row,numeric_profile::main_abs_f12);
    }
    return result;
  }
 public:
  Transformer(Numerics& num,ModelShape dims,Weights& w,std::vector<Row> cosine,std::vector<Row> sine,Word attn)
   :n(num),s(num.ops()),shape(dims),weights(w),cos_table(std::move(cosine)),sin_table(std::move(sine)),attention_scale(attn){
    shape.validate();if(cos_table.size()!=64||sin_table.size()!=64)throw std::invalid_argument("BAD_SHAPE");
    for(std::size_t i=0;i<64;++i)if(cos_table[i].size()!=1280||sin_table[i].size()!=1280)throw std::invalid_argument("BAD_SHAPE");
  }
  // Complete layer used by the fixed-budget forward and the isolated M1 entry.
  // Exposes shares only; no per-layer opening or clear reference is available.
  Rows layer_from_shares(const Rows& x,const std::vector<Bit>& valid,const Row& positions,size_t layer,Cache* cache=nullptr,size_t write_start=0){
    const size_t L=x.size(),prefix=cache?write_start+L:L;
    n.local();
    if(!L||L>1024||valid.size()!=L||positions.size()!=L||layer>=28||prefix>1280)throw std::invalid_argument("BAD_SHAPE");
    for(const auto& row:x)if(row.size()!=shape.hidden)throw std::invalid_argument("BAD_SHAPE");
    if(cache&&(cache->owner!=&s||shape.hidden!=2048))throw std::invalid_argument("BAD_STATE");
    try{

      n.stage("Transformer_layer",true);
      auto w=weights.layer(layer);require_layer(w);
      std::vector<Rows> q(16,Rows(L)),k(8,Rows(prefix)),v(8,Rows(prefix));
      Rows normalized=n.rms_many(x,w.in_norm);
      auto qkv=project_joint_many(normalized,{w.q,w.k,w.v});
      auto& projected_q=qkv[0];auto& projected_k=qkv[1];auto& projected_v=qkv[2];
      const auto bounds=shape.qk_bounds();Rows q_heads,k_heads;
      q_heads.reserve(16*L);k_heads.reserve(8*L);
      for(std::size_t row=0;row<L;++row){
        const auto& q0=projected_q[row];const auto& k0=projected_k[row];
        if(q0.size()!=2048||k0.size()!=1024||projected_v[row].size()!=1024)throw std::invalid_argument("BAD_SHAPE");
        for(size_t h=0;h<16;++h)q_heads.emplace_back(q0.begin()+h*128,q0.begin()+(h+1)*128);
        for(size_t h=0;h<8;++h)k_heads.emplace_back(k0.begin()+h*128,k0.begin()+(h+1)*128);
      }
      q_heads=n.rms_many(q_heads,w.q_norm,bounds.q_norm);
      k_heads=n.rms_many(k_heads,w.k_norm,bounds.k_norm);
      Row public_rope(1280*128);
      for(size_t j=0;j<1280;++j)for(size_t c=0;c<64;++c){public_rope[j*128+c]=cos_table[c][j];public_rope[j*128+c+64]=sin_table[c][j];}
      auto coefficients=n.rope_coefficients(positions,public_rope);
      q_heads=n.rope_many(q_heads,coefficients,16,bounds.q_rope);
      k_heads=n.rope_many(k_heads,coefficients,8,bounds.k_rope);
      for(std::size_t row=0;row<L;++row){
        auto& q0=projected_q[row];auto& k0=projected_k[row];auto& v0=projected_v[row];
        if(q0.size()!=2048||k0.size()!=1024||v0.size()!=1024)throw std::invalid_argument("BAD_SHAPE");
        for(std::size_t h=0;h<16;++h)q[h][row]=q_heads[row*16+h];
        for(std::size_t h=0;h<8;++h){
          Row kh=k_heads[row*8+h],vh(v0.begin()+h*128,v0.begin()+(h+1)*128);
          if(cache){cache->write(false,layer,h,write_start+row,kh);cache->write(true,layer,h,write_start+row,vh);}
          else{k[h][row]=std::move(kh);v[h][row]=std::move(vh);}
        }
      }
      if(cache)for(size_t h=0;h<8;++h){auto keys=cache->read(false,layer,h,prefix),values=cache->read(true,layer,h,prefix);for(size_t j=0;j<prefix;++j){k[h][j]=Row(keys.begin()+j*128,keys.begin()+(j+1)*128);v[h][j]=Row(values.begin()+j*128,values.begin()+(j+1)*128);}}
      Rows concatenated(L,Row(2048));
      for(size_t h=0;h<16;++h){
        LogicalBuffer<Word> shared_keys,shared_values;
        shared_keys.reserve(prefix*128);shared_values.reserve(prefix*128);
        for(size_t j=0;j<prefix;++j){shared_keys.insert(shared_keys.end(),k[h/2][j].begin(),k[h/2][j].end());shared_values.insert(shared_values.end(),v[h/2][j].begin(),v[h/2][j].end());}
        LogicalBuffer<Word> shared_queries;shared_queries.reserve(L*128);
        LogicalBuffer<uint8_t> allowed(L*prefix),query_valid(L);
        for(size_t row=0;row<L;++row){
          shared_queries.insert(shared_queries.end(),q[h][row].begin(),q[h][row].end());query_valid[row]=valid[row];
          for(size_t j=0;j<prefix;++j)allowed[row*prefix+j]=j<=write_start+row?uint8_t(cache?cache->valid[j]:valid[j]):uint8_t(s.public_bit(false));
        }
        // Production prefill/Embedding step 0; Decode write slots 1024..1278 give 1..255.
        const Word step=cache&&write_start>=1024?write_start-1023:0;
        LogicalBuffer<Word> values=n.attention_logical(shared_queries,shared_keys,shared_values,L,prefix,allowed,query_valid,bounds,attention_scale,layer,h,step);
        for(size_t row=0;row<L;++row)std::copy(values.begin()+row*128,values.begin()+(row+1)*128,concatenated[row].begin()+h*128);
      }
      auto attention=project_many(concatenated,w.o);
      Rows next(L,Row(shape.hidden)),mlp_inputs;mlp_inputs.reserve(L);
      for(std::size_t row=0;row<L;++row){
        next[row]=n.checked_many(n.local().add(x[row],attention[row]),shape.residual_abs_f12());
      }
      mlp_inputs=n.rms_many(next,w.ffn_norm);
      auto gate_up=project_joint_many(mlp_inputs,{w.gate,w.up});
      auto& gate=gate_up[0];auto& up=gate_up[1];
      gate=n.swiglu_rows(gate,up);
      auto down=project_many(gate,w.down,true,numeric_profile::swiglu_joint_abs_f12,shape.residual_abs_f12());
      for(size_t row=0;row<L;++row)next[row]=n.checked_many(n.local().add(next[row],down[row]),shape.residual_abs_f12());
      n.stage("Transformer_layer",false);
      return next;
    }catch(...){s.abort_session();if(cache)cache->clear();throw;}
  }
  Forward forward(const Row& ids,const std::vector<Bit>& valid,const Row& positions,Cache* cache=nullptr,std::size_t write_start=0){
    std::size_t L=ids.size();
    if(L!=valid.size()||L!=positions.size()||L==0)throw std::invalid_argument("BAD_SHAPE");
    if(cache){
      if(shape.hidden!=2048)throw std::invalid_argument("BAD_SHAPE");
      if(cache->owner!=&s||!cache->usable)throw std::invalid_argument("BAD_STATE");
      if(!((L==1024&&write_start==0&&cache->next_write==0)||(L==1&&write_start==cache->next_write&&write_start>=1024&&write_start<=1278)))throw std::invalid_argument("REPLAY_OR_ORDER");
      cache->validate_storage();
      // No retry/resume after any partially executed forward call.
      cache->usable=false;
    }
    else if(shape.hidden!=1024||(L!=160&&L!=257)||write_start!=0)throw std::invalid_argument("BAD_SHAPE");
    try {
      n.local();
      Rows x;x.reserve(L);
      // Public fixed budgets yield four-slot batches and, for 257-slot
      // preparation / single Decode, one single-slot tail. Only S's
      // deterministic weight transforms are reused within a batch.
      for(size_t first=0;first<L;first+=4){
        const size_t count=std::min<size_t>(4,L-first);
        if(count!=1&&count!=4)throw std::invalid_argument("BAD_SHAPE");
        const char* phase=count==4?"Token_lookup_batch_4":"Token_lookup";
        n.stage(phase,true);
        auto block=tokens(Row(ids.begin()+first,ids.begin()+first+count));
        for(auto& row:block)x.push_back(std::move(row));
        n.stage(phase,false);
      }
      if(cache)for(std::size_t i=0;i<L;++i)cache->valid[write_start+i]=valid[i];
      const std::size_t prefix=cache?write_start+L:L;
      for(size_t layer=0;layer<28;++layer)x=layer_from_shares(x,valid,positions,layer,cache,write_start);
      auto gamma=weights.final_norm();require_matrix(gamma,shape.hidden,1);x=n.rms_many(x,gamma);
      if(cache){cache->next_write=prefix;cache->usable=true;}
      return {std::move(x),valid};
    } catch(...) {
      // Token lookup and final RMS can fail outside layer_from_shares. The
      // caller may retain its Cache object, so clear here as well as on scope
      // destruction. No partially executed call can leave a reusable cache.
      if(cache)cache->clear();
      s.abort_session();
      throw;
    }
  }
  std::pair<Row,Bit> last_hidden(const Forward& y){
    if(y.hidden.empty()||y.hidden.size()!=y.valid.size())throw std::invalid_argument("BAD_SHAPE");
    for(const auto& row:y.hidden)if(row.size()!=shape.hidden)throw std::invalid_argument("BAD_SHAPE");
    Row output(shape.hidden,n.constant(0));Bit seen=s.public_bit(false);
    for(std::size_t i=y.hidden.size();i-->0;){
      Bit choose=s.band(y.valid[i],s.bnot(seen));seen=s.bor(seen,y.valid[i]);
      output=n.select_many(std::vector<uint8_t>(shape.hidden,choose),y.hidden[i],output);
    }
    return {output,seen};
  }
  std::pair<Row,Bit> embedding(const Row& ids,const std::vector<Bit>& valid,const Row& positions,Bit record_valid){
    if(shape.hidden!=1024)throw std::invalid_argument("BAD_SHAPE");
    auto [h,v]=last_hidden(forward(ids,valid,positions));return n.l2(h,record_valid,v);
  }
  Word greedy(const Row& hidden){
    if(shape.hidden!=2048)throw std::invalid_argument("BAD_SHAPE");
    const auto& table=weights.token_table();require_matrix(table,shape.vocabulary,shape.hidden);
    auto logits=project(hidden,table,false); // tied output head, raw f24
    Row scores,indices;
    for(size_t i=0;i<=151645;++i)if(i!=151644){scores.push_back(logits[i]);indices.push_back(n.constant(i));}
    // Balanced fixed tournament. Strict greater chooses the right candidate;
    // ties keep the left (smaller ID) throughout every public tree level.
    while(scores.size()>1){
      Row left,right,li,ri;
      for(size_t i=0;i+1<scores.size();i+=2){left.push_back(scores[i]);right.push_back(scores[i+1]);li.push_back(indices[i]);ri.push_back(indices[i+1]);}
      auto take=n.compare_many(right,left);
      auto next=n.select_many(take,right,left),ix=n.select_many(take,ri,li);
      if(scores.size()%2){next.push_back(scores.back());ix.push_back(indices.back());}
      scores=std::move(next);indices=std::move(ix);
    }
    return indices[0];
  }
};
}
