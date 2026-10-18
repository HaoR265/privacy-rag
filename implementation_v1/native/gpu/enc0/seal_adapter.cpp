#include "enc0.hpp"
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <array>
#include <seal/util/rlwe.h>
#include <seal/util/polyarithsmallmod.h>
#include <stdexcept>
namespace prag::enc0_gpu {
PublicZero::PublicZero(const seal::SEALContext& context,const seal::PublicKey& pk,const std::string& uuid,const std::string& pci){
 constexpr std::array<uint64_t,7> locked={1152921504599080961ULL,1152921504600260609ULL,
 1152921504604979201ULL,1152921504605962241ULL,1152921504606584833ULL,
 1152921504606683137ULL,1152921504606748673ULL};
 auto require=[](bool v){if(!v)throw std::invalid_argument("E_NTT_CONTEXT");};
 const auto key=context.key_context_data(),data=context.first_context_data();
 require(context.parameters_set()&&key&&data);
 for(const auto& level:{key,data}){
  const auto& p=level->parms();
  require(p.scheme()==seal::scheme_type::bfv&&p.poly_modulus_degree()==degree&&
   p.plain_modulus().value()==(uint64_t(1)<<59)&&level->qualifiers().sec_level==seal::sec_level_type::tc128);
 }
 require(key->parms().coeff_modulus().size()==7&&data->parms().coeff_modulus().size()==6);
 for(size_t i=0;i<7;++i)require(key->parms().coeff_modulus()[i].value()==locked[i]);
 for(size_t i=0;i<6;++i)require(data->parms().coeff_modulus()[i].value()==locked[i]);
 const auto level=key;const size_t primes=level->parms().coeff_modulus().size();
 std::vector<uint64_t> q(primes),forward(primes*degree*2),inverse(forward.size()),invn(primes*2);
 for(size_t k=0;k<primes;++k){
  const auto& table=level->small_ntt_tables()[k];q[k]=table.modulus().value();
  require(table.coeff_count_power()==14&&q[k]==locked[k]);
  for(size_t i=0;i<degree;++i){
   auto f=table.get_from_root_powers()[i],r=table.get_from_inv_root_powers()[i];
   forward[(k*degree+i)*2]=f.operand;forward[(k*degree+i)*2+1]=f.quotient;
   inverse[(k*degree+i)*2]=r.operand;inverse[(k*degree+i)*2+1]=r.quotient;
  }
  auto n=table.inv_degree_modulo();invn[k*2]=n.operand;invn[k*2+1]=n.quotient;
 }
 require(seal::is_valid_for(pk,context)&&pk.data().size()==2&&pk.data().is_ntt_form()&&pk.parms_id()==context.key_parms_id());
 std::vector<uint64_t> flat(pk.data().data(),pk.data().data()+2*7*degree),pi(12);
 for(size_t i=0;i<6;++i){auto value=key->rns_tool()->inv_q_last_mod_q()[i];pi[2*i]=value.operand;pi[2*i+1]=value.quotient;}
 context_=std::make_shared<seal::SEALContext>(context);
 initialize(q,forward,inverse,invn,pi,flat,uuid,pci);
}

Samples::~Samples(){volatile uint64_t* p=words.data();for(size_t i=0;i<words.size();++i)p[i]=0;}
void sample(const seal::SEALContext& context,Samples& s){
 const auto& p=context.key_context_data()->parms();if(s.words.size()!=3*7*degree)throw std::invalid_argument("E_ENC0_SAMPLE_SHAPE");
 auto rng=p.random_generator()->create();seal::util::sample_poly_ternary(rng,p,s.words.data());
 seal::util::SEAL_NOISE_SAMPLER(rng,p,s.words.data()+7*degree);seal::util::SEAL_NOISE_SAMPLER(rng,p,s.words.data()+2*7*degree);
}
seal::Ciphertext PublicZero::encrypt_from_samples(const Samples& s){
 if(s.words.size()!=3*7*degree)throw std::invalid_argument("E_ENC0_SAMPLE_SHAPE");const auto& q=context_->key_context_data()->parms().coeff_modulus();
 for(size_t i=0;i<s.words.size();++i)if(s.words[i]>=q[(i/degree)%7].value())throw std::invalid_argument("E_ENC0_SAMPLE_CANONICAL");
 auto flat=evaluate(s.words);struct W{std::vector<uint64_t>&v;~W(){volatile uint64_t*p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}}wipe{flat};
 seal::Ciphertext out;out.resize(*context_,context_->first_parms_id(),2);std::copy(flat.begin(),flat.end(),out.data());out.is_ntt_form()=false;out.scale()=1;out.correction_factor()=1;return out;
}
seal::Ciphertext PublicZero::encrypt_zero(){Samples s;sample(*context_,s);return encrypt_from_samples(s);}
seal::Ciphertext reference(const seal::SEALContext& context,const seal::PublicKey& pk,const Samples& s){
 auto key=context.key_context_data();const auto& q=key->parms().coeff_modulus();auto pool=seal::MemoryManager::GetPool(seal::mm_prof_opt::mm_force_new,true);
 auto u=seal::util::allocate_poly(degree,7,pool);std::copy_n(s.words.data(),7*degree,u.get());seal::Ciphertext temp(pool);temp.resize(context,context.key_parms_id(),2);
 for(size_t p=0;p<7;++p){seal::util::ntt_negacyclic_harvey(u.get()+p*degree,key->small_ntt_tables()[p]);for(size_t c=0;c<2;++c){auto dst=temp.data(c)+p*degree;seal::util::dyadic_product_coeffmod(u.get()+p*degree,pk.data().data(c)+p*degree,degree,q[p],dst);seal::util::inverse_ntt_negacyclic_harvey(dst,key->small_ntt_tables()[p]);seal::util::add_poly_coeffmod(dst,s.words.data()+(c+1)*7*degree+p*degree,degree,q[p],dst);}}
 seal::Ciphertext out;out.resize(context,context.first_parms_id(),2);for(size_t c=0;c<2;++c){key->rns_tool()->divide_and_round_q_last_inplace(seal::util::RNSIter(temp.data(c),degree),pool);std::copy_n(temp.data(c),6*degree,out.data(c));}return out;
}
}
