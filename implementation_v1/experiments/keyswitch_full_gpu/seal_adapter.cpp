#include "switch.hpp"
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <array>
#include <seal/util/galois.h>
#include <stdexcept>
namespace prag::experimental_full_switch {
Switch::Switch(const seal::SEALContext& context,const seal::GaloisKeys& keys,uint32_t element){
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
 require(element%2==1&&element<2*degree&&keys.parms_id()==context.key_parms_id()&&keys.has_key(element));
 const auto& selected=keys.data()[seal::GaloisKeys::get_index(element)];require(selected.size()==6);
 std::vector<uint64_t> flat(6*2*7*degree),pi(12);
 struct Wiper {std::vector<uint64_t>& v;~Wiper(){volatile uint64_t* p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}} wipe{flat};
 for(size_t j=0;j<6;++j){const auto& ct=selected[j].data();require(seal::is_valid_for(selected[j],context)&&ct.size()==2&&ct.is_ntt_form()&&ct.parms_id()==context.key_parms_id());std::copy_n(ct.data(),2*7*degree,flat.data()+j*2*7*degree);}
 for(size_t i=0;i<6;++i){auto value=key->rns_tool()->inv_q_last_mod_q()[i];pi[2*i]=value.operand;pi[2*i+1]=value.quotient;}
 context_=std::make_shared<seal::SEALContext>(context);element_=element;
 initialize(q,forward,inverse,invn,pi,flat);
}
seal::Ciphertext Switch::apply(const seal::Ciphertext& input){
 const auto& context=*context_;
 if(!seal::is_valid_for(input,context)||input.size()!=2||input.is_ntt_form()||input.parms_id()!=context.first_parms_id())throw std::invalid_argument("E_SWITCH_CIPHERTEXT");
 std::vector<uint64_t> transformed(2*6*degree);
 struct Wiper {std::vector<uint64_t>& v;~Wiper(){volatile uint64_t* p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}} wipe{transformed};
 auto cd=context.first_context_data();auto tool=context.key_context_data()->galois_tool();
 for(size_t c=0;c<2;++c)tool->apply_galois(seal::util::ConstRNSIter(input.data(c),degree),6,element_,cd->parms().coeff_modulus(),seal::util::RNSIter(transformed.data()+c*6*degree,degree));
 auto flat=evaluate(transformed);Wiper wipe_result{flat};seal::Ciphertext result;result.resize(context,context.first_parms_id(),2);std::copy(flat.begin(),flat.end(),result.data());return result;
}
seal::Ciphertext Switch::packing_pair(const seal::Ciphertext& left,const seal::Ciphertext& right,Switch& other,unsigned level){
 const auto& context=*context_;
 for(auto ct:{&left,&right})if(!seal::is_valid_for(*ct,context)||ct->size()!=2||ct->is_ntt_form()||ct->parms_id()!=context.first_parms_id())throw std::invalid_argument("E_PACK_CIPHERTEXT");
 auto flat=packing_evaluate(left.data(),right.data(),other,level);
 struct Wipe{std::vector<uint64_t>& v;~Wipe(){volatile uint64_t* p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}}wipe{flat};
 seal::Ciphertext result;result.resize(context,context.first_parms_id(),2);std::copy(flat.begin(),flat.end(),result.data());return result;
}
}
