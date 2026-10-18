#include "ntt.hpp"
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <array>
#include <stdexcept>
namespace prag::experimental::ntt {
NttBatch::NttBatch(const seal::SEALContext& context,bool key_level,size_t batches){
 constexpr std::array<uint64_t,7> locked={1152921504599080961ULL,1152921504600260609ULL,
 1152921504604979201ULL,1152921504605962241ULL,1152921504606584833ULL,
 1152921504606683137ULL,1152921504606748673ULL};
 auto require=[](bool v){if(!v)throw std::invalid_argument("E_NTT_CONTEXT");};
 const auto key=context.key_context_data(),data=context.first_context_data();
 require(context.parameters_set()&&key&&data&&batches>=1&&batches<=19);
 for(const auto& level:{key,data}){
  const auto& p=level->parms();
  require(p.scheme()==seal::scheme_type::bfv&&p.poly_modulus_degree()==degree&&
   p.plain_modulus().value()==(uint64_t(1)<<59)&&level->qualifiers().sec_level==seal::sec_level_type::tc128);
 }
 require(key->parms().coeff_modulus().size()==7&&data->parms().coeff_modulus().size()==6);
 for(size_t i=0;i<7;++i)require(key->parms().coeff_modulus()[i].value()==locked[i]);
 for(size_t i=0;i<6;++i)require(data->parms().coeff_modulus()[i].value()==locked[i]);
 const auto level=key_level?key:data;const size_t primes=level->parms().coeff_modulus().size();
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
 initialize(batches,q,forward,inverse,invn);
}
}

namespace prag::experimental::ntt {
void NttBatch::bind_client_key(const seal::SEALContext& context,const std::vector<uint64_t>& key){
 constexpr uint64_t locked[]={1152921504599080961ULL,1152921504600260609ULL,1152921504604979201ULL,1152921504605962241ULL,1152921504606584833ULL,1152921504606683137ULL};
 const auto cd=context.first_context_data();
 if(!cd||key.size()!=6*degree)throw std::invalid_argument("E_CLIENT_KEY_SHAPE");
 if(cd->parms().coeff_modulus().size()!=6||cd->parms().poly_modulus_degree()!=degree)throw std::invalid_argument("E_CLIENT_KEY_CONTEXT");
 for(size_t k=0;k<6;++k)if(cd->parms().coeff_modulus()[k].value()!=locked[k])throw std::invalid_argument("E_CLIENT_KEY_CONTEXT");
 std::vector<uint64_t> pairs(key.size()*2);
 struct Wipe {std::vector<uint64_t>& x;~Wipe(){volatile uint64_t* p=x.data();for(size_t i=0;i<x.size();++i)p[i]=0;}} wipe{pairs};
 for(size_t k=0;k<6;++k)for(size_t i=0;i<degree;++i){
  auto q=cd->parms().coeff_modulus()[k];auto v=key[k*degree+i];
  if(v>=q.value())throw std::invalid_argument("E_CLIENT_KEY_CANONICAL");
  seal::util::MultiplyUIntModOperand a;a.set(v,q);pairs[2*(k*degree+i)]=a.operand;pairs[2*(k*degree+i)+1]=a.quotient;
 }
 upload_client_key(pairs);
}
}
