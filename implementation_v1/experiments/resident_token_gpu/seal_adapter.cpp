#include "resident.hpp"
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <seal/util/uintarithsmallmod.h>
#include <algorithm>
#include <array>
#include <stdexcept>
namespace prag::experimental::resident {
namespace {
void require(bool value){if(!value)throw std::invalid_argument("E_GPU_TOKEN_CONTEXT_LAYOUT");}
struct LocalWiper {std::vector<uint64_t>& values;~LocalWiper(){volatile uint64_t* p=values.data();for(size_t i=0;i<values.size();++i)p[i]=0;}};
void validate_context(const seal::SEALContext& context){
 constexpr std::array<uint64_t,7> locked={1152921504599080961ULL,1152921504600260609ULL,1152921504604979201ULL,1152921504605962241ULL,1152921504606584833ULL,1152921504606683137ULL,1152921504606748673ULL};
 const auto key=context.key_context_data(),data=context.first_context_data();
 require(context.parameters_set()&&key&&data);
 for(const auto& level:{key,data})require(level->parms().scheme()==seal::scheme_type::bfv&&level->parms().poly_modulus_degree()==degree&&level->parms().plain_modulus().value()==(uint64_t(1)<<59)&&level->qualifiers().sec_level==seal::sec_level_type::tc128);
 require(key->parms().coeff_modulus().size()==7&&data->parms().coeff_modulus().size()==6);
 for(size_t i=0;i<7;++i)require(key->parms().coeff_modulus()[i].value()==locked[i]);
 for(size_t i=0;i<6;++i)require(data->parms().coeff_modulus()[i].value()==locked[i]);
}
}
TokenWeights::TokenWeights(const seal::SEALContext& context,const std::vector<std::vector<seal::Ciphertext>>& inputs){
 validate_context(context);const size_t slots=inputs.size(),limbs=degree*primes;
 require(slots>=1&&slots<=4);
 // Complete structural validation happens before allocation/transform/upload.
 for(const auto& row:inputs){require(row.size()==blocks);for(const auto& ct:row)require(ct.size()==2&&!ct.is_ntt_form()&&ct.parms_id()==context.first_parms_id()&&ct.poly_modulus_degree()==degree&&ct.coeff_modulus_size()==primes&&seal::is_valid_for(ct,context));}
 std::vector<uint64_t> flat(blocks*slots*2*limbs);LocalWiper wiper{flat};
 const auto cd=context.first_context_data();
 for(size_t block=0;block<blocks;++block)for(size_t slot=0;slot<slots;++slot)for(size_t component=0;component<2;++component){
  auto* dest=flat.data()+((block*slots+slot)*2+component)*limbs;
  std::copy_n(inputs[slot][block].data(component),limbs,dest);
  for(size_t prime=0;prime<primes;++prime)seal::util::ntt_negacyclic_harvey(dest+prime*degree,cd->small_ntt_tables()[prime]);
 }
 context_=std::make_shared<seal::SEALContext>(context);
 std::vector<uint64_t> moduli(primes),roots(primes*degree*2),inverse(roots.size()),invn(2*primes);
 for(size_t k=0;k<primes;++k){moduli[k]=cd->parms().coeff_modulus()[k].value();for(size_t j=0;j<degree;++j){const auto r=cd->small_ntt_tables()[k].get_from_root_powers()[j];roots[(k*degree+j)*2]=r.operand;roots[(k*degree+j)*2+1]=r.quotient;}}
 for(size_t k=0;k<primes;++k){const auto& table=cd->small_ntt_tables()[k];
  for(size_t j=0;j<degree;++j){auto v=table.get_from_inv_root_powers()[j];inverse[(k*degree+j)*2]=v.operand;inverse[(k*degree+j)*2+1]=v.quotient;}
  auto n=table.inv_degree_modulo();invn[k*2]=n.operand;invn[k*2+1]=n.quotient;}
 initialize(slots,flat,moduli,roots,inverse,invn);
}

DeviceCipherBatch TokenWeights::evaluate_finished(const std::vector<uint64_t>& weights,const std::vector<uint64_t>& local){
 const auto cd=context_->first_context_data();const auto& q=cd->parms().coeff_modulus();
 std::vector<uint64_t> scaled(local.size()*primes);LocalWiper wipe{scaled};
 for(size_t slot=0;slot<local.size();++slot){auto m=local[slot];require(m<(uint64_t(1)<<59));
  // SEAL scalingvariant.cpp: floor((r*m + plain_upper_half_threshold)/t).
  const auto numerator=static_cast<unsigned __int128>(cd->coeff_modulus_mod_plain_modulus())*m+cd->plain_upper_half_threshold();
  const uint64_t fix=uint64_t(numerator/(uint64_t(1)<<59));
  for(size_t k=0;k<primes;++k)scaled[slot*primes+k]=seal::util::multiply_add_uint_mod(m,cd->coeff_div_plain_modulus()[k],fix,q[k]);
 }
 return evaluate_device(weights,scaled);
}
void TokenWeights::bind_local_shares(const std::vector<std::vector<uint64_t>>& shares){
 const auto cd=context_->first_context_data();std::vector<uint64_t> flat,delta;LocalWiper wipe{flat};
 require(!shares.empty()&&shares.size()<=4);flat.reserve(shares.size()*blocks*block_width);
 for(const auto& row:shares){require(row.size()==blocks*block_width);for(auto x:row){require(x<(uint64_t(1)<<59));flat.push_back(x);}}
 for(size_t k=0;k<primes;++k){const auto x=cd->coeff_div_plain_modulus()[k];delta.push_back(x.operand);delta.push_back(x.quotient);}
 initialize_local(flat,delta,cd->coeff_modulus_mod_plain_modulus(),cd->plain_upper_half_threshold());
}
DeviceCipherBatch TokenWeights::evaluate_local_finished(const std::vector<uint64_t>& weights){return evaluate_device(weights,{});}

}
