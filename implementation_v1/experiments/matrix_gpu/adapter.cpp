#include "matrix.hpp"
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <array>
#include <algorithm>
#include <stdexcept>
namespace matrix_gpu { namespace {void require(bool v){if(!v)throw std::invalid_argument("MATRIX_GPU_CONTEXT");}
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

Matrix::Matrix(const seal::SEALContext& context,const std::vector<seal::Ciphertext>& inputs){validate_context(context);require(!inputs.empty()&&inputs.size()<=512);const auto cd=context.first_context_data();const size_t limbs=degree*primes;std::vector<uint64_t> flat(inputs.size()*2*limbs);
 struct Wipe{std::vector<uint64_t>& v;~Wipe(){volatile uint64_t* p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}} wipe{flat};
 for(size_t b=0;b<inputs.size();++b){const auto& ct=inputs[b];require(ct.size()==2&&!ct.is_ntt_form()&&ct.parms_id()==context.first_parms_id()&&ct.poly_modulus_degree()==degree&&ct.coeff_modulus_size()==primes&&seal::is_valid_for(ct,context));std::copy_n(ct.data(),2*limbs,flat.data()+b*2*limbs);}
 std::vector<uint64_t> q(primes),f(primes*degree*2),r(f.size()),ni(primes*2);for(size_t k=0;k<primes;++k){auto& t=cd->small_ntt_tables()[k];q[k]=cd->parms().coeff_modulus()[k].value();for(size_t j=0;j<degree;++j){auto a=t.get_from_root_powers()[j],b=t.get_from_inv_root_powers()[j];f[(k*degree+j)*2]=a.operand;f[(k*degree+j)*2+1]=a.quotient;r[(k*degree+j)*2]=b.operand;r[(k*degree+j)*2+1]=b.quotient;}auto n=t.inv_degree_modulo();ni[k*2]=n.operand;ni[k*2+1]=n.quotient;}
 initialize(flat,q,f,r,ni);
}
}
