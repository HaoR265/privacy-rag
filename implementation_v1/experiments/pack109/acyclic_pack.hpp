// Fixed-parameter acyclic switching keys. No cyclic/KDM evaluation-key graph.
// Only the standalone candidate includes this until protocol evidence exists.
#pragma once
#include <seal/util/rlwe.h>
#include <seal/util/galois.h>
#include <seal/util/polyarithsmallmod.h>
namespace prag {
inline void monomial_shift(seal::Ciphertext& ct,size_t exponent,const seal::SEALContext& ctx) {
  constexpr size_t n=16384;
  const auto& primes=ctx.get_context_data(ct.parms_id())->parms().coeff_modulus();
  std::vector<uint64_t> tmp(n);
  exponent%=2*n;
  for(size_t c=0;c<2;++c)for(size_t p=0;p<primes.size();++p) {
    auto *a=ct.data(c)+p*n;const uint64_t q=primes[p].value();
    for(size_t j=0;j<n;++j) {
      const size_t dest=(j+exponent)%(2*n);
      tmp[dest%n]=dest>=n?(a[j]?q-a[j]:0):a[j];
    }
    std::copy(tmp.begin(),tmp.end(),a);
  }
}
inline seal::GaloisKeys acyclic_keys(const seal::SecretKey& old,
    const seal::SecretKey& next,const seal::SEALContext& ctx,unsigned level) {
  constexpr size_t n=16384,k=7;
  const auto cd=ctx.key_context_data();const auto& primes=cd->parms().coeff_modulus();
  seal::GaloisKeys result;result.parms_id()=ctx.key_parms_id();result.data().resize(n);
  std::vector<uint64_t> rotated(n*k),tmp(n);
  for(uint32_t elt:{uint32_t(1),uint32_t((1u<<(level+1))+1)}) {
    cd->galois_tool()->apply_galois_ntt(seal::util::ConstRNSIter(old.data().data(),n),k,elt,seal::util::RNSIter(rotated.data(),n));
    auto& keys=result.data()[seal::GaloisKeys::get_index(elt)];keys.resize(6);
    for(size_t j=0;j<6;++j) {
      seal::util::encrypt_zero_symmetric(next,ctx,ctx.key_parms_id(),true,false,keys[j].data());
      const uint64_t factor=primes.back().value()%primes[j].value();
      seal::util::multiply_poly_scalar_coeffmod(rotated.data()+j*n,n,factor,primes[j],tmp.data());
      auto *dst=keys[j].data().data(0)+j*n;
      seal::util::add_poly_coeffmod(dst,tmp.data(),n,primes[j],dst);
    }
  }
  OPENSSL_cleanse(rotated.data(),rotated.size()*8);
  return result;
}
inline seal::Ciphertext pack_pair(seal::Ciphertext left,seal::Ciphertext right,
      unsigned level,const seal::GaloisKeys& keys,const seal::SEALContext& ctx) {
  seal::Evaluator eval(ctx);
  monomial_shift(right,16384>>(level+1),ctx);
  seal::Ciphertext difference;eval.sub(left,right,difference);eval.add_inplace(left,right);
  // Both summands switch to the same INDEPENDENT next secret key.
  eval.apply_galois_inplace(left,1,keys);
  eval.apply_galois_inplace(difference,(1u<<(level+1))+1,keys);
  eval.add_inplace(left,difference);return left;
}
inline size_t packed_target(size_t coordinate) {
  size_t r=0;for(unsigned j=0;j<9;++j)r=(r<<1)|((coordinate>>j)&1);
  return r*32;
}
}
