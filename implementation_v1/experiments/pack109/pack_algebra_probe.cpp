#include "prag/bfv.hpp"
#include <openssl/crypto.h>
#include "acyclic_pack.hpp"
#include <iostream>
using namespace prag;
int main() {
  seal::SEALContext ctx(locked_bfv_parameters(),true,seal::sec_level_type::tc128);
  seal::KeyGenerator first(ctx);auto old=first.secret_key();seal::PublicKey pk;first.create_public_key(pk);
  seal::Encryptor enc(ctx,pk);std::vector<seal::Ciphertext> work(16);
  // Public synthetic polynomials with nonzero nuisance coefficients. A real
  // ciphertext check of all intermediate keys, not a private protocol run.
  for(size_t i=0;i<work.size();++i) {
    seal::Plaintext m(16384);for(size_t j=0;j<16384;++j)m[j]=(i*197+j*71)%4096;
    m[0]=(i%2?((uint64_t(1)<<59)-1):1)*(i+1)&((uint64_t(1)<<59)-1);enc.encrypt(m,work[i]);
  }
  for(unsigned l=0;l<4;++l) {
    seal::KeyGenerator next(ctx);auto keys=acyclic_keys(old,next.secret_key(),ctx,l);
    for(size_t i=0;i<work.size()/2;++i)work[i]=pack_pair(std::move(work[2*i]),std::move(work[2*i+1]),l,keys,ctx);
    work.resize(work.size()/2);old=next.secret_key();
  }
  seal::Decryptor dec(ctx,old);seal::Plaintext out;dec.decrypt(work.front(),out);
  for(size_t i=0;i<16;++i) {
    size_t r=0;for(unsigned b=0;b<4;++b)r=(r<<1)|((i>>b)&1);
    uint64_t expected=(i%2?((uint64_t(1)<<59)-1):1)*(i+1)*16&((uint64_t(1)<<59)-1);
    if(out[r*1024]!=expected)throw std::runtime_error("PACK_ALGEBRA_FAILED");
  }
  std::cout<<"{\"coordinates\":16,\"acyclic_levels\":4,\"result\":\"exact\"}\n";
}
