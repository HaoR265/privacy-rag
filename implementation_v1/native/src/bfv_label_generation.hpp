#pragma once
#include "utils/prg.h"
#include <openssl/rand.h>
#include <openssl/crypto.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace label_batch {
struct Wipe {
 void *p;size_t n;~Wipe() noexcept {OPENSSL_cleanse(p,n);}
 Wipe(void*p,size_t n):p(p),n(n){} Wipe(const Wipe&)=delete;Wipe&operator=(const Wipe&)=delete;
};
inline void generate(sci::block128* out,size_t count) {
 if(!out || !count || count>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("LABEL_SHAPE");
 alignas(16) unsigned char seed[16]{};Wipe seed_wipe(seed,sizeof seed);
 if(RAND_priv_bytes(seed,sizeof seed)!=1)throw std::runtime_error("LABEL_RNG");
 // Explicit per-input fresh secret seed. Never persisted, reused, or public.
 sci::PRG128 prg(seed);static_assert(std::is_trivially_destructible<sci::AES_KEY>::value,"AES_KEY must have no owned pointers");
 Wipe aes_wipe(&prg.aes,sizeof prg.aes);Wipe counter_wipe(&prg.counter,sizeof prg.counter);
 OPENSSL_cleanse(seed,sizeof seed);
 prg.random_block(out,int(count));
}
}
