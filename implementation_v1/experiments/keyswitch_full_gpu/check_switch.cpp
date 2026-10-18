#include "switch.hpp"
#include <seal/seal.h>
#include "../../native/src/bfv_acyclic_pack.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
using namespace prag::experimental_full_switch;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){try{
 seal::EncryptionParameters p(seal::scheme_type::bfv);p.set_poly_modulus_degree(degree);p.set_plain_modulus(uint64_t(1)<<59);p.set_coeff_modulus({seal::Modulus(1152921504599080961ULL),seal::Modulus(1152921504600260609ULL),seal::Modulus(1152921504604979201ULL),seal::Modulus(1152921504605962241ULL),seal::Modulus(1152921504606584833ULL),seal::Modulus(1152921504606683137ULL),seal::Modulus(1152921504606748673ULL)});seal::SEALContext context(p,true,seal::sec_level_type::tc128);require(context.parameters_set(),"context invalid");
 seal::KeyGenerator from(context),to(context);seal::Encryptor encryptor(context,from.secret_key());seal::Evaluator eval(context);
 std::cout<<"{\"scope\":\"independent actual encrypted key-switch operator, no model/protocol/decryption\",\"cases\":[";bool first=true;std::vector<double> cleanup_times;
 for(unsigned level=0;level<9;++level){
  auto start=Clock::now();auto keys=prag::acyclic_keys(from.secret_key(),to.secret_key(),context,level);double key_ms=ms(start);
  for(uint32_t elt:{uint32_t(1),uint32_t((1u<<(level+1))+1)}){
   start=Clock::now();Switch gpu(context,keys,elt);double construct_ms=ms(start);
   for(size_t pattern=0;pattern<3;++pattern){
    seal::Plaintext plain(degree);for(size_t i=0;i<degree;++i)plain[i]=pattern==0?0:pattern==1?(i==degree-1?((uint64_t(1)<<59)-1):0):((i*7919+123456)&((uint64_t(1)<<59)-1));seal::Ciphertext input;encryptor.encrypt_symmetric(plain,input);
    auto expected=input;start=Clock::now();eval.apply_galois_inplace(expected,elt,keys);double cpu_ms=ms(start);start=Clock::now();auto actual=gpu.apply(input);double gpu_ms=ms(start);
    require(actual.size()==expected.size()&&actual.parms_id()==expected.parms_id()&&!actual.is_ntt_form(),"metadata mismatch");require(std::equal(expected.data(),expected.data()+2*6*degree,actual.data()),"GPU != SEAL switch coefficients");
    auto t=gpu.timing();if(!first)std::cout<<",";first=false;std::cout<<"{\"level\":"<<level<<",\"element\":"<<elt<<",\"pattern\":"<<pattern<<",\"coefficients\":"<<2*6*degree<<",\"key_generation_ms\":"<<key_ms<<",\"constructor_ms\":"<<construct_ms<<",\"setup_upload_ms\":"<<t.setup_upload_ms<<",\"cpu_full_galois_switch_ms\":"<<cpu_ms<<",\"gpu_full_galois_switch_ms\":"<<gpu_ms<<",\"upload_ms\":"<<t.upload_ms<<",\"rns_forward_ntt_ms\":"<<t.rns_ntt_ms<<",\"product_ms\":"<<t.product_ms<<",\"inverse_ntt_ms\":"<<t.inverse_ms<<",\"moddown_ms\":"<<t.moddown_ms<<",\"download_ms\":"<<t.download_ms<<",\"device_bytes\":"<<t.device_bytes<<",\"passed\":true}";
   }
   auto t=gpu.timing();start=Clock::now();auto clean=gpu.close();cleanup_times.push_back(ms(start));require(clean.closed&&!clean.first_cuda_error&&clean.wipe_synchronized_bytes==t.sensitive_bytes&&clean.freed_bytes==t.sensitive_bytes,"cleanup failure");require(gpu.close().freed_bytes==t.sensitive_bytes,"nonidempotent close");
  }
 }
 std::cout<<"],\"cleanup_ms\":[";for(size_t i=0;i<cleanup_times.size();++i){if(i)std::cout<<",";std::cout<<cleanup_times[i];}std::cout<<"],\"all_coefficients_equal\":true,\"cleanup_api_success_not_physical_erasure\":true}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
