#include "resident.hpp"
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <seal/util/polyarithsmallmod.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace prag::experimental::resident;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
struct Wipe{std::vector<uint64_t>& v;~Wipe(){volatile uint64_t* p=v.data();for(size_t i=0;i<v.size();++i)p[i]=0;}};
int main(int argc,char** argv){try{
 require(argc==2,"expected ephemeral canonical weight-column file");
 uint64_t endian=1;require(*reinterpret_cast<unsigned char*>(&endian)==1,"little endian host required");
 const auto context_start=Clock::now();seal::EncryptionParameters parms(seal::scheme_type::bfv);parms.set_poly_modulus_degree(degree);parms.set_plain_modulus(uint64_t(1)<<59);
 parms.set_coeff_modulus({seal::Modulus(1152921504599080961ULL),seal::Modulus(1152921504600260609ULL),seal::Modulus(1152921504604979201ULL),seal::Modulus(1152921504605962241ULL),seal::Modulus(1152921504606584833ULL),seal::Modulus(1152921504606683137ULL),seal::Modulus(1152921504606748673ULL)});
 seal::SEALContext context(parms,true,seal::sec_level_type::tc128);require(context.parameters_set(),"context rejected");
 const auto cd=context.first_context_data();const auto& moduli=cd->parms().coeff_modulus();
 const auto context_ms=ms(context_start);
 std::vector<uint64_t> actual_columns(3*blocks*block_width);Wipe wipe_columns{actual_columns};auto read_start=Clock::now();
 {std::ifstream input(argv[1],std::ios::binary);require(bool(input),"cannot open weight columns");input.read(reinterpret_cast<char*>(actual_columns.data()),actual_columns.size()*8);require(input.gcount()==std::streamsize(actual_columns.size()*8)&&input.peek()==std::char_traits<char>::eof(),"column file shape mismatch");}
 const auto input_file_read_ms=ms(read_start);
 std::mt19937_64 random(0x52534944454e54ULL);bool first=true;
 struct Close{size_t slots;double ms;Cleanup c;};std::vector<Close> closes;
 std::cout<<"{\"scope\":\"real S-owned weight columns, synthetic coefficient ciphertexts, local operator only\",\"context_ms\":"<<context_ms<<",\"input_file_read_ms\":"<<input_file_read_ms<<",\"cases\":[";
 for(size_t slots:{size_t(1),size_t(4)}){
  const size_t limbs=primes*degree,words=slots*2*limbs;
  auto t=Clock::now();std::vector<uint64_t> input(blocks*words);
  std::vector<std::vector<seal::Ciphertext>> coefficient(slots);
  for(size_t slot=0;slot<slots;++slot)for(size_t block=0;block<blocks;++block){
   seal::Ciphertext ct;ct.resize(context,context.first_parms_id(),2);
   for(size_t c=0;c<2;++c)for(size_t k=0;k<primes;++k)for(size_t j=0;j<degree;++j){uint64_t q=moduli[k].value(),value=random()%q;if(j%13==0)value=0;if(j%13==1)value=q-1;if(j%13==2)value=1;ct.data(c)[k*degree+j]=value;}
   coefficient[slot].push_back(std::move(ct));
  }
  const double fixture_generation_ms=ms(t);t=Clock::now();TokenWeights gpu(context,coefficient);const double constructor_ms=ms(t);
  // Independent original CPU input NTT, excluded from per-column comparison.
  t=Clock::now();for(size_t slot=0;slot<slots;++slot)for(size_t block=0;block<blocks;++block)for(size_t c=0;c<2;++c){
   auto* dst=input.data()+((block*slots+slot)*2+c)*limbs;std::copy_n(coefficient[slot][block].data(c),limbs,dst);
   for(size_t k=0;k<primes;++k)seal::util::ntt_negacyclic_harvey(dst+k*degree,cd->small_ntt_tables()[k]);
  }
  const double cpu_input_ntt_ms=ms(t);
  for(size_t column=0;column<4;++column){
   std::vector<uint64_t> ring(blocks*block_width);Wipe wipe_ring{ring};constexpr uint64_t mask=(uint64_t(1)<<59)-1;
   if(column<3)std::copy_n(actual_columns.data()+column*ring.size(),ring.size(),ring.data());
   else for(size_t i=0;i<ring.size();++i){const int64_t boundary[]={0,1,-1,1<<20,-(1<<20),(1<<20)-1,-((1<<20)-1)};ring[i]=uint64_t(boundary[i%7])&mask;}
   std::vector<uint64_t> expected(words),lifted(degree),product(degree);Wipe wipe_lifted{lifted};Wipe wipe_product{product};Wipe wipe_expected{expected};
   double lift_ntt_ms=0,dyadic_ms=0;t=Clock::now();
   for(size_t block=0;block<blocks;++block)for(size_t k=0;k<primes;++k){
    auto phase=Clock::now();std::fill(lifted.begin(),lifted.end(),0);uint64_t q=moduli[k].value();
    for(size_t j=0;j<block_width;++j){uint64_t v=ring[block*block_width+j];lifted[block_width-1-j]=v+((q-(uint64_t(1)<<59))&(uint64_t(0)-(v>>58)));}
    seal::util::ntt_negacyclic_harvey(lifted.data(),cd->small_ntt_tables()[k]);lift_ntt_ms+=ms(phase);phase=Clock::now();
    for(size_t slot=0;slot<slots;++slot)for(size_t c=0;c<2;++c){const size_t offset=(slot*2+c)*limbs+k*degree;
     seal::util::dyadic_product_coeffmod(input.data()+block*words+offset,lifted.data(),degree,moduli[k],product.data());
     seal::util::add_poly_coeffmod(expected.data()+offset,product.data(),degree,moduli[k],expected.data()+offset);}
    dyadic_ms+=ms(phase);
   }
   const double cpu_api_ms=ms(t);t=Clock::now();auto result=gpu.evaluate(ring);Wipe wipe_result{result};const double gpu_api_ms=ms(t);require(result==expected,"resident result differs from actual SEAL path");
   auto stats=gpu.timing();if(!first)std::cout<<",";first=false;
   std::cout<<"{\"slots\":"<<slots<<",\"column_case\":"<<column<<",\"actual_weight_column\":"<<(column<3?"true":"false")<<",\"coefficients_compared\":"<<words<<",\"passed\":true,\"fixture_generation_ms\":"<<fixture_generation_ms<<",\"constructor_ms\":"<<constructor_ms<<",\"cpu_input_ntt_ms\":"<<cpu_input_ntt_ms<<",\"cpu_api_ms\":"<<cpu_api_ms<<",\"cpu_lift_ntt_ms\":"<<lift_ntt_ms<<",\"cpu_dyadic_ms\":"<<dyadic_ms<<",\"gpu_api_ms\":"<<gpu_api_ms<<",\"initial_input_upload_ms\":"<<stats.initial_input_upload_ms<<",\"weight_upload_ms\":"<<stats.weight_upload_ms<<",\"lift_ms\":"<<stats.lift_ms<<",\"ntt_ms\":"<<stats.ntt_ms<<",\"dyadic_ms\":"<<stats.dyadic_ms<<",\"download_ms\":"<<stats.download_ms<<",\"ring_weight_upload_bytes\":"<<stats.ring_weight_bytes<<",\"ntt_weight_resident_bytes\":"<<stats.ntt_weight_bytes<<",\"device_bytes\":"<<stats.device_bytes<<",\"ntt_stage_ms\":[";
   for(size_t i=0;i<14;++i){if(i)std::cout<<",";std::cout<<stats.ntt_stages_ms[i];}std::cout<<"]}";
  }
  auto reject=[&](std::vector<uint64_t>& bad){bool rejected=false;try{gpu.evaluate(bad);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid weight accepted");};
  std::vector<uint64_t> bad(blocks*block_width);bad[0]=uint64_t(1)<<59;reject(bad);bad[0]=(uint64_t(1)<<20)+1;reject(bad);bad[0]=(uint64_t(1)<<59)-(uint64_t(1)<<20)-1;reject(bad);bad.pop_back();reject(bad);
  t=Clock::now();auto c=gpu.close();closes.push_back({slots,ms(t),c});auto stats=gpu.timing();size_t data_bytes=stats.input_bytes+stats.ring_weight_bytes+stats.ntt_weight_bytes+stats.output_bytes;
  require(c.closed&&!c.first_cuda_error&&c.wipe_synchronized_bytes==data_bytes&&c.freed_bytes==data_bytes,"cleanup failure");require(gpu.close().freed_bytes==data_bytes,"double close mismatch");bool rejected=false;try{gpu.evaluate(bad);}catch(const std::runtime_error&){rejected=true;}require(rejected,"closed backend accepted");
  coefficient[0][0].is_ntt_form()=true;rejected=false;try{TokenWeights invalid(context,coefficient);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"NTT-flagged coefficient input accepted");coefficient[0][0].is_ntt_form()=false;
  auto wrong_parms=parms;auto wrong_q=wrong_parms.coeff_modulus();std::swap(wrong_q[0],wrong_q[1]);wrong_parms.set_coeff_modulus(wrong_q);seal::SEALContext wrong(wrong_parms,true,seal::sec_level_type::tc128);rejected=false;try{TokenWeights invalid(wrong,coefficient);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"wrong context accepted");
 }
 std::cout<<"],\"cleanup\":[";for(size_t i=0;i<closes.size();++i){if(i)std::cout<<",";const auto& c=closes[i];std::cout<<"{\"slots\":"<<c.slots<<",\"ms\":"<<c.ms<<",\"wipe_synchronized_bytes\":"<<c.c.wipe_synchronized_bytes<<",\"freed_bytes\":"<<c.c.freed_bytes<<",\"error\":"<<c.c.first_cuda_error<<",\"closed\":true}";}
 std::cout<<"],\"all_coefficients_equal\":true,\"input_rejections_checked\":true,\"all_cleanup_api_success\":true,\"full_protocol\":false,\"full_model\":false}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
