// Independent public phase-boundary diagnostic; never linked into production.
#include "../../native/src/bfv.cpp"
#include <chrono>
#include <csignal>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int run(Role role,int fd,const std::string& prefix,const std::string& cert) {
  std::array<unsigned char,16> session{};session[0]=109;
  std::array<unsigned char,32> config{};config[0]=109;
  Channel io(fd,session,config);
  const auto own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  auto primes=locked_bfv_parameters().coeff_modulus();primes.pop_back();
  const mpz_class q=modulus_product(primes),t=mpz_class(1)<<59,
                  e0=8194*(19*(mpz_class(1)<<71)+1)+1,
                  bound=(mpz_class(1)<<18)*e0+174762*(mpz_class(1)<<24)+8194;
  mpz_class inverse;mpz_invert(inverse.get_mpz_t(),t.get_mpz_t(),q.get_mpz_t());
  std::vector<mpz_class> us{0,1,q-1},ms{-(mpz_class(1)<<20),-1,0,1,mpz_class(1)<<20},
                         masks{0,1,t/2-1,t/2,t-1};
  const std::vector<mpz_class> errors{-bound,-1,0,1,bound};
  for(int quarter=0;quarter<=4;++quarter)for(int offset=-2;offset<=2;++offset) {
    mpz_class u=((quarter*q/4+offset)*inverse)%q;if(u<0)u+=q;us.push_back(u);
  }
  std::vector<mpz_class> local;std::vector<Word> expected,server_masks;
  unsigned corrections=0;
  for(const auto& m:ms)for(const auto& mask:masks)for(const auto& e:errors)for(const auto& u:us) {
    mpz_class message=(512*m-mask)%t;if(message<0)message+=t;
    mpz_class phase=((q*message+t/2)/t+e)%q;if(phase<0)phase+=q;
    mpz_class v=(phase-u)%q;if(v<0)v+=q;
    mpz_class exact=((phase*t+q/2)/q)%t;
    if(exact!=message)throw std::runtime_error("PHASE_DIAGNOSTIC_CONSTRUCTION");
    mpz_class ra=(u*t)%q,rb=(v*t)%q,k=((4*ra)/q+(4*rb)/q+2)/4;
    corrections|=1u<<k.get_ui();
    local.push_back(role==Role::Client?u:v);
    mpz_class residue=m%t;if(residue<0)residue+=t;expected.push_back(residue.get_ui());
    server_masks.push_back(role==Role::Server?mask.get_ui():0);
  }
  if(corrections!=7)throw std::runtime_error("PHASE_CORRECTION_COVERAGE");
  const size_t constructed=local.size();
  while(local.size()%512){local.push_back(local.back());expected.push_back(expected.back());server_masks.push_back(server_masks.back());}
  op.reserve_material("token109_phase_independent_OT",1);Decoder decoder(role,io);
  const auto before=io.bytes(),fs=io.frames_sent(),fr=io.frames_received(),dirs=io.direction_runs();
  auto start=std::chrono::steady_clock::now();std::vector<Word> result;size_t tiles=0;
  for(size_t first=0;first<local.size();) {
    const size_t count=512;
    op.reserve_material("token109_phase_GC",count);
    auto part=decoder.decode_token_tile({local.begin()+first,local.begin()+first+count},
        {server_masks.begin()+first,server_masks.begin()+first+count},q);
    result.insert(result.end(),part.begin(),part.end());first+=count;++tiles;
  }
  const auto bytes=io.bytes()-before,sent=io.frames_sent()-fs,recv=io.frames_received()-fr,turns=io.direction_runs()-dirs;
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  io.set_operation(1000000000);
  if(role==Role::Server)io.send_data(result.data(),int(result.size()*8));
  else {
    std::vector<Word> other(result.size());io.recv_data(other.data(),int(other.size()*8));
    for(size_t i=0;i<result.size();++i)if(((result[i]+other[i])&MASK)!=expected[i])
      throw std::runtime_error("TOKEN_PHASE_UNSCALE_MISMATCH");
  }
  op.close_session();
  if(role==Role::Client)std::cout<<"{\"result\":\"matched_exact_token_unscale_boundaries\",\"coordinates\":"<<local.size()
    <<",\"constructed_cases\":"<<constructed<<",\"tile_GC_instances\":"<<tiles<<",\"AND_gates\":"<<358*local.size()<<",\"OT_choices\":"<<61*local.size()
    <<",\"correction_values_covered\":[0,1,2],\"noise_extrema_covered\":true,\"mTLS\":true,\"isolated_development\":false"
    <<",\"application_bytes\":"<<bytes<<",\"frames_sent\":"<<sent<<",\"frames_received\":"<<recv
    <<",\"native_version\":\"1.0.9\",\"direct_public_valid_phase_input\":true,\"ciphertext_lookup_executed\":false,\"negative_and_weight_extrema\":true,\"reconstruction_excluded_from_measurement\":true,\"direction_runs\":"<<turns<<",\"elapsed_seconds\":"<<elapsed<<",\"full_model_R2\":false}\n";
  return 0;
}
int main(int argc,char**argv) {
  std::signal(SIGPIPE,SIG_IGN);if(argc!=3)return 2;int sv[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,sv))return 3;
  auto child=fork();if(child<0)return 3;int result=1;
  if(child==0){close(sv[0]);try{result=run(Role::Server,sv[1],argv[1],argv[2]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}_exit(result);}
  close(sv[1]);try{result=run(Role::Client,sv[0],argv[1],argv[2]);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}
  int status=0;waitpid(child,&status,0);return result||!WIFEXITED(status)||WEXITSTATUS(status);
}
