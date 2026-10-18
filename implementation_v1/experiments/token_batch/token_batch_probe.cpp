// Independent diagnostic only. Public fixture indices and output
// reconstruction are excluded from production/model interfaces.
#include "prag/secure_ops.hpp"
#include "artifacts.hpp"
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/rand.h>
#include <sstream>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
static constexpr const char* token_backend="CPU_SEAL_SCI_RTX2060_RESIDENT_TOKEN_AND_PACKING";
#else
static constexpr const char* token_backend="CPU_SEAL_SCI";
#endif

static void cost_file(const std::string& path,Role role,size_t slots,size_t vocabulary,
    size_t hidden,const TokenLookupStats& s){
  std::ostringstream out;out<<std::setprecision(17)
    <<"{\"schema\":\"token_lookup_batch_cost_v1\",\"party\":\""<<(role==Role::Client?"C":"S")
    <<"\",\"call\":0,\"batch_slots\":"<<slots<<",\"vocabulary\":"<<vocabulary<<",\"padded_vocabulary\":155648,\"hidden\":"<<hidden
    <<",\"native_version\":\""<<PRAG_NATIVE_VERSION<<"\",\"crypto_backend\":\""<<token_backend<<"\",\"full_model_executed\":false"
    <<",\"wall_time_includes_waiting\":true,\"server_raw_includes_weight_read\":true"
    <<",\"onehot_seconds\":"<<s.onehot_seconds<<",\"initial_key_seconds\":"<<s.initial_key_seconds
    <<",\"switch_key_seconds\":"<<s.switch_key_seconds<<",\"input_seconds\":"<<s.input_seconds
    <<",\"server_raw_seconds\":"<<s.server_raw_seconds<<",\"server_weight_read_seconds\":"<<s.server_weight_read_seconds
    <<",\"server_weight_lift_ntt_seconds\":"<<s.server_weight_lift_ntt_seconds<<",\"server_dyadic_seconds\":"<<s.server_dyadic_seconds
    <<",\"server_packing_seconds\":"<<s.server_packing_seconds<<",\"return_phase_seconds\":"<<s.return_phase_seconds
    <<",\"gc_seconds\":"<<s.gc_seconds<<",\"total_seconds\":"<<s.total_seconds
    <<",\"gpu_dyadic_used\":"<<(s.gpu_dyadic_used?"true":"false")<<",\"gpu_cleanup_closed\":"<<(s.gpu_cleanup_closed?"true":"false")
    <<",\"gpu_constructor_seconds\":"<<s.gpu_constructor_seconds<<",\"gpu_input_upload_seconds\":"<<s.gpu_input_upload_seconds
    <<",\"gpu_packing_constructor_seconds\":"<<s.gpu_packing_constructor_seconds<<",\"gpu_packing_device_bytes\":"<<s.gpu_packing_device_bytes
    <<",\"gpu_weight_lift_seconds\":"<<s.gpu_weight_lift_seconds<<",\"gpu_weight_ntt_seconds\":"<<s.gpu_weight_ntt_seconds<<",\"gpu_dyadic_seconds\":"<<s.gpu_dyadic_seconds
    <<",\"gpu_weight_upload_seconds\":"<<s.gpu_weight_upload_seconds<<",\"gpu_kernel_seconds\":"<<s.gpu_kernel_seconds
    <<",\"gpu_output_download_seconds\":"<<s.gpu_output_download_seconds<<",\"gpu_cleanup_seconds\":"<<s.gpu_cleanup_seconds
    <<",\"gpu_device_bytes\":"<<s.gpu_device_bytes<<",\"gpu_wipe_synchronized_bytes\":"<<s.gpu_wipe_synchronized_bytes
    <<",\"gpu_freed_bytes\":"<<s.gpu_freed_bytes<<",\"gpu_cleanup_error\":"<<s.gpu_cleanup_error
    <<",\"server_weight_blocks\":"<<s.server_weight_blocks<<",\"server_weight_values\":"<<s.server_weight_values
    <<",\"server_weight_ntts\":"<<s.server_weight_ntts<<",\"server_pack_merges\":"<<s.server_pack_merges
    <<",\"input_ciphertexts\":"<<s.input_ciphertexts<<",\"returned_tiles\":"<<s.returned_tiles<<"}\n";
  const auto line=out.str();int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  if(fd<0)throw std::runtime_error("COST_OUTPUT_NOT_FRESH");
  size_t done=0;while(done<line.size()){auto count=write(fd,line.data()+done,line.size()-done);
    if(count<=0){close(fd);throw std::runtime_error("COST_WRITE_FAILED");}done+=size_t(count);}
  if(close(fd))throw std::runtime_error("COST_CLOSE_FAILED");
}

static int run(Role role,int fd,const std::string& prefix,const std::string& cert,
    size_t hidden,const std::vector<Word>& ids,const std::vector<Word>& client_shares,
    const std::string& reference,const std::string& folder,
    const std::array<unsigned char,16>& session,bool profile){
  std::array<unsigned char,32> config{};config[0]=4;config[1]=uint8_t(ids.size());
  Channel io(fd,session,config);const std::string own=role==Role::Client?"U":"S-compute";
  io.enable_tls(role,cert+"/ca.pem",cert+"/"+own+".pem",cert+"/"+own+".key","localhost");
  SecureOps op(role,io,prefix+(role==Role::Client?".C.ledger":".S.ledger"),false);
  const size_t vocabulary=hidden==1024?151669:151936;
  prag::model::CheckpointWeights weights(folder,hidden==1024?prag::model::ModelShape::embedding():prag::model::ModelShape::generation(),role);
  size_t calls=0,visits=0;
  PrivateWeightBlock reader=[&](size_t begin,size_t count,size_t first,size_t rows){
    if(role!=Role::Server)throw std::runtime_error("C_WEIGHT_ACCESS");
    ++calls;visits+=count*rows;std::vector<std::vector<Word>> result(rows,std::vector<Word>(count));
    for(size_t j=0;j<count;++j)for(size_t r=0;r<rows;++r)
      result[r][j]=begin+j<vocabulary?weights.token_table().at(begin+j,first+r):0;
    return result;
  };
  std::vector<Word> safe;Bit failure=op.public_bit(false);bool expected_failure=false;
  // Same semantic policy as Transformer::token: invalid real-vocabulary ID is
  // secretly replaced with 0, and a secret failure bit is accumulated.
  for(size_t slot=0;slot<ids.size();++slot){
    Word index=role==Role::Client?client_shares[slot]:(ids[slot]-client_shares[slot])&MASK;
    const Bit valid=op.bnot(op.gt(index,op.public_word(vocabulary-1),false));
    failure=op.bor(failure,op.bnot(valid));safe.push_back(op.select(valid,index,op.public_word(0)));
    expected_failure=expected_failure||(ids[slot]>=vocabulary);
  }
  const auto before=io.bytes(),sent=io.frames_sent(),received=io.frames_received();
  const auto start=std::chrono::steady_clock::now();TokenLookupStats stats;
  auto result=op.private_token_lookup_many(safe,vocabulary,hidden,reader,&stats);
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  const auto bytes=io.bytes()-before,frames_sent=io.frames_sent()-sent,frames_received=io.frames_received()-received;
  if(role==Role::Server&&(calls!=19*hidden||visits!=155648*hidden))throw std::runtime_error("SCAN_SHAPE");
  if(profile)cost_file(prefix+(role==Role::Client?".C.cost.jsonl":".S.cost.jsonl"),role,ids.size(),vocabulary,hidden,stats);
  rusage usage{};getrusage(RUSAGE_SELF,&usage);uint64_t memory=usage.ru_maxrss*uint64_t(1024),server_memory=0;
  // Deliberate independent diagnostic reconstruction, outside measured cost.
  io.set_operation(UINT64_C(1000000000));
  if(role==Role::Server){
    io.send_data(&memory,sizeof(memory));uint8_t bit=failure;io.send_data(&bit,1);
    for(const auto& row:result)io.send_data(row.data(),row.size()*sizeof(Word));
  }else{
    io.recv_data(&server_memory,sizeof(server_memory));uint8_t other_failure;io.recv_data(&other_failure,1);
    if((bool(other_failure)^failure)!=expected_failure)throw std::runtime_error("FAILURE_BIT_MISMATCH");
    std::ifstream expected(reference,std::ios::binary);if(!expected)throw std::runtime_error("REFERENCE_MISSING");
    for(size_t slot=0;slot<ids.size();++slot){std::vector<Word> other(hidden),want(hidden);
      io.recv_data(other.data(),hidden*sizeof(Word));expected.read(reinterpret_cast<char*>(want.data()),hidden*sizeof(Word));
      if(!expected)throw std::runtime_error("REFERENCE_SHAPE");
      for(size_t j=0;j<hidden;++j)if(((other[j]+result[slot][j])&MASK)!=want[j])throw std::runtime_error("TOKEN_MISMATCH");
    }
    if(expected.peek()!=EOF)throw std::runtime_error("REFERENCE_SHAPE");
  }
  op.close_session();
  if(role==Role::Client)std::cout<<"[{\"native_version\":\""<<PRAG_NATIVE_VERSION
    <<"\",\"crypto_backend\":\""<<token_backend<<"\",\"result\":\"matched_CUDA_R1\",\"batch_slots\":"<<ids.size()<<",\"hidden\":"<<hidden
    <<",\"application_bytes\":"<<bytes<<",\"elapsed_seconds\":"<<std::setprecision(17)<<elapsed
    <<",\"initial_key_bytes\":"<<stats.initial_key_bytes<<",\"switch_key_bytes\":"<<stats.switch_key_bytes
    <<",\"onehot_bytes\":"<<stats.onehot_bytes<<",\"online_bfv_gc_bytes\":"<<stats.online_bfv_gc_bytes
    <<",\"frames_sent\":"<<frames_sent<<",\"frames_received\":"<<frames_received
    <<",\"client_peak_RSS\":"<<memory<<",\"server_peak_RSS\":"<<server_memory
    <<",\"model_invalid_id_replaced_with_zero\":true,\"failure_bit_matched\":true"
    <<",\"full_model_R2\":false,\"diagnostic_reconstruction_excluded\":true}]\n";
  return 0;
}

int main(int argc,char** argv){
  std::signal(SIGPIPE,SIG_IGN);if(argc!=7&&argc!=8)return 2;
  const bool profile=argc==8;if(profile&&std::string(argv[7])!="--cost-profile")return 2;
  size_t hidden=std::stoul(argv[3]);if(hidden!=1024&&hidden!=2048)return 2;
  std::vector<Word> ids;std::stringstream list(argv[4]);std::string field;
  while(std::getline(list,field,',')){auto id=std::stoull(field);if(id>MASK)return 2;ids.push_back(id);}
  if(ids.empty()||ids.size()>4)return 2;
  std::array<unsigned char,16> session{};std::vector<Word> shares(ids.size());
  if(RAND_priv_bytes(session.data(),session.size())!=1||RAND_priv_bytes(reinterpret_cast<unsigned char*>(shares.data()),shares.size()*sizeof(Word))!=1)return 3;
  for(auto& share:shares)share&=MASK;
  int pair[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,pair))return 3;auto pid=fork();if(pid<0)return 3;
  int result=1;
  if(pid==0){close(pair[0]);try{result=run(Role::Server,pair[1],argv[1],argv[2],hidden,ids,shares,argv[5],argv[6],session,profile);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}_exit(result);}
  close(pair[1]);try{result=run(Role::Client,pair[0],argv[1],argv[2],hidden,ids,shares,argv[5],argv[6],session,profile);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}
  int status;waitpid(pid,&status,0);return result||!WIFEXITED(status)||WEXITSTATUS(status);
}
