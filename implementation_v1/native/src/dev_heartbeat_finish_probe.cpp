// Public-data mTLS lifecycle diagnostic. No model or inference is executed.
// Build separately against prag_crypto; does not change any production gate.
#include "prag/heartbeat.hpp"
#include "prag/secure_ops.hpp"
#include <openssl/rand.h>
#include <csignal>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static void run(Role role,int fds[3],const std::string& prefix,
    const std::string& pki,const std::string& scenario,
    const std::array<unsigned char,16>& session){
  std::array<unsigned char,32> configuration{};configuration[0]=4;
  Channel compute(fds[0],session,configuration),first(fds[1],session,configuration),second(fds[2],session,configuration);
  const std::string own=role==Role::Client?"U":"S-compute";
  for(auto* lane:{&compute,&first,&second})
    lane->enable_tls(role,pki+"/ca.pem",pki+"/"+own+".pem",pki+"/"+own+".key","localhost");
  SecureOps op(role,compute,prefix+(role==Role::Client?".C.ledger":".S.ledger"));
  Channel& out=role==Role::Client?first:second;
  Channel& in=role==Role::Client?second:first;
  Heartbeat hb(out,in,compute,[&]{op.abort_session();});
  try{
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(!hb.received()&&std::chrono::steady_clock::now()<deadline){hb.check();std::this_thread::sleep_for(std::chrono::milliseconds(5));}
    if(!hb.received())throw std::runtime_error("E_PROBE_HEARTBEAT_START");
    unsigned char ready=1,peer=0;compute.send_data(&ready,1);compute.recv_data(&peer,1);
    if(peer!=ready)throw std::runtime_error("E_PROBE_READY");
    const bool client=role==Role::Client;
    const bool abort=scenario=="early_eof_C"||scenario=="early_eof_S";
    const bool selected=(scenario.back()=='C')==client;
    if(abort&&selected){op.abort_session();hb.close();return;}
    if(!abort&&selected)std::this_thread::sleep_for(std::chrono::milliseconds(300));
    if(abort){
      bool rejected=false;try{hb.finish();}catch(const std::exception&){rejected=true;}
      hb.close();
      if(!rejected||!hb.failed()||op.session_active())throw std::runtime_error("E_PROBE_EOF_ACCEPTED");
      return;
    }
    hb.finish();
    if(hb.failed()||!op.session_active())throw std::runtime_error("E_PROBE_GRACEFUL_ABORTED");
    op.close_session();
  }catch(...){op.abort_session();hb.close();throw;}
}
int main(int argc,char** argv){
  std::signal(SIGPIPE,SIG_IGN);
  if(argc!=4)return 2;
  const std::string scenario=argv[3];
  if(scenario!="delayed_C"&&scenario!="delayed_S"&&scenario!="early_eof_C"&&scenario!="early_eof_S")return 2;
  std::array<unsigned char,16> session{};
  if(RAND_bytes(session.data(),session.size())!=1)return 2;
  int pairs[3][2];for(auto& p:pairs)if(socketpair(AF_UNIX,SOCK_STREAM,0,p))return 2;
  const auto child=fork();if(child<0)return 2;
  int status=0;
  try{
    int fds[3];for(int j=0;j<3;++j){close(pairs[j][child==0?0:1]);fds[j]=pairs[j][child==0?1:0];}
    run(child==0?Role::Server:Role::Client,fds,argv[1],argv[2],scenario,session);
  }catch(const std::exception&){status=2;}
  if(child==0)_exit(status);
  int state=0;if(waitpid(child,&state,0)!=child)return 2;
  if(status||!WIFEXITED(state)||WEXITSTATUS(state))return 2;
  std::cout<<"{\"scenario\":\""<<scenario<<"\",\"matched\":true,\"full_R2\":false,\"model_executed\":false}\n";
}
