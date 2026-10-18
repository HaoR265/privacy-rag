#pragma once
#include "prag/secure_ops.hpp"
#include <numeric>
#include <stdexcept>
namespace prag {
// A call-scoped joint projection; input ciphertexts are never retained beyond
// this invocation. The descriptor binds every public output segment.
inline std::vector<std::vector<Word>> private_joint_matrix(SecureOps& op,
    const std::vector<std::vector<Word>>& x,const std::vector<size_t>& outputs,
    unsigned fx,unsigned fw,const std::vector<PrivateWeightBlock>& readers) {
  if(x.empty()||x.size()>16||x[0].empty()||x[0].size()>8192||fx>58||fw>58||fx+fw>58||outputs.size()<2||outputs.size()>3||readers.size()!=outputs.size())
    throw std::runtime_error("E_JOINT_SHAPE");
  for(const auto& row:x)if(row.size()!=x[0].size())throw std::runtime_error("E_JOINT_SHAPE");
  if(op.role()==Role::Server)for(const auto& reader:readers)if(!reader)throw std::runtime_error("E_JOINT_READER");
  size_t total=0;for(size_t o:outputs){if(!o||o>155648-total)throw std::runtime_error("E_JOINT_SHAPE");total+=o;}
  if(!op.session_active())throw std::runtime_error("E_SESSION_CLOSED");
  if(!op.isolated_development()&&!op.channel().authenticated())throw std::runtime_error("E_MTLS_REQUIRED");
  try {
    std::array<unsigned char,80> descriptor{},other{};
    const uint64_t values[]={0x505241474a4e5433ULL,3,x.size(),x[0].size(),fx,fw,outputs.size(),outputs[0],outputs[1],outputs.size()==3?outputs[2]:0};
    for(size_t j=0;j<10;++j)for(size_t b=0;b<8;++b)descriptor[8*j+b]=values[j]>>(56-8*b);
    op.reserve_material("bfv_joint_descriptor",1);
    auto& io=op.channel();
    if(op.role()==Role::Client){io.send_data(descriptor.data(),80);io.recv_data(other.data(),80);}
    else {io.recv_data(other.data(),80);if(other!=descriptor)throw std::runtime_error("E_JOINT_DESCRIPTOR");io.send_data(descriptor.data(),80);}
    if(other!=descriptor)throw std::runtime_error("E_JOINT_DESCRIPTOR");
    PrivateWeightBlock joined=[&](size_t begin,size_t columns,size_t first,size_t count){
      if(op.role()!=Role::Server)throw std::runtime_error("E_PRIVATE_WEIGHT_OWNER");
      std::vector<std::vector<Word>> out;out.reserve(count);size_t offset=0;
      for(size_t g=0;g<outputs.size();++g) {
        const size_t lo=std::max(first,offset),hi=std::min(first+count,offset+outputs[g]);
        if(lo<hi){if(!readers[g])throw std::runtime_error("E_JOINT_READER");
          auto w=readers[g](begin,columns,lo-offset,hi-lo);
          if(w.size()!=hi-lo)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
          out.insert(out.end(),std::make_move_iterator(w.begin()),std::make_move_iterator(w.end()));}
        offset+=outputs[g];
      }
      return out;
    };
    return op.private_matrix(x,total,fx,fw,joined);
  }catch(...){op.abort_session();throw;}
}
}
