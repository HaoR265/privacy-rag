// Read-only registration over actual pinned S weights; no secure model is run.
#include "artifacts.hpp"
#include <iostream>
using namespace prag::model;
int main(int argc,char** argv){
  if(argc!=2)return 2;
  try{
    const std::filesystem::path root=argv[1];
    require_compiled_numeric_profile(root/"artifacts/constants/numeric_profile_v1_qk.json");
    std::size_t count=0,rejected=0;
    for(const auto& name:{std::string("embedding"),std::string("generation")}){
      auto shape=name=="embedding"?ModelShape::embedding():ModelShape::generation();
      const auto& bounds=name=="embedding"?numeric_profile::embedding_linear:numeric_profile::generation_linear;
      // C has the same public bound metadata and never opens a weight file.
      CheckpointWeights client("/nonexistent-public-shape-only",shape,prag::Role::Client);
      if(client.token_table().owned())throw std::runtime_error("CLIENT_WEIGHT_OPENED");
      for(const auto& bound:bounds){
        MappedWeights weight(root/"artifacts/models"/name/"quantized_f12"/(std::string(bound.tensor)+".npy"),bound.rows,bound.cols,false);
        weight.require_linear_bound(bound);++count;
        if(std::string(bound.tensor)=="model.layers.0.self_attn.q_proj.weight"){
          for(unsigned variant=0;variant<3;++variant){
            auto wrong=bound;
            if(variant==0)wrong.row_l1++;
            if(variant==1)wrong.sha256="0000000000000000000000000000000000000000000000000000000000000000";
            if(variant==2)wrong.input_abs_f12=(Word{1}<<56)/wrong.row_l1+1;
            bool caught=false;
            try{weight.require_linear_bound(wrong);}catch(const std::runtime_error&){caught=true;++rejected;}
            if(!caught)throw std::runtime_error("WRONG_BOUND_ACCEPTED");
          }
        }
      }
    }
    std::cout<<"{\"entry\":\"read_only_native_linear_registration\",\"files_checked\":"<<count
      <<",\"actual_sha256_and_all_row_l1_matched\":true,\"invalid_registration_rejections\":"<<rejected
      <<",\"client_provider_has_no_weights\":true,\"numeric_version\":\""<<numeric_profile::version
      <<"\",\"profile_sha256\":\""<<numeric_profile::sha256<<"\",\"full_model_run\":false,\"R2_run\":false}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
