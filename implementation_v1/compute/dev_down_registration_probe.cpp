// Read-only, S-local artifact registration check. No R2 computation is run.
#include "artifacts.hpp"
#include <iostream>
using namespace prag::model;
int main(int argc,char** argv){
  if(argc!=2)return 2;
  try{
    const std::filesystem::path root=argv[1];
    require_compiled_numeric_profile(root/"artifacts/constants/numeric_profile_v1_qk.json");
    std::size_t count=0;
    for(const auto& name:{std::string("embedding"),std::string("generation")}){
      auto shape=name=="embedding"?ModelShape::embedding():ModelShape::generation();
      const auto& bounds=name=="embedding"?numeric_profile::embedding_down:numeric_profile::generation_down;
      for(std::size_t layer=0;layer<28;++layer){
        MappedWeights weight(root/"artifacts/models"/name/"quantized_f12"/("model.layers."+std::to_string(layer)+".mlp.down_proj.weight.npy"),shape.hidden,shape.ffn,false);
        weight.require_down_bound(bounds[layer]);++count;
        if(layer==0){
          auto wrong=bounds[layer];wrong.row_l1++;
          bool rejected=false;try{weight.require_down_bound(wrong);}catch(const std::runtime_error&){rejected=true;}
          if(!rejected)throw std::runtime_error("WRONG_BOUND_ACCEPTED");
          wrong=bounds[layer];wrong.sha256="0000000000000000000000000000000000000000000000000000000000000000";
          rejected=false;try{weight.require_down_bound(wrong);}catch(const std::runtime_error&){rejected=true;}
          if(!rejected)throw std::runtime_error("WRONG_HASH_ACCEPTED");
        }
      }
    }
    std::cout<<"{\"entry\":\"read_only_native_down_registration\",\"files_checked\":"<<count
      <<",\"actual_sha256_and_all_row_l1_matched\":true,\"wrong_hash_and_l1_rejected\":true,\"numeric_version\":\""<<numeric_profile::version
      <<"\",\"profile_sha256\":\""<<numeric_profile::sha256<<"\",\"full_model_run\":false,\"R2_run\":false}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
