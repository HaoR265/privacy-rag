#pragma once
#include "prag/secure_ops.hpp"
namespace prag {
struct TokenCandidateStats {uint64_t onehot_bytes=0,initial_key_bytes=0,switch_key_bytes=0,online_bfv_gc_bytes=0;double onehot_seconds=0;};
std::vector<Word> token_candidate(SecureOps&,Word,size_t,size_t,const PrivateWeightBlock&,TokenCandidateStats&);
}
