#pragma once
#include "halfgate.hpp"
#include <array>
#include <functional>
#include <istream>
namespace gpu_hg {
struct Phase {uint32_t gate_begin=0,gates=0;std::vector<Instruction> nodes;std::vector<uint32_t> levels,live_in,live_out;};
struct StreamPlan {uint32_t wires=0;std::array<Phase,3> phases;std::vector<uint32_t> roots;};
// Callback runs only on the caller/TLS-owner thread. Generator supplies bytes;
// evaluator asks callback to completely initialize the supplied span.
using Transfer=std::function<void(void*,size_t)>;
Times stream_complete(bool,const DeviceBinding&,const StreamPlan&,size_t n,
 const void*ci,const void*si,void*out,Label seed,Label delta,Label zero,Label one,
 uint64_t first,const Transfer&);
StreamPlan read_plan(std::istream&);
StreamPlan read_plan(const std::string&);
}
