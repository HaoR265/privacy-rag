#include "stream.hpp"
#include <fstream>
#include <stdexcept>
namespace gpu_hg {
StreamPlan read_plan(std::istream&f){auto word=[&](){uint32_t x=0;for(unsigned j=0;j<4;++j){int b=f.get();if(b<0)throw std::runtime_error("PLAN_TRUNCATED");x|=uint32_t(b)<<(8*j);}return x;};auto vec=[&](){uint32_t n=word();if(n>100000)throw std::runtime_error("PLAN_LIMIT");std::vector<uint32_t>v(n);for(auto&x:v)x=word();return v;};if(word()!=0x53504731)throw std::runtime_error("PLAN_MAGIC");StreamPlan p;p.wires=word();p.roots=vec();for(auto&q:p.phases){q.gate_begin=word();q.gates=word();auto n=word();if(n>100000)throw std::runtime_error("PLAN_LIMIT");q.nodes.resize(n);for(auto&o:q.nodes)o={word(),word(),word(),word(),word()};q.levels=vec();q.live_in=vec();q.live_out=vec();}if(f.get()!=EOF)throw std::runtime_error("PLAN_TRAILING");return p;}
StreamPlan read_plan(const std::string&path){std::ifstream f(path,std::ios::binary);return read_plan(f);}
}
