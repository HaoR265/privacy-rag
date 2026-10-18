#pragma once
#include <cstdint>
#include <vector>
#include <string>
namespace gpu_hg {
struct DeviceBinding { std::string uuid, pci_bus_id; };
struct Label { uint32_t w[4]; };
struct Input { Label a,b; };
struct Garbled { Label zero,table0,table1; };
struct EvalInput { Label a,b,table0,table1; };
struct Times {double initialize=0,upload=0,kernel=0,download=0,cleanup=0,whole_api=0;uint64_t device_bytes=0,upload_bytes=0,download_bytes=0;bool cleaned=false,host_config_wiped=false;};
Times garble(const DeviceBinding&,const std::vector<Input>&,Label seed,Label delta,uint64_t first_gate,std::vector<Garbled>&);
Times evaluate(const DeviceBinding&,const std::vector<EvalInput>&,Label seed,uint64_t first_gate,std::vector<Label>&);
struct AddInput {Label a[59],b[59];};
struct AddGarbled {Label bits[59],tables[116];};
struct AddEvalInput {Label a[59],b[59],tables[116];};
struct AddEvaluated {Label bits[59];};
Times garble_add59(const DeviceBinding&,const std::vector<AddInput>&,Label seed,Label delta,Label zero,uint64_t first_gate,std::vector<AddGarbled>&);
Times evaluate_add59(const DeviceBinding&,const std::vector<AddEvalInput>&,Label seed,Label zero,uint64_t first_gate,std::vector<AddEvaluated>&);
struct Instruction {uint32_t op,a,b,out,gate;};
struct Plan {uint32_t lanes,input_count,wire_count,gates;uint32_t input_ops=0,body_ops=0,output_ops=0,flag_ops=0,input_gates=0,body_gates=0,output_gates=0;std::vector<Instruction> nodes;std::vector<uint32_t> level_offsets,outputs;};
struct CompleteOutput {std::vector<Label> labels,tables;};
Times garble_complete_tile(const DeviceBinding&,const Plan&,const std::vector<Label>&,Label seed,Label delta,Label zero,Label one,uint64_t first,size_t full_n,size_t offset,CompleteOutput&);
Times evaluate_complete_tile(const DeviceBinding&,const Plan&,const std::vector<Label>&,const std::vector<Label>& tables,Label seed,Label zero,Label one,uint64_t first,size_t full_n,size_t offset,std::vector<Label>&);
std::string device(const DeviceBinding&);
// Full-batch byte spans use existing per-role label buffers; lengths validated.
Times complete_direct(bool sender,const DeviceBinding&,const Plan&,size_t lanes,
 const void* ci,size_t ci_bytes,const void* si,size_t si_bytes,
 void* tables,size_t table_bytes,void* outputs,size_t output_bytes,
 Label seed,Label delta,Label zero,Label one,uint64_t first);
}
