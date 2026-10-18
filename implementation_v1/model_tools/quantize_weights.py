"""Stream exact checkpoint values through floor(x*4096) to signed int32 NPY.

Private-weight storage uses signed f12 integers; R2 adapters explicitly convert
these to 59-bit residues. No full model, GPU allocation, or pickle deserialization.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from model_tools.fetch_models import MODELS, sha256


def tensor_headers(path):
    with path.open("rb") as f:
        size=struct.unpack("<Q",f.read(8))[0]
        header=json.loads(f.read(size))
    return size+8,{k:v for k,v in header.items() if k!="__metadata__"}


def verify_tied_checkpoint(checkpoint):
    found={}
    for path in sorted(Path(checkpoint).glob("*.safetensors")):
        base,headers=tensor_headers(path)
        for name in ("model.embed_tokens.weight","embed_tokens.weight","lm_head.weight"):
            if name in headers:found[name]=(path,base,headers[name])
    if "lm_head.weight" not in found:return True
    a=found.get("model.embed_tokens.weight",found.get("embed_tokens.weight")); b=found["lm_head.weight"]
    if a is None or a[2]["shape"]!=b[2]["shape"] or a[2]["dtype"]!=b[2]["dtype"]:return False
    def bytes_view(item):
        path,base,header=item; start,end=header["data_offsets"]
        return np.memmap(path,dtype="u1",mode="r",offset=base+start,shape=end-start)
    x,y=bytes_view(a),bytes_view(b)
    return x.shape==y.shape and all(np.array_equal(x[i:i+1048576],y[i:i+1048576]) for i in range(0,x.size,1048576))


def expected_shapes(kind):
    hidden,ffn,vocab=(1024,3072,151669) if kind=="embedding" else (2048,6144,151936)
    shapes={"model.embed_tokens.weight": [vocab,hidden],"model.norm.weight":[hidden]}
    for layer in range(28):
        prefix=f"model.layers.{layer}."
        for name,shape in {"input_layernorm.weight":[hidden],"post_attention_layernorm.weight":[hidden],
            "self_attn.q_proj.weight":[2048,hidden],"self_attn.k_proj.weight":[1024,hidden],
            "self_attn.v_proj.weight":[1024,hidden],"self_attn.o_proj.weight":[hidden,2048],
            "self_attn.q_norm.weight":[128],"self_attn.k_norm.weight":[128],
            "mlp.gate_proj.weight":[ffn,hidden],"mlp.up_proj.weight":[ffn,hidden],
            "mlp.down_proj.weight":[hidden,ffn]}.items():
            shapes[prefix+name]=shape
    return shapes


def quantize(root:Path,kind:str):
    checkpoint=root/kind/"checkpoint"
    dest=root/kind/"quantized_f12"
    dest.mkdir(parents=True,exist_ok=True)
    expected=expected_shapes(kind)
    tied_equal=verify_tied_checkpoint(checkpoint)
    if not tied_equal:raise ValueError("TIED_OUTPUT_HEAD_CHECKPOINT_MISMATCH")
    manifest={"format_version":"quantized-private-weights-v1","model":MODELS[kind][0],
              "revision":MODELS[kind][1],"fractional_bits":12,"stored_dtype":"little_endian_int32",
              "semantic_ring_bits":59,"quantization":"floor(exact_checkpoint_value*4096)",
              "tied_output_head":"model.embed_tokens.weight","duplicate_lm_head_byte_equality_checked":tied_equal,
              "tensors":[],"weight_domain_pass":True}
    present=set()
    for path in sorted(checkpoint.glob("*.safetensors")):
        base,headers=tensor_headers(path)
        for source_name,header in sorted(headers.items()):
            if source_name=="lm_head.weight":continue  # exact byte-identical tied alias, verified above
            name=source_name if source_name.startswith("model.") else "model."+source_name
            if name not in expected or header["shape"]!=expected[name]:
                raise ValueError("MODEL_TENSOR_SHAPE_MISMATCH: "+name)
            if name in present:
                raise ValueError("DUPLICATE_MODEL_TENSOR")
            present.add(name)
            shape=header["shape"]
            dtype={"BF16":"<u2","F16":"<f2","F32":"<f4"}.get(header["dtype"])
            if dtype is None: raise ValueError("UNSUPPORTED_CHECKPOINT_DTYPE")
            raw=np.memmap(path,dtype=dtype,mode="r",offset=base+header["data_offsets"][0],shape=tuple(shape))
            output_path=dest/(name+".npy")
            temp_path=dest/(name+".partial.npy")
            out=np.lib.format.open_memmap(temp_path,mode="w+",dtype="<i4",shape=tuple(shape))
            row_width=shape[-1] if len(shape)>1 else 1
            row_count=shape[0] if len(shape)>1 else shape[0]
            rows=max(1,1048576//row_width)
            low,high,l1max=0,0,0
            for start in range(0,row_count,rows):
                source=np.asarray(raw[start:start+rows])
                floats=(source.astype(np.uint32)<<16).view(np.float32) if header["dtype"]=="BF16" else source.astype(np.float32)
                if not np.isfinite(floats).all(): raise ValueError("NONFINITE_WEIGHT")
                q=np.floor(floats.astype(np.float64)*4096)
                if np.any(q<np.iinfo(np.int32).min) or np.any(q>np.iinfo(np.int32).max):
                    raise ValueError("WEIGHT_INT32_OVERFLOW")
                integers=q.astype(np.int32)
                out[start:start+rows]=integers
                low=min(low,int(integers.min(initial=0))); high=max(high,int(integers.max(initial=0)))
                l1=np.abs(integers.astype(np.int64))
                if len(shape)>1: l1=l1.sum(axis=-1)
                l1max=max(l1max,int(l1.max(initial=0)))
            out.flush(); del out,raw
            temp_path.replace(output_path)
            domain=low>=-(256<<12) and high<=(256<<12)
            manifest["weight_domain_pass"] &= domain
            manifest["tensors"].append({"name":name,"shape":shape,"path":output_path.name,
                "source_file":path.name,"source_tensor":source_name,"source_dtype":header["dtype"],"sha256":sha256(output_path),
                "minimum_integer":low,"maximum_integer":high,"maximum_row_l1_integer":l1max,
                "domain_abs_le_256":domain})
        print(json.dumps({"model":kind,"processed_checkpoint":path.name,"tensors":len(present)}),flush=True)
    if present!=set(expected): raise ValueError("INCOMPLETE_MODEL_TENSOR_SET")
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"model":kind,"tensors":len(present),"weight_domain_pass":manifest["weight_domain_pass"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("artifacts/models"))
    parser.add_argument("--model",choices=list(MODELS),required=True)
    args=parser.parse_args(); quantize(args.root,args.model)
