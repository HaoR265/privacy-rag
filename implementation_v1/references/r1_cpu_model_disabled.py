"""Complete fixed-shape Qwen3 integer reference. R1 plaintext only, never R2."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from references.r1_numeric import IntegerNumerics,MAIN,WEIGHT,arr,dot,floor_shift,multiply,last_valid,logical_positions,greedy
from references.resource_policy import require_model_device


class IntegerQwen:
    def __init__(self,model_dir:Path,constants=Path("artifacts/constants/constants_v1.json"),
                 numeric_profile=Path("artifacts/constants/numeric_profile_v1_qk.json")):
        raise RuntimeError("REFERENCE_CPU_MODEL_INFERENCE_DISABLED: historical CPU model source is read-only development evidence")
        self.root=Path(model_dir)
        self.manifest=json.loads((self.root/"manifest.json").read_text())
        if not self.manifest["weight_domain_pass"]:
            raise ValueError("PRIVATE_WEIGHT_DOMAIN_INVALID")
        self.generation=self.manifest["model"]=="Qwen/Qwen3-1.7B"
        self.hidden=2048 if self.generation else 1024
        self.num=IntegerNumerics.load(constants)
        profile=json.loads(Path(numeric_profile).read_text())
        if (profile["format_version"]!="numeric-profile-v1-coordinated" or profile["main_abs_f12"]!=MAIN
            or profile['weight_abs_f12']!=WEIGHT):
            raise ValueError("R1_NUMERIC_PROFILE_MISMATCH")
        kind="generation" if self.generation else "embedding"
        self.qk_domain=next(m for m in profile["models"] if m["model"]==kind)
        if (self.qk_domain["quantized_manifest_sha256"]!=hashlib.sha256((self.root/"manifest.json").read_bytes()).hexdigest()
            or profile["constants_json_sha256"]!=hashlib.sha256(Path(constants).read_bytes()).hexdigest()):
            raise ValueError("R1_NUMERIC_PROFILE_PROVENANCE_MISMATCH")
        self.profile_id=profile["profile_id"]
        self.swiglu_domain=profile["swiglu"]
        if (self.swiglu_domain["joint_and_down_proj_input_abs_f12"]!=(1<<34)
            or hashlib.sha256(Path(self.swiglu_domain["evidence_path"]).read_bytes()).hexdigest()!=self.swiglu_domain["evidence_sha256"]):
            raise ValueError("R1_SWIGLU_PROFILE_MISMATCH")
        self.residual_bound=self.qk_domain['residual_abs_f12']
        self.weight_metadata={t['name']:t for t in self.manifest['tensors']}
        self.linear_bounds={t['tensor']:t for t in self.qk_domain['linear_bounds']}
        self.verified_weights=set()
        self.cache={}
        self.progress=None

    def weight(self,name):
        # Streamed memory mapping prevents loading a 6.9 GB f12 model in RAM.
        path=self.root/(name+".npy")
        if name not in self.verified_weights:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=self.weight_metadata[name]['sha256']:
                raise ValueError('R1_WEIGHT_HASH_MISMATCH')
            self.verified_weights.add(name)
        return np.load(path,mmap_mode="r",allow_pickle=False)

    def linear(self,x,name,raw=False):
        bound=self.linear_bounds[name]
        if (bound['sha256']!=self.weight_metadata[name]['sha256']
            or bound['maximum_row_l1_f12_integer']!=self.weight_metadata[name]['maximum_row_l1_integer']
            or not self.weight_metadata[name]['domain_abs_le_256']):
            raise ValueError('R1_LINEAR_BOUND_HASH_MISMATCH')
        return self.num.linear(x,self.weight(name),raw=raw,input_bound=bound['input_abs_f12'],
            output_bound=self.residual_bound if name.endswith('.mlp.down_proj.weight') else MAIN,
            row_l1_bound=bound['maximum_row_l1_f12_integer'],weight_domain_verified=True)

    def forward(self,tokens,validity,positions=None,write_position=None):
        tokens,validity=arr(tokens),arr(validity)
        if tokens.ndim!=1 or validity.shape!=tokens.shape or np.any((validity!=0)&(validity!=1)):
            raise ValueError("R1_INPUT_SHAPE")
        if positions is None: positions=logical_positions(validity)
        positions=arr(positions)
        vocab=151936 if self.generation else 151669
        if np.any(tokens<0) or np.any(tokens>=vocab): raise ValueError("R1_TOKEN_RANGE")
        x=self.num.checked(self.weight("model.embed_tokens.weight")[tokens].astype(np.int64))
        length=len(tokens)
        physical=np.arange(length) if write_position is None else np.arange(write_position,write_position+length)
        for layer in range(28):
            prefix=f"model.layers.{layer}."
            w=lambda name:self.weight(prefix+name+".weight")
            linear=lambda value,name:self.linear(value,prefix+name+".weight")
            a=self.num.rms_norm(x,w("input_layernorm"))
            q=linear(a,"self_attn.q_proj").reshape(length,16,128).transpose(1,0,2)
            k=linear(a,"self_attn.k_proj").reshape(length,8,128).transpose(1,0,2)
            v=linear(a,"self_attn.v_proj").reshape(length,8,128).transpose(1,0,2)
            q=self.num.rope(self.num.rms_norm(q,w("self_attn.q_norm"),self.qk_domain["qk_norm_abs_candidate_f12"]["q"]),positions,
                self.qk_domain["qk_norm_abs_candidate_f12"]["q"],self.qk_domain["rope_abs_candidate_f12"]["q"])
            k=self.num.rope(self.num.rms_norm(k,w("self_attn.k_norm"),self.qk_domain["qk_norm_abs_candidate_f12"]["k"]),positions,
                self.qk_domain["qk_norm_abs_candidate_f12"]["k"],self.qk_domain["rope_abs_candidate_f12"]["k"])
            if self.generation:
                if write_position is None:
                    cache_k=np.zeros((8,1280,128),dtype=np.int64); cache_v=np.zeros_like(cache_k)
                    cache_m=np.zeros(1280,dtype=np.int64)
                else:
                    cache_k,cache_v,cache_m=self.cache[layer]
                cache_k[:,physical]=k; cache_v[:,physical]=v; cache_m[physical]=validity
                self.cache[layer]=(cache_k,cache_v,cache_m)
                end=int(physical[-1])+1
                k,v,key_valid=cache_k[:,:end],cache_v[:,:end],cache_m[:end]
            else: key_valid=validity
            allowed=(validity[:,None]!=0)&(key_valid[None,:]!=0)&(np.arange(len(key_valid))[None,:]<=physical[:,None])
            outputs=[]
            for head in range(16):
                raw_score=self.num.checked(dot(q[head],k[head//2].T),self.qk_domain["QK_raw_abs_f24_bound"])
                score=self.num.checked(floor_shift(raw_score,12),self.qk_domain["QK_after_floor12_abs_f12_bound"])
                scaled=self.num.checked(multiply(score,self.num.constants["attention_scale_f20"]),self.qk_domain["attention_scale_raw_abs_f32_bound"])
                score=floor_shift(scaled,20)
                p=self.num.softmax(score,allowed,validity)
                pv=floor_shift(self.num.probability_value(p,v[head//2]),24)
                outputs.append(self.num.checked(pv*validity[:,None]))
            combined=np.concatenate(outputs,axis=-1)
            projected=linear(combined,"self_attn.o_proj")
            y=self.num.checked(x+projected,self.residual_bound)
            t=self.num.rms_norm(y,w("post_attention_layernorm"))
            gate=linear(t,"mlp.gate_proj"); up=linear(t,"mlp.up_proj")
            joint_bound=self.swiglu_domain["joint_and_down_proj_input_abs_f12"]
            joint=self.num.checked(floor_shift(multiply(self.num.silu(gate),up),12),joint_bound)
            x=self.num.checked(y+linear(joint,"mlp.down_proj"),self.residual_bound)
            if self.progress is not None and length==1024:
                self.progress({'event':'generation_prefill_layer','completed_layers':layer+1,'numeric_failure':self.num.failed})
        return self.num.rms_norm(x,self.weight("model.norm.weight"))

    def embedding(self,tokens,validity,record_valid=True):
        if self.generation or len(tokens) not in (160,257): raise ValueError("R1_EMBEDDING_PROFILE")
        self.num.failed=False  # new independent preparation/query reference invocation
        self.num.first_sum_failure=None
        hidden=self.forward(tokens,validity)
        last,exists=last_valid(hidden,validity)
        return self.num.l2(last,record_valid,exists)

    def generate(self,tokens,validity):
        if not self.generation or len(tokens)!=1024: raise ValueError("R1_GENERATION_PROFILE")
        self.num.failed=False
        self.num.first_sum_failure=None
        self.cache={}
        if self.progress is not None:self.progress({'event':'generation_prefill_start','physical_slots':len(tokens)})
        hidden=self.forward(tokens,validity)
        if self.progress is not None:self.progress({'event':'generation_prefill_complete','numeric_failure':self.num.failed})
        last,exists=last_valid(hidden,validity)
        # Keep the full 256-output / 255-decode schedule even for no valid prompt.
        self.num.failed |= not exists
        effective=int(np.sum(validity))
        answers,output_validity=[],[]
        alive=True
        for output_index in range(256):
            logits=self.linear(last,"model.embed_tokens.weight",raw=True)
            chosen=greedy(logits)
            alive=bool(alive and chosen not in (151643,151645))
            answers.append(chosen if alive else 0); output_validity.append(int(alive))
            if output_index<255:
                last=self.forward([chosen if alive else 0],[int(alive)],
                    [effective+output_index if alive else 0],1024+output_index)[0]
            if self.progress is not None and (output_index+1)%4==0:
                self.progress({'event':'generation_outputs','output_slots_completed':output_index+1,
                               'decode_calls_completed':min(output_index+1,255),'numeric_failure':self.num.failed})
        failed=self.num.failed
        return {"tokens":([0]*256 if failed else answers),
                "validity":([0]*256 if failed else output_validity),
                "numeric_failure":failed,"length_limit":bool(alive and not failed)}
