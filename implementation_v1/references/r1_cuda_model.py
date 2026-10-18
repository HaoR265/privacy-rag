"""Complete independent R1 CUDA model; no CPU activation arithmetic or fallback."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from references.resource_policy import require_model_device
from references.r1_cuda_numeric import CUDAIntegerNumerics,MAIN,WEIGHT


class CUDAIntegerQwen:
    def __init__(self,model_dir,constants=Path('artifacts/constants/constants_v1.json'),
                 numeric_profile=Path('artifacts/constants/numeric_profile_v1_qk.json')):
        self.device=require_model_device('R1')
        self.root=Path(model_dir);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['weight_domain_pass']:raise ValueError('PRIVATE_WEIGHT_DOMAIN_INVALID')
        self.generation=self.manifest['model']=='Qwen/Qwen3-1.7B';self.hidden=2048 if self.generation else 1024
        self.num=CUDAIntegerNumerics.load(constants,self.device)
        profile=json.loads(Path(numeric_profile).read_text())
        if profile['format_version']!='numeric-profile-v1-coordinated' or profile['main_abs_f12']!=MAIN or profile['weight_abs_f12']!=WEIGHT:
            raise ValueError('R1_NUMERIC_PROFILE_MISMATCH')
        kind='generation' if self.generation else 'embedding'
        self.qk_domain=next(m for m in profile['models'] if m['model']==kind)
        if self.qk_domain['quantized_manifest_sha256']!=hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest() or profile['constants_json_sha256']!=hashlib.sha256(Path(constants).read_bytes()).hexdigest():
            raise ValueError('R1_NUMERIC_PROFILE_PROVENANCE_MISMATCH')
        self.profile_id=profile['profile_id'];self.swiglu_domain=profile['swiglu']
        if self.swiglu_domain['joint_and_down_proj_input_abs_f12']!=(1<<34) or hashlib.sha256(Path(self.swiglu_domain['evidence_path']).read_bytes()).hexdigest()!=self.swiglu_domain['evidence_sha256']:
            raise ValueError('R1_SWIGLU_PROFILE_MISMATCH')
        self.residual_bound=self.qk_domain['residual_abs_f12']
        self.weight_metadata={t['name']:t for t in self.manifest['tensors']}
        self.matrix_bounds={t['tensor']:t for t in self.qk_domain['matrix_bounds']}
        self.linear_bounds={t['tensor']:t for t in self.qk_domain['linear_bounds']}
        self.weights={};self.cache={};self.progress=None;self.layer_context=None
        self.arr=self.num.arr

    def close(self):
        # The development failure callback captures this model; break that
        # cycle explicitly so the next model cannot retain its checkpoint.
        self.num.failure_callback=None
        self.weights.clear();self.cache.clear();self.progress=None

    def weight(self,name):
        if name not in self.weights:
            path=self.root/(name+'.npy');metadata=self.weight_metadata[name]
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=metadata['sha256'] or not metadata['domain_abs_le_256']:
                raise ValueError('R1_WEIGHT_HASH_OR_DOMAIN_MISMATCH')
            source=np.load(path,mmap_mode='r',allow_pickle=False)
            # A host copy is file delivery only. All narrowing is on CUDA and is
            # justified by the manifest bound tied to the exact file hash.
            tensor=torch.from_numpy(np.array(source,copy=True)).to(self.device)
            if source.ndim==2:
                bound=self.matrix_bounds[name]
                if bound['sha256']!=metadata['sha256'] or list(source.shape)!=bound['shape']:
                    raise ValueError('R1_MATRIX_REGISTRATION_MISMATCH')
                if bound['maximum_absolute_weight_f12']<=32767:
                    tensor=tensor.to(torch.int16)
            self.weights[name]=tensor
        return self.weights[name]

    def linear(self,x,name,raw=False):
        bound=self.linear_bounds[name];metadata=self.weight_metadata[name]
        if bound['sha256']!=metadata['sha256'] or bound['maximum_row_l1_f12_integer']!=metadata['maximum_row_l1_integer'] or not metadata['domain_abs_le_256']:
            raise ValueError('R1_LINEAR_BOUND_HASH_MISMATCH')
        return self.num.linear(x,self.weight(name),raw=raw,input_bound=bound['input_abs_f12'],
            output_bound=self.residual_bound if name.endswith('.mlp.down_proj.weight') else MAIN,
            row_l1_bound=bound['maximum_row_l1_f12_integer'])

    @torch.inference_mode()
    def forward(self,tokens,validity,positions=None,write_position=None):
        tokens,validity=self.arr(tokens),self.arr(validity)
        if tokens.ndim!=1 or validity.shape!=tokens.shape or bool(torch.any((validity!=0)&(validity!=1)).item()):raise ValueError('R1_INPUT_SHAPE')
        positions=self.num.logical_positions(validity) if positions is None else self.arr(positions)
        vocab=151936 if self.generation else 151669
        if bool(torch.any((tokens<0)|(tokens>=vocab)).item()):raise ValueError('R1_TOKEN_RANGE')
        x=self.num.checked(self.weight('model.embed_tokens.weight')[tokens].to(torch.int64));length=len(tokens)
        start=0 if write_position is None else write_position
        physical=torch.arange(start,start+length,device=self.device)
        for layer in range(28):
            self.layer_context={'layer':layer,'write_position':write_position}
            prefix=f'model.layers.{layer}.'
            w=lambda name:self.weight(prefix+name+'.weight')
            linear=lambda value,name:self.linear(value,prefix+name+'.weight')
            a=self.num.rms_norm(x,w('input_layernorm'))
            q=linear(a,'self_attn.q_proj').reshape(length,16,128).permute(1,0,2)
            k=linear(a,'self_attn.k_proj').reshape(length,8,128).permute(1,0,2)
            v=linear(a,'self_attn.v_proj').reshape(length,8,128).permute(1,0,2)
            q=self.num.rope(self.num.rms_norm(q,w('self_attn.q_norm'),self.qk_domain['qk_norm_abs_candidate_f12']['q']),positions,
                self.qk_domain['qk_norm_abs_candidate_f12']['q'],self.qk_domain['rope_abs_candidate_f12']['q'])
            k=self.num.rope(self.num.rms_norm(k,w('self_attn.k_norm'),self.qk_domain['qk_norm_abs_candidate_f12']['k']),positions,
                self.qk_domain['qk_norm_abs_candidate_f12']['k'],self.qk_domain['rope_abs_candidate_f12']['k'])
            if self.generation:
                if write_position is None:
                    cache_k=torch.zeros((8,1280,128),dtype=torch.int64,device=self.device);cache_v=torch.zeros_like(cache_k)
                    cache_m=torch.zeros(1280,dtype=torch.int64,device=self.device)
                else:cache_k,cache_v,cache_m=self.cache[layer]
                cache_k[:,physical]=k;cache_v[:,physical]=v;cache_m[physical]=validity
                self.cache[layer]=(cache_k,cache_v,cache_m)
                end=start+length;k,v,key_valid=cache_k[:,:end],cache_v[:,:end],cache_m[:end]
            else:key_valid=validity
            allowed=(validity[:,None]!=0)&(key_valid[None,:]!=0)&(torch.arange(len(key_valid),device=self.device)[None,:]<=physical[:,None])
            outputs=[]
            for head in range(16):
                raw=self.num.checked(self.num.dot(q[head],k[head//2].T),self.qk_domain['QK_raw_abs_f24_bound'])
                score=self.num.checked(self.num.floor_shift(raw,12),self.qk_domain['QK_after_floor12_abs_f12_bound'])
                scaled=self.num.checked(self.num.multiply(score,self.num.constants['attention_scale_f20']),self.qk_domain['attention_scale_raw_abs_f32_bound'])
                score=self.num.floor_shift(scaled,20);p=self.num.softmax(score,allowed,validity)
                pv=self.num.floor_shift(self.num.probability_value(p,v[head//2]),24)
                outputs.append(self.num.checked(pv*validity[:,None]))
            combined=torch.cat(outputs,dim=-1);projected=linear(combined,'self_attn.o_proj')
            y=self.num.checked(x+projected,self.residual_bound);t=self.num.rms_norm(y,w('post_attention_layernorm'))
            gate=linear(t,'mlp.gate_proj');up=linear(t,'mlp.up_proj')
            joint=self.num.checked(self.num.floor_shift(self.num.multiply(self.num.silu(gate),up),12),self.swiglu_domain['joint_and_down_proj_input_abs_f12'])
            x=self.num.checked(y+linear(joint,'mlp.down_proj'),self.residual_bound)
            if self.progress is not None and length==1024:
                self.progress({'event':'generation_prefill_layer','completed_layers':layer+1,'numeric_failure':self.num.failed})
        self.layer_context={'layer':'final_norm','write_position':write_position}
        return self.num.rms_norm(x,self.weight('model.norm.weight'))

    def _begin(self):
        self.num.failed=False;self.num.first_failure=None;self.num.first_sum_failure=None
        def report(event):
            if self.progress is not None:self.progress({'event':'first_numeric_failure',**(self.layer_context or {}),**event})
        self.num.failure_callback=report if self.progress is not None else None

    def embedding(self,tokens,validity,record_valid=True):
        if self.generation or len(tokens) not in (160,257):raise ValueError('R1_EMBEDDING_PROFILE')
        self._begin();hidden=self.forward(tokens,validity);last,exists=self.num.last_valid(hidden,validity)
        vector,eligible=self.num.l2(last,record_valid,exists)
        # The independent retrieval/reference caller receives the final 1024
        # coordinates only; all Transformer and L2 arithmetic stayed on CUDA.
        return vector.cpu().numpy(),int(eligible.item())

    @torch.inference_mode()
    def generate(self,tokens,validity):
        if not self.generation or len(tokens)!=1024:raise ValueError('R1_GENERATION_PROFILE')
        self._begin();self.cache={}
        if self.progress is not None:self.progress({'event':'generation_prefill_start','physical_slots':len(tokens)})
        hidden=self.forward(tokens,validity)
        if self.progress is not None:self.progress({'event':'generation_prefill_complete','numeric_failure':self.num.failed})
        last,exists=self.num.last_valid(hidden,validity);self.num._mark(~exists)
        effective=torch.sum(self.arr(validity));answers=[];masks=[];alive=torch.ones((),dtype=torch.bool,device=self.device)
        for index in range(256):
            chosen=self.num.greedy(self.linear(last,'model.embed_tokens.weight',raw=True))
            alive=alive&(chosen!=151643)&(chosen!=151645);token=torch.where(alive,chosen,0)
            answers.append(token);masks.append(alive.to(torch.int64))
            if index<255:
                last=self.forward(token.reshape(1),alive.to(torch.int64).reshape(1),torch.where(alive,effective+index,0).reshape(1),1024+index)[0]
            if self.progress is not None and (index+1)%4==0:
                self.progress({'event':'generation_outputs','output_slots_completed':index+1,'decode_calls_completed':min(index+1,255),'numeric_failure':self.num.failed})
        failed=self.num.failed;permitted=~self.num._failed
        return {'tokens':torch.where(permitted,torch.stack(answers),0).cpu().tolist(),
            'validity':torch.where(permitted,torch.stack(masks),0).cpu().tolist(),'numeric_failure':failed,
            'length_limit':bool((alive&permitted).item()),'compute_device':str(self.device),
            'compute_device_name':torch.cuda.get_device_name(self.device),'cuda_integer_gemm_counts':dict(self.num.gemm_counts)}
