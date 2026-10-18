"""R0 ONLY: streamed Transformers 4.51.3 layers, canonical masks and greedy policy.

Every complete weight and activation is plaintext in this independent reference.
Never callable from the management API or selected by production configuration.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
import transformers
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RMSNorm,Qwen3RotaryEmbedding
from model_tools.quantize_weights import verify_tied_checkpoint
from references.resource_policy import require_model_device


class PrefixCache:
    def __init__(self): self.values={}
    def update(self,k,v,layer,kwargs):
        positions=kwargs["cache_position"]
        if layer not in self.values:
            shape=(1,8,1280,128)
            self.values[layer]=(torch.zeros(shape,dtype=k.dtype,device=k.device),
                                torch.zeros(shape,dtype=v.dtype,device=v.device))
        cache_k,cache_v=self.values[layer]
        cache_k[:,:,positions,:]=k; cache_v[:,:,positions,:]=v
        end=int(positions[-1])+1
        return cache_k[:,:,:end,:],cache_v[:,:,:end,:]


class FloatQwen:
    def __init__(self,checkpoint:Path,device=None):
        self.device=require_model_device("R0",device)
        if transformers.__version__!="4.51.3": raise RuntimeError("R0_TRANSFORMERS_VERSION_MISMATCH")
        self.root=Path(checkpoint)
        self.config=AutoConfig.from_pretrained(self.root,local_files_only=True)
        self.config._attn_implementation="eager"
        self.keys={}
        if not verify_tied_checkpoint(self.root):raise ValueError("R0_TIED_HEAD_MISMATCH")
        for path in sorted(self.root.glob("*.safetensors")):
            with safe_open(path,framework="pt",device="cpu") as f:
                for key in f.keys():
                    if key=="lm_head.weight":continue
                    canonical=key if key.startswith("model.") else "model."+key
                    self.keys[canonical]=(path,key)
        if len(self.keys)!=310: raise ValueError("R0_INCOMPLETE_TENSORS")
        self.cache=PrefixCache()
        self.cache_valid=torch.zeros(1280,dtype=torch.bool,device=self.device)
        self.rope=Qwen3RotaryEmbedding(self.config,device=self.device)
        self.generation=self.config.hidden_size==2048

    def weight(self,name):
        path,key=self.keys[name]
        with safe_open(path,framework="pt",device="cpu") as f:
            return f.get_tensor(key).to(device=self.device,dtype=torch.float32)

    @torch.inference_mode()
    def forward(self,tokens,validity,positions=None,write_position=None):
        ids=torch.as_tensor(tokens,dtype=torch.long,device=self.device)
        valid=torch.as_tensor(validity,dtype=torch.bool,device=self.device)
        length=ids.numel()
        if ids.ndim!=1 or valid.shape!=ids.shape: raise ValueError("R0_SHAPE")
        pos=(torch.where(valid,torch.cumsum(valid,0)-1,0) if positions is None else
             torch.as_tensor(positions,dtype=torch.long,device=self.device))
        # Only the current token rows persist after lookup; source table remains S/reference-local.
        x=self.weight("model.embed_tokens.weight")[ids].unsqueeze(0)
        physical=torch.arange(length,device=self.device) if write_position is None else torch.arange(write_position,write_position+length,device=self.device)
        if self.generation:
            self.cache_valid[physical]=valid
            key_valid=self.cache_valid[:int(physical[-1])+1]
        else: key_valid=valid
        allowed=valid[:,None]&key_valid[None,:]&(torch.arange(key_valid.numel(),device=self.device)[None,:]<=physical[:,None])
        allowed[~valid,0]=True
        mask=torch.where(allowed,0.0,-torch.inf).reshape(1,1,length,-1)
        embeddings=self.rope(x,pos.unsqueeze(0))
        for index in range(28):
            prefix=f"model.layers.{index}."
            # meta construction avoids random initialization and its memory/work overhead.
            with torch.device("meta"):
                layer=Qwen3DecoderLayer(self.config,index)
            state={key[len(prefix):]:self.weight(key) for key in self.keys if key.startswith(prefix)}
            layer.load_state_dict(state,strict=True,assign=True); layer.eval()
            def mask_invalid_query(module,inputs,output):
                # o_proj has no bias, so masking here equals zeroing PV before projection.
                return (output[0]*valid.reshape(1,length,1),output[1])
            handle=layer.self_attn.register_forward_hook(mask_invalid_query)
            x=layer(x,attention_mask=mask,position_ids=pos.unsqueeze(0),
                    past_key_value=self.cache if self.generation else None,
                    cache_position=physical,position_embeddings=embeddings,use_cache=self.generation)[0]
            handle.remove(); del layer,state
        norm=Qwen3RMSNorm(self.config.hidden_size,eps=1e-6).to(self.device)
        norm.weight=torch.nn.Parameter(self.weight("model.norm.weight"),requires_grad=False)
        return norm(x)[0]

    @torch.inference_mode()
    def embedding(self,tokens,validity):
        if self.generation or len(tokens) not in (160,257): raise ValueError("R0_EMBEDDING_PROFILE")
        hidden=self.forward(tokens,validity)
        positions=[i for i,v in enumerate(validity) if v]
        last=hidden[positions[-1]] if positions else torch.zeros(1024,device=self.device)
        norm=torch.linalg.vector_norm(last)
        return torch.where(norm>0,last/torch.where(norm>0,norm,1),0).cpu()

    @torch.inference_mode()
    def generate(self,tokens,validity):
        if not self.generation or len(tokens)!=1024: raise ValueError("R0_GENERATION_PROFILE")
        self.cache=PrefixCache(); self.cache_valid.zero_()
        hidden=self.forward(tokens,validity)
        indices=[i for i,v in enumerate(validity) if v]
        last=hidden[indices[-1]] if indices else torch.zeros(2048,device=self.device)
        allowed=torch.cat((torch.arange(151644,device=self.device),torch.tensor([151645],device=self.device)))
        answers,masks=[],[]; alive=True; effective=sum(validity)
        for t in range(256):
            logits=torch.nn.functional.linear(last,self.weight("model.embed_tokens.weight"))
            chosen=int(allowed[torch.argmax(logits[allowed])])
            alive=alive and chosen not in (151643,151645)
            answers.append(chosen if alive else 0); masks.append(int(alive))
            if t<255: last=self.forward([chosen if alive else 0],[int(alive)],[effective+t if alive else 0],1024+t)[0]
        return {"tokens":answers,"validity":masks,"length_limit":alive}
