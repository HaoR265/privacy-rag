"""Observation-only hooks for independent public R0 diagnostics."""
import weakref
import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RMSNorm


class ActivationObserver:
    def __init__(self, validity):
        self.valid=np.asarray(validity,dtype=bool)
        self.names=weakref.WeakKeyDictionary()
        self.statistics={}
        self.layers=0

    def measure(self,label,value,squares=False):
        all_values=value.detach().cpu().numpy()[0].astype(np.float64)
        if all_values.shape[0]!=len(self.valid) or not np.isfinite(all_values).all():
            raise RuntimeError('R0_OBSERVER_SHAPE_OR_FINITE')
        active=all_values[self.valid]
        excess=np.abs(active)>256
        maxima=np.max(np.abs(active).reshape(-1,active.shape[-1]),axis=0)
        counts=np.sum(excess.reshape(-1,active.shape[-1]),axis=0)
        item={'shape':list(all_values.shape),'valid_physical_positions':int(self.valid.sum()),
              'valid_min_real':float(active.min()),'valid_max_real':float(active.max()),
              'max_abs_all_positions':float(np.abs(all_values).max()),
              'valid_scalar_count':int(active.size),'valid_scalars_abs_over_256':int(excess.sum()),
              'feature_count_with_abs_over_256':int(np.count_nonzero(counts)),
              'largest_features':[{'feature':int(i),'max_abs_real':float(maxima[i]),'count_over_256':int(counts[i])}
                                  for i in np.argsort(maxima)[-8:][::-1]]}
        if squares:
            for scope,a in [('valid',active),('all',all_values)]:
                q=np.floor(a*4096).astype(np.int64)
                sums=[sum(int(x)*int(x) for x in row) for row in q.reshape(-1,q.shape[-1])]
                denominators=[s//q.shape[-1]+16 for s in sums]
                k=[(d.bit_length()-1-24)//2 for d in denominators]
                item[scope+'_floor_f12_square_sum_max']=max(sums)
                item[scope+'_floor_f12_square_sums_below_2pow56']=max(sums)<(1<<56)
                item[scope+'_inferred_RMS_k_min']=min(k)
                item[scope+'_inferred_RMS_k_max']=max(k)
        self.statistics[label]=item

    def before(self,module,inputs):
        if isinstance(module,Qwen3DecoderLayer):
            layer=module.self_attn.layer_idx
            self.names[module]=f'layer{layer}.residual_after_mlp'
            for name,child in module.named_modules():
                if name:self.names[child]=f'layer{layer}.{name}'
            self.measure(f'layer{layer}.residual_input',inputs[0],True)
        elif isinstance(module,Qwen3RMSNorm):
            self.measure(self.names.get(module,'final_norm')+'.input',inputs[0],True)
        elif isinstance(module,torch.nn.Linear) and self.names.get(module,'').endswith('.mlp.down_proj'):
            self.measure(self.names[module]+'.input_joint',inputs[0])

    def after(self,module,inputs,output):
        if isinstance(module,Qwen3DecoderLayer):
            self.measure(self.names[module],output[0],True)
            self.layers+=1
        elif isinstance(module,torch.nn.Linear):
            name=self.names[module]
            self.measure(name+'.output',output,name.endswith('.mlp.down_proj'))
        elif isinstance(module,Qwen3RMSNorm):
            name=self.names.get(module,'final_norm')
            self.measure(name+'.output',output,name=='final_norm')

    def __enter__(self):
        self.pre=torch.nn.modules.module.register_module_forward_pre_hook(self.before)
        self.post=torch.nn.modules.module.register_module_forward_hook(self.after)
        return self

    def __exit__(self,*args):
        self.pre.remove();self.post.remove()
