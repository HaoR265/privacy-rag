"""Public CUDA R1 full Embedding diagnostic for the active numeric profile."""
import json,time,os
from pathlib import Path
import numpy as np
import torch
from private_rag.model_input import InputPreparer
from references.r1_model import IntegerQwen
from references.resource_policy import require_model_device
from model_tools.reference_provenance import identify,unchanged


def run(kind='question',record_prefix='models_coordinated_cuda'):
    device=require_model_device('R1')
    identity=identify('model_tools/check_qk_embedding_revision.py')
    print(json.dumps({'event':'CUDA_R1_embedding_start','pid':os.getpid(),'input_kind':kind,
                      'device':str(device),'device_name':torch.cuda.get_device_name(device)}),flush=True)
    if not record_prefix.startswith('models_') or not all(c.isalnum() or c=='_' for c in record_prefix):raise ValueError('INVALID_DEVELOPER_RECORD_PREFIX')
    prep=InputPreparer(Path('artifacts/models'),Path('artifacts/template/template_v1.json'))
    texts={'question':'隐私计算如何保护问题？','document1':'资料使用 HPKE Base 模式交付给获得授权的用户。','document2':'交付方案使用 X25519、HKDF-SHA256 和 AES-128-GCM。'}
    text=texts[kind];slots=prep.question(text).embedding if kind=='question' else prep.document_chunks(text,'public-coordinated-cuda')[0].embedding
    model=IntegerQwen(Path('artifacts/models/embedding/quantized_f12'));events=[];model.progress=events.append
    try:
        torch.cuda.reset_peak_memory_stats(device);start=time.monotonic()
        vector,eligible=model.embedding(slots.tokens,slots.validity);torch.cuda.synchronize(device)
        result={'scope':'Independent public CUDA R1 complete embedding developer check, not formal acceptance or R2',
                'numeric_profile':model.profile_id,'public_input':text,'input_kind':kind,'physical_slots':len(slots.tokens),'layers':28,
                'start_provenance':identity,'start_to_finish_identity_unchanged':unchanged(identity,'model_tools/check_qk_embedding_revision.py'),
                'elapsed_seconds':time.monotonic()-start,'output_shape':list(vector.shape),'eligible':int(eligible),
                'numeric_failure':bool(model.num.failed),'output_nonzero_count':int(np.count_nonzero(vector)),
                'output_l2_norm_after_f20':float(np.linalg.norm(vector.astype(np.float64)/(1<<20))),
                'first_failure':next((e for e in events if e['event']=='first_numeric_failure'),None),
                'pid':os.getpid(),'device':str(device),'device_name':torch.cuda.get_device_name(device),'model_CPU_fallback':False,
                'peak_allocated_device_bytes':torch.cuda.max_memory_allocated(device),
                'peak_reserved_device_bytes':torch.cuda.max_memory_reserved(device),
                'cuda_integer_gemm_counts':model.num.gemm_counts,'r2_executed':False}
        Path(f'records/{record_prefix}_{kind}_run.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        np.save(f'records/{record_prefix}_{kind}_vector.npy',vector,allow_pickle=False)
        print(json.dumps({k:v for k,v in result.items() if k!='start_provenance'},ensure_ascii=False),flush=True)
    finally:
        model.close()
        del model
        torch.cuda.empty_cache()



if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-kind',choices=['question','document1','document2'],default='question')
    parser.add_argument('--record-prefix',default='models_coordinated_cuda')
    args=parser.parse_args();run(args.input_kind,args.record_prefix)
