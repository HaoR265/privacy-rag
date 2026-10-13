"""Actual mTLS batch RMS vs exact CUDA R1; separate immutable run records."""
import argparse
import hashlib
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
import numpy as np
import torch
from private_rag.pki import create_local_pki
from references.resource_policy import require_model_device
from references.r1_cuda_numeric import CUDAIntegerNumerics

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(d, b, mode):
    device = require_model_device('R1')
    model = 'embedding' if d == 1024 else 'generation'
    tensor = 'model.layers.0.self_attn.q_norm.weight' if d == 128 else 'model.layers.0.input_layernorm.weight'
    folder = ROOT / f'artifacts/models/{model}/quantized_f12'
    path = folder / (tensor + '.npy')
    entry = next(e for e in json.loads((folder / 'manifest.json').read_text())['tensors'] if e['path'] == path.name)
    if sha(path) != entry['sha256']:
        raise RuntimeError('weight hash mismatch')
    w = torch.as_tensor(np.load(path).copy(), device=device, dtype=torch.int64)
    x = ((((torch.arange(d*b, device=device)*13+7)%255)-127)*4096).reshape(b,d)
    if mode == 'boundaries':
        if b != 16:
            raise ValueError('boundary batch is 16 rows')
        cap = {128:8388608,1024:67108863,2048:94906265}[d]
        x[0].zero_()
        for row,value in [(1,cap),(2,-cap),(3,cap+1),(4,-cap-1),(5,1),(6,-1)]:
            x[row].zero_();x[row,0]=value
        x[7].fill_(cap);x[8].fill_(-cap)
        x[9].fill_(1);x[10].fill_(-1)
        x[11] = torch.where(torch.arange(d,device=device)%2==0,1,-1)
    n = CUDAIntegerNumerics({'exp_f24':[],'rope_cos_f20':[],'rope_sin_f20':[]},device)
    profile = json.loads((ROOT/'artifacts/constants/numeric_profile_v1_qk.json').read_text())
    # The generation Q bound is compiled from this same registered artifact.
    from re import search
    bound = int(search(r'QKBounds generation\{(\d+)',(ROOT/'compute/numeric_profile.hpp').read_text()).group(1)) if d==128 else 8388608
    y = n.rms_norm(x,w,output_bound=bound)
    reference_failed = n.failed
    row_l1 = int(w.abs().max().item())
    prefix = ROOT/f'records/batch111_rms_{d}_{b}_{mode}_{uuid.uuid4().hex}'
    with tempfile.TemporaryDirectory(prefix='prag-batch-rms-') as temp:
        temp=Path(temp);cert=temp/'pki';create_local_pki(cert)
        expected=temp/'expected.u64'
        expected.write_bytes((y&((1<<59)-1)).cpu().numpy().astype('<u8').tobytes())
        Path(str(expected)+'.flag').write_bytes(bytes([reference_failed]))
        Path(str(expected)+'.input').write_bytes((x&((1<<59)-1)).cpu().numpy().astype('<u8').tobytes())
        expected_sha=sha(expected);input_sha=sha(Path(str(expected)+'.input'))
        del w,x,y,n;torch.cuda.empty_cache()
        cmd=[str(ROOT/'experiments/batch111/build/prag_batch111_rms'),str(prefix),str(cert),str(path),str(expected),'1',entry['sha256'],str(row_l1),str(d),str(b)]
        cp=subprocess.run(cmd,capture_output=True,text=True,timeout=330)
    lines=cp.stdout.strip().splitlines()
    result=json.loads(lines[-1]) if lines else {'result':'process_error','completed':False}
    result.update(native_version='1.0.11',production_Numerics_rms_many=True,scope='isolated complete RMS kernel, no Transformer',dimension=d,batch=b,case=mode,exit_code=cp.returncode,stderr=cp.stderr,
        weight_sha256=entry['sha256'],tensor=tensor,actual_row_L1=row_l1,
        CUDA_R1_device=torch.cuda.get_device_name(device),reference_failed=reference_failed,reference_sha256=expected_sha,synthetic_input_sha256=input_sha,
        full_model_R2=False,private_GPU_backend=False,production_gate='BACKEND_NOT_READY',deadline_seconds=300)
    sources=['native/src/batch.cpp','native/src/batch_matrix.inc','native/include/prag/batch.hpp','native/src/secure_ops.cpp','compute/batch_numerics.hpp','compute/numerics.hpp','compute/model.hpp','compute/numeric_profile.hpp',
             'experiments/batch111/rms_probe.cpp','experiments/batch111/run_rms.py','experiments/batch111/build/prag_batch111_rms',
             'experiments/batch111/build/native/libprag_crypto.a','references/r1_cuda_numeric.py','experiments/batch110_baseline/meter.hpp']
    result['source_and_binary_sha256']={s:sha(ROOT/s) for s in sources}
    result['evidence']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(ROOT.glob(str(prefix.relative_to(ROOT))+'.*'))]
    target=Path(str(prefix)+'.json');target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'record':str(target.relative_to(ROOT)),**{k:v for k,v in result.items() if k not in ('source_and_binary_sha256','evidence')}},ensure_ascii=False),flush=True)
    if cp.returncode or result['result']!='matched_CUDA_R1':
        raise RuntimeError(f'RMS not verified: {target}')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dimension',type=int,choices=[128,1024,2048],default=128);parser.add_argument('--batch',type=int,choices=[1,16],default=1);parser.add_argument('--mode',choices=['normal','boundaries'],default='normal');args=parser.parse_args()
    run(args.dimension,args.batch,args.mode)
