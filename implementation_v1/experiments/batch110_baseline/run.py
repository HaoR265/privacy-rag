import hashlib,json,subprocess,tempfile,uuid
from pathlib import Path
from private_rag.pki import create_local_pki
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'records/batch110_baseline_profile.json'
    if out.exists():raise RuntimeError('immutable evidence exists')
    cases=[]
    with tempfile.TemporaryDirectory() as t:
        cert=Path(t)/'pki';create_local_pki(cert)
        for d in (128,1024,2048):
            model='embedding' if d==1024 else 'generation'
            tensor='model.layers.0.self_attn.q_norm.weight' if d==128 else 'model.layers.0.input_layernorm.weight'
            folder=ROOT/f'artifacts/models/{model}/quantized_f12'
            weight=folder/(tensor+'.npy')
            manifest=json.loads((folder/'manifest.json').read_text())
            entry=next(e for e in manifest['tensors'] if e['path']==weight.name)
            assert sha(weight)==entry['sha256']
            # Hashing and byte inspection only; no host model arithmetic.
            import numpy as np
            import torch
            from references.resource_policy import require_model_device
            w=torch.as_tensor(np.load(weight).copy(),device=require_model_device('R1'),dtype=torch.int64)
            l1=int(w.abs().max().item());del w;torch.cuda.empty_cache()
            for b in (1,16):
                prefix=ROOT/f'records/batch110_baseline_{d}_{b}_{uuid.uuid4().hex}'
                cmd=[str(ROOT/'experiments/batch110_baseline/probe'),str(prefix),str(cert),str(weight),str(Path(t)/'unused'), '1',entry['sha256'],str(l1),str(d),str(b)]
                cp=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
                raw=json.loads(cp.stdout.strip().splitlines()[-1]);raw.update(dimension=d,batch=b,exit_code=cp.returncode,stderr=cp.stderr,tensor=tensor,weight_sha256=entry['sha256'])
                raw['evidence']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in ROOT.glob(str(prefix.relative_to(ROOT))+'.*')]
                cases.append(raw);print(json.dumps(raw,ensure_ascii=False),flush=True)
    out.write_text(json.dumps({'scope':'bounded actual scalar v109 RMS profiling; not completed; no Transformer','cases':cases,'source_and_binary_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'experiments/batch110_baseline/probe',ROOT/'experiments/batch110_baseline/probe.cpp',ROOT/'experiments/batch110_baseline/profile_numerics.hpp',ROOT/'experiments/batch110_baseline/meter.hpp',ROOT/'experiments/pack109_production/build/native/libprag_crypto.a']},'unreached_phases':['floor_mean','epsilon','exponent','mantissa','NR','normalize','gamma','output_guard']},indent=2)+'\n')
if __name__=='__main__':main()
