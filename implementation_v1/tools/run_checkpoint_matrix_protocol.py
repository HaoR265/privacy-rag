"""Actual generation layer0 Q and Q-gamma maps vs GPU R1; no full model."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import uuid
import numpy as np
import torch
from private_rag.pki import create_local_pki
from references.resource_policy import require_model_device
from references.r1_cuda_numeric import CUDAIntegerNumerics

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=ROOT/'records/matrix107_checkpoint_runs.json'
    if target.exists():raise RuntimeError('refusing to overwrite evidence')
    device=require_model_device('R1')
    library=ROOT/'native/build-matrix107/libprag_crypto.a'
    source=ROOT/'compute/dev_checkpoint_matrix_probe.cpp'
    binary=ROOT/'native/build-matrix107/prag_checkpoint_matrix_probe'
    command=['c++','-O3','-std=c++17','-maes','-msse4.1','-mavx','-mavx2','-DEMP_USE_RANDOM_DEVICE=1',
             '-I'+str(ROOT/'native/include'),'-I'+str(ROOT/'third_party/EzPC/SCI/src'),str(source),str(library),
             str(ROOT/'third_party/SEAL/build/lib/libseal-4.1.a'),'-lssl','-lcrypto','-lgmpxx','-lgmp','-o',str(binary)]
    subprocess.run(command,capture_output=True,check=True)
    sources=[source,library,binary,Path(__file__),ROOT/'compute/artifacts.hpp',ROOT/'compute/numeric_profile.hpp',
             ROOT/'references/r1_cuda_numeric.py']
    identities={str(p.relative_to(ROOT)):sha(p) for p in sources}
    folder=ROOT/'artifacts/models/generation/quantized_f12'
    manifest=json.loads((folder/'manifest.json').read_text())
    cases=[]
    with tempfile.TemporaryDirectory(prefix='prag-checkpoint-matrix-') as temp:
        pki=Path(temp)/'pki';create_local_pki(pki)
        for diagonal,name in [(False,'model.layers.0.self_attn.q_proj.weight'),(True,'model.layers.0.self_attn.q_norm.weight')]:
            path=folder/(name+'.npy')
            entry=next(e for e in manifest['tensors'] if e['path']==path.name)
            if sha(path)!=entry['sha256']:raise RuntimeError('actual weight hash mismatch')
            # Disk/host holds bytes only. All reference arithmetic runs on GPU.
            w=torch.as_tensor(np.load(path).copy(),device=device,dtype=torch.int64)
            dimension=w.shape[0]
            x=(((torch.arange(dimension,device=device)*13+7)%255)-127).reshape(1,-1)*4096
            row_l1=int(w.abs().max().item()) if diagonal else int(w.abs().sum(1).max().item())
            n=CUDAIntegerNumerics({'exp_f24':[],'rope_cos_f20':[],'rope_sin_f20':[]},device)
            y=x*w if diagonal else n.exact_matmul(x,w.T,127*4096*row_l1,right_l1_bound=row_l1)
            expected=Path(temp)/'expected.u64'
            expected.write_bytes((y&((1<<59)-1)).cpu().numpy().astype('<u8').tobytes())
            output_sha=sha(expected)
            del w,x,y,n;torch.cuda.empty_cache()
            prefix=ROOT/'records'/('matrix107_checkpoint_'+('gamma' if diagonal else 'Q')+'_'+uuid.uuid4().hex)
            process=subprocess.Popen(['nice','-n','10',str(binary),str(prefix),str(pki),str(path),str(expected),str(int(diagonal)),entry['sha256'],str(row_l1)],
                                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:stdout,stderr=process.communicate(timeout=300)
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.communicate(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.communicate()
                raise
            if process.returncode:raise RuntimeError(stderr)
            data=json.loads(stdout)
            data.update(tensor=name,weight_sha256=entry['sha256'],actual_row_L1=row_l1,
                        R1_device=torch.cuda.get_device_name(device),R1_CPU_matmul=False,R1_output_sha256=output_sha,
                        protocol_weight_mapped_at_S_only=True,synthetic_input=True,
                        native_weight_hash_and_L1_checked=True,full_RMS_or_layer_executed=False,
                        exit_code=process.returncode,stderr=stderr)
            data['ledgers']=[]
            for party in ['C','S']:
                ledger=Path(str(prefix)+'.'+party+'.ledger')
                assert ledger.read_text().splitlines()[-1]=='terminal=closed all_reserved=burned'
                data['ledgers'].append({'path':str(ledger.relative_to(ROOT)),'sha256':sha(ledger)})
            cases.append(data)
            print(json.dumps({'tensor':name,'result':data['result'],'seconds':data['elapsed_seconds']}),flush=True)
    if identities!={str(p.relative_to(ROOT)):sha(p) for p in sources}:raise RuntimeError('source changed')
    report={'native_version':'1.0.7','scope':'Two isolated real checkpoint weight maps, no model or full RMS',
            'model':manifest['model'],'revision':manifest['revision'],'cases':cases,
            'source_and_binary_sha256':identities,'compile_command':command,
            'temporary_PKI_and_reference_removed':not Path(temp).exists(),'full_model_R2':False}
    target.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
