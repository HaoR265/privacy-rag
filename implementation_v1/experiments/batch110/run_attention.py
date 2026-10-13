"""Single complete attention head, synthetic shares, real constants, CUDA R1."""
import argparse,hashlib,json,struct,subprocess,tempfile,uuid
from pathlib import Path
import torch
from private_rag.pki import create_local_pki
from references.resource_policy import require_model_device
from references.r1_cuda_numeric import CUDAIntegerNumerics
ROOT=Path(__file__).resolve().parents[2];MASK=(1<<59)-1
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(rows,keys,mode):
    device=require_model_device('R1');torch.manual_seed(110)
    n=CUDAIntegerNumerics.load(ROOT/'artifacts/constants/constants_v1.json',device)
    q=((torch.arange(rows*128,device=device)%37)-18).reshape(rows,128)*257
    k=((torch.arange(keys*128,device=device)%41)-20).reshape(keys,128)*263
    v=((torch.arange(keys*128,device=device)%61)-30).reshape(keys,128)*271
    allowed=torch.arange(keys,device=device)[None,:]<=torch.arange(rows,device=device)[:,None]+keys-rows
    valid=torch.ones(rows,device=device,dtype=torch.bool)
    if mode=='mask_boundaries':
        if rows<4:raise ValueError('requires at least four query rows')
        valid[0]=False;allowed[1]=False;allowed[2]=False;allowed[2,0]=True
        q[3].fill_(-1236996);k[0].fill_(8912916);k[1].fill_(-8912916)
    raw=n.dot(q,k.T);scores=n.floor_shift(n.floor_shift(raw,12)*n.constants['attention_scale_f20'],20)
    probabilities=n.softmax(scores,allowed,valid)
    out=n.checked(torch.where(valid[:,None],n.floor_shift(n.probability_value(probabilities,v),24),0))
    failure=n.failed
    def words(x):return (x&MASK).cpu().numpy().astype('<u8').tobytes()
    fixture=struct.pack('<2Q',rows,keys)+words(q)+words(k)+words(v)+allowed.to(torch.uint8).cpu().numpy().tobytes()+valid.to(torch.uint8).cpu().numpy().tobytes()+words(out)+bytes([failure])
    prefix=ROOT/f'records/batch110_attention_{rows}_{keys}_{mode}_{uuid.uuid4().hex}'
    del q,k,v,raw,scores,probabilities,out,n;torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory() as t:
        t=Path(t);cert=t/'pki';create_local_pki(cert);inp=t/'fixture';inp.write_bytes(fixture)
        cp=subprocess.run([str(ROOT/'experiments/batch110/build/prag_batch110_attention'),str(prefix),str(cert),str(inp),str(ROOT/'artifacts/constants/constants_v1.bin')],capture_output=True,text=True,timeout=330)
    result=json.loads(cp.stdout.strip()) if cp.stdout.strip() else {'result':'process_error'}
    result.update(exit_code=cp.returncode,stderr=cp.stderr,case=mode,rows=rows,keys=keys,reference_failed=failure,
        fixture_sha256=hashlib.sha256(fixture).hexdigest(),CUDA_R1_device=torch.cuda.get_device_name(device),deadline_seconds=300,
        full_layer_R2=False,full_model_R2=False,private_GPU_backend=False)
    paths=['native/src/batch.cpp','native/include/prag/batch.hpp','compute/batch_numerics.hpp','compute/numerics.hpp','compute/model.hpp','compute/numeric_profile.hpp','experiments/batch110/attention_probe.cpp','experiments/batch110/run_attention.py','experiments/batch110/build/prag_batch110_attention','experiments/batch110/build/native/libprag_crypto.a','references/r1_cuda_numeric.py','artifacts/constants/constants_v1.json','artifacts/constants/constants_v1.bin']
    result['source_and_binary_sha256']={p:sha(ROOT/p) for p in paths}
    result['evidence']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(ROOT.glob(str(prefix.relative_to(ROOT))+'.*'))]
    target=Path(str(prefix)+'.json');target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'record':str(target.relative_to(ROOT)),**{k:v for k,v in result.items() if k not in ('source_and_binary_sha256','evidence')}},ensure_ascii=False),flush=True)
    if cp.returncode or result['result']!='matched_CUDA_R1':raise RuntimeError('attention not verified')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',type=int,default=4);p.add_argument('--keys',type=int,default=16);p.add_argument('--mode',choices=['normal','mask_boundaries'],default='normal');a=p.parse_args();run(a.rows,a.keys,a.mode)
