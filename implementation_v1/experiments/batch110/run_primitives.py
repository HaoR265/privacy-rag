"""Independent exact CUDA integer fixtures for production shared batch APIs."""
import hashlib,json,struct,subprocess,tempfile,uuid
from pathlib import Path
import torch
from private_rag.pki import create_local_pki
from references.resource_policy import require_model_device
ROOT=Path(__file__).resolve().parents[2];MASK=(1<<59)-1
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    device=require_model_device('R1');torch.manual_seed(110)
    def arr(v):return torch.as_tensor(v,device=device,dtype=torch.int64)
    def ring(v):return v&MASK
    def signed(v):return torch.where((v&(1<<58))!=0,v-(1<<59),v)
    x=arr([0,1,-1,4095,-4095,4096,-4096,(1<<56)-1,-((1<<56)-1),1<<56,-(1<<56),(1<<58)-1,-(1<<58)]+[1<<k for k in range(59)])
    x=ring(torch.cat([x,torch.randint(0,1<<59,(121,),device=device)]));n=x.numel();sx=signed(x)
    cases=[];names=[]
    def add(name,code,rows,cols,p,w,b,ow,ob,table=None):
        w=ring(w).flatten();b=b.to(torch.uint8).flatten();ow=ring(ow).flatten();ob=ob.to(torch.uint8).flatten()
        fields=[code,rows,cols,*[v&MASK for v in p],w.numel(),b.numel(),ow.numel(),ob.numel()]
        public_table=b'' if table is None else table.cpu().numpy().astype('<u8').tobytes()
        cases.append(struct.pack('<10Q',*fields)+public_table+w.cpu().numpy().astype('<u8').tobytes()+b.cpu().numpy().tobytes()+ow.cpu().numpy().astype('<u8').tobytes()+ob.cpu().numpy().tobytes());names.append(name)
    empty=arr([])
    for shift in (0,1,12,24,58):add(f'exact_signed_floor_{shift}',2,1,n,[shift,0,0],x,empty,torch.bitwise_right_shift(sx,shift),empty)
    bad=(sx < -4096)|(sx>4096);add('range_and_dummy',1,1,n,[-4096,4096,3],x,empty,torch.where(bad,3,x),bad)
    y=torch.flip(x,[0]);sy=signed(y)
    for sg in (0,1):add(f'compare_signed_{sg}',3,1,n,[sg,0,0],torch.cat([x,y]),empty,empty,(sx>sy) if sg else x>y)
    bits=torch.arange(n,device=device)%2;add('select',4,1,n,[0,0,0],torch.cat([x,y]),bits,torch.where(bits.bool(),x,y),empty)
    xb=torch.bitwise_right_shift(x[:,None],torch.arange(59,device=device))&1
    add('A2B_all59',5,1,n,[0,0,0],x,empty,empty,xb)
    add('B2A_all59',6,1,n,[0,0,0],empty,xb,x,empty)
    rb=torch.randint(0,2,(16,128),device=device);rb[0]=0
    add('row_boolean_or',7,16,128,[0,0,0],empty,rb,empty,rb.any(1))
    positive=sx>0;safe=torch.where(positive,sx,1);e=torch.zeros_like(safe);remaining=safe.clone()
    for step in (32,16,8,4,2,1):
        high=remaining>=(1<<step);remaining=torch.where(high,remaining>>step,remaining);e+=high*step
    for sqrt in (0,1):
        k=torch.div(e-24,2,rounding_mode='floor') if sqrt else e-24;bad=(~positive)|(k < -10)|(k>8)
        add(f'exponent_sqrt_{sqrt}',8,1,n,[-10,8,sqrt],x,empty,torch.cat([torch.where(bad,1<<24,x),torch.where(bad,0,k)]),bad)
    for exponent in (-40,-32,-1,0,1,32,40):
        shift=-exponent;invalid=abs(shift)>32;amount=0 if invalid else abs(shift)
        if shift>=0:
            overflow=(sx.abs()>(((1<<56)-1)>>amount));output=torch.where(overflow,0,sx)<<amount
        else:
            output=sx>>amount;overflow=output.abs()>=(1<<56);output=torch.where(overflow,0,output)
        add(f'secret_shift_k_{exponent}',9,1,n,[0,0,0],torch.cat([x,arr([exponent])]),empty,output,overflow|invalid)
    limit=(1<<53)-32769
    terms=arr([[limit,0],[limit-1,1],[limit,1],[limit//2,limit//2+1],[limit+1,0],[0,0],[1,1],[limit//2+1,limit//2+1]])
    cb=torch.zeros_like(terms);cb[6,0]=1;total=terms.sum(1);bad=(total>limit)|cb.any(1)
    add('guarded_sum_exact_boundary',10,8,2,[limit,0,0],terms,cb,torch.where(bad,0,total),bad)
    add('beaver_vector_full_ring',100,1,n,[0,0,0],torch.cat([x,y]),empty,x*y,empty)
    constants=json.loads((ROOT/'artifacts/constants/constants_v1.json').read_text())
    indices=arr(list(range(256))+[256,257,(1<<58)-1,1<<58,MASK]);eligible=indices<256;safe=indices.clamp_max(255)
    for digit in range(3):
        table=arr(constants['exp_f24'][digit]);add(f'public_LUT_digit_{digit}_complete',11,1,indices.numel(),[256,0,0],indices,empty,torch.where(eligible,table[safe],0),empty,table)
    ex=arr([-1,0,1,255,256,65535,65536,131071,131072,131073,1<<55,-(1<<55)])
    negative=ex<0;off=negative|(ex>131072);safe=torch.where(off,0,ex)
    add('exp_indices_tail_and_negative',13,1,ex.numel(),[0,0,0],ex,empty,torch.cat([safe&255,(safe>>8)&255,(safe>>16)&255]),torch.cat([off,negative]))
    prefix=ROOT/f'records/batch110_primitives_{uuid.uuid4().hex}'
    with tempfile.TemporaryDirectory() as t:
        t=Path(t);pki=t/'pki';create_local_pki(pki);fixture=t/'fixture.bin';fixture.write_bytes(struct.pack('<Q',len(cases))+b''.join(cases));fixture_sha=sha(fixture)
        cp=subprocess.run([str(ROOT/'experiments/batch110/build/prag_batch110_primitives'),str(prefix),str(pki),str(fixture)],capture_output=True,text=True,timeout=90)
    result=json.loads(cp.stdout) if cp.returncode==0 else {'result':'failed','stdout':cp.stdout}
    result.update(exit_code=cp.returncode,stderr=cp.stderr,case_names=names,fixture_sha256=fixture_sha,oracle='CUDA exact integer',device=torch.cuda.get_device_name(device),full_model_R2=False)
    result['files']={s:sha(ROOT/s) for s in ['experiments/batch110/primitive_probe.cpp','experiments/batch110/run_primitives.py','experiments/batch110/build/prag_batch110_primitives','experiments/batch110/build/native/libprag_crypto.a','native/src/batch.cpp','native/include/prag/batch.hpp']}
    result['ledgers']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in ROOT.glob(str(prefix.relative_to(ROOT))+'*.ledger')]
    Path(str(prefix)+'.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'record':str(prefix.relative_to(ROOT))+'.json','result':result['result'],'case_count':len(cases),'stderr':cp.stderr}),flush=True)
    if cp.returncode:raise RuntimeError('native primitive mismatch')
if __name__=='__main__':main()
