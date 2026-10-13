"""Isolated whole-layer development sample. R2 workers never read the oracle.

Public sample registration, separate one-party sealed inputs and final outputs.
No runtime environment switch can enable this path in the production worker.
"""
import argparse,fcntl,hashlib,json,os,socket,subprocess,tempfile,time,uuid
from pathlib import Path
import numpy as np
import torch
from private_rag.pki import create_local_pki
from private_rag.memfd import create_memfd, seal_memfd
from references.r1_cuda_model import CUDAIntegerQwen
from experiments.hybrid112.registered_build import registered_build, freeze, copy_binary, add_sources
ROOT=Path(__file__).resolve().parents[2];MASK=(1<<59)-1

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def serialize(t):return (t&MASK).cpu().numpy().astype('<u8').tobytes()
def sealed(data):
    fd=create_memfd('prag112-own-input',allow_sealing=True)
    os.write(fd,data);os.lseek(fd,0,0)
    seal_memfd(fd)
    return fd
@torch.inference_mode()
def reference(m,x,valid,positions,layer):
    n=m.num;length=x.shape[0];p=f'model.layers.{layer}.'
    w=lambda name:m.weight(p+name+'.weight');linear=lambda value,name:m.linear(value,p+name+'.weight')
    a=n.rms_norm(x,w('input_layernorm'))
    q=linear(a,'self_attn.q_proj').reshape(length,16,128).permute(1,0,2)
    k=linear(a,'self_attn.k_proj').reshape(length,8,128).permute(1,0,2)
    v=linear(a,'self_attn.v_proj').reshape(length,8,128).permute(1,0,2)
    q=n.rope(n.rms_norm(q,w('self_attn.q_norm'),m.qk_domain['qk_norm_abs_candidate_f12']['q']),positions,m.qk_domain['qk_norm_abs_candidate_f12']['q'],m.qk_domain['rope_abs_candidate_f12']['q'])
    k=n.rope(n.rms_norm(k,w('self_attn.k_norm'),m.qk_domain['qk_norm_abs_candidate_f12']['k']),positions,m.qk_domain['qk_norm_abs_candidate_f12']['k'],m.qk_domain['rope_abs_candidate_f12']['k'])
    physical=torch.arange(length,device=m.device)
    allowed=(valid[:,None]!=0)&(valid[None,:]!=0)&(physical[None,:]<=physical[:,None]);outputs=[]
    for h in range(16):
        raw=n.checked(n.dot(q[h],k[h//2].T),m.qk_domain['QK_raw_abs_f24_bound'])
        score=n.checked(n.floor_shift(raw,12),m.qk_domain['QK_after_floor12_abs_f12_bound'])
        scaled=n.checked(n.multiply(score,n.constants['attention_scale_f20']),m.qk_domain['attention_scale_raw_abs_f32_bound'])
        prob=n.softmax(n.floor_shift(scaled,20),allowed,valid)
        outputs.append(n.checked(n.floor_shift(n.probability_value(prob,v[h//2]),24)*valid[:,None]))
    y=n.checked(x+linear(torch.cat(outputs,dim=-1),'self_attn.o_proj'),m.residual_bound)
    t=n.rms_norm(y,w('post_attention_layernorm'));gate=linear(t,'mlp.gate_proj');up=linear(t,'mlp.up_proj')
    joint=n.checked(n.floor_shift(n.multiply(n.silu(gate),up),12),m.swiglu_domain['joint_and_down_proj_input_abs_f12'])
    return n.checked(y+linear(joint,'mlp.down_proj'),m.residual_bound)

def main(kind,rows,layer,build_directory):
    os.chdir(ROOT)
    binary,build_identities=registered_build(build_directory,'prag_hybrid_layer')
    binary_sha256=build_identities[str(binary.relative_to(ROOT))]
    torch.manual_seed(112)
    m=CUDAIntegerQwen(ROOT/f'artifacts/models/{kind}/quantized_f12');device=m.device
    ids=103+torch.arange(rows,device=device)%17
    x=m.weight('model.embed_tokens.weight')[ids].to(torch.int64)
    validity=torch.ones(rows,dtype=torch.int64,device=device);positions=torch.arange(rows,device=device)*7
    expected=reference(m,x,validity,positions,layer);expected_flag=m.num.failed
    expected_raw=serialize(expected);expected_sha=hashlib.sha256(expected_raw).hexdigest()
    # Sample values are public development data; each worker still receives
    # exactly one uniformly randomized additive/XOR share, not clear samples.
    word_input=torch.cat([x.flatten(),positions]);c=torch.randint(0,1<<59,word_input.shape,device=device,dtype=torch.int64);s=(word_input-c)&MASK
    cb=torch.randint(0,2,validity.shape,device=device,dtype=torch.int64);sb=cb^validity
    shares=[serialize(c)+cb.cpu().numpy().astype('u1').tobytes(),serialize(s)+sb.cpu().numpy().astype('u1').tobytes()]
    sample={'model':kind,'rows':rows,'layer':layer,'hidden':m.hidden,'token_source':'public ids 103+i%17','positions':'7*i','validity':'all true','fixed_model_dimensions':True,'full_embedding_budget_run':False,'whole_layer':True,'full_model_R2':False}
    sources=[str(p.relative_to(ROOT)) for folder in ('compute','native/src','native/include/prag','experiments/hybrid112') for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.hpp','.cpp','.cu','.inc','.txt')]
    sources+=['references/r1_cuda_model.py','references/r1_cuda_numeric.py','artifacts/constants/constants_v1.bin','artifacts/constants/numeric_profile_v1_qk.json',f'artifacts/models/{kind}/quantized_f12/manifest.json']
    identities=add_sources(build_identities,sources)
    cfg=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).digest();snap=hashlib.sha256(json.dumps(sample,sort_keys=True).encode()).digest();sid=uuid.uuid4().bytes
    prefix=ROOT/f'records/hybrid112_layer_{kind}_{rows}_{layer}_{sid.hex()}'
    reg={**sample,'session':sid.hex(),'configuration':cfg.hex(),'snapshot':snap.hex(),'source_and_binary_sha256':identities,'expected_sha256':expected_sha,'expected_numeric_failure':expected_flag,'device':torch.cuda.get_device_name(device)}
    Path(str(prefix)+'.registration.json').write_text(json.dumps(reg,indent=2)+'\n')
    freeze(prefix,identities)
    expected_cpu=expected_raw
    m.close();del m,x,expected,word_input,c,s,cb,sb,validity,positions,ids;torch.cuda.empty_cache()
    start=time.monotonic();processes=[];fds=[];streams=[];pairs=[];result={}
    try:
        with tempfile.TemporaryDirectory(prefix='prag-hybrid112-') as td:
            td=Path(td)
            binary=copy_binary(binary,td/binary.name,binary_sha256)
            pki=td/'pki';pki_ids=create_local_pki(pki)
            pairs=[socket.socketpair() for _ in range(3)]
            try:
                for index,role in enumerate(('C','S')):
                    side=[pairs[0][index],pairs[1 if index==0 else 2][index],pairs[2 if index==0 else 1][index]]
                    header=b'PR112IN1'+bytes([index])+bytes(7)+sid+cfg+snap
                    input_fd=sealed(header+shares[index]);output_fd=create_memfd('prag112-own-final');fds.extend([input_fd,output_fd])
                    own_name='U' if role=='C' else 'S-compute';peer_name='S-compute' if role=='C' else 'U'
                    output=(td/(role+'.stdout')).open('w');error=(td/(role+'.stderr')).open('w');streams.extend([output,error])
                    cmd=[str(binary),role,*[str(s.fileno()) for s in side],str(input_fd),str(output_fd),str(pki),pki_ids[peer_name]['certificate_sha256'],sid.hex(),cfg.hex(),snap.hex(),str(ROOT),kind,'layer',str(rows),str(layer),str(prefix)]
                    processes.append(subprocess.Popen(cmd,pass_fds=(*[s.fileno() for s in side],input_fd,output_fd),stdout=output,stderr=error))
                for pair in pairs:
                    for sock in pair:sock.close()
                shares.clear()
                print(json.dumps({'event':'R2_layer_started','record':str(prefix.relative_to(ROOT)),'workers':[p.pid for p in processes],'shape':sample}),flush=True)
                for p in processes:p.wait()
                for stream in streams:stream.close()
                result={'exit_codes':[p.returncode for p in processes],'parties':{},'stderr':{role:(td/(role+'.stderr')).read_text() for role in ('C','S')},'elapsed_seconds':time.monotonic()-start}
                for role in ('C','S'):
                    value=(td/(role+'.stdout')).read_text();result['parties'][role]=json.loads(value) if value.strip() else {}
                if all(p.returncode==0 for p in processes):
                    finals=[];flags=[]
                    for fd in (fds[1],fds[3]):
                        os.lseek(fd,0,0);raw=os.read(fd,rows*sample['hidden']*8+1)
                        if len(raw)!=rows*sample['hidden']*8+1:raise RuntimeError('FINAL_SHAPE')
                        finals.append(torch.as_tensor(np.frombuffer(raw[:-1],dtype='<u8').copy().view(np.int64),device=device));flags.append(raw[-1])
                    # Independent oracle reconstructs only after both R2 endpoints
                    # completed and burned their ledgers; no feedback enters R2.
                    actual=(finals[0]+finals[1])&MASK;wanted=torch.as_tensor(np.frombuffer(expected_cpu,dtype='<u8').copy().view(np.int64),device=device)
                    result.update(final_hidden_matched=bool(torch.equal(actual,wanted)),failure_matched=((flags[0]^flags[1])==expected_flag),numeric_failure=bool(flags[0]^flags[1]),output_words=actual.numel())
                    result['result']='matched_CUDA_R1' if result['final_hidden_matched'] and result['failure_matched'] else 'mismatch'
                    del finals,actual,wanted
                else:result['result']='aborted'
            finally:
                for p in processes:
                    if p.poll() is None:p.terminate()
                for p in processes:
                    if p.poll() is None:p.wait(timeout=30)
                for pair in pairs:
                    for sock in pair:sock.close()
                for fd in fds:os.close(fd)
                for stream in streams:stream.close()
    except BaseException as exc:
        result.update(result='aborted',exception_type=type(exc).__name__)
        raise
    finally:
        result.update(reg)
        result['evidence']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(prefix.parent.glob(prefix.name+'.*'))]
        Path(str(prefix)+'.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'record':str(prefix.relative_to(ROOT))+'.json',**{k:v for k,v in result.items() if k not in ('source_and_binary_sha256','evidence')}}),flush=True)
    if result['result']!='matched_CUDA_R1':raise RuntimeError('Whole layer not verified')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['embedding','generation'],required=True);p.add_argument('--rows',type=int,default=2);p.add_argument('--layer',type=int,default=0);p.add_argument('--build-directory',required=True,help='fresh development build from tools.build_hybrid');a=p.parse_args();main(a.model,a.rows,a.layer,a.build_directory)
