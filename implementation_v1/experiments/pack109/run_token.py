"""Real candidate Token lookup; GPU reference is separate and public only."""
import argparse,json,subprocess,tempfile,uuid,os,signal,time
from pathlib import Path
import torch
from tools.run_tile_matrix_protocol import ROOT,sha
from references.resource_policy import require_model_device
from private_rag.pki import create_local_pki
def main():
    p=argparse.ArgumentParser();p.add_argument('--hidden',type=int,choices=[1024,2048],required=True);p.add_argument('--id',type=int,required=True);a=p.parse_args()
    name=f'pack109_token_candidate_H{a.hidden}_id{a.id}';dest=ROOT/'records'/f'{name}.json'
    if dest.exists():raise RuntimeError('existing '+name)
    paths=['token_probe.cpp','token_candidate.cpp','token_candidate.hpp','token_engine.inc','token_decoder.inc','token_api.inc','acyclic_pack.hpp','bfv_matrix.inc','bfv_tile_decoder.inc','layout.hpp','secure_ops_candidate.cpp','include/prag/secure_ops.hpp','run_token.py','libprag_crypto_v108.a','build/prag_pack109_token']
    paths=[ROOT/'experiments/pack109'/f for f in paths];identities={str(f.relative_to(ROOT)):sha(f) for f in paths}
    print(json.dumps({'starting':name,'reference':'RTX2060','protocol':'isolated SEAL/SCI primitive'}),flush=True)
    with tempfile.TemporaryDirectory(prefix='prag-token109-') as tmp:
        tmp=Path(tmp);pki=tmp/'pki';create_local_pki(pki);expected=tmp/'expected.u64'
        device=require_model_device('R1');torch.cuda.reset_peak_memory_stats(device)
        values=(a.id*71+torch.arange(a.hidden,device=device,dtype=torch.int64)*197)%65535-32767
        vocab=151669 if a.hidden==1024 else 151936
        if a.id>=vocab:values.zero_()
        expected.write_bytes((values&((1<<59)-1)).cpu().numpy().astype('<u8').tobytes())
        ref={'device':torch.cuda.get_device_name(device),'output_sha256':sha(expected),'peak_bytes':torch.cuda.max_memory_allocated(device),'public_synthetic_table':True,'model_checkpoint_used':False}
        del values;torch.cuda.empty_cache()
        prefix=ROOT/'records'/(name+'_'+uuid.uuid4().hex)
        proc=subprocess.Popen(['nice','-n','10',str(paths[-1]),str(prefix),str(pki),str(a.hidden),str(a.id),str(expected)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:out,err=proc.communicate(timeout=3600)
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM);proc.communicate();raise
        if proc.returncode:raise RuntimeError(err)
        data=json.loads(out)
    ledger=[]
    for party in ['C','S']:
        f=Path(str(prefix)+f'.{party}.ledger');lines=f.read_text().splitlines();assert lines[-1]=='terminal=closed all_reserved=burned';counts={}
        for line in lines:
            fields=line.split()
            if len(fields)==5 and fields[1]=='reserved':counts[fields[2]]=counts.get(fields[2],0)+1
        ledger.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f),'counts':counts})
    assert ledger[0]['counts']==ledger[1]['counts'];counts=ledger[0]['counts']
    assert counts['token_fixed_input']==19 and counts['token_return_tile_mask_Enc0']==a.hidden//512 and counts['token_exact_tile_GC']==a.hidden//512
    if identities!={str(f.relative_to(ROOT)):sha(f) for f in paths}:raise RuntimeError('changed source during run')
    data.update(source_sha256=identities,ledgers=ledger,R1_reference=ref,stderr=err,candidate_only=True,production_integrated=False,child_waited=True,temporary_PKI_removed=not tmp.exists(),formal_acceptance=False,
                setup_not_hidden=True,per_session_setup=data['initial_key_bytes']+data['switch_key_bytes'])
    dest.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:v for k,v in data.items() if k not in ['source_sha256','ledgers']}),flush=True)
if __name__=='__main__':main()
