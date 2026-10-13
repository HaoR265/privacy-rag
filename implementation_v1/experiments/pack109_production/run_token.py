"""Production library, real registered Token table, CUDA reference, no model."""
import argparse,json,subprocess,tempfile,uuid,os,signal
from pathlib import Path
import numpy as np
import torch
from tools.run_tile_matrix_protocol import ROOT,sha
from references.resource_policy import require_model_device
from private_rag.pki import create_local_pki

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hidden',type=int,choices=[1024,2048],required=True)
    p.add_argument('--ids',required=True)
    a=p.parse_args();ids=[int(x) for x in a.ids.split(',')]
    assert ids and all(0<=x<(1<<59) for x in ids)
    name=f'pack109_token_production_H{a.hidden}_ids'+a.ids.replace(',','_')
    dest=ROOT/'records'/f'{name}.json'
    if dest.exists():raise RuntimeError('existing '+name)
    model='embedding' if a.hidden==1024 else 'generation'
    folder=ROOT/f'artifacts/models/{model}/quantized_f12'
    manifest=json.loads((folder/'manifest.json').read_text())
    entry=next(t for t in manifest['tensors'] if t['path']=='model.embed_tokens.weight.npy')
    weight=folder/entry['path'];assert sha(weight)==entry['sha256']
    binary=ROOT/'experiments/pack109_production/build/prag_pack109_token'
    paths=[Path(__file__).resolve(),binary,ROOT/'experiments/pack109_production/token_probe.cpp',
           ROOT/'experiments/pack109_production/build/native/libprag_crypto.a']
    paths+=list((ROOT/'native/src').glob('*'))+list((ROOT/'native/include/prag').glob('*'))
    paths+=[ROOT/'compute'/f for f in ['artifacts.hpp','model.hpp','numerics.hpp','numeric_profile.hpp']]
    identities={str(f.relative_to(ROOT)):sha(f) for f in paths}
    print(json.dumps({'starting':name,'reference':'RTX2060','protocol':'isolated production SEAL/SCI primitive'}),flush=True)
    with tempfile.TemporaryDirectory(prefix='prag-token109-production-') as tmp:
        tmp=Path(tmp);pki=tmp/'pki';create_local_pki(pki);expected=tmp/'expected.u64'
        device=require_model_device('R1');torch.cuda.reset_peak_memory_stats(device)
        table=np.load(weight,mmap_mode='r');vocab=table.shape[0]
        with expected.open('wb') as stream:
            for index in ids:
                values=torch.as_tensor(np.array(table[index]),device=device,dtype=torch.int64) if index<vocab else torch.zeros(a.hidden,device=device,dtype=torch.int64)
                stream.write((values&((1<<59)-1)).cpu().numpy().astype('<u8').tobytes())
        ref={'device':torch.cuda.get_device_name(device),'output_sha256':sha(expected),
             'peak_bytes':torch.cuda.max_memory_allocated(device),'model_checkpoint_used':True,
             'model':manifest['model'],'revision':manifest['revision'],'tensor_sha256':entry['sha256'],
             'CPU_reference_arithmetic':False,'full_model_executed':False}
        del values,table;torch.cuda.empty_cache()
        prefix=ROOT/'records'/(name+'_'+uuid.uuid4().hex)
        proc=subprocess.Popen(['nice','-n','10',str(binary),str(prefix),str(pki),str(a.hidden),a.ids,str(expected),str(folder)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:out,err=proc.communicate(timeout=3600)
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM);proc.communicate();raise
        if proc.returncode:raise RuntimeError(err)
        data=json.loads(out)
    ledgers=[]
    for party in ['C','S']:
        f=Path(str(prefix)+f'.{party}.ledger');lines=f.read_text().splitlines()
        assert lines[-1]=='terminal=closed all_reserved=burned';counts={}
        for line in lines:
            fields=line.split()
            if len(fields)==5 and fields[1]=='reserved':counts[fields[2]]=counts.get(fields[2],0)+1
        ledgers.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f),'counts':counts})
    assert ledgers[0]['counts']==ledgers[1]['counts'];counts=ledgers[0]['counts']
    assert counts['token_fixed_input']==19*len(ids)
    assert counts['token_return_tile_mask_Enc0']==counts['token_exact_tile_GC']==a.hidden//512*len(ids)
    assert counts['token_acyclic_keys_9_levels']==counts['bfv_key_and_GC_independent_OT']==1
    assert data[0]['switch_key_bytes']==200026336
    for case in data[1:]:assert case['switch_key_bytes']==case['initial_key_bytes']==0
    assert len({c['online_bfv_gc_bytes'] for c in data[1:]})<=1
    assert identities=={str(f.relative_to(ROOT)):sha(f) for f in paths}
    assert sha(weight)==entry['sha256']
    report=dict(native_version='1.0.9',cases=data,source_sha256=identities,ledgers=ledgers,
        R1_reference=ref,stderr=err,candidate_only=False,production_integrated=True,
        child_waited=True,temporary_PKI_removed=not tmp.exists(),formal_acceptance=False,
        setup_not_hidden=True,full_model_R2=False,private_weights_mapped_only_at_S=True,
        actual_scan_entries_per_call=155648,actual_weight_visits_per_call=155648*a.hidden,
        per_session_key_setup=data[0]['initial_key_bytes']+data[0]['switch_key_bytes'])
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'record':str(dest),'cases':data}),flush=True)

if __name__=='__main__':main()
