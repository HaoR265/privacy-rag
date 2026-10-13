"""Production-linked primitive diagnostic; records exact source and ledger identities."""
import argparse,json,hashlib,subprocess,tempfile,uuid,os,signal
from pathlib import Path
from tools.run_tile_matrix_protocol import reference, ROOT, sha
from private_rag.pki import create_local_pki
def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['small','shapes','single'],default='small');p.add_argument('--prefix',default='pack109_repeat_'+uuid.uuid4().hex);a=p.parse_args()
    if not a.prefix or len(a.prefix)>120 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.prefix):p.error('prefix must contain only ASCII letters, digits, underscore or hyphen')
    cases=[(3,17,7,0),(3,17,7,1),(16,17,33,0)] if a.suite in ['small','single'] else [(1,2048,4096,2),(1,2048,12288,3),(16,2048,2048,0),(16,2048,6144,0),(16,6144,2048,0)]
    if a.suite=='single':cases=cases[:1]
    sources=[ROOT/'experiments/pack109_production/matrix_probe.cpp']+list((ROOT/'native/src').glob('*'))+list((ROOT/'native/include/prag').glob('*'))
    binary=ROOT/'experiments/pack109_production/build/prag_pack109_matrix';sources+=[Path(__file__).resolve(),binary,ROOT/'experiments/pack109_production/build/native/libprag_crypto.a']
    identities={str(q.relative_to(ROOT)):sha(q) for q in sources}
    for b,k,o,mode in cases:
        name=f'{a.prefix}_{b}x{k}x{o}_m{mode}';dest=ROOT/'records'/f'{name}.json'
        if dest.exists():raise RuntimeError('existing record '+name)
        print(json.dumps({'starting':name}),flush=True)
        with tempfile.TemporaryDirectory(prefix='prag-pack109-') as tmp:
            tmp=Path(tmp);pki=tmp/'pki';create_local_pki(pki);expected=tmp/'expected.u64'
            ref=reference(b,k,o,1 if mode==1 else 0,expected)
            prefix=ROOT/'records'/(name+'_'+uuid.uuid4().hex)
            proc=subprocess.Popen(['nice','-n','10',str(binary),str(prefix),str(pki),str(b),str(k),str(o),str(mode),str(expected)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:stdout,stderr=proc.communicate(timeout=3600)
            except BaseException:
                os.killpg(proc.pid,signal.SIGTERM);proc.communicate();raise
            if proc.returncode:raise RuntimeError(stderr)
            data=json.loads(stdout)
            (ROOT/'records'/f'{name}_raw_output.json').write_text(json.dumps(data,indent=2)+'\n')
        ledgers=[]
        for party in ['C','S']:
            f=Path(str(prefix)+f'.{party}.ledger');lines=f.read_text().splitlines();assert lines[-1]=='terminal=closed all_reserved=burned'
            counts={}
            for line in lines:
                fields=line.split()
                if len(fields)==5 and fields[1]=='reserved':counts[fields[2]]=counts.get(fields[2],0)+1
            ledgers.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f),'reservation_instances':counts})
        assert ledgers[0]['reservation_instances']==ledgers[1]['reservation_instances']
        counts=ledgers[0]['reservation_instances']
        assert counts['bfv_matrix_input']==data['input_ciphertexts']
        assert counts['bfv_matrix_tile_mask_Enc0']==data['returned_ciphertext_groups']
        if identities!={str(q.relative_to(ROOT)):sha(q) for q in sources}:raise RuntimeError('source changed during run')
        data.update(candidate_only=False,production_integrated=True,source_sha256=identities,R1_reference=ref,ledgers=ledgers,
            GC_instances=counts['bfv_matrix_tile_GC'],GC_ANDs=122*b*o,OT_choices=2*b*o,
            joint_groups=[2048,1024,1024] if mode==2 else ([6144,6144] if mode==3 else [o]),
            temporary_resources_removed=not tmp.exists(),child_waited=True,stderr=stderr)
        dest.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'complete':name,**{k:data[k] for k in ['input_ciphertexts','returned_ciphertext_groups','application_bytes','elapsed_seconds']}}),flush=True)
if __name__=='__main__':main()
