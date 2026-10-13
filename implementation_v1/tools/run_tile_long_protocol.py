"""Independent authenticated public phase diagnostic, no model execution."""
import hashlib,json,os,signal,subprocess,tempfile,uuid
from pathlib import Path
from private_rag.pki import create_local_pki
ROOT=Path(__file__).resolve().parents[1]
def main():
    target=ROOT/'records/tile108_long_mtls.json'
    if target.exists():raise RuntimeError('evidence already exists')
    paths=['native/src/bfv.cpp','native/src/bfv_matrix.inc',
           'native/src/bfv_tile_decoder.inc','native/src/dev_tile_long_probe.cpp',
           'tools/run_tile_long_protocol.py','native/build-tile108/prag_tile_long_probe',
           'native/build-tile108/libprag_crypto.a']
    sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    identities={p:sha(p) for p in paths}
    prefix=ROOT/'records'/('tile108_long_'+uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix='prag-tile-phase-') as temp:
        pki=Path(temp)/'pki';create_local_pki(pki)
        p=subprocess.Popen([str(ROOT/paths[-2]),str(prefix),str(pki)],stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:out,err=p.communicate(timeout=180)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM);p.communicate(timeout=5);raise
        if p.returncode:raise RuntimeError(err)
    data=json.loads(out);data.update(source_and_binary_sha256=identities,stderr=err,exit_code=p.returncode,
        temporary_PKI_removed=not Path(temp).exists(),child_waited=True,production_primitive_integrated=True,full_system_activated=False)
    data['ledgers']=[]
    for role in ['C','S']:
        ledger=Path(str(prefix)+f'.{role}.ledger')
        assert ledger.read_text().splitlines()[-1]=='terminal=closed all_reserved=burned'
        name=str(ledger.relative_to(ROOT));data['ledgers'].append({'path':name,'sha256':sha(name)})
    assert identities=={name:sha(name) for name in paths}
    counts={}
    for line in (ROOT/data['ledgers'][0]['path']).read_text().splitlines():
        fields=line.split()
        if len(fields)==5 and fields[1]=='reserved':counts[fields[2]]=counts.get(fields[2],0)+1
    assert counts.get('bfv_linear_tile_GC')==3 and counts.get('bfv_matrix_tile_GC')==1
    assert counts.get('bfv_target_GC',0)==0
    data['material_counts']=counts
    target.write_text(json.dumps(data,indent=2)+'\n');print(out.strip())
if __name__=='__main__':main()
