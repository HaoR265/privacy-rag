"""Independent authenticated public phase diagnostic, no model execution."""
import hashlib,json,os,signal,subprocess,tempfile,uuid
from pathlib import Path
from private_rag.pki import create_local_pki
ROOT=Path(__file__).resolve().parents[2]
def main():
    target=ROOT/'records/pack109_token_phase_mtls.json'
    if target.exists():raise RuntimeError('evidence already exists')
    paths=['native/src/bfv.cpp','native/src/bfv_matrix.inc','native/src/bfv_tile_decoder.inc',
           'native/src/bfv_token_decoder.inc','native/src/bfv_acyclic_pack.hpp',
           'native/src/bfv_token_api.inc','native/src/bfv_token_engine.inc',
           'experiments/pack109_production/token_phase_probe.cpp',
           'experiments/pack109_production/run_token_phase.py',
           'experiments/pack109_production/build/token_phase_probe',
           'experiments/pack109_production/build/native/libprag_crypto.a']
    sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    identities={p:sha(p) for p in paths}
    prefix=ROOT/'records'/('pack109_token_phase_'+uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix='prag-token-phase-') as temp:
        pki=Path(temp)/'pki';create_local_pki(pki)
        p=subprocess.Popen([str(ROOT/paths[-2]),str(prefix),str(pki)],stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:out,err=p.communicate(timeout=120)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM);p.communicate(timeout=5);raise
        if p.returncode:raise RuntimeError(err)
    data=json.loads(out);data.update(source_and_binary_sha256=identities,stderr=err,exit_code=p.returncode,
        temporary_PKI_removed=not Path(temp).exists(),child_waited=True,production_activated=False)
    data['ledgers']=[]
    for role in ['C','S']:
        ledger=Path(str(prefix)+f'.{role}.ledger')
        assert ledger.read_text().splitlines()[-1]=='terminal=closed all_reserved=burned'
        name=str(ledger.relative_to(ROOT));data['ledgers'].append({'path':name,'sha256':sha(name)})
    assert identities=={name:sha(name) for name in paths}
    target.write_text(json.dumps(data,indent=2)+'\n');print(out.strip())
if __name__=='__main__':main()
