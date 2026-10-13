"""Actual exception cleanup of retained GPU Cache; no model inference."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from private_rag.pki import create_local_pki

ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def main():
    os.chdir(ROOT)
    prefix=f'records/hybrid112_cache_cleanup_{uuid.uuid4().hex}'
    sources=['experiments/hybrid112_cleanup/run.py','experiments/hybrid112_cleanup/CMakeLists.txt',
             'experiments/hybrid112_cleanup/cache_failure.cpp','compute/model.hpp','compute/local_shares.cu',
             'compute/local_shares.hpp','build/hybrid112cleanup/prag_hybrid112_cleanup',
             'build/hybrid112cleanup/native/libprag_crypto.a','build/hybrid112cleanup/libprag_cleanup_local.a']
    results={'model_executed':False,'production_enabled':False,'cases':[],
             'source_and_binary_sha256':{p:sha(p) for p in sources}}
    with tempfile.TemporaryDirectory(prefix='prag-cache-abort-') as td:
        pki=Path(td)/'pki';create_local_pki(pki)
        for mode in range(2):
            name=prefix+f'.{mode}'
            run=subprocess.run([str(ROOT/'build/hybrid112cleanup/prag_hybrid112_cleanup'),str(ROOT/name),str(pki),str(mode)],capture_output=True,text=True)
            if run.returncode:raise RuntimeError(run.stderr)
            for role in ('C','S'):
                if (ROOT/(name+'.'+role+'.ledger')).read_text().splitlines()[-1]!='terminal=aborted all_reserved=burned':
                    raise RuntimeError('NOT_ABORTED')
            results['cases'].append(json.loads(run.stdout))
    results['evidence']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p.relative_to(ROOT))} for p in sorted(ROOT.glob(prefix+'.*.ledger'))]
    (ROOT/(prefix+'.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'record':prefix+'.json',**results}),flush=True)

if __name__=='__main__':main()
