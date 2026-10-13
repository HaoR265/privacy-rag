"""Finite new RoPE table opcode boundary debug; no model is executed."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import uuid
import torch
from private_rag.pki import create_local_pki
from references.resource_policy import require_model_device

ROOT=Path(__file__).resolve().parents[2]
MASK=(1<<59)-1

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    os.chdir(ROOT)
    device=require_model_device('R1')
    indices=torch.tensor([0,1,1278,1279,1280,MASK,1279+(1<<40),1<<58],device=device,dtype=torch.int64)
    table=(((torch.arange(1280*128,device=device,dtype=torch.int64)*123457)%((1<<30)-1))-(1<<29)).reshape(1280,128)
    bad=indices>1279
    expected=torch.where(bad[:,None],0,table[torch.where(bad,0,indices)])
    raw=lambda t:(t&MASK).cpu().numpy().astype('<u8').tobytes()
    fixture=struct.pack('<Q',8)+raw(indices)+raw(table.flatten())+raw(expected.flatten())+bad.cpu().numpy().astype('u1').tobytes()
    sid=uuid.uuid4().hex;prefix=ROOT/f'records/hybrid112_lookup_{sid}'
    paths=['experiments/hybrid112/lookup_probe.cpp','experiments/hybrid112/CMakeLists.txt',
           'experiments/hybrid112/run_lookup_probe.py','build/hybrid112edge/prag_hybrid112_lookup',
           'build/hybrid112edge/native/libprag_crypto.a','native/src/batch.cpp','native/include/prag/batch.hpp']
    registration={'source_and_binary_sha256':{p:sha(ROOT/p) for p in paths},'fixture_sha256':hashlib.sha256(fixture).hexdigest(),
                  'positions':indices.cpu().tolist(),'expected_bad':bad.cpu().tolist(), 'device':torch.cuda.get_device_name(device)}
    with tempfile.TemporaryDirectory(prefix='prag112-lookup-') as td:
        td=Path(td);create_local_pki(td/'pki');(td/'public.bin').write_bytes(fixture)
        run=subprocess.run([str(ROOT/'build/hybrid112edge/prag_hybrid112_lookup'),str(prefix),str(td/'pki'),str(td/'public.bin'),sid],capture_output=True,text=True)
    result={**registration,'exit_code':run.returncode,'stderr':run.stderr,'output':json.loads(run.stdout) if run.stdout else None,
            'evidence':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(prefix.parent.glob(prefix.name+'.*'))]}
    Path(str(prefix)+'.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'record':str(prefix.relative_to(ROOT))+'.json','exit_code':run.returncode,'output':result['output'],'stderr':run.stderr}),flush=True)
    if run.returncode:raise RuntimeError('LOOKUP_DEBUG_FAILED')

if __name__=='__main__':main()
