import json,subprocess,tempfile,uuid
from pathlib import Path
import torch
from tools.run_tile_matrix_protocol import ROOT,sha
from references.resource_policy import require_model_device
from private_rag.pki import create_local_pki
def main():
    dest=ROOT/'records/pack109_diagonal_candidate.json'
    if dest.exists():raise RuntimeError('record exists')
    files=[ROOT/'experiments/pack109'/f for f in ['diagonal_probe.cpp','secure_ops_candidate.cpp','include/prag/secure_ops.hpp','run_diagonal.py','build/prag_pack109_diagonal','libprag_crypto_v108.a']]
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    with tempfile.TemporaryDirectory(prefix='prag-diag109-') as tmp:
        tmp=Path(tmp);pki=tmp/'pki';create_local_pki(pki);ref=tmp/'reference.u64';device=require_model_device('R1')
        with ref.open('wb') as stream:
            for n in [128,1024,2048]:
                j=torch.arange(n,device=device,dtype=torch.int64);value=((j%8191-4095)*((j*197)%2097151-1048575))&((1<<59)-1)
                stream.write(value.cpu().numpy().astype('<u8').tobytes())
        reference={'device':torch.cuda.get_device_name(device),'sha256':sha(ref),'CPU_model_or_reference':False}
        del value,j;torch.cuda.empty_cache();prefix=ROOT/'records'/('pack109_diagonal_'+uuid.uuid4().hex)
        proc=subprocess.run([str(files[-2]),str(prefix),str(pki),str(ref)],capture_output=True,text=True,timeout=120)
        if proc.returncode:raise RuntimeError(proc.stderr)
        data=json.loads(proc.stdout)
    ledgers=[]
    for party in ['C','S']:
        f=Path(str(prefix)+f'.{party}.ledger');assert f.read_text().splitlines()[-1]=='terminal=closed all_reserved=burned';ledgers.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f)})
    assert hashes=={str(f.relative_to(ROOT)):sha(f) for f in files}
    data.update(source_sha256=hashes,ledgers=ledgers,R1_reference=reference,temporary_resources_removed=not tmp.exists(),child_waited=True)
    dest.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data['cases']))
if __name__=='__main__':main()
