"""Generate actual public bindings; this tool never marks a backend enabled."""
from pathlib import Path
import json
import struct
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from private_rag.configuration import canonical,sha256_file
from private_rag.model_input import public_prompt_layout

def main():
    template_path=ROOT/'artifacts/template/template_v1.json'
    template=json.loads(template_path.read_text())
    private=[s for s in template['segments'] if s['kind']!='public']
    if [p['capacity'] for p in private]!=[128,256,256]: raise ValueError('TEMPLATE_SHAPE')
    tokens,valid=public_prompt_layout(template)
    data=b'PRTMPL01'+struct.pack('<4Q',*(p['offset'] for p in private),template['dummy_token_id'])
    data+=struct.pack('<1024Q',*tokens)+bytes(valid)
    native=ROOT/'artifacts/template/template_v1.bin';native.write_bytes(data)
    (native.parent/'native_manifest.json').write_text(json.dumps({
        'format_version':'PRTMPL01','path':native.name,'bytes':len(data),'sha256':sha256_file(native),
        'layout':'magic8, LEu64 question/evidence1/evidence2 offsets and dummy, LEu64 public_tokens[1024], u8 public_validity[1024]',
        'source_template_sha256':sha256_file(template_path)},indent=2)+'\n')
    files=[]
    def add(path,scope):
        p=ROOT/path;files.append({'path':path,'scope':scope,'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    for path in ('artifacts/template/template_v1.json','artifacts/template/template_v1.bin',
                 'artifacts/constants/constants_v1.json','artifacts/constants/constants_v1.bin',
                 'artifacts/constants/numeric_profile_v1_qk.json',
                 'artifacts/constants/swiglu_domain_evidence_v1.json',
                 'artifacts/constants/coordinated_domain_evidence_v1.json',
                 'artifacts/constants/range_evidence_v1_coordinated.json',
                 'artifacts/bfv_context.json','artifacts/bfv_correctness_bound.json',
                 'artifacts/bfv_native_manifest.json','artifacts/interface_delivery.json'):
        add(path,'public')
    for name in ('embedding','generation'):
        checkpoint=f'artifacts/models/{name}'
        checkpoint_manifest=json.loads((ROOT/checkpoint/'checkpoint_manifest.json').read_text())
        add(checkpoint+'/checkpoint_manifest.json','public')
        for item in checkpoint_manifest['files']:
            # Tokenizer/config files are actual owner-local inputs, not merely
            # descriptions of server weights. Verify their pinned identities
            # before binding them; never register altered bytes as a new norm.
            if item['path'].endswith('.safetensors'): continue
            p=ROOT/checkpoint/'checkpoint'/item['path']
            if p.stat().st_size!=item['bytes'] or sha256_file(p)!=item['sha256']:
                raise ValueError('CHECKPOINT_PUBLIC_FILE_MISMATCH')
            add(checkpoint+'/checkpoint/'+item['path'],'public')
        relative=f'artifacts/models/{name}/quantized_f12'
        manifest=json.loads((ROOT/relative/'manifest.json').read_text())
        add(relative+'/manifest.json','server_manifest')
        for item in manifest['tensors']:
            p=ROOT/relative/item['path']
            # Quantization already hashed every byte. Startup verification reads
            # bytes again; do not mistake this registration for a repeated check.
            files.append({'path':relative+'/'+item['path'],'scope':'server_weights','bytes':p.stat().st_size,'sha256':item['sha256']})
    config={'format_version':'private-rag-configuration-v1','protocol':'prag-v1','ring_bits':59,
      'numeric_version':json.loads((ROOT/'artifacts/constants/numeric_profile_v1_qk.json').read_text())['profile_id'],'template_sha256':sha256_file(template_path),
      'model_revisions':{'embedding':'97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3','generation':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'},
      'budgets':{'query_embedding':160,'document_embedding':257,'question':128,'evidence':[256,256],'prefill':1024,'outputs':256,'decode':255,'cache':1280},
      'files':sorted(files,key=lambda f:f['path'])}
    p=ROOT/'artifacts/configuration.json';p.write_bytes(canonical(config))
    (ROOT/'artifacts/configuration.sha256').write_text(sha256_file(p)+'\n')
    print(json.dumps({'configuration_sha256':sha256_file(p),'files':len(files),'production_enabled':False}))
if __name__=='__main__':main()
