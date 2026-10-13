"""Public schedules and application-frame bytes, never a model benchmark."""
import hashlib,json,math
from pathlib import Path
from tools.build_tile_cost import INPUT,RETURN,DESCRIPTOR,GC_FIXED,GC_PER_COORD,GC_CHUNK,chunks
ROOT=Path(__file__).resolve().parents[1]
BASE_OT=57761
INITIAL_PK=1835104
SWITCH_KEYS=200026336
def ot_bytes(n,l=59,wordbytes=8):
    return sum(128*((math.ceil(c/256)*256)//8+96)+wordbytes*math.ceil(2*c*l/(8*wordbytes))+96
               for i in range(0,n,1024) for c in [min(1024,n-i)])
ONEHOT=318*ot_bytes(1,1,1)+2*sum(ot_bytes(1<<i,1,1) for i in range(18))+ot_bytes(155648)
TOKEN_GC=240+2*128*(2048+96)+32*61*512+31*96+16*120*512+96+math.ceil(358*512*32/65536)*GC_CHUNK+59*512+96
def layout(b,k,o):
    best=None;d=16;gc=512//b
    while d<=8192//b:
        c=min(gc,8192//(b*d));full,tail=divmod(o,gc)
        ret=full*math.ceil(gc/c)+(math.ceil(tail/c) if tail else 0);inp=math.ceil(k/d)
        candidate=(inp*INPUT+ret*RETURN,d,c,inp,ret)
        if best is None or candidate<best:best=candidate
        d*=2
    return best[1:]
def empty():
    return dict(inputs=0,returns=0,matrix_calls=0,joint_calls=0,matrix_targets=0,matrix_gc_tiles=0,matrix_gc_chunks=0,
                token_calls=0,token_targets=0,token_gc_tiles=0,diagonal_coordinates=0,
                matrix_bytes=0,token_bfv_gc_bytes=0,token_onehot_bytes=0,diagonal_bytes=0)
def add(a,b,times=1):
    for k in a:a[k]+=b[k]*times
def matrix(b,k,o,joint=False):
    d,c,inputs,returns=layout(b,k,o);g=512//b;full,tail=divmod(o,g)
    tiles=full+bool(tail);gcchunks=full*chunks(b*g)+(chunks(b*tail) if tail else 0)
    x=empty();x.update(inputs=inputs,returns=returns,matrix_calls=1,joint_calls=int(joint),matrix_targets=b*o,
                      matrix_gc_tiles=tiles,matrix_gc_chunks=gcchunks,
                      matrix_bytes=inputs*INPUT+returns*RETURN+b*o*GC_PER_COORD+tiles*GC_FIXED+gcchunks*GC_CHUNK+DESCRIPTOR*(1+joint))
    return x
def token(h):
    x=empty();x.update(inputs=19,returns=h//512,token_calls=1,token_targets=h,token_gc_tiles=h//512,
                      token_bfv_gc_bytes=DESCRIPTOR+19*INPUT+h//512*(RETURN+TOKEN_GC),token_onehot_bytes=ONEHOT)
    return x
def diagonal(h):
    x=empty();x.update(diagonal_coordinates=h,diagonal_bytes=2*(40+96)+ot_bytes(59*h));return x
def sequence(length,h,ffn):
    result=empty();add(result,token(h),length)
    add(result,diagonal(h),length*(2*28+1));add(result,diagonal(128),length*28*24)
    for begin in range(0,length,16):
        b=min(16,length-begin)
        for k,o,joint in [(h,4096,True),(2048,h,False),(h,2*ffn,True),(ffn,h,False)]:
            add(result,matrix(b,k,o,joint),28)
    return result
def total(c):return sum(c[k] for k in ['matrix_bytes','token_bfv_gc_bytes','token_onehot_bytes','diagonal_bytes'])
def main():
    counts={'prepare_embedding_257':sequence(257,1024,3072),'query_embedding_160':sequence(160,1024,3072),
            'generation_prefill_1024':sequence(1024,2048,6144),'generation_decode_255':empty(),'generation_logits_256':empty()}
    add(counts['generation_decode_255'],sequence(1,2048,6144),255)
    add(counts['generation_logits_256'],matrix(1,2048,151936),256)
    counts['one_decode_token_including_logits']=sequence(1,2048,6144)
    add(counts['one_decode_token_including_logits'],matrix(1,2048,151936))
    counts['generation_total']=empty()
    for k in ['generation_prefill_1024','generation_decode_255','generation_logits_256']:add(counts['generation_total'],counts[k])
    counts['nonempty_query_total']=dict(counts['generation_total']);add(counts['nonempty_query_total'],counts['query_embedding_160'])
    measured=[]
    for p in sorted((ROOT/'records').glob('pack109_production_*x*_m?.json')):
        e=json.loads(p.read_text());b,k,o=e['shape_B_K_O'];mode=int(p.stem[-1])
        expected=matrix(b,k,o,mode in [2,3])['matrix_bytes']+INITIAL_PK+BASE_OT
        assert expected==e['application_bytes'],(p,expected,e['application_bytes'])
        measured.append({'record':str(p.relative_to(ROOT)),'bytes':expected})
    for p in sorted((ROOT/'records').glob('pack109_token_production_H*.json')):
        e=json.loads(p.read_text())
        for i,case in enumerate(e['cases']):
            expected=total(token(case['hidden']))+(INITIAL_PK+SWITCH_KEYS+3*BASE_OT if i==0 else 0)
            assert expected==case['application_bytes'],(p,i,expected,case['application_bytes'])
            measured.append({'record':str(p.relative_to(ROOT)),'case':i,'bytes':expected})
    old=json.loads((ROOT/'records/crypto_static_cost_v1_0_8.json').read_text())
    oldbytes=old['BFV_bytes_including_GC_input_OT_excluding_setup_and_other_operators']
    paths=['tools/build_pack109_cost.py','tools/build_tile_cost.py','native/include/prag/bfv_layout.hpp',
           'native/src/bfv_matrix.inc','native/src/bfv.cpp','native/src/secure_ops.cpp',
           'native/src/bfv_token_engine.inc','native/src/bfv_token_decoder.inc','compute/model.hpp','compute/numerics.hpp']
    result={'kind':'static_changed_operator_scope_not_full_model_or_runtime','native_version':'1.0.9','counts':counts,
      'online_application_bytes':{k:total(c) for k,c in counts.items()},
      'same_functional_scope_excluding_onehot_and_setup':{k:total(c)-c['token_onehot_bytes'] for k,c in counts.items()},
      'previous_v108_same_functional_scope_bytes':oldbytes,
      'setup_per_session':{'input_public_key':INITIAL_PK,'acyclic_switch_keys_and_final_public_key':SWITCH_KEYS,
                           'base_OT_streams':3,'all_base_OT_bytes':3*BASE_OT,'total':INITIAL_PK+SWITCH_KEYS+3*BASE_OT},
      'setup_accounting':'Paid once for one shared SecureOps session; if phases have separate sessions, charge each session again.',
      'per_unit_bytes':{'input':INPUT,'return_c1':RETURN,'token_GC_512':TOKEN_GC,'token_onehot_warm':ONEHOT,'base_OT_stream':BASE_OT},
      'measured_transcript_reconciliation':measured,'all_available_measured_bytes_match':True,
      'full_model_R2':False,'formal_acceptance':False,
      'excludes':['secret Attention QK/PV and other Beaver operations','range checks, validity/failure handling, floor, RMS/Softmax/NR and greedy selection',
                  'management, retrieval, heartbeat and TLS/TCP/IP transport overhead','runtime and hardware performance extrapolation'],
      'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}}
    (ROOT/'records/crypto_static_cost_v1_0_9.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'query_inputs':counts['nonempty_query_total']['inputs'],'query_online_bytes':total(counts['nonempty_query_total']),
                      'per_session_setup':result['setup_per_session']['total'],'measured_cases':len(measured)}))
if __name__=='__main__':main()
