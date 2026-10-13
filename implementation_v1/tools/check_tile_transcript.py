"""Reconcile actual tile frames and byte counts, no protocol or model run."""
import hashlib,json
from pathlib import Path
from tools.build_tile_cost import INPUT,RETURN,tile_bytes,chunks,bytes_for,empty
ROOT=Path(__file__).resolve().parents[1]
def main():
    paths=[f'records/tile108_small_{s}.json' for s in ('3x17x7_m0','3x17x7_m1','16x17x33_m0')]
    paths += [f'records/tile108_shapes_{s}_m0.json' for s in ('1x2048x2048','1x2048x6144','1x6144x2048','1x2048x151936')]
    cases=[]
    for path in paths:
        r=json.loads((ROOT/path).read_text());b,k,o=r['shape_B_K_O'];cap=512//b
        ts=[min(cap,o-first)*b for first in range(0,o,cap)]
        h,c=r['input_ciphertexts'],len(ts)
        expected={'application_bytes':h*INPUT+c*RETURN+sum(tile_bytes(t) for t in ts)+1893217,
                  'frames_sent':128*c+h+259,'frames_received':sum(chunks(t) for t in ts)+6*c+129,
                  'transcript_direction_runs':2*c+6,'tile_GC_instances':c,'GC_AND_gates':122*b*o,
                  'GC_client_OT_choices':2*b*o,'GC_OT_extension_batches':c}
        assert all(r[key]==value for key,value in expected.items()),(path,expected)
        cases.append({'record':path,'measured_counts_match':True,**expected})
    # The long-input diagnostic reuses its setup from the first short matrix.
    long=json.loads((ROOT/'records/tile108_long_mtls.json').read_text())
    for case in long['cases'][1:]:
        t=case['rows'];ts=[min(512,t-i) for i in range(0,t,512)]
        wanted=case['input_blocks']*INPUT+t*RETURN+sum(tile_bytes(z) for z in ts)+352
        assert case['application_bytes_before_reconstruction']==wanted
    sources=paths+['records/tile108_long_mtls.json','tools/check_tile_transcript.py','tools/build_tile_cost.py',
                   'native/src/bfv_tile_decoder.inc','native/src/bfv.cpp','third_party/EzPC/SCI/src/OT/iknp.h']
    result={'result':'all_measured_application_counts_match','native_version':'1.0.8','cases':cases,
            'long_input_cases_verified':2,'fresh_matrix_setup_bytes_excluding_descriptor':1893217-352,
            'matrix_direction_runs':'2*tile_count+6 in these fresh sessions; not RTT',
            'GC_division_rounds':0,'GC_AND_gates_per_coordinate':122,'one_GC_per_tile':True,
            'excludes':['TLS/TCP/IP','separate diagnostic reconstruction'],'full_model_R2':False,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (ROOT/'records/tile108_transcript_cost_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':result['result'],'matrix_cases':len(cases),'long_cases':2}))
if __name__=='__main__':main()
