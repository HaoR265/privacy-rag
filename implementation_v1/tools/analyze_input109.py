"""Disaggregate the immutable v108 public schedule; no model execution."""
import json
from pathlib import Path
from tools.build_tile_cost import INPUT, RETURN, empty, add, matrix, token, bytes_for
ROOT=Path(__file__).resolve().parents[1]
def parts(length,h,f):
    out={name:empty() for name in ['token','gamma_main','gamma_QK_128','Q','K','V','O','gate','up','down','logits','attention_secret_products']}
    add(out['token'],token(h),length)
    add(out['gamma_main'],matrix(1,h,h),length*57)
    add(out['gamma_QK_128'],matrix(1,128,128),length*28*24)
    for start in range(0,length,16):
        b=min(16,length-start)
        for name,k,o in [('Q',h,2048),('K',h,1024),('V',h,1024),('O',2048,h),('gate',h,f),('up',h,f),('down',f,h)]:
            add(out[name],matrix(b,k,o),28)
    return out
def main():
    stages={'query_embedding':parts(160,1024,3072),'prefill_1024':parts(1024,2048,6144),'decode_255':parts(1,2048,6144)}
    for c in stages['decode_255'].values():
        for k in c:c[k]*=255
    stages['logits_256']={'logits':empty()};add(stages['logits_256']['logits'],matrix(1,2048,151936),256)
    total=empty(); by={}
    for stage,ops in stages.items():
        for name,c in ops.items():
            add(total,c);by.setdefault(name,empty());add(by[name],c)
    baseline=json.loads((ROOT/'records/crypto_static_cost_v1_0_8.json').read_text())
    assert total==baseline['counts']['nonempty_query_total']
    def annotate(c):
        return dict(c,input_bytes=c['inputs']*INPUT,input_share_of_all_inputs=c['inputs']/total['inputs'],
                    return_bytes=c['ciphertexts']*RETURN,bfv_gc_subset_bytes=bytes_for(c))
    report={'kind':'v108_baseline_disaggregation_not_new_protocol','total':annotate(total),
            'by_operator':{k:annotate(v) for k,v in by.items()},
            'by_stage':{s:{k:annotate(v) for k,v in p.items()} for s,p in stages.items()},
            'attention_note':'QK and PV use shared Beaver products, no BFV inputs; gamma_QK_128 is head RMS gamma, not attention score GEMM'}
    (ROOT/'records/input109_v108_baseline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['by_operator'],indent=2))
if __name__=='__main__':main()
