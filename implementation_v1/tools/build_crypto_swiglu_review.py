#!/usr/bin/env python3
"""Independent, read-only integer review of all locked down-projection weights."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()
candidate_path=ROOT/'records/models_swiglu_domain_candidate.json'
candidate=json.loads(candidate_path.read_text())
models=[]
for source in candidate['models']:
    kind=source['model']; base=ROOT/'artifacts/models'/kind/'quantized_f12'
    manifest_path=base/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    assert sha(manifest_path)==source['quantized_manifest_sha256']
    expected={r['tensor']:r for r in source['layers']}
    selected=[e for e in manifest['tensors'] if e['name'].endswith('mlp.down_proj.weight')]
    assert len(selected)==len(expected)==28
    records=[]
    for item in selected:
        path=base/item['path']; entry=expected[item['name']]
        digest=sha(path); assert digest==item['sha256']==entry['sha256']
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        assert list(a.shape)==entry['shape'] and a.dtype.kind=='i'
        maxima=[]; maximum_abs=0
        for start in range(0,a.shape[0],64):
            b=a[start:start+64].astype(np.int64)
            absolute=np.abs(b); rows=absolute.sum(axis=1,dtype=np.int64)
            maxima.extend(int(v) for v in rows)
            maximum_abs=max(maximum_abs,int(absolute.max()))
        l1=max(maxima); row=maxima.index(l1)
        assert l1==entry['maximum_row_l1_f12_integer']==item['maximum_row_l1_integer']
        assert row==entry['maximum_l1_row'] and maximum_abs==entry['maximum_absolute_weight_f12_integer']
        bound=(1<<28)*l1
        loose=(1<<28)*int(a.shape[1])*maximum_abs
        assert bound==entry['full_dot_and_any_partial_sum_abs_f24_bound']
        assert loose==entry['alternative_bound_using_dimension_and_max_scalar']
        assert max(bound,loose,1<<40)<(1<<56)
        records.append({'tensor':item['name'],'sha256':digest,'shape':list(a.shape),
                        'maximum_row_l1':l1,'maximum_l1_row':row,'maximum_absolute_weight':maximum_abs,
                        'raw_dot_and_any_subset_abs_bound':bound,'dimension_maximum_abs_loose_bound':loose,
                        'post_floor12_abs_integer_bound':(bound+4095)//4096})
    models.append({'model':kind,'manifest_sha256':sha(manifest_path),'layers_verified':len(records),
                   'maximum_row_l1':max(e['maximum_row_l1'] for e in records),
                   'maximum_raw_dot_bound':max(e['raw_dot_and_any_subset_abs_bound'] for e in records),
                   'layers':records})
report={'kind':'independent_implementation_review_not_acceptance_standard',
        'candidate_sha256':sha(candidate_path),'script_sha256':sha(Path(__file__)),
        'integer_backend':'NumPy CPU int64 chunks of64 rows, exact Python integer products',
        'GPU_attempt':'torch2.6.0+cu124 reported CUDA unknown error / unavailable; no GPU environment changes',
        'upstream_absolute_f12':1<<20,'joint_abs_f12':1<<28,'elementwise_product_bound':1<<40,
        'exclusive_intermediate_budget':1<<56,'models':models,
        'conclusion':'All56 actual files and every row verified. Down-projection raw sums and any subset/partial sum fit budget under joint cap2^28.',
        'not_established':'Does not imply down output<=256, residual/RMS input domain, successful later layers, or full-model execution.'}
(ROOT/'records/numeric_swiglu_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'all_files_verified':sum(m['layers_verified'] for m in models),'models':[{k:m[k] for k in ('model','maximum_row_l1','maximum_raw_dot_bound')} for m in models]}))
