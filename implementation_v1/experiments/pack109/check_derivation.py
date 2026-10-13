"""Exact public algebra bounds; finite checks supplement the written proof."""
import json,hashlib
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    ctx=json.loads((ROOT/'artifacts/bfv_context.json').read_text());n=16384;t=1<<59;q=int(ctx['levels'][1]['Q']);p=int(ctx['levels'][0]['primes'][-1])
    switch=Fraction(6*n*((1<<60)-1)*21,p)+Fraction(n+1,2)
    assert switch<(1<<24)
    e0=8194*(19*(1<<71)+1)+1
    eb=(1<<18)*e0+174762*(1<<24)+8194
    assert 4*t*eb<q and 512*(1<<20)<(1<<58)
    checks=0
    for b in range(1,17):
        d=16
        while d<=8192//b:
            c=min(512//b,8192//(b*d));s=2*d-1
            assert c and c*d<=8192 and b*c*s-1<n and (b-1)*s+d<=n
            # Target intervals are disjoint by s=2d-1. Enumerate every row,
            # output and +/- (d-1) interval endpoint, not secret values.
            centers=[(o*b+r)*s+d-1 for o in range(c) for r in range(b)]
            assert len(set(centers))==b*c
            assert all(centers[i+1]-centers[i]>d-1 for i in range(len(centers)-1))
            checks+=1;d*=2
    for level in range(9):
        step=n>>(level+1);g=(1<<(level+1))+1
        for m in range(1<<(level+1)):
            actual=(g*m*step)%(2*n)
            assert actual%n==m*step and (actual>=n)==bool(m&1)
        for j in range(n):assert ((j*g)%(2*n))%step==0 if j%step==0 else ((j*g)%(2*n))%step!=0
    paths=['experiments/pack109/check_derivation.py','experiments/pack109/acyclic_pack.hpp','records/pack109_candidate_derivation.md',
        'third_party/SEAL/native/src/seal/keygenerator.cpp','third_party/SEAL/native/src/seal/evaluator.cpp','third_party/SEAL/native/src/seal/util/rlwe.cpp']
    result={'kind':'candidate_exact_bound_and_index_checks_not_security_audit','matrix_public_layout_cases':checks,
       'key_switch_error_fraction':str(switch),'strict_key_switch_error_bound':1<<24,'packing_levels':9,'packing_scale':512,
       'token_return_error_bound':str(eb),'token_return_error_bound_bits':eb.bit_length(),'four_t_E':str(4*t*eb),'q':str(q),
       'four_t_E_less_q':True,'token_unscale_no_wrap':True,'acyclic_switching_no_circular_key_graph':True,
       'GC_ANDs_per_Token_coordinate':358,'input_ciphertexts_per_Token':19,'return_objects_H1024':2,'return_objects_H2048':4,
       'finite_routing_checks_passed':True,'full_R2':False,'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}}
    (ROOT/'records/pack109_candidate_algebra.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['matrix_public_layout_cases','token_return_error_bound_bits','four_t_E_less_q']}))
if __name__=='__main__':main()
