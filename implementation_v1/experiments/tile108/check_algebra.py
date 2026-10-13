"""Public exact integer checks supporting the tile carry theorem, not acceptance."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
q=int(json.loads((ROOT/'artifacts/bfv_context.json').read_text())['levels'][1]['Q'])
t=1<<59;B=8194*(512*(1<<71)+1)
assert 4*t*B<q
def delta(a,b):
    carry1=(a&1)&(b&1);p1=(a>>1)^(b>>1)
    s1=p1^carry1;s2=((a>>1)&(b>>1))^(p1&carry1)
    return (s1^s2)+2*(s1&s2)
for a in range(4):
    for b in range(4):assert delta(a,b)==(a+b+2)//4
def check(q,t,u,v,wanted):
    A,ra=divmod(u*t,q);C,rb=divmod(v*t,q)
    out=(A+C+delta(4*ra//q,4*rb//q))%t
    exact=(((u+v)%q)*t+q//2)//q%t
    assert out==exact==wanted
# Exhaust all local splits of every noiseless small-ring message.
small=0
for Q,T in [(257,4),(509,8),(1021,16)]:
    for m in range(T):
        phase=(Q*m+T//2)//T%Q
        for u in range(Q):check(Q,T,u,(phase-u)%Q,m);small+=1
# Actual q/t, all quartile/high-part boundaries and maximum proven phase error.
rng=random.Random(108);actual=0
messages=[0,1,2,t//2-1,t//2,t-2,t-1]+[rng.randrange(t) for _ in range(100)]
us=[0,1,q-1]+[rng.randrange(q) for _ in range(64)]
inv=pow(t,-1,q)
for quarter in range(5):
    for offset in [-2,-1,0,1,2]:us.append(((quarter*q//4+offset)*inv)%q)
for m in messages:
    for e in [-B,-1,0,1,B]:
        phase=((q*m+t//2)//t+e)%q
        for u in us:check(q,t,u,(phase-u)%q,m);actual+=1
report={'kind':'exact_public_integer_derivation_checks','result':'matched','q':str(q),'t':str(t),'B':str(B),
 'four_t_B_less_q':True,'small_noiseless_cases':small,'actual_profile_cases':actual,'carry_truth_table_cases':16,
 'GC_AND_gates_per_coordinate':122,'GC_instance_unit':'whole tile of at most 512 coordinates',
 'GC_division_rounds':0,'local_quotient_remainder':'GMP exact integer division, never float',
 'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/tile108/check_algebra.py','experiments/tile108/tile_decoder.inc']},
 'full_model_R2':False,'formal_acceptance':False}
(ROOT/'records/tile108_algebra_candidate.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['result','small_noiseless_cases','actual_profile_cases','four_t_B_less_q']}))
