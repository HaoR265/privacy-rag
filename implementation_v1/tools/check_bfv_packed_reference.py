#!/usr/bin/env python3
"""Small public algebra diagnostics, not a BFV/R2 execution or acceptance suite."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
N, T = 16384, 1 << 59
for d in range(1, 8193):
    stride = 2*d-1
    capacity = min(N//stride, 8192//d)
    assert capacity >= 1 and capacity*d <= 8192 and capacity*stride <= N
    assert (capacity-1)*stride+d-1 < N
cases = []
for d in (1, 2, 3, 5, 9):
    for m in (1, 2, 3):
        x = [((13*j+7) % 19)-9 for j in range(d)]
        w = [[0 if r == 1 else ((j*(r+3)+5) % 11)-5
              for j in range(d)] for r in range(m)]
        s = 2*d-1
        packed = [0]*(m*s)
        for r in range(m):
            for j in range(d):
                packed[r*s+d-1-j] = w[r][j]
        product = [0]*(m*s+d-1)
        for j in range(d):
            for k in range(len(packed)):
                product[j+k] += x[j]*packed[k]
        targets = [product[r*s+d-1] for r in range(m)]
        expected = [sum(a*b for a,b in zip(x,row)) for row in w]
        assert targets == expected
        # All padded high message coefficients vanish.
        assert not any(product[m*s:])
        cases.append({'d':d, 'rows':m, 'targets':targets})
probe = []
for d, m in ((4096,3),(2730,4)):
    x = [((13*j+7) % 101)-50 for j in range(d)]
    expected = [sum(x[j]*(0 if r in (1,m-1) else ((j*(r+3)+5)%17)-8)
                    for j in range(d)) for r in range(m)]
    probe.append({'d':d,'rows':m,'signed_raw_expected':expected,
                  'ring59_expected':[v%T for v in expected]})
report = {'kind':'public_integer_algebra_not_R2', 'all_8192_dimensions_valid':True,
          'small_polynomial_cases':cases,'probe_expected':probe,
          'message_support':'[r*(2*d-1),(r+1)*(2*d-1)-1]',
          'noise_may_wrap_but_L1_bound_still_applies':True,
          'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                           for p in ('native/src/bfv.cpp','native/include/prag/bfv.hpp',
                                     'tools/check_bfv_packed_reference.py')}}
(ROOT/'records/bfv_packed105_public_reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'public_layout_dimensions':8192,'small_polynomial_cases':len(cases),
                  'all_matched':True,'not_BFV_or_R2':True}))
