#!/usr/bin/env python3
"""Tiny public-integer diagnostics for the locked BFV decode; no R2 claim."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMES = [1152921504599080961, 1152921504600260609,
          1152921504604979201, 1152921504605962241,
          1152921504606584833, 1152921504606683137]
Q = 1
for prime in PRIMES:
    Q *= prime
T = 1 << 59
H = Q // 2


def locked_decode60(u, c0):
    assert 0 <= u < Q and 0 <= c0 < Q
    phase = u + c0
    assert phase < 1 << 361
    if phase >= Q:
        phase -= Q
    numerator = phase * T + H
    assert 0 <= numerator < Q * (1 << 60) and numerator < 1 << 420
    remainder = numerator >> 60
    assert 0 <= remainder < Q
    quotient = 0
    for step in range(59, -1, -1):
        remainder = (remainder << 1) | ((numerator >> step) & 1)
        assert remainder < 2 * Q and remainder < 1 << 361
        greater = remainder >= Q
        remainder -= int(greater) * Q
        quotient |= int(greater) << step
        assert 0 <= remainder < Q
    assert (quotient, remainder) == divmod(numerator, Q)
    return quotient, quotient % T


first = (Q - H + T - 1) // T
last = (T * Q - H + T - 1) // T
cases = [(0, 0), (first - 1, 0), (0, first), (last - 1, 0),
         (0, last), (Q - 1, 0), (Q - 1, 1), (Q - 1, Q - 1)]
result = []
for u, c0 in cases:
    whole, low = locked_decode60(u, c0)
    result.append({'u': str(u), 'c0': str(c0),
                   'full_quotient': str(whole), 'low59': str(low)})
assert [int(item['low59']) for item in result] == [0, 0, 1, T-1, 0, 0, 0, 0]
assert Q.bit_length() == 360 and result[5]['full_quotient'] == str(T)
report = {'kind': 'public_integer_diagnostic_not_R2', 'cases': result,
          'all_matched_divmod': True, 'locked_q': str(Q),
          'q_bits': 360, 'phase_bits': 361, 'numerator_bits': 420,
          'remainder_bits': 361, 'fixed_rounds': 60,
          'quotient_bit59_required_in_current_initialization': True,
          'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                            for path in ['native/src/bfv.cpp', 'tools/check_bfv_decode60_reference.py']}}
(ROOT / 'records/bfv_decode60_public_reference.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'public_integer_cases': len(result), 'all_matched_divmod': True,
                  'not_GC_or_R2': True}))
