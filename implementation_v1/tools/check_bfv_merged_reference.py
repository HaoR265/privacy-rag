#!/usr/bin/env python3
"""Public integer derivation checks; not BFV/R2 execution or acceptance criteria."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T, N, L = 1 << 59, 16384, 1 << 71
context = json.loads((ROOT / 'artifacts/bfv_context.json').read_text())
q = int(context['levels'][1]['Q'])
bounds = []
for k in range(1, 20):
    bound = 8194 * (k * L + 1)
    left = 2 * T * bound + 1
    assert left < q
    bounds.append({'K': k, 'B_K': str(bound), '2tB_K_plus_1': str(left)})
for d in range(1, 8193):
    stride = 2 * d - 1
    capacity = min(N // stride, 8192 // d)
    assert capacity >= 1 and capacity * d <= 8192
    assert capacity * stride - 1 < N
probe_expected = []
for case, (columns, rows) in enumerate(((3, 3), (8193, 3), (155648, 1))):
    def x_at(j):
        return (7 if j in (0, 8192) else 0) if case == 1 else (j * 13 + 7) % 101 - 50
    def w_at(row, j):
        if case < 2 and row == 1:
            return 0
        if case == 1:
            return ({0: 11, 8192: -11} if row == 0 else {0: 3, 8192: -5}).get(j, 0)
        return (j * (row + 3) + 5) % 17 - 8
    sums = [0] * rows
    blocks = (columns + 8191) // 8192
    d = min(columns, 8192)
    for block in range(blocks):
        begin = block * 8192
        actual = min(d, columns - begin)
        # Single-group short case and singleton merged groups have fixed target.
        for row in range(rows):
            sums[row] += sum(x_at(begin + j) * w_at(row, begin + j) for j in range(actual))
    direct = [sum(x_at(j) * w_at(row, j) for j in range(columns)) for row in range(rows)]
    assert sums == direct
    probe_expected.append({'columns': columns, 'rows': rows, 'input_blocks': blocks,
                           'signed_raw_expected': direct,
                           'ring59_expected': [value % T for value in direct]})
assert probe_expected[1]['signed_raw_expected'] == [0, 0, -14]
report = {'kind': 'public_integer_algebra_not_R2', 'native_version': '1.0.6',
          'all_8192_dimensions_valid': True, 'all_19_noise_bounds_valid': True,
          'Q': str(q), 'noise_bounds': bounds, 'probe_expected': probe_expected,
          'raw_input_ciphertext_cache_max_bytes': 19 * N * 6 * 2 * 8,
          'excludes_from_cache_count': ['product', 'accumulator', 'NTT scratch', 'context'],
          'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                            for p in ('native/src/bfv.cpp', 'native/include/prag/secure_ops.hpp',
                                      'tools/check_bfv_merged_reference.py')}}
(ROOT / 'records/bfv_merged106_public_reference.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'public_layout_dimensions': 8192, 'noise_bounds': 19,
                  'probe_cases': 3, 'all_matched': True, 'not_BFV_or_R2': True}))
