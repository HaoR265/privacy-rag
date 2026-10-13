"""Finite public index/bound checks supporting the matrix layout derivation.

No model, private data or CPU matrix multiplication; not an acceptance suite.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = []
    for batch in range(1, 17):
        columns = 512 // batch
        # Each pair has an entire disjoint length-31 convolution interval.
        intervals = [(31*(o*batch+b), 31*(o*batch+b)+30,
                      31*(o*batch+b)+15)
                     for o in range(columns) for b in range(batch)]
        assert len({p[2] for p in intervals}) == batch*columns
        assert all(a[1] < b[0] for a, b in zip(intervals, intervals[1:]))
        assert intervals[-1][1] < 16384 and 16*columns <= 8192
        # Within every translated pair, only equal inner indices contribute
        # to its target. This includes all public zero-padded tails.
        for actual in range(1, 17):
            for j in range(actual):
                for k in range(actual):
                    assert (j + 15-k == 15) == (j == k)
        cases.append({'batch': batch, 'output_tile_columns': columns,
                      'target_count': len(intervals),
                      'maximum_message_degree': intervals[-1][1],
                      'maximum_weight_coefficients': 16*columns})
    q = int(json.loads((ROOT/'artifacts/bfv_context.json').read_text())['levels'][1]['Q'])
    bound = 8194*(512*(1 << 71)+1)
    assert 4*(1 << 59)*bound < q
    paths = ['native/src/bfv_matrix.inc', 'native/src/bfv.cpp',
             'native/include/prag/bfv.hpp', 'tools/check_tile_layout.py']
    report = {'kind': 'public_index_and_exact_integer_bound_check',
              'native_version': '1.0.8', 'result': 'matched_derived_invariants',
              'batch_cases': cases, 'all_inner_tail_lengths_checked': list(range(1, 17)),
              'maximum_reduction_blocks': 512,
              'noise_bound': str(bound), 'noise_bound_bits': bound.bit_length(),
              'decode_lhs': str(4*(1 << 59)*bound), 'Q': str(q),
              'deterministic_decode_inequality': True,
              'encrypted_input_coefficient_storage_max_bytes_per_party': 512*2*16384*6*8,
              'memory_bound_scope': 'Ciphertext coefficient arrays only; excludes allocator, keys, NTT tables, GC and process RSS',
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
              'full_model_R2': False}
    (ROOT/'records/tile108_layout_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['result', 'noise_bound', 'noise_bound_bits']}))


if __name__ == '__main__':
    main()
