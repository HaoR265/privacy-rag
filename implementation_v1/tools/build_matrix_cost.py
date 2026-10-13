"""Public v1.0.7 matrix schedule: static BFV communication subset, not time."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = 2*16384*6*8+96
RETURN = 16384*6*8+96
TARGET = (32+96)+(16+96)+34*(65536+96)+(16*361+96)+(59+96)
DESCRIPTOR = 2*(80+96)
# Locked SCI IKNP block_size=16384 bits: 128 extension rows per target,
# then one 2*361-label response, independent of secret choices or phases.
GC_INPUT_OT = 128*(16384//8+96)+(2*361*16+96)


def add(a, b, times=1):
    for key in a:
        a[key] += b[key]*times


def empty():
    return dict(inputs=0, ciphertexts=0, targets=0, matrix_calls=0)


def matrix(batch, inner, output):
    cap = 512//batch
    return dict(inputs=(inner+15)//16, ciphertexts=(output+cap-1)//cap,
                targets=batch*output, matrix_calls=1)


def sequence(length, hidden, ffn):
    result = empty()
    # Each token table lookup remains the fixed 19-block streamed protocol.
    add(result, dict(inputs=19, ciphertexts=hidden, targets=hidden, matrix_calls=0), length)
    # Gamma maps are diagonal private matrices, one row per RMS invocation.
    gamma = matrix(1, hidden, hidden)
    add(result, gamma, length*(2*28+1))
    add(result, matrix(1, 128, 128), length*28*(16+8))
    maps = [(hidden, 2048), (hidden, 1024), (hidden, 1024), (2048, hidden),
            (hidden, ffn), (hidden, ffn), (ffn, hidden)]
    for begin in range(0, length, 16):
        batch = min(16, length-begin)
        for inner, output in maps:
            add(result, matrix(batch, inner, output), 28)
    expected = length*(2*hidden+28*(4*hidden+7168+2*ffn))
    assert result['targets'] == expected
    return result


def main():
    counts = {'prepare_embedding_257': sequence(257, 1024, 3072),
              'query_embedding_160': sequence(160, 1024, 3072),
              'generation_prefill_1024': sequence(1024, 2048, 6144),
              'generation_decode_255': empty(), 'generation_logits_256': empty()}
    add(counts['generation_decode_255'], sequence(1, 2048, 6144), 255)
    add(counts['generation_logits_256'], matrix(1, 2048, 151936), 256)
    counts['one_decode_token_including_logits'] = sequence(1, 2048, 6144)
    add(counts['one_decode_token_including_logits'], matrix(1, 2048, 151936))
    counts['generation_total'] = empty()
    for key in ['generation_prefill_1024', 'generation_decode_255', 'generation_logits_256']:
        add(counts['generation_total'], counts[key])
    counts['nonempty_query_total'] = dict(counts['generation_total'])
    add(counts['nonempty_query_total'], counts['query_embedding_160'])
    lower = {key: c['inputs']*INPUT+c['ciphertexts']*RETURN+c['targets']*TARGET+c['matrix_calls']*DESCRIPTOR
             for key, c in counts.items()}
    old = json.loads((ROOT/'records/crypto_static_cost_v1_0_6.json').read_text())
    assert counts['nonempty_query_total']['targets'] == old['BFV_target_decode_counts']['nonempty_query_total']
    sources = ['compute/model.hpp', 'compute/numerics.hpp', 'native/src/bfv_matrix.inc',
               'native/src/bfv.cpp', 'third_party/EzPC/SCI/src/OT/iknp.h',
               'tools/build_matrix_cost.py']
    report = {'kind': 'static_public_schedule_BFV_communication_subset_not_benchmark',
              'native_version': '1.0.7', 'counts': counts,
              'application_bytes_lower': lower,
              'BFV_bytes_including_GC_input_OT_excluding_setup_and_other_operators': {
                  key: lower[key]+c['targets']*GC_INPUT_OT for key,c in counts.items()},
              'per_unit_bytes': dict(input_ciphertext=INPUT, returned_c1=RETURN,
                                     target_excluding_input_OT=TARGET, matrix_descriptor=DESCRIPTOR,
                                     GC_input_OT_per_target=GC_INPUT_OT),
              'previous_v106_query_lower_bytes': old['bfv_only_communication_lower_bytes']['nonempty_query_total'],
              'excludes': ['GC input OT', 'base OT, keys and setup', 'all shared arithmetic, boolean, range checks and lookups',
                           'TLS, TCP, IP, management and heartbeat', 'wall time and throughput'],
              'matrix_batching': 'At most 16 public input rows; all output tiles per matrix reuse ceil(K/16) ciphertexts',
              'gamma': 'One diagonal matrix per RMS invocation; no cross-row gamma batching',
              'token_lookup': 'Unchanged 19-block full scan, one returned ciphertext per hidden coordinate',
              'target_GC_count_reduced': False,
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              'full_model_R2': False}
    (ROOT/'records/crypto_static_cost_v1_0_7.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'query_lower_bytes': lower['nonempty_query_total'],
                      'one_decode_lower_bytes': lower['one_decode_token_including_logits'],
                      'query_counts': counts['nonempty_query_total']}))


if __name__ == '__main__':
    main()
