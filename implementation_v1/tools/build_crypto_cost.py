#!/usr/bin/env python3
"""Static public-schedule communication subset, never a performance benchmark."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = 96
POLYNOMIAL = 16384
MODULI = 6
PHASE_BITS = 361
NUMERATOR_BITS = 420
# Locked BFV-only decode: N=v*2^59+floor(q/2)<q*2^60. Widths remain
# 361/420/361; the public-zero high quotient prefix needs no trial subtract.
DIVISION_ROUNDS = 60
AND_GATES = 5 * PHASE_BITS + 2 * NUMERATOR_BITS + 3 * PHASE_BITS * DIVISION_ROUNDS
CHUNK = 65536
GC_CHUNKS = (32 * AND_GATES + CHUNK - 1) // CHUNK
INPUT_BYTES = 2 * POLYNOMIAL * MODULI * 8 + HEADER
C1_BYTES = POLYNOMIAL * MODULI * 8 + HEADER
PARTS = {
    'fresh_GC_initialization_frames': (32 + HEADER) + (16 + HEADER),
    'fixed_padded_garbled_table_frames': GC_CHUNKS * (CHUNK + HEADER),
    'server_input_labels_frame': 16 * PHASE_BITS + HEADER,
    'C_output_permutation_frame': 59 + HEADER,
}
TARGET_BYTES = sum(PARTS.values())

def capacity(columns):
    d = min(8192, columns)
    assert d > 0
    return min(16384 // (2*d-1), 8192 // d)

def projection(rows, columns):
    # Public 32-row tiles remain separate private_linear calls.
    blocks = (columns + 8191) // 8192
    cap = capacity(columns)
    full, tail = divmod(rows, 32)
    ciphertext_groups = full * ((32 + cap - 1) // cap) + (tail + cap - 1) // cap
    return rows, ((rows+31)//32)*blocks, ciphertext_groups

def per_token(hidden, ffn):
    # Each scalar gamma call remains one ciphertext despite d=1 capacity.
    # Token selection has one streamed call: nineteen inputs, one return per row.
    scalar_gamma = hidden + 28 * (2*hidden + 3072)
    targets = ciphertexts = hidden + scalar_gamma
    inputs = 19 + scalar_gamma
    matrices = [(2048, hidden), (1024, hidden), (1024, hidden),
                (hidden, 2048), (ffn, hidden), (ffn, hidden), (hidden, ffn)]
    for rows, columns in matrices:
        target, encrypted_input, group = projection(rows, columns)
        targets += 28*target
        inputs += 28*encrypted_input
        ciphertexts += 28*group
    assert targets == 2*hidden + 28*(4*hidden + 7168 + 2*ffn)
    assert inputs == 19 + hidden + 28*(2*hidden + 3200 + hidden//16 + ffn//16)
    return targets, inputs, ciphertexts

E_RETURN, E_INPUT, E_GROUP = per_token(1024, 3072)
G_RETURN, G_INPUT, G_GROUP = per_token(2048, 6144)
returns = {
    'prepare_embedding_257': E_RETURN * 257,
    'query_embedding_160': E_RETURN * 160,
    'generation_1279_forward_tokens': G_RETURN * 1279,
    'generation_256_logits': 256 * 151936,
}
inputs = {
    'prepare_embedding_257': E_INPUT * 257,
    'query_embedding_160': E_INPUT * 160,
    'generation_1279_forward_tokens': G_INPUT * 1279,
    'generation_256_logits': 256 * ((151936 + 31) // 32),
}
groups = {
    'prepare_embedding_257': E_GROUP * 257,
    'query_embedding_160': E_GROUP * 160,
    'generation_1279_forward_tokens': G_GROUP * 1279,
    'generation_256_logits': 256 * projection(151936, 2048)[2],
}
for counts in (returns, inputs, groups):
    counts['generation_total'] = counts['generation_1279_forward_tokens'] + counts['generation_256_logits']
    counts['nonempty_query_total'] = counts['query_embedding_160'] + counts['generation_total']
cost = {name: count * TARGET_BYTES + groups[name] * C1_BYTES + inputs[name] * INPUT_BYTES for name, count in returns.items()}
paths = ['native/src/bfv.cpp', 'native/src/gc_batch_channel.cpp', 'native/src/secure_ops.cpp',
         'native/include/prag/gc_batch_channel.hpp', 'native/src/channel.cpp',
         'third_party/EzPC/SCI/src/GC/halfgate_gen.h', 'compute/model.hpp',
         'compute/numerics.hpp', 'compute/pipeline.hpp', 'tools/build_crypto_cost.py',
         'records/crypto_decode60_implementation.md', 'records/crypto_packed105_implementation.md',
         'records/crypto_merged106_implementation.md']
def ot_frame_bytes(count, bits, storage_bytes):
    # SCI SplitIKNP: no precomputation; adapters limit each invocation to1024.
    assert 0 < count <= 1024
    padded = ((count + 255) // 256) * 256
    extension = 128 * (HEADER + padded // 8)
    post = storage_bytes * ((2 * count * bits + 8 * storage_bytes - 1) // (8 * storage_bytes))
    return extension + HEADER + post

def lookup_cost(length):
    width = (length - 1).bit_length()
    high = 59 - width
    boolean_counts = [1] * 236  # One 59-bit A2B,118 ANDs,2 directions each.
    while high > 1:
        for first in range(0, high // 2, 1024):
            count = min(1024, high // 2 - first)
            boolean_counts.extend([count, count])
        high = (high + 1) // 2
    for depth in range(width):
        for first in range(0, 1 << depth, 1024):
            count = min(1024, (1 << depth) - first)
            boolean_counts.extend([count, count])
    arithmetic_counts = [min(1024, length - first) for first in range(0, length, 1024)]
    old = length * (590 * ot_frame_bytes(1, 1, 1) + ot_frame_bytes(1, 59, 8))
    new = (sum(ot_frame_bytes(n, 1, 1) for n in boolean_counts)
           + sum(ot_frame_bytes(n, 59, 8) for n in arithmetic_counts))
    return {'length': length, 'old_repeated_A2B_calls': 2 * length, 'new_A2B_calls': 1,
            'old_boolean_AND_gates': 295 * length,
            'new_boolean_AND_gates': sum(boolean_counts) // 2,
            'old_OT_invocations': 591 * length,
            'new_OT_invocations': len(boolean_counts) + len(arithmetic_counts),
            'old_application_frame_bytes_excluding_base_OT': old,
            'new_application_frame_bytes_excluding_base_OT': new,
            'one_bit_message_OT_count': sum(boolean_counts),
            '59_bit_message_OT_count': sum(arithmetic_counts)}

report = {
    'kind': 'static_public_shape_cost_not_runtime_benchmark',
    'profile': 'prag-native-v1.0.6',
    'division_rounds': DIVISION_ROUNDS,
    'integer_widths': {'phase': PHASE_BITS, 'numerator': NUMERATOR_BITS, 'remainder': PHASE_BITS},
    'AND_gates_per_target': AND_GATES,
    'fixed_64KiB_GC_chunks_per_target': GC_CHUNKS,
    'actual_table_bytes': AND_GATES * 32,
    'public_padding_bytes': GC_CHUNKS * CHUNK - AND_GATES * 32,
    'target_frame_parts_lower_bytes_excluding_ciphertext_and_input_OT': PARTS,
    'per_target_lower_bytes_excluding_ciphertext_and_input_OT': TARGET_BYTES,
    'per_ciphertext_c1_frame_bytes': C1_BYTES,
    'per_input_ciphertext_frame_bytes': INPUT_BYTES,
    'BFV_target_decode_counts': returns,
    'BFV_returned_ciphertext_group_counts': groups,
    'BFV_input_ciphertext_counts': inputs,
    'bfv_only_communication_lower_bytes': cost,
    'per_token': {'embedding': {'returns': E_RETURN, 'input_ciphertexts': E_INPUT, 'returned_ciphertext_groups': E_GROUP},
                  'generation': {'returns': G_RETURN, 'input_ciphertexts': G_INPUT, 'returned_ciphertext_groups': G_GROUP}},
    'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    'lookup_public_shapes': [lookup_cost(n) for n in (1, 3, 256, 1280)],
    'lookup_cost_scope': 'Exact application Channel frames under locked SplitIKNP, excluding one-time base OT, TLS and local fsync. Public table/index schedule only; not elapsed-time or full-system estimate.',
    'excludes': ['GC input OT', 'base OT and setup/public key',
                 'all remaining shared/boolean/lookup/range operations',
                 'TLS TCP IP heartbeat and management', 'wall time and throughput'],
    'assumes': ['N>0 query and successful public fixed schedule',
                'current 32-output project tiles and scalar private gamma',
                'no early exit for private EOS or numeric failure',
                'K independently encrypted input blocks reused across output groups within one streamed call',
                'independent full polynomial mask and fresh Enc(0) for every returned ciphertext',
                'independent fresh GC object for every target, continuing one-time OT stream',
                'at most 19 same-level products merged before final protection; long tail remains 8192-wide'],
}
(ROOT / 'records/crypto_static_cost_v1_0_6.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'bfv_only_query_lower_bytes': cost['nonempty_query_total'],
                  'bfv_only_prepare_lower_bytes': cost['prepare_embedding_257'],
                  'target_decode_count': returns['nonempty_query_total'],
                  'returned_ciphertext_groups': groups['nonempty_query_total']}))
