"""Derive public operator counts for registered M1-M3 development shapes.

This is an implementation schedule, not a test standard, timing prediction or
complete wire-size model. Secret validity and numerical failure never enter it.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def ceildiv(n, d):
    return (n + d - 1) // d


def plan(model, rows, layers=28, decode=False, standalone_layer=False):
    hidden, ffn = {'embedding': (1024, 3072), 'generation': (2048, 6144)}[model]
    keys = 1024 + rows if decode else rows
    batches = ceildiv(rows, 16)
    attention = layers * 16 * batches
    rms_per_layer = 2 * batches + ceildiv(16 * rows, 16) + ceildiv(8 * rows, 16)
    rms = layers * rms_per_layer + (0 if standalone_layer else batches)
    return {
        'model': model, 'rows': rows, 'keys': keys, 'layers': layers, 'hidden': hidden,
        'Q_heads': 16, 'KV_heads': 8, 'head_dimension': 128, 'ffn': ffn,
        'token_lookups': 0 if standalone_layer else rows,
        'token_fixed_scan_each': 155648,
        'private_weight_projection_calls': 4 * batches * layers + int(decode),
        'projection_segments': {'QKV': [2048, 1024, 1024], 'gate_up': [ffn, ffn]},
        'RMS_batches': rms,
        'BFV_matrix_Beaver_triples': 2 * attention,
        'BFV_random_cross_term_matrix_calls': attention * (1 + ceildiv(keys, 16) + 1 + 8),
        'public_trace_completed_phase_counts': {
            'Transformer_layer': layers,
            'RoPE_shared_public_coefficients': layers * batches,
            'shared_QK': attention, 'shared_PV': attention,
            'SiLU': layers * rows * ceildiv(ffn, 4096),
            'coordinate_range': rms,
        },
        'L2': not standalone_layer and not decode,
        'LM_head_output_count': 151936 if decode else 0,
        'allowed_argmax_candidates': 151645 if decode else 0,
        'allowed_argmax_pair_comparisons': 151644 if decode else 0,
        'KV_storage_words_per_party': 2 * 28 * 8 * 1280 * 128 if decode else 0,
        'secret_validity_can_reduce_counts': False,
    }


def phase_counts(path):
    from collections import Counter
    result = Counter()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'end':
                result[row['phase']] += 1
    return result


def check_final_trace(prefix, expected, *, token_trace):
    """Inspect final, closed executions only; no inference from partial logs."""
    from tools.hybrid112_status import trace_status
    checked = []
    for party in ('C', 'S'):
        path = ROOT / (prefix + '.' + party + '.trace.jsonl')
        state = trace_status(path)
        if (not state.get('trace_available') or state['partial_last_line'] or
                state['public_active_phases'] or state['public_aborted_phases']):
            raise RuntimeError('PUBLIC_TRACE_NOT_COMPLETE: ' + prefix + ':' + party)
        actual = phase_counts(path)
        counts = dict(expected['public_trace_completed_phase_counts'])
        if token_trace:
            counts['Token_lookup'] = expected['token_lookups']
        if any(actual[k] != value for k, value in counts.items()):
            raise RuntimeError('PUBLIC_SCHEDULE_MISMATCH: ' + prefix + ':' + party)
        checked.append({'party': party, 'trace': str(path.relative_to(ROOT)),
            'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'counts': counts, 'token_lookup_trace_available': token_trace})
    return checked


def main():
    plans = {'M1_embedding': plan('embedding', 2, 1, standalone_layer=True),
             'M1_generation': plan('generation', 2, 1, standalone_layer=True),
             'M2_single_Decode': plan('generation', 1, decode=True),
             'M3_each_document': plan('embedding', 257),
             'M3_query': plan('embedding', 160)}
    # Compare already completed M1 public traces. Do not compare partial runs
    # against their final expected counts or classify them as failures.
    checked = []
    for model, prefix in [
        ('embedding', 'hybrid112_layer_embedding_2_0_1ff4142dbc84438db1e7382d6715cf32'),
        ('generation', 'hybrid112_layer_generation_2_0_2296cff92f6345f28be2dfde120bec00'),
    ]:
        record_path = ROOT / 'records' / (prefix + '.json')
        record = json.loads(record_path.read_text())
        if record['result'] != 'matched_CUDA_R1':
            raise RuntimeError('M1_NOT_COMPLETE')
        expected = plans['M1_' + model]['public_trace_completed_phase_counts']
        for party in ('C', 'S'):
            path = ROOT / 'records' / (prefix + '.' + party + '.trace.jsonl')
            actual = phase_counts(path)
            if any(actual[k] != v for k, v in expected.items()):
                raise RuntimeError('PUBLIC_SCHEDULE_MISMATCH: ' + model + ':' + party)
            checked.append({'model': model, 'party': party, 'trace': str(path.relative_to(ROOT)),
                'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'counts': expected})
    # This command can be rerun after the existing registered executions end.
    # Their own archived source identity and terminal evidence are checked by
    # build_hybrid112_evidence; public counts alone never establish a pass.
    from tools.build_hybrid112_evidence import M2, M3, verify_finished_record, verify_archive, read
    later_checked, pending = [], []
    for name, prefix, archive in [('M2', M2, 'records/hybrid112_M2_running_source_snapshot.tar.gz'),
                                  ('M3', M3, M3 + '.source.tar.gz')]:
        if not (ROOT / (prefix + '.json')).exists():
            pending.append(name)
            continue
        record = read(prefix + '.json')
        verify_archive(read(prefix + '.registration.json'), archive)
        if name == 'M2' and record.get('result') == 'matched_CUDA_R1' and record.get('numeric_failure') is False:
            verify_finished_record(record)
            later_checked.extend(check_final_trace(prefix, plans['M2_single_Decode'], token_trace=False))
        elif name == 'M3' and record.get('result') == 'M3_matched_CUDA_R1' and len(record.get('stages', [])) == 3:
            for stage in record['stages']:
                verify_finished_record(stage)
                expected = plans['M3_each_document' if stage['mode'] == 'prepare' else 'M3_query']
                later_checked.extend(check_final_trace(prefix + '.' + stage['session'], expected, token_trace=True))
        else:
            pending.append(name + ':not_successful')
    paths = ['compute/model.hpp', 'compute/numerics.hpp', 'native/src/batch_matrix.inc',
             'native/include/prag/joint_matrix.hpp', 'tools/hybrid112_schedule.py']
    result = {'kind': 'public_operator_schedule_not_complete_wire_plan', 'plans': plans,
              'completed_M1_traces_checked': checked,
              'completed_M2_M3_traces_checked': later_checked, 'pending': pending,
              'scope': 'reviewed loop structure and named finished public traces only; pending runs are not passes',
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
              'not_estimated': ['TLS bytes', 'whole-query time', 'whole-query bytes', 'formal acceptance'],
              'full_R2_complete': False}
    (ROOT / 'records/hybrid112_public_schedule.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed_M1_party_traces_checked': len(checked), 'plans': list(plans),
                      'full_R2_complete': False}))


if __name__ == '__main__':
    main()
