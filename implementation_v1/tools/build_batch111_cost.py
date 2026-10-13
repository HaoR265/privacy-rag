"""Separate measured kernel costs from unchanged historical static scopes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    stage = json.loads((ROOT / 'records/batch111_stage.json').read_text())
    prior = json.loads((ROOT / 'records/crypto_static_cost_v1_0_9.json').read_text())
    old = prior['counts']['nonempty_query_total']
    record = {'kind': 'measured_kernel_breakdown_not_full_query_estimate', 'native_version': '1.0.11',
        'full_query_bytes': None, 'full_query_seconds': None, 'full_model_R2_executed': False,
        'source_stage_sha256': hashlib.sha256((ROOT / 'records/batch111_stage.json').read_bytes()).hexdigest(),
        'historical_v109_static_scope': {
            'record': 'records/crypto_static_cost_v1_0_9.json', 'runtime_version': '1.0.9',
            'weight_linear_matrix_bytes': old['matrix_bytes'],
            'private_Token_BFV_GC_bytes': old['token_bfv_gc_bytes'],
            'private_Token_onehot_bytes': old['token_onehot_bytes'],
            'old_gamma_bytes_not_added_to_current_RMS': old['diagonal_bytes'],
            'boundary': 'Historical fixed-shape formulas; not a run of v111, not full current model cost'},
        'measured_current_RMS': [], 'measured_current_attention': [],
        'unmeasured_current_scopes': ['remaining elementwise SiLU/SwiGLU', 'RoPE', 'L2', 'private retrieval/Top2/I7',
            'residual guards outside RMS/attention', 'argmax and output selection', 'full model initialization/session scheduling',
            'TLS/TCP/IP transport bytes'],
        'units': 'Application-layer bytes count both directions once on C; excludes final diagnostic reconstruction',
        'do_not_add': 'RMS already includes gamma; phase subtrees are inclusive; historical scopes and current kernels do not form a complete query total',
        'runtime_boundary': 'Observed local timings include real CPU crypto, IO and profiling; CUDA integer oracle is separate; no private GPU or full model execution'}
    for key, target, parent in [('rms_runs', 'measured_current_RMS', 'full_RMS'),
                                 ('attention_runs', 'measured_current_attention', 'complete_attention_head')]:
        for run in stage[key]:
            record[target].append({'record': run['record'], 'shape': {k: run[k] for k in
                ('dimension', 'batch', 'rows', 'keys', 'case') if k in run},
                'application_bytes': run['application_bytes'], 'seconds': run.get('seconds', run.get('elapsed_seconds')),
                'stages_nonoverlapping': {p: v for p, v in run['phases_inclusive'].items()
                                        if p.startswith(parent + '/') and p.count('/') == 1},
                'full_phase_including_material_counts': run['phases_inclusive'][parent]})
    (ROOT / 'records/batch111_cost_scope.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
