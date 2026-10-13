"""Read-only arithmetic candidate from existing upstream domains and all weights.

No R1/R2 domain is changed by this script. It is evidence for a design discussion.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from model_tools.fetch_models import sha256


def derive():
    upstream = 256 << 12
    product_bound = upstream * upstream
    joint_bound = product_bound >> 12
    budget = 1 << 56
    models = []
    for kind in ('embedding', 'generation'):
        root = Path('artifacts/models') / kind / 'quantized_f12'
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        entries = []
        for tensor in manifest['tensors']:
            if not tensor['name'].endswith('mlp.down_proj.weight'):
                continue
            path = root / tensor['path']
            digest = sha256(path)
            if digest != tensor['sha256']:
                raise ValueError('DOWN_PROJ_FILE_HASH_MISMATCH')
            weight = np.load(path, mmap_mode='r', allow_pickle=False)
            maximum_l1 = 0
            maximum_row = -1
            maximum_abs = 0
            for first in range(0, weight.shape[0], 128):
                absolute = np.abs(weight[first:first+128].astype(np.int64))
                l1 = np.sum(absolute, axis=1, dtype=np.int64)
                offset = int(np.argmax(l1))
                if int(l1[offset]) > maximum_l1:
                    maximum_l1 = int(l1[offset]);maximum_row = first + offset
                maximum_abs = max(maximum_abs, int(absolute.max()))
            if maximum_l1 != tensor['maximum_row_l1_integer']:
                raise ValueError('DOWN_PROJ_ROW_L1_MANIFEST_MISMATCH')
            dot_bound = maximum_l1 * joint_bound
            simple_bound = weight.shape[1] * maximum_abs * joint_bound
            if max(product_bound, dot_bound, simple_bound) >= budget:
                raise ArithmeticError('CANDIDATE_DOES_NOT_FIT_BUDGET')
            entries.append({'tensor': tensor['name'], 'sha256': digest, 'shape': list(weight.shape),
                            'maximum_absolute_weight_f12_integer': maximum_abs,
                            'maximum_row_l1_f12_integer': maximum_l1, 'maximum_l1_row': maximum_row,
                            'full_dot_and_any_partial_sum_abs_f24_bound': dot_bound,
                            'alternative_bound_using_dimension_and_max_scalar': simple_bound,
                            'post_floor12_abs_f12_bound': (dot_bound + 4095) // 4096,
                            'all_bounds_strictly_less_than_2pow56': True})
        if len(entries) != 28:
            raise ValueError('DOWN_PROJ_TENSOR_COUNT')
        maximum = max(entries, key=lambda x: x['full_dot_and_any_partial_sum_abs_f24_bound'])
        models.append({'model': kind, 'model_id': manifest['model'], 'revision': manifest['revision'],
                       'quantized_manifest_sha256': sha256(manifest_path), 'layers': entries,
                       'maximum_row_l1_f12_integer': maximum['maximum_row_l1_f12_integer'],
                       'maximum_raw_dot_abs_f24_bound': maximum['full_dot_and_any_partial_sum_abs_f24_bound'],
                       'maximum_bound_tensor': maximum['tensor'],
                       'minimum_budget_headroom_factor': budget / maximum['full_dot_and_any_partial_sum_abs_f24_bound']})
    native = Path('native/src/bfv.cpp')
    native_text = native.read_text()
    if 'begin += 8192' not in native_text:
        raise ValueError('RECHECK_ACTUAL_NATIVE_BLOCK_SCHEDULE')
    report = {'status': 'PROPOSED_FOR_DISCUSSION_NOT_APPLIED',
              'scope': 'Only SwiGLU joint and the input domain of its immediately following down_proj; down_proj output, residual and other main values remain abs<=256',
              'basis': 'Existing checked upstream domain, not observed activation maximum',
              'upstream_silu_and_up_abs_f12': upstream,
              'raw_elementwise_product_abs_f24_bound': product_bound,
              'proposed_joint_and_down_proj_input_abs_f12': joint_bound,
              'proposed_joint_real_abs_bound': 65536,
              'engineering_budget_exclusive': budget, 'ring_bits_unchanged': 59,
              'scale_and_floor_order_unchanged': True,
              'bound_semantics': 'Bounds apply to signed reconstructed mathematical quantities, not individual additive shares; each local share remains an unrestricted ring59 residue and local modular wrap is intentional.',
              'derivation': 'Both already-checked SiLU(gate) and up are in [-2^20,2^20]. Their product is in [-2^40,2^40]; floor(product/2^12) stays in [-2^28,2^28]. For every down_proj output row r and any subset/prefix J, abs(sum(j in J) X_j W_rj)<=2^28 sum(j in J)|W_rj|<=2^28 L1_r.',
              'models': models,
              'schedule': {'native_source_sha256': sha256(native), 'column_block_capacity': 8192,
                           'embedding_columns': 3072, 'generation_columns': 6144,
                           'blocks_per_down_proj_row': 1, 'padding': 'public zero to 8192',
                           'output_row_tile': 32,
                           'partial_accumulation': 'Any column sub-block and any accumulated union is bounded by full-row L1. Sum all raw f24 block contributions before the one floor12; never truncate each block.',
                           'new_schedule_required': False},
              'crypto_scope': {'bfv_bound_artifact_sha256': sha256(Path('artifacts/bfv_correctness_bound.json')),
                               'observation': 'The current BFV interface takes full ring59 shares and the existing conditional noise artifact bounds plaintext weight coefficients up to full centered ring59. This proposed activation-domain change does not enlarge those already-stated share/weight supports or alter the native polynomial shape.',
                               'limitation': 'Arithmetic candidate only; no new BFV correctness or overall security certification is claimed.'},
              'remaining_conditions': ['Requires explicit authorization for this further locked-domain change',
                                       'Bind all derived per-layer bounds to exact quantized weight hashes and configuration',
                                       'Only down_proj may accept this larger input domain; general linear inputs remain unchanged',
                                       'Keep down_proj output/residual guards at 256 and continue actual full-model checks',
                                       'Successful normal outputs for later layers and the generation model have not been established']}
    target = Path('records/models_swiglu_domain_candidate.json')
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'joint_abs_f12': joint_bound,
                      'models': [{k: m[k] for k in ('model', 'maximum_row_l1_f12_integer', 'maximum_raw_dot_abs_f24_bound', 'minimum_budget_headroom_factor')} for m in models]}))


if __name__ == '__main__':
    derive()
