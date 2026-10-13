"""Public same-input R0 counterpart of the new R1 down_proj domain failure."""
import hashlib
import json
from pathlib import Path
import time
import weakref

import numpy as np
import torch

from private_rag.model_input import InputPreparer
from references.r0_model import FloatQwen
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RMSNorm
from references.resource_policy import require_model_device


def main():
    require_model_device('R0')
    torch.set_num_threads(4)
    prep = InputPreparer(Path('artifacts/models'), Path('artifacts/template/template_v1.json'))
    question = '隐私计算如何保护问题？'
    slots = prep.question(question).embedding
    model = FloatQwen(Path('artifacts/models/embedding/checkpoint'))
    down_count, captured = 0, None
    module_names, statistics = weakref.WeakKeyDictionary(), {}
    valid = np.asarray(slots.validity, dtype=bool)

    def measure(label, values, square_evidence=True):
        a=values.detach().cpu().numpy()[0].astype(np.float64)
        if a.shape[0]!=len(valid) or not np.isfinite(a).all():
            raise RuntimeError('R0_OBSERVATION_SHAPE_OR_FINITE')
        active=a[valid]
        q=np.floor(active*4096).astype(np.int64)
        rows=q.reshape(-1,q.shape[-1]) if square_evidence else []
        # Python integers prevent a diagnostic square-sum overflow from hiding evidence.
        squares=[sum(int(x)*int(x) for x in row) for row in rows]
        epsilon_a=[s//q.shape[-1]+16 for s in squares]
        exponents=[(x.bit_length()-1-24)//2 for x in epsilon_a]
        exceeding=np.abs(active)>256
        per_feature=np.sum(exceeding.reshape(-1,active.shape[-1]),axis=0)
        maxima=np.max(np.abs(active).reshape(-1,active.shape[-1]),axis=0)
        top=np.argsort(maxima)[-8:][::-1]
        statistics[label]={
            'shape':list(a.shape),'valid_physical_positions':int(valid.sum()),
            'valid_min_real':float(active.min()),'valid_max_real':float(active.max()),
            'max_abs_all_positions':float(np.max(np.abs(a))),
            'valid_scalar_count':int(active.size),'valid_scalars_abs_over_256':int(exceeding.sum()),
            'feature_count_with_abs_over_256':int(np.count_nonzero(per_feature)),
            'largest_features':[{'feature':int(i),'max_abs_real':float(maxima[i]),'count_over_256':int(per_feature[i])} for i in top],
            'floor_f12_valid_min_integer':int(q.min()),'floor_f12_valid_max_integer':int(q.max()),
        }
        if square_evidence:
            statistics[label].update(
                floor_f12_row_square_sum_f24_max=max(squares),
                floor_f12_row_square_sum_f24_min=min(squares),
                floor_f12_row_square_sum_all_below_2pow56=all(s<(1<<56) for s in squares),
                floor_f12_mean_square_plus_epsilon_f24_max=max(epsilon_a),
                inferred_RMS_exponent_k_min=min(exponents),inferred_RMS_exponent_k_max=max(exponents),
                inferred_RMS_exponents_within_locked_minus10_to8=min(exponents)>=-10 and max(exponents)<=8)

    def before(module,inputs):
        if isinstance(module,Qwen3DecoderLayer):
            layer=module.self_attn.layer_idx
            module_names[module]=f'layer{layer}.residual_after_mlp'
            for name,child in module.named_modules():
                if name: module_names[child]=f'layer{layer}.{name}'
            measure(f'layer{layer}.residual_input',inputs[0])
        elif isinstance(module,Qwen3RMSNorm):
            measure(module_names.get(module,'final_norm')+'.input',inputs[0])
        elif isinstance(module,torch.nn.Linear) and module.in_features==3072 and module.out_features==1024:
            measure(module_names[module]+'.input_joint',inputs[0],False)

    def observe(module, inputs, output):
        nonlocal down_count, captured
        # This dimension pair uniquely identifies the 28 Embedding down_proj modules.
        if isinstance(module, torch.nn.Linear) and module.in_features == 3072 and module.out_features == 1024:
            measure(module_names[module]+'.output',output)
            if down_count == 2:
                captured = output.detach().cpu().numpy()[0].copy()
            down_count += 1
        elif isinstance(module,Qwen3DecoderLayer):
            measure(module_names[module],output[0])
        elif isinstance(module,torch.nn.Linear):
            measure(module_names[module]+'.output',output,False)
        elif isinstance(module,Qwen3RMSNorm):
            name=module_names.get(module,'final_norm')
            measure(name+'.output',output,name=='final_norm')
        # Observation hook returns None and does not change computation.

    handle = torch.nn.modules.module.register_module_forward_hook(observe)
    pre_handle = torch.nn.modules.module.register_module_forward_pre_hook(before)
    start = time.monotonic()
    try:
        z = model.embedding(slots.tokens, slots.validity)
    finally:
        handle.remove()
        pre_handle.remove()
    elapsed = time.monotonic() - start
    if down_count != 28 or captured is None:
        raise RuntimeError('R0_DOWN_PROJ_OBSERVATION_INCOMPLETE')
    r1 = np.load('records/models_swiglu_revision_question_first_failure.npy', allow_pickle=False)
    first = json.loads(Path('records/models_swiglu_revision_question_run.json').read_text())['first_failure']
    if first['layer'] != 2 or not first['weight_path'].endswith('model.layers.2.mlp.down_proj.weight.npy'):
        raise RuntimeError('R1_FAILURE_CHANGED_RECHECK_OPERATOR')
    index = tuple(first['max_abs_index'])
    profile_path = Path('artifacts/constants/numeric_profile_v1_qk.json')
    profile = json.loads(profile_path.read_text())
    result = {
        'scope': 'public same-input same-operator independent R0/R1 diagnosis; no further domain expansion',
        'numeric_profile_version': profile['profile_version'],
        'numeric_profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        'public_question': question, 'operator': 'layer2 mlp.down_proj output before residual',
        'physical_slots': 160, 'R0_complete_layers': down_count, 'elapsed_seconds': elapsed,
        'shape': list(r1.shape), 'same_max_abs_index': list(index),
        'same_index_is_valid': bool(valid[index[0]]),
        'r1_integer': int(r1[index]), 'r1_real': float(r1[index]) / 4096,
        'r0_real': float(captured[index]), 'current_output_domain_real_abs': 256,
        'r1_excess_over_256': abs(float(r1[index])) / 4096 - 256,
        'r0_excess_over_256': abs(float(captured[index])) - 256,
        'r1_valid_min_real': float(r1[valid].min()) / 4096,
        'r1_valid_max_real': float(r1[valid].max()) / 4096,
        'r0_valid_min_real': float(captured[valid].min()),
        'r0_valid_max_real': float(captured[valid].max()),
        'max_absolute_R0_R1_difference': float(np.max(np.abs(captured - r1 / 4096))),
        'r0_output_norm': float(torch.linalg.vector_norm(z)),
        'R2_executed': False, 'further_numeric_domains_modified': False,
        'R0_floor_quantization_statistics_are_not_R1_states':True,
        'activation_statistics_path':'records/models_swiglu_revision_r0_activation_statistics.json',
    }
    np.save('records/models_swiglu_revision_r0_down_proj.npy', captured, allow_pickle=False)
    Path('records/models_swiglu_revision_down_proj_conflict.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    Path('records/models_swiglu_revision_r0_activation_statistics.json').write_text(
        json.dumps({'scope':'one public R0 full 28-layer forward observation; empirical, not a general activation bound',
                    'public_question':question,'numeric_profile_version':profile['profile_version'],
                    'quantization_for_statistics':'floor(float32 observed value * 4096); these are not actual R1 propagated states',
                    'RMS_rows':'Last feature dimension; for Q/K norm each head is one row',
                    'attention_residual_output':'layerN.post_attention_layernorm.input',
                    'statistics':statistics},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
