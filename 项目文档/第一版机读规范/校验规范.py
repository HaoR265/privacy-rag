#!/usr/bin/env python3
"""只读检查本规范包内部一致性；不运行模型、密码协议或系统测试。

依赖 Python 3 和 PyYAML。本脚本不访问父目录、不联网、不安装依赖、不写文件。
"""
from pathlib import Path
import json
import hashlib
import math
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit('缺少 PyYAML；本脚本不自动安装依赖。')

ROOT = Path(__file__).resolve().parent
errors = []


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'YAML 重复键：{key!r}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def require(condition, message):
    if not condition:
        errors.append(message)


def package_file(relative, description):
    target = ROOT / relative
    if not target.resolve().is_relative_to(ROOT):
        errors.append(f'{description} 指向规范包外：{relative}')
        return None
    if not target.is_file():
        errors.append(f'{description} 不存在：{relative}')
        return None
    return target


def unique_ids(items, label):
    ids = [item['id'] for item in items]
    require(len(ids) == len(set(ids)), f'{label}编号重复')
    return set(ids)


def main():
    loaded = {}
    documents = {}
    # 不跟随包外链接读取任何内容。
    for path in sorted(ROOT.rglob('*')):
        if path.suffix not in {'.md', '.yaml'} or not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        safe = package_file(relative, '规范文件')
        if safe is None:
            continue
        content = safe.read_text(encoding='utf-8')
        documents[relative] = content
        if path.suffix == '.yaml':
            try:
                loaded[relative] = yaml.load(content, Loader=UniqueLoader)
            except (yaml.YAMLError, ValueError) as exc:
                errors.append(f'{relative}: {exc}')

    required = ['规范索引.yaml', '技术规则.yaml', '参数与接口.yaml', '代理规则.yaml', '决策与工件.yaml']
    for name in required:
        require(name in loaded, f'缺少可解析文件：{name}')
    if not all(name in loaded for name in required):
        return {'check': 'standalone_specification_consistency', 'errors': errors}

    index = loaded['规范索引.yaml']
    for relative in index['entrypoints']:
        package_file(relative, '入口文件')
    for key in ['rule_catalog', 'parameter_catalog', 'agent_policy', 'decision_registry', 'protocol_contract', 'startup_prompt']:
        package_file(index[key], f'索引字段 {key}')
    require(index['distribution']['unit'] == 'this_directory', '交付范围不是当前文件夹')
    require(index['distribution']['external_design_documents_required'] is False, '存在外部设计文档依赖')
    require(index['distribution']['prior_conversation_required'] is False, '存在历史对话依赖')

    rules = loaded['技术规则.yaml']['rules']
    require(len(rules) == 62, '技术规则应保持62条')
    rule_ids = unique_ids(rules, '技术规则')
    architecture = documents.get('02_架构与技术约束.md', '')
    headings = re.findall(r'^### ([A-Z]+-\d+) ', architecture, re.M)
    require(headings == [r['id'] for r in rules], '架构Markdown与YAML规则编号或顺序不一致')
    for rule in rules:
        require(rule['requirement'] in architecture, f"{rule['id']} 的Markdown/YAML正文不一致")

    registry = loaded['决策与工件.yaml']
    decisions = registry['decisions']
    decision_ids = unique_ids(decisions, '决策')
    artifact_ids = unique_ids(registry['artifacts'], '工件')
    for artifact in registry['artifacts']:
        relative, expected = artifact.get('path'), artifact.get('sha256')
        if relative is None:
            require(expected is None, f"{artifact['id']} 有哈希但没有工件路径")
            continue
        actual = package_file(relative, f"{artifact['id']} 工件")
        if actual is not None:
            require(hashlib.sha256(actual.read_bytes()).hexdigest() == expected,
                    f"{artifact['id']} 包内工件哈希不一致")
            if actual.suffix == '.json':
                try:
                    json.loads(actual.read_text(encoding='utf-8'))
                except (ValueError, UnicodeError):
                    errors.append(f"{artifact['id']} 包内JSON无法解析")
    normalization_ids = unique_ids(registry['implementation_normalizations'], '实现约定')
    for decision in decisions:
        if decision['status'] in {'pending_user', 'proposed_for_discussion'}:
            require(decision.get('selected') is None, f"{decision['id']} 未批准却已有生效选择")
    policy = loaded['代理规则.yaml']
    unique_ids(policy['coding_rules'], '编码规则')
    gpu = policy.get('gpu_preference', {})
    require(gpu.get('device') == 'RTX2060'
            and gpu.get('cpu_model_inference_allowed') is False
            and gpu.get('cpu_model_fallback_allowed') is False,
            '当前模型资源约束须明确RTX2060且禁止CPU模型推理与回退')
    for rule in policy['coding_rules']:
        require(rule['requirement'] in documents.get('03_行为准则与编码规范.md', ''),
                f"{rule['id']} 的Markdown/YAML正文不一致")
    for field in ['authorization_decision', 'scope_decision', 'standalone_decision', 'implementation_detail_decision']:
        require(policy[field] in decision_ids, f'代理规则中的决策引用无效：{field}')

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'rule_ids':
                    require(isinstance(item, list) and set(item) <= rule_ids, f'{path} 规则引用无效')
                if key == 'normalization_id':
                    require(item in normalization_ids, f'{path} 实现约定引用无效')
                if key == 'decision' and isinstance(item, str) and item.startswith('DEC-'):
                    require(item in decision_ids, f'{path} 决策引用无效')
                if key in {'source_file', 'source_sha256', 'source_line_count', 'source_sections', 'line_start', 'line_end'}:
                    errors.append(f'{path}.{key} 残留外部文档定位字段')
                walk(item, f'{path}.{key}')
        elif isinstance(value, list):
            for position, item in enumerate(value):
                walk(item, f'{path}[{position}]')
    for name, value in loaded.items():
        walk(value, name)

    parameters = loaded['参数与接口.yaml']
    bfv = parameters['cryptographic_backends']['BFV']
    layout = bfv['controlled_decode']['actual_integer_circuit_layout']
    bfv_artifact = next(a for a in registry['artifacts'] if a['id'] == 'ART-002')
    if bfv_artifact.get('path'):
        bundle_path = package_file(bfv_artifact['path'], 'BFV工件包')
        if bundle_path is not None:
            try:
                entries = json.loads(bundle_path.read_text(encoding='utf-8'))['contents']
                bounds = [e['content'] for e in entries if e.get('name') == 'bfv_correctness_bound.json']
                require(len(bounds) == 1, 'BFV工件包缺少唯一的当前正确性依据')
                if len(bounds) == 1:
                    circuit = bounds[0]['exact_decode_circuit']
                    require(circuit['profile'] == bfv['controlled_decode']['profile'] == 'bounded_phase_exact_tile_carry_v2',
                            'BFV整块解码配置不一致')
                    for declared, actual in (
                        ('local_phase_bits', 'local_component_bits'),
                        ('local_scaled_product_bits', 'local_scaled_product_bits'),
                        ('local_quarter_product_bits', 'local_quarter_product_bits'),
                        ('carry_input_bits_per_party_per_target', 'C_input_bits_per_coordinate'),
                        ('output_bits_per_target', 'output_bits_per_coordinate'),
                        ('restoring_division_rounds', 'division_rounds'),
                        ('carry_AND_gates_per_target', 'carry_AND_gates'),
                        ('mask_subtraction_AND_gates_per_target', 'mask_subtraction_AND_gates'),
                        ('GC_instances_per_tile', 'GC_instances_per_tile'),
                        ('evaluator_input_OT_batches_per_tile', 'GC_OT_extension_batches_per_tile'),
                    ):
                        require(layout[declared] == circuit[actual], f'BFV定义与实际整块工件不一致：{declared}')
                    require(circuit['division_rounds'] == 0 and circuit['AND_gates_per_coordinate'] == 122
                            and circuit['carry_AND_gates'] + circuit['mask_subtraction_AND_gates'] == 122
                            and layout['half_gate_AND_count_expression'] == '122*T'
                            and circuit['C_input_bits_per_coordinate'] == 2
                            and circuit['S_input_bits_per_coordinate'] == 61
                            and circuit['GC_instances_per_tile'] == circuit['GC_OT_extension_batches_per_tile'] == 1,
                            '整块GC/OT门数或输入布局不一致')
                    packing = bfv['packing']; actual_packing = bounds[0]['long_input_layout']
                    for declared, actual in (
                        ('maximum_input_blocks_per_call', 'maximum_input_blocks'),
                        ('maximum_input_columns_per_call', 'maximum_input_columns'),
                        ('controlled_GC_instances_expression', 'GC_instances'),
                    ):
                        require(packing[declared] == actual_packing[actual], f'长输入整块定义与工件不一致：{declared}')
                    require(actual_packing['block_dimension'] == 8192 and actual_packing['target_coefficient'] == 8191
                            and actual_packing['maximum_input_blocks'] == 19 and actual_packing['maximum_input_columns'] == 155648
                            and packing['decode_tile_coordinates_max'] == circuit['tile_coordinates_max'] == 512
                            and actual_packing['short_input_routes_to_matrix'] is True,
                            '长输入或短向量路由不一致')
                    require(packing['descriptor']['fields'] == ['PRAGLIN2',2,1,'K','O','f_X','f_W',8192,512,19],
                            '长输入v2公开绑定字段不一致')
                    matrix = bfv['matrix_packing']; actual_matrix = bounds[0]['matrix_coefficient_layout']
                    for key in ('batch_rows_max','input_columns_max','output_columns_max','reduction_tile','stride',
                                'tile_output_coordinates_max','output_tile_columns_expression','target_coefficient',
                                'maximum_message_degree','maximum_weight_slots_per_polynomial','maximum_input_blocks_per_call',
                                'noise_bound_expression','input_ciphertext_coefficient_storage_max_bytes_per_party'):
                        require(matrix[key] == actual_matrix[key], f'矩阵布局定义与工件不一致：{key}')
                    require(matrix['reduction_tile'] == 'public_power_of_two_16_to_floor8192divB' and matrix['stride'] == '2*d-1' and matrix['batch_rows_max'] == 16
                            and matrix['maximum_input_blocks_per_call'] == 512
                            and matrix['controlled_GC_instances_expression'] == actual_matrix['GC_instances'],
                            '矩阵布局超出已形成依据')
                    require(matrix['descriptor']['fields'] == ['PRAGMAT1',3,'B','K','O','f_X','f_W','d','c',512],
                            '矩阵v3公开绑定字段不一致')
                    require(actual_matrix['per_ciphertext_full_mask_and_fresh_Enc0'] is True
                            and actual_matrix['per_target_fresh_GC_object'] is False
                            and actual_matrix['per_tile_fresh_GC_object'] is True,
                            '整块返回保护单位不一致')
                    token = bfv['token_packing']; actual_token = bounds[0]['token_coefficient_layout']
                    for key in ('fixed_scan','tile_coordinates','key_transitions','independent_auxiliary_keys',
                                'switch_key_ciphertexts','weight_integer_abs_max','internal_scale',
                                'internal_scaled_abs_max','target_coefficient','GC_ANDs_per_coordinate',
                                'C_input_bits_per_coordinate','S_input_bits_per_coordinate','GC_OT_batches_per_tile',
                                'no_ring_inverse_of_512'):
                        require(token[key] == actual_token[key], f'Token专用定义与工件不一致：{key}')
                    require(token['fixed_scan'] == 155648 and token['input_blocks'] == actual_token['input_ciphertexts'] == 19
                            and token['tile_coordinates'] == 512 and token['key_transitions'] == 9
                            and token['switch_key_ciphertexts'] == 108 and actual_token['acyclic_only'] is True
                            and token['auxiliary_key_setup_bytes'] == actual_token['all_auxiliary_key_setup_bytes'] == 200026336,
                            'Token全扫描、实际返回布局或换钥初始化不一致')
                    require(token['descriptor']['fields'] == ['PRAGTOK3',3,'vocabulary',155648,'H',512,8192,19,9,12],
                            'Token v3描述符不一致')
                    token_error = 4**9*(8194*(19*2**71+1)+1)+174762*2**24+8194
                    require(int(actual_token['error_bound']) == token_error and 4*2**59*token_error < int(bounds[0]['Q'])
                            and token['internal_scale']*token['weight_integer_abs_max'] == token['internal_scaled_abs_max'] < 2**58,
                            'Token独立噪声界或精确去尺度条件不成立')
                    require(token['GC_ANDs_per_coordinate'] == 358 and token['C_input_bits_per_coordinate'] == 61
                            and token['S_input_bits_per_coordinate'] == 120 and token['GC_OT_choices_per_tile'] == 61*512
                            and token['GC_OT_post_chunk_choices'] == 1024 and token['GC_OT_batches_per_tile'] == 1,
                            'Token GC门数或输入OT布局不一致')
                    require(bfv['joint_projection']['descriptor']['fields'] == ['PRAGJNT3',3,'B','K','f_X','f_W','groups','O0','O1','O2_or_zero']
                            and bfv['private_diagonal']['descriptor']['fields'] == ['PRAGDIA3',3,'n','f_X','f_W']
                            and bounds[0]['private_gamma']['OT_choices_per_coordinate'] == 59,
                            '联合投影或gamma OT绑定不一致')
                    expected_bound = 8194*(512*8192*2**58+1)
                    require(int(bounds[0]['returned_phase_error_abs_strict_bound']) == expected_bound,
                            'BFV正确性界未覆盖512块矩阵合并')
                    require(bounds[0]['decode_required_inequality'] == '4*t*B < Q'
                            and int(bounds[0]['left_hand_side']) == 4*2**59*expected_bound
                            and 4*2**59*expected_bound < int(bounds[0]['Q']),
                            'BFV两位进位的更强不等式不成立')
            except (ValueError, KeyError, UnicodeError):
                errors.append('BFV工件包不能提供当前受控解码及打包/合并绑定')
    require('DEC-014' in decision_ids and any(
        d['id'] == 'DEC-014' and d['status'] == 'confirmed' for d in decisions),
        '多目标打包与分块合并的DEC-014授权登记缺失')
    current_profile = 'v1-f12-coordinated-20261013'
    require(index['spec_version'] == current_profile, '索引不是当前成套数值配置')
    require(loaded['技术规则.yaml']['spec_version'] == current_profile, '技术规则配置版本不一致')
    numeric = parameters['numeric_pipeline']
    require(numeric['profile_version'] == current_profile, '参数不是当前成套数值配置')
    ranges = numeric['range_policy']
    coordinated = ranges['coordinated_domains']
    require(coordinated['profile_version'] == current_profile, '成套数值域版本不一致')
    require(coordinated['ordinary_activation_real_abs_max'] == 2048
            and coordinated['ordinary_activation_f12_abs_max'] == 2**23,
            '普通激活2048/f12范围不一致')
    require(coordinated['parameter_real_abs_max'] == 256
            and coordinated['parameter_f12_abs_max'] == 2**20,
            '权重、gamma与Token参数范围不一致')
    swiglu = ranges['swiglu_local_domain']
    qk = ranges['qk_local_domains']
    # The standalone copies, parameter bindings and ART-006 bundle must name
    # the same actual files; a valid bundle hash alone does not prove this.
    direct_bindings = [
        (qk['profile'], qk['sha256']),
        (swiglu['profile'], swiglu['profile_sha256']),
        (swiglu['evidence'], swiglu['evidence_sha256']),
        (coordinated['evidence_artifact'], coordinated['evidence_sha256']),
    ]
    require(qk['sha256'] == swiglu['profile_sha256'] == coordinated['profile_sha256'],
            '三个数值域不是同一实际profile哈希')
    model_artifact = next((a for a in registry['artifacts'] if a['id'] == 'ART-006'), None)
    bundled = []
    if model_artifact is not None and model_artifact.get('path'):
        bundle_path = package_file(model_artifact['path'], '模型工件包')
        if bundle_path is not None:
            try:
                bundled = json.loads(bundle_path.read_text(encoding='utf-8'))['contents']
            except (ValueError, KeyError, UnicodeError):
                errors.append('模型工件包不能提供数值文件绑定')
    for relative, expected in direct_bindings:
        path = package_file(relative, '独立数值工件')
        if path is None:
            continue
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected,
                f'独立数值工件哈希不一致：{relative}')
        matches = [item for item in bundled if item.get('name') == path.name]
        require(len(matches) == 1 and matches[0].get('sha256') == expected,
                f'模型工件包与独立数值文件身份不一致：{relative}')
        if len(matches) == 1:
            try:
                require(matches[0].get('content') == json.loads(raw),
                        f'模型工件包与独立数值文件内容不一致：{relative}')
            except (ValueError, UnicodeError):
                errors.append(f'独立数值JSON无法解析：{relative}')
    require(swiglu['joint_and_down_proj_input_real_abs_max'] == 4194304
            and swiglu['joint_and_down_proj_input_f12_abs_max'] == 2**34
            and swiglu['raw_elementwise_product_f24_abs_max'] == 2**46,
            'SwiGLU成套整数界不一致')
    for d in (128, 1024, 2048):
        limit = d * (2**42 - 16) - 1
        require(coordinated['C_d_f12_abs_max'][d] == math.isqrt(limit),
                f'C_{d}不是定义的整数平方根')
        require(coordinated['RMS_square_sum_limit_f24'][d] == limit,
                f'RMS维数{d}的预累加预算不一致')
    require(coordinated['L2_square_sum_limit_f24'] == 2**52 - 1,
            'L2预累加预算不一致')
    require(coordinated['whole_row_reset_required'] is True
            and coordinated['safe_denominator_f24'] == 2**24
            and coordinated['safe_exponent'] == 0, '归一化失败整行替换定义不一致')
    require('DEC-013' in decision_ids and any(
        d['id'] == 'DEC-013' and d['selected'] == current_profile
        and d['status'] == 'confirmed' for d in decisions), 'DEC-013当前采纳记录缺失')
    for name, content in documents.items():
        for field_path in re.findall(r'参数与接口\.yaml的([A-Za-z_][A-Za-z0-9_.]*)', content):
            node = parameters
            for key in field_path.rstrip('.').split('.'):
                if not isinstance(node, dict) or key not in node:
                    errors.append(f'{name} 引用了不存在的参数字段：{field_path}')
                    break
                node = node[key]
        # 仅查特定的外部定位形式，不误删资料原文或模型来源等业务概念。
        if re.search(r'§|参见原文|(?:\.\./)', content):
            errors.append(f'{name} 残留外部说明定位或父目录引用')
        if name.endswith('.md'):
            for target in re.findall(r'\[[^\]]*\]\(([^\n]+?)\)', content):
                target = target.strip('<>').split('#', 1)[0]
                if not target:
                    continue
                if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                    errors.append(f'{name} 包含包外链接：{target}')
                    continue
                relative = (Path(name).parent / target).as_posix()
                package_file(relative, f'{name} 中的链接')
    return {
        'check': 'standalone_specification_consistency',
        'yaml_files': len(loaded),
        'rules': len(rules),
        'decision_count': len(decision_ids),
        'artifact_count': len(artifact_ids),
        'unmaterialized_artifacts': [a['id'] for a in registry['artifacts'] if a['status'] == 'not_materialized'],
        'system_tests_executed': False,
        'errors': errors,
    }


if __name__ == '__main__':
    report = main()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(1 if errors else 0)
