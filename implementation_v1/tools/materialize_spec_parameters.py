"""Export formed nonnumeric contracts into the standalone specification.

Reads local public artifacts and reviewed ABI sources; never runs a model,
activates a backend, changes numerical rules, or claims full R2 validation.
Large weight hashes are copied from the formed manifests, not recomputed here.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from private_rag import compute_input, delivery as delivery_module, heartbeat, model_input, types, wire
from private_rag.errors import ErrorCode
from tokenizers import Tokenizer

SPEC = ROOT.parent / '项目文档/第一版机读规范'
PARAMETERS = SPEC / '参数与接口.yaml'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(relative: str, *required: str):
    path = ROOT / relative
    text = path.read_text(encoding='utf-8')
    for item in required:
        require(item in text, f'review required: {relative}: {item}')
    return {'implementation_module': relative, 'implementation_sha256': sha(path)}


def artifact(relative: str, bundle: str, **extra):
    return {'artifact': '实施工件/' + bundle, 'component': Path(relative).name,
            'sha256': sha(ROOT / relative), **extra}


def model_artifacts(name: str, expected: dict):
    prefix = f'artifacts/models/{name}/'
    checkpoint = read(prefix + 'checkpoint_manifest.json')
    quant = read(prefix + 'quantized_f12/manifest.json')
    for manifest in (checkpoint, quant):
        require(manifest['model'] == expected['model_id'] and manifest['revision'] == expected['revision'], 'model binding mismatch')
    require(checkpoint['weights_present'] and quant['weight_domain_pass'], 'missing model artifacts')
    # Verify every checkpoint size and the actual public tokenizer/config files.
    # Hashes for large secret-model payloads remain the declared manifest hashes.
    for entry in checkpoint['files']:
        path = ROOT / prefix / 'checkpoint' / entry['path']
        require(path.is_file() and path.stat().st_size == entry['bytes'], 'checkpoint file missing or size mismatch')
        if not entry['path'].endswith('.safetensors'):
            require(sha(path) == entry['sha256'], 'public checkpoint hash mismatch')
    require(len(quant['tensors']) == 310, 'quantized tensor inventory changed')
    for tensor in quant['tensors']:
        require((ROOT / prefix / 'quantized_f12' / tensor['path']).is_file(), 'quantized tensor missing')
    weights = [entry for entry in checkpoint['files'] if entry['path'].endswith('.safetensors')]
    tokenizers = [entry for entry in checkpoint['files'] if Path(entry['path']).name in {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}]
    return {
        'checkpoint_manifest': artifact(prefix + 'checkpoint_manifest.json', 'models.json', model=checkpoint['model'], revision=checkpoint['revision'], file_count=len(checkpoint['files'])),
        'weight_hashes': {'algorithm': 'SHA256', 'files': weights, 'validation_status': 'formed_manifest_hashes_sizes_checked_large_payloads_not_rehashed_by_this_export'},
        'tokenizer_hashes': {'algorithm': 'SHA256', 'files': tokenizers, 'validation_status': 'actual_public_files_rehashed_and_matched'},
        'quantized_weight_manifest': artifact(prefix + 'quantized_f12/manifest.json', 'models.json', model=quant['model'], revision=quant['revision'], tensor_count=len(quant['tensors']), stored_dtype=quant['stored_dtype'], fractional_bits=quant['fractional_bits'], semantic_ring_bits=quant['semantic_ring_bits'], quantization=quant['quantization'], weight_domain_pass=quant['weight_domain_pass'], validation_status='manifest_formed_all_payload_paths_exist_not_rehashed_by_this_export'),
    }


def checked_template():
    template = read('artifacts/template/template_v1.json')
    manifest = read('artifacts/template/manifest.json')
    native = read('artifacts/template/native_manifest.json')
    require(sha(ROOT / 'artifacts/template/template_v1.json') == manifest['sha256'] == native['source_template_sha256'], 'template hash mismatch')
    binary = ROOT / 'artifacts/template' / native['path']
    require(binary.stat().st_size == native['bytes'] and sha(binary) == native['sha256'], 'native template hash mismatch')
    tokenizer = Tokenizer.from_file(str(ROOT / 'artifacts/models/generation/checkpoint/tokenizer.json'))
    config = read('artifacts/models/generation/checkpoint/tokenizer_config.json')
    require(hashlib.sha256(config['chat_template'].encode()).hexdigest() == template['official_template_sha256'], 'official role template changed')
    cursor = 0
    for segment in template['segments']:
        require(segment['offset'] == cursor, 'noncontiguous template slots')
        if segment['kind'] == 'public':
            actual = tokenizer.encode(segment['text'], add_special_tokens=False).ids
            require(actual == segment['token_ids'] and len(actual) == segment['capacity'], 'public segment tokenizer mismatch')
        cursor += segment['capacity']
    require(cursor == template['used_capacity'] == template['padding_offset'], 'template capacity mismatch')
    require(cursor + template['padding_capacity'] == template['physical_slots'] == 1024, 'template physical budget mismatch')
    actual_reserved = sorted(tokenizer.token_to_id(text) for text in template['reserved_control_strings'])
    require(actual_reserved == template['reserved_control_ids'], 'reserved control inventory mismatch')
    return template, native


def reviewed_schemas(delivery):
    require(delivery['share_payload']['bytes'] == delivery_module.RECORD_BYTES == 12296, 'share-record ABI mismatch')
    require(delivery['hpke']['total_bytes'] == delivery_module.HPKE_BYTES == 12344, 'HPKE envelope ABI mismatch')
    require(delivery['binding']['fields'] == [f.name for f in fields(delivery_module.DeliveryBinding)], 'delivery-binding fields changed')
    require(delivery_module.INFO_DOMAIN == b'private-rag/delivery/v1\x00', 'HPKE info domain changed')
    channel_source = source('native/src/channel.cpp', 'unsigned char h[96] = {};', 'memcpy(h, "PRGCF001", 8);', 'memcpy(h + 8, session_.data(), 16);', 'memcpy(h + 24, config_.data(), 32);', 'put64(h + 56, sent_++);', 'put64(h + 64, uint64_t(n));', 'put64(h + 72, stage_);', 'put64(h + 80, operation_);', 'put64(h + 88, 1);')
    native = {**channel_source, 'format': 'PRGCF001', 'actual_worker_transport': True,
        'header_bytes': 96, 'header_fields': [
            {'name': name, 'offset_bytes': offset, 'bytes': size, 'encoding': encoding}
            for name, offset, size, encoding in [
                ('magic', 0, 8, 'ASCII PRGCF001'), ('session_id', 8, 16, 'raw UUID bytes'),
                ('configuration', 24, 32, 'raw SHA256 bytes'), ('directional_sequence', 56, 8, 'uint64 big endian'),
                ('payload_bytes', 64, 8, 'uint64 big endian'), ('stage', 72, 8, 'uint64 big endian'),
                ('operation', 80, 8, 'uint64 big endian'), ('format_version', 88, 8, 'uint64 big endian, value 1')]],
        'payload': '由公开操作确定固定 n 字节；接收方先核对预期 n、session、configuration、方向序号、stage、operation、version，再读取 n 字节',
        'total_bytes': '96+n；不包括 TLS 记录开销', 'all_operations_have_one_global_payload_size': False,
        'compression': 'none', 'p256_point_bytes': 65, 'supported_native_host_ABI': 'x86_64 little endian',
        'whole_model_message_plan_verified': False}
    python = {**source('private_rag/wire.py'), 'format': 'PRG1', 'actual_worker_transport': False,
        'scope': '独立 Python 通用固定帧编码器；原生 worker 使用 PRGCF001',
        'prefix_struct': wire.PREFIX.format, 'prefix_bytes': wire.PREFIX.size,
        'prefix_fields': ['magic=PRG1', 'version=1', 'flags=0', 'directional_sequence', 'header_json_bytes', 'payload_bytes'],
        'header_area_bytes': wire.HEADER_BYTES, 'payload_area_bytes': wire.PAYLOAD_BYTES, 'frame_bytes': wire.FRAME_BYTES,
        'header_fields': sorted(wire.REQUIRED), 'header_encoding': 'ASCII canonical JSON, ensure_ascii=true, sorted keys, compact separators, allow_nan=false',
        'padding': '未用 header/payload 字节全零；解码拒绝非规范编码、重复或未知字段以及非零填充'}
    worker_source = source('compute/worker_main.cpp', 'if(argc!=22)', 'std::array<unsigned char,104> h{};', '"PRAGIN01"', 'std::array<unsigned char,2312>', 'encoded[8200]=result.client_failed.value();', 'argv[20]', 'argv[21]')
    argv = ['party_C_or_S', 'mode_prepare_or_query', 'compute_fd', 'outbound_heartbeat_fd', 'inbound_heartbeat_fd',
            'sealed_input_memfd', 'local_output_pipe_write_fd', 'session_uuid_hex32', 'configuration_hex64', 'snapshot_sha256_hex64',
            'expected_peer_leaf_certificate_sha256_hex64', 'CA_pem_path', 'local_certificate_pem_path', 'local_private_key_pem_path',
            'expected_peer_hostname', 'models_root', 'constants_binary', 'template_binary', 'native_ledger_path', 'lifecycle_event_write_fd', 'lifecycle_ack_read_fd']
    worker = {**worker_source, 'program': 'prag_worker', 'argc_including_program': len(argv) + 1,
        'argv': [{'index': i, 'name': name} for i, name in enumerate(argv, 1)],
        'physical_lane_order': ['compute', 'C_out', 'S_out'], 'C_fd_order': ['compute', 'C_out', 'S_out'], 'S_fd_order': ['compute', 'S_out', 'C_out'],
        'input': {'transport': 'same-party sealed Linux memfd; never management HTTP', 'magic': compute_input.MAGIC.decode(), 'header_struct': compute_input.HEADER.format, 'header_bytes': compute_input.HEADER.size,
            'header_fields': ['magic8', 'mode_u8_prepare1_query2', 'party_u8_C0_S1', 'reserved6_zero', 'session_UUID16', 'configuration_SHA256_32', 'snapshot_SHA256_32', 'record_count_u64LE'],
            'local_slots': 'uint64LE tokens[count] followed by uint8 validity[count]; full fixed capacity',
            'C_prepare': '257 embedding slots; zero records', 'C_query': '160 embedding slots then 128 body slots then all N records',
            'S_prepare': 'zero local slots and zero records', 'S_query': 'zero local slots followed by all N own records',
            'bytes': '104 + C_local_slot_count*9 + N*12296', 'seals': ['F_SEAL_WRITE', 'F_SEAL_GROW', 'F_SEAL_SHRINK', 'F_SEAL_SEAL'],
            'record': delivery['share_payload']},
        'public_snapshot_handshake': '41 bytes = snapshot SHA256[32] || N uint64LE || mode uint8, inside authenticated native Channel',
        'events': [{'value': 1, 'name': 'AUTHENTICATED', 'when': '三 lane mTLS 与公开快照匹配后、首次 OT 前'}, {'value': 2, 'name': 'RELEASING', 'when': '最终释放前'}],
        'event_ACK': 'each event one byte, same-byte ACK after own persistent callback; reject duplicates/order mismatch/EOF; no automatic retry',
        'output': {'C_query': {'bytes': 2312, 'layout': '256 uint64LE token IDs || 256 uint8 validity || one uint64LE status'},
            'S_query': {'bytes': 0}, 'C_prepare': {'bytes': 8201, 'layout': '1024 uint64LE own z shares || one uint64LE own e share || one uint8 private client_failed'},
            'S_prepare': {'bytes': 8200, 'layout': '1024 uint64LE own z shares || one uint64LE own e share; no private failure bit'}},
        'completion': delivery['compute_coordinator']['completion'], 'full_R2_executed': False}
    hb = {**source('native/src/heartbeat.cpp', '0x4842', 'set_operation(0)', 'std::array<unsigned char, 8> payload{}'),
        'format': 'PRGCF001', 'stage': 0x4842, 'operation': 0, 'payload': 'uint64 big endian monotonically increasing counter, initially 0',
        'payload_bytes': 8, 'header_bytes': 96, 'application_frame_bytes': 104, 'includes_TLS_overhead': False,
        'lanes': 'C_out and S_out independent authenticated TLS connections, bound to compute peer/session/configuration',
        'interval_seconds': int(heartbeat.INTERVAL), 'inactivity_seconds': int(heartbeat.INACTIVITY),
        'python_helper': {**source('private_rag/heartbeat.py'), 'actual_worker_transport': False, 'struct': heartbeat.MESSAGE.format,
                          'frame_bytes': heartbeat.MESSAGE.size, 'fields': ['ASCII PHB1', 'UUID16', 'uint64BE sequence', 'four zero bytes']}}
    probe = read('records/crypto-heartbeat-probe02.json')
    require(probe['TLS13_mTLS_connections'] == 3 and probe['threads_joined'] is True and probe['full_system'] is False, 'heartbeat evidence changed')
    hb['development_probe'] = {'result_sha256': sha(ROOT / 'records/crypto-heartbeat-probe02.json'), **probe}
    return native, python, worker, hb


def replace_blocks(original: str, data: dict, keys: tuple[str, ...]) -> str:
    text = original
    for key in keys:
        pattern = rf'(?ms)^{re.escape(key)}:\n.*?(?=^[A-Za-z_][A-Za-z_0-9]*:|\Z)'
        block = yaml.safe_dump({key: data[key]}, allow_unicode=True, sort_keys=False, width=110) + '\n'
        text, n = re.subn(pattern, lambda _: block, text, count=1)
        require(n == 1, f'cannot locate top level block {key}')
    return text


def main():
    original = PARAMETERS.read_text(encoding='utf-8')
    params = yaml.safe_load(original)
    for name in ('embedding', 'generation'):
        params['models'][name]['actual_artifacts'] = model_artifacts(name, params['models'][name])
        params['models'][name]['validation_status'] = 'manifests_and_payload_inventory_materialized_public_files_hash_checked_full_R2_not_run'
    template, native_template = checked_template()
    preparation = params['input_templates_and_capacity']['text_preparation']
    preparation['reserved_control_strings'] = template['reserved_control_strings']
    preparation['reserved_control_ids_complete_list'] = template['reserved_control_ids']
    preparation['actual_text_normalization_definition'] = {**source('private_rag/model_input.py', 'errors="strict"', 'text = text[1:]', 'unicodedata.normalize("NFC"', 'RESERVED_CONTROL_STRING', 'RESERVED_CONTROL_TOKEN'),
        'order': ['bytes 按 UTF-8 strict 解码；str 经 UTF-8 strict 编码检查', '仅删除开头一个 U+FEFF', 'CRLF 替换成 LF，再把单独 CR 替换成 LF', 'Unicode NFC', '拒绝任意保留控制串', '正文以生成 tokenizer 独立分词，不添加 special Token；拒绝 ID 不在 0..151642'],
        'trim_whitespace': False, 'invalid_input': '仅输入持有方本地拒绝，无原文或控制串回显',
        'embedding_postprocessor': '实际 Token 流必须恰为不加 special Token 的分词结果加一个 EOS 151643',
        'unicode_database_version': model_input.unicodedata.unidata_version}
    preparation['validation_status'] = 'materialized_owner_local_implementation_control_ids_checked_against_tokenizer'
    generation = params['input_templates_and_capacity']['generation_template']
    generation['exact_whitespace_and_newlines'] = {s['name']: s['text'] for s in template['segments'] if s['kind'] == 'public'}
    offsets = {s['name']: {'offset': s['offset'], 'capacity': s['capacity'], 'kind': s['kind']} for s in template['segments']}
    offsets['padding'] = {'offset': template['padding_offset'], 'capacity': template['padding_capacity'], 'kind': 'invalid_dummy'}
    generation['fixed_subslot_offsets'] = offsets
    generation['dummy_token_id'] = template['dummy_token_id']
    generation['dummy_validity'] = template['dummy_validity']
    generation['budget_validation_result'] = {'public_token_count': sum(s['capacity'] for s in template['segments'] if s['kind'] == 'public'),
        'private_slot_capacity': sum(s['capacity'] for s in template['segments'] if s['kind'] != 'public'),
        'used_capacity': template['used_capacity'], 'padding_capacity': template['padding_capacity'], 'physical_slots': template['physical_slots'],
        'actual_tokenizer_reencoded_public_fragments': True, 'contiguous_offsets_checked': True, 'official_role_template_sha256': template['official_template_sha256'],
        'native_binary': {'artifact': '实施工件/template.json', 'component': 'native_manifest.json', 'binary_format': native_template['format_version'], 'bytes': native_template['bytes'], 'sha256': native_template['sha256']},
        'canonical_stream_rule': template['canonical_rule'], 'full_R2_executed': False}
    generation['activation_status'] = 'template_prerequisite_materialized_full_model_requires_independent_activation'
    params['positions_masks_and_cache']['invalid_query_attention']['dummy_key_physical_index'] = template['dummy_key_physical_index']
    delivery = read('artifacts/interface_delivery.json')
    native, python, worker, hb = reviewed_schemas(delivery)
    descriptor = {**source('private_rag/types.py'), 'python_class': 'Descriptor',
        'fields': [{'name': f.name, 'type': str(f.type).replace("<class '", '').replace("'>", '')} for f in fields(types.Descriptor)],
        'implemented_representations': ['arithmetic_share', 'boolean_share', 'private_weight', 'public_value'],
        'not_supported_by_this_python_class': ['gc_bitvector', 'bfv_ciphertext'],
        'arithmetic_validation': 'ring_bits=59, signed_interpretation=twos_complement, integer fractional_bits in 0..58',
        'boolean_validation': 'positive integer bit_width; fractional_bits=None; BooleanShare carries XOR-share bools',
        'scope': '本方类型检查，不是全部原生算子统一线协议；GC/BFV 类型边界仍由对应后端及调用约定约束'}
    td = params['representation']['tensor_descriptor']
    td['actual_schema'] = descriptor
    td['wire_encoding'] = {'arithmetic_residue': 'uint64 little endian, 0<=value<2^59', 'descriptor': 'Python Descriptor 本身无通用二进制序列化；已实现帧、交付和本方 worker ABI 见 interfaces.common.serialization_schema', 'complete_all_representation_ABI': False}
    td['validation_status'] = 'partially_materialized_python_and_native_boundaries_not_unified_all_types'
    channels = params['transport_storage_and_dependencies']['channels']
    channels['exact_byte_order'] = {'ring_residues': 'uint64 little endian', 'native_channel_header': 'uint64 big endian', 'native_heartbeat_payload': 'uint64 big endian', 'worker_memfd': 'little endian; UUID/SHA256 raw bytes', 'HPKE_info': delivery['binding']['encoding']}
    channels['exact_field_serialization_and_frame_schema'] = {'native_worker': native, 'python_generic_helper': python}
    channels['validation_status'] = 'concrete_schemas_materialized_full_R2_schedule_not_verified'
    common = params['interfaces']['common']
    params['interfaces']['status'] = 'logical_contracts_with_materialized_delivery_channel_worker_ABI_incomplete_unified_tensor_ABI'
    common['exact_type_definitions'] = {'Descriptor': descriptor,
        'python_local_share_types': {name: [{'name': f.name, 'type': str(f.type).replace("<class '", '').replace("'>", '')} for f in fields(getattr(types, name))] for name in ['ArithmeticShare', 'BooleanShare', 'SecretToken', 'PrivateWeight', 'Public']},
        'WorkerBinding': [{'name': f.name, 'type': str(f.type)} for f in fields(compute_input.WorkerBinding)],
        'native_tensor': {**source('compute/secure_tensor.hpp'), 'fields': {'values': 'one-party vector<uint64_t> residues', 'shape': 'public vector<size_t>, internal only', 'fractional_bits': 'unsigned', 'configuration': 'public string'}, 'internal_kernels': '部分内部函数仍以 Word/Bit 和静态调用约定传递；不是所有秘密值已包成统一描述符'},
        'complete_all_representation_ABI': False}
    common['exact_error_enum'] = {'python': [error.value for error in ErrorCode], 'native_worker_exit_failure': 2, 'native_worker_stderr_codes': ['BACKEND_NOT_READY', 'COMPUTE_ABORTED'], 'native_output_forwarding': '父进程不转发 native stdout/stderr；秘密数值失败走固定最终输出/仅 D 的准备失败位'}
    hpke = {key: value for key, value in delivery['hpke'].items() if key != 'official_api_source'}
    ledger = {**delivery['material_ledger'], **source('private_rag/lifecycle.py', 'operator_confirmed_worker_and_lanes_stopped: bool', 'if operator_confirmed_worker_and_lanes_stopped is not True:')}
    ledger['crash'] = 'reserved entries burned only on terminal cleanup; operator recovery requires old child and all three lanes confirmed stopped plus exclusive compute-owner lock; no automatic replay/resume'
    common['serialization_schema'] = {'artifact': artifact('artifacts/interface_delivery.json', 'interfaces.json'), 'canonical_public_JSON': delivery['binding']['encoding'],
        'delivery_binding': delivery['binding'], 'share_record': delivery['share_payload'], 'HPKE': hpke,
        'native_channel': native, 'python_generic_frame': python, 'same_party_worker': worker,
        'document_vault': delivery['vault'], 'public_lifecycle': delivery['compute_coordinator'], 'material_ledger': ledger}
    common['validation_status'] = 'formed_schemas_and_source_bindings_checked_not_full_R2_validation'
    params['interfaces']['modules']['AssemblePrompt']['exact_offsets'] = offsets
    params['interfaces']['modules']['AssemblePrompt']['activation_status'] = 'template_prerequisite_materialized_full_model_requires_independent_activation'
    params['runtime_and_lifecycle']['heartbeat']['actual_heartbeat_frame_bytes'] = hb['application_frame_bytes']
    params['runtime_and_lifecycle']['heartbeat']['actual_frame_schema'] = hb
    params['runtime_and_lifecycle']['heartbeat']['validation_status'] = 'native_three_TLS_probe_recorded_no_claim_of_elapsed_180_second_inactivity_run_or_full_R2'
    # Do not rewrite unrelated/numerical top-level blocks, activation, or evidence.
    keys = ('representation', 'models', 'input_templates_and_capacity', 'positions_masks_and_cache', 'transport_storage_and_dependencies', 'interfaces', 'runtime_and_lifecycle')
    updated = replace_blocks(original, params, keys)
    old_data, new_data = yaml.safe_load(original), yaml.safe_load(updated)
    for key in old_data:
        if key not in keys:
            require(old_data[key] == new_data[key], f'out of scope mutation: {key}')
    for key in ('numeric_pipeline', 'nonlinear_algorithms', 'cryptographic_backends'):
        match = rf'(?ms)^{key}:\n.*?(?=^[A-Za-z_][A-Za-z_0-9]*:|\Z)'
        require(re.search(match, original).group() == re.search(match, updated).group(), f'numeric text changed: {key}')
    require(new_data['runtime_and_lifecycle']['scheduler']['active_compute_total_timeout_seconds'] is None, 'total deadline must stay disabled')
    require(new_data['models']['shared_architecture']['rope_scaling'] is None, 'rope scaling must stay disabled')
    PARAMETERS.write_text(updated, encoding='utf-8')
    record = {'status': 'nonnumeric_formed_parameters_materialized', 'parameter_file': '参数与接口.yaml',
        'parameter_sha256': hashlib.sha256(updated.encode()).hexdigest(), 'modified_top_level_blocks': list(keys),
        'template_sha256': sha(ROOT / 'artifacts/template/template_v1.json'), 'model_quantized_tensor_counts': {'embedding': 310, 'generation': 310},
        'checks': ['public checkpoint files rehashed', 'large checkpoint sizes checked', 'quantized payload paths present', 'template/native binary hashes', 'public fragments reencoded with real tokenizer', 'all reserved controls mapped to actual IDs', 'reviewed ABI source fingerprints', 'YAML parse', 'numeric sections byte-identical'],
        'large_weight_payloads_rehashed_this_run': False, 'formal_test_or_acceptance': False, 'full_R2_executed': False,
        'remaining': ['unified GC/BFV tensor descriptor ABI not implemented', 'complete fixed model message/access plan not materialized', 'full R2 correctness/performance/acceptance not established']}
    (ROOT / 'records/spec_parameter_materialization.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
