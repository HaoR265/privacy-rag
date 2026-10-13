"""Bind completed hybrid diagnostics to their actual archived execution sources.

Active registrations are explicitly separate. This command runs no model and
cannot certify a pending M2/M3 stage or enable production.
"""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
COMPLETED = [
    ('M1_embedding', 'records/hybrid112_layer_embedding_2_0_1ff4142dbc84438db1e7382d6715cf32.json', 'records/hybrid112_M1_source_snapshot.tar.gz'),
    ('M1_generation', 'records/hybrid112_layer_generation_2_0_2296cff92f6345f28be2dfde120bec00.json', 'records/hybrid112_M1_source_snapshot.tar.gz'),
    ('L2_retrieval_component', 'records/hybrid112_postprocess_fcc69841b66f4fa08782cef499ac1c8a.json', 'records/hybrid112_postprocess_source_snapshot.tar.gz'),
]
M2 = 'records/hybrid112_decode_e245f7c77f914382806a5a8e7a07689e'
M3 = 'records/hybrid112_M3_0ef42a29ee7b4f499a4a70681b65c868'


def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def verify_archive(record, archive):
    with tarfile.open(ROOT / archive) as tar:
        for name, expected in record['source_and_binary_sha256'].items():
            stream = tar.extractfile(name)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                raise RuntimeError('ARCHIVE_IDENTITY: ' + name)


def verify_finished_record(record, terminal='closed', ledger_count=2):
    """Check immutable evidence; require a terminal only for a claimed success.

    A killed process can leave reservations durably burned without appending a
    terminal line. Report that observation on an unsuccessful run; never write
    a synthetic terminal or promote the run to a pass.
    """
    ledgers = []
    for item in record.get('evidence', []):
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('EVIDENCE_IDENTITY: ' + item['path'])
        if item['path'].endswith('.ledger'):
            with (ROOT / item['path']).open('rb') as f:
                f.seek(max(0, f.seek(0, 2) - 100))
                lines = f.read().splitlines()
                last = lines[-1] if lines else b''
                observed = {b'terminal=closed all_reserved=burned': 'closed_and_burned',
                            b'terminal=aborted all_reserved=burned': 'aborted_and_burned'}.get(
                                last, 'terminal_marker_not_confirmed')
                ledgers.append({'path': item['path'], 'observation': observed})
                if terminal is not None and last != f'terminal={terminal} all_reserved=burned'.encode():
                    raise RuntimeError('LEDGER_NOT_TERMINAL')
    if terminal is not None and len(ledgers) != ledger_count:
        raise RuntimeError('LEDGER_EVIDENCE_COUNT')
    return ledgers


def main():
    output = {'scope': 'finite actual hybrid R2 development evidence; no formal acceptance',
              'development_native_version': 'prag-native-v1.0.12', 'full_R2_complete': False,
              'production_enabled': False, 'M4_executed_in_this_stage': False,
              'completed': [], 'pending': []}
    for name, path, archive in COMPLETED:
        record = read(path)
        verify_archive(record, archive)
        verify_finished_record(record)
        if record.get('result') != 'matched_CUDA_R1' or record.get('numeric_failure') is not False:
            raise RuntimeError('UNSUCCESSFUL_FINITE_SAMPLE')
        output['completed'].append({'name': name, 'record': path, 'record_sha256': sha(path),
            'execution_source_archive': archive, 'archive_sha256': sha(archive),
            'shape': {k: record[k] for k in ('model', 'rows', 'hidden', 'layer', 'layers', 'output_words') if k in record},
            'parties': record['parties'], 'numeric_failure': record['numeric_failure']})
    edge = 'records/hybrid112_lookup_97ece7006ee243c0912d17ed95acb7a3.json'
    record = read(edge)
    for path, digest in record['source_and_binary_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('LOOKUP_SOURCE_CHANGED')
    verify_finished_record(record)
    if record['exit_code'] != 0 or record['output']['result'] != 'matched_CUDA_oracle':
        raise RuntimeError('LOOKUP_NOT_VERIFIED')
    output['completed'].append({'name': 'RoPE_lookup_boundary_component', 'record': edge,
        'record_sha256': sha(edge), 'result': record['output'],
        'boundary': 'public positions 0,1,1278,1279,1280,ring -1,high bit,sign bit; not a full model'})
    cleanup = 'records/hybrid112_cache_cleanup_54c56e0a08434c7d8b0675c856f4b054.json'
    record = read(cleanup)
    archive = 'records/hybrid112_release_cleanup_source_snapshot.tar.gz'
    verify_archive(record, archive)
    verify_finished_record(record, 'aborted', ledger_count=4)
    if len(record['cases']) != 2 or any(not all(case[key] for key in
            ('cache_released_before_destructor', 'cache_reuse_rejected', 'crypto_aborted'))
            or case['model_arithmetic_executed'] for case in record['cases']):
        raise RuntimeError('CACHE_ABORT_NOT_VERIFIED')
    output['completed'].append({'name': 'retained_GPU_cache_abort_cleanup', 'record': cleanup,
        'record_sha256': sha(cleanup), 'execution_source_archive': archive,
        'archive_sha256': sha(archive), 'cases': record['cases'],
        'boundary': 'faults injected before model arithmetic; does not certify any model output'})
    build = 'records/hybrid112_production_build.json'
    record = read(build)
    verify_archive(record, archive)
    if record['exit_code'] != 2 or record['stderr'] != 'BACKEND_NOT_READY' or not all(record[k] for k in
            ('rejects_before_interpreting_input', 'production_library_excludes_development_decode',
             'hybrid_protocol_library_matches_M3_build')):
        raise RuntimeError('PRODUCTION_GATE_NOT_VERIFIED')
    output['completed'].append({'name': 'production_build_closed_gate', 'record': build,
        'record_sha256': sha(build), 'execution_source_archive': archive,
        'archive_sha256': sha(archive), 'boundary': 'compilation and input-before-gate rejection only; no model run'})
    build = 'records/hybrid112_fresh_build_entry.json'
    record = read(build)
    archive = 'records/hybrid112_fresh_build_source_snapshot.tar.gz'
    identities = {**record['source_and_dependency_sha256'], **record['output_sha256']}
    verify_archive({'source_and_binary_sha256': identities}, archive)
    if (record.get('result') != 'compiled_context_matched_gate_closed' or
            record.get('model_executed') is not False or record.get('production_enabled') is not False or
            len(record.get('gate_cases', [])) != 4 or any(
                case['exit_code'] != 2 or not case['stdout_empty'] or case['stderr'] != 'BACKEND_NOT_READY'
                for case in record['gate_cases'])):
        raise RuntimeError('FRESH_BUILD_NOT_VERIFIED')
    output['completed'].append({'name': 'fresh_directory_build_entry', 'record': build,
        'record_sha256': sha(build), 'execution_source_archive': archive,
        'archive_sha256': sha(archive), 'boundary': 'independent build identity and closed gates; no model run'})
    build = 'records/hybrid112_registered_development_build.json'
    record = read(build)
    archive = record['execution_source_archive']
    verify_archive(record, archive)
    if (record.get('result') != 'compiled_context_matched_gate_closed_and_registration_verified' or
            record.get('model_executed') is not False or record.get('production_enabled') is not False or
            len(record.get('binaries', [])) != 3 or not all(b['copied_and_verified'] for b in record['binaries']) or
            sha(archive) != record['archive_sha256']):
        raise RuntimeError('DEVELOPMENT_BUILD_REGISTRATION_NOT_VERIFIED')
    output['completed'].append({'name': 'registered_development_build_entry', 'record': build,
        'record_sha256': sha(build), 'execution_source_archive': archive,
        'archive_sha256': sha(archive), 'boundary': 'build and execution source binding; no M1/M2/M3 run with these entry changes'})
    for name, prefix, archive in [('M2_decode', M2, 'records/hybrid112_M2_running_source_snapshot.tar.gz'),
                                  ('M3_private_embedding_delivery_retrieval', M3, M3 + '.source.tar.gz')]:
        registration = read(prefix + '.registration.json')
        verify_archive(registration, archive)
        binding = {'name': name, 'registration': prefix + '.registration.json',
                   'registration_sha256': sha(prefix + '.registration.json'),
                   'execution_source_archive': archive, 'archive_sha256': sha(archive)}
        if (ROOT / (prefix + '.json')).exists():
            record = read(prefix + '.json')
            if name == 'M2_decode':
                passed = record.get('result') == 'matched_CUDA_R1' and record.get('numeric_failure') is False
                binding['ledger_observations'] = verify_finished_record(record, 'closed' if passed else None)
            else:
                passed = record.get('result') == 'M3_matched_CUDA_R1'
                if passed and (len(record.get('stages', [])) != 3 or any(
                        stage.get('result') != 'completed_actual_private_stage' or
                        not all(stage.get('oracle', {}).get(k) is True for k in
                                ('final_integers_match', 'failure_matches')) or
                        stage.get('oracle', {}).get('numeric_failure') is not False
                        for stage in record['stages'])):
                    raise RuntimeError('INCOMPLETE_M3_STAGE_EVIDENCE')
                binding['ledger_observations'] = []
                for stage in record.get('stages', []):
                    binding['ledger_observations'].extend(verify_finished_record(stage, 'closed' if passed else None))
                if not passed:
                    # An exception can leave the controller summary without
                    # stages. Inspect its immutable stage records separately;
                    # absent or nonterminal evidence remains unconfirmed.
                    binding['stage_evidence'] = []
                    for stage_path in sorted((ROOT / 'records').glob(Path(prefix).name + '.*.json')):
                        if stage_path.name.endswith(('.registration.json', '.progress.json')):
                            continue
                        stage = json.loads(stage_path.read_text())
                        if stage.get('configuration') != registration['configuration']:
                            raise RuntimeError('M3_STAGE_CONFIGURATION_MISMATCH')
                        relative = str(stage_path.relative_to(ROOT))
                        binding['stage_evidence'].append({'record': relative, 'record_sha256': sha(relative),
                            'result': stage.get('result'), 'ledger_observations': verify_finished_record(stage, None)})
            binding.update(record=prefix + '.json', record_sha256=sha(prefix + '.json'), result=record.get('result'))
            if not passed:
                output['pending'].append({**binding, 'status': 'not_passed_or_incomplete'})
            else:
                output['completed'].append(binding)
        else:
            output['pending'].append({**binding, 'status': 'registered_execution_in_progress_not_a_pass'})
    output['M1_complete_finite_shapes'] = True
    output['M2_complete'] = any(r['name'] == 'M2_decode' for r in output['completed'])
    output['M3_complete'] = any(r['name'] == 'M3_private_embedding_delivery_retrieval' for r in output['completed'])
    output['reference_boundary'] = 'CUDA R1 final oracle only; plaintext/reconstructed oracle data never input to native R2 participants'
    output['resource_policy'] = {'R0_R1': 'RTX2060 required; no CPU model fallback',
        'R2_CPU': 'pinned SEAL/SCI BFV, OT, GC, Beaver cryptographic operations',
        'R2_RTX2060': 'exact local share arithmetic, public layout, KV own-share storage',
        'transfers': 'own shares, S-owned weights, public constants only'}
    (ROOT / 'artifacts/hybrid112_evidence.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ('M1_complete_finite_shapes', 'M2_complete', 'M3_complete', 'full_R2_complete', 'production_enabled')}))


if __name__ == '__main__':
    main()
