"""Bind actual batch kernel runs, current builds and preserved v109 evidence.

This materializes engineering evidence only. It never runs or enables a model.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    files = set()

    def bind(path):
        files.add(path)
        return read(path)

    def verify(mapping):
        for path, expected in mapping.items():
            if sha(path) != expected:
                raise RuntimeError('Evidence identity changed: ' + path)
            files.add(path)

    def evidence(record, terminal):
        verify(record.get('source_and_binary_sha256', record.get('source_sha256', record.get('files', {}))))
        for item in record.get('evidence', record.get('ledgers', [])):
            path = item['path']
            verify({path: item['sha256']})
            if path.endswith('.ledger'):
                actual = (ROOT / path).read_text().splitlines()[-1]
                wanted = terminal(path) if callable(terminal) else terminal
                if actual != f'terminal={wanted} all_reserved=burned':
                    raise RuntimeError('Unterminated ledger: ' + path)

    def phases(record):
        path = next(e['path'] for e in record['evidence'] if e['path'].endswith('.C.trace.jsonl'))
        rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
        phases = {}
        stack = []
        for row in rows:
            if row['event'] == 'begin':
                stack.append(row['phase'])
                continue
            if not stack or stack[-1] != row['phase']:
                raise RuntimeError('Unbalanced phase trace')
            phase_name = '/'.join(stack)
            stack.pop()
            group = phases.setdefault(phase_name, {'occurrences': 0, 'seconds': 0.0, 'application_bytes': 0,
                'frames_sent': 0, 'frames_received': 0, 'direction_runs': 0, 'peak_rss_kib': 0, 'reservations': {}})
            group['occurrences'] += 1
            for key in ('seconds', 'application_bytes', 'frames_sent', 'frames_received', 'direction_runs'):
                group[key] += row.get(key, 0)
            group['peak_rss_kib'] = max(group['peak_rss_kib'], row.get('peak_rss_kib', 0))
            materials = Counter(group['reservations'])
            materials.update(row.get('reservations', {}))
            group['reservations'] = dict(materials)
            group['last_event'] = row['event']
        return phases

    # A historical run is checked against its archived sources, never current files.
    archive = 'records/v110_source_snapshot.tar.gz'
    files.add(archive)
    with tarfile.open(ROOT / archive) as tar:
        old_manifest = json.load(tar.extractfile('artifacts/bfv_native_manifest.json'))
        for path, expected in old_manifest['files'].items():
            data = tar.extractfile(path).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Historical archive identity changed: ' + path)
        old_bound = json.load(tar.extractfile('artifacts/bfv_correctness_bound.json'))
        # BFV and the pre-existing OT/GC implementations are byte-identical.
        unchanged = [p for p in old_manifest['files'] if
                     p.startswith(('native/src/', 'native/include/prag/', 'third_party/')) and p not in ('native/src/batch.cpp', 'native/include/prag/batch.hpp')]
        verify({p: old_manifest['files'][p] for p in unchanged})
        historical_runs = [p for p in old_manifest['files'] if p.startswith('records/') and p.endswith('.json') and p != 'records/private_model_device_policy.json']
        verify({p: old_manifest['files'][p] for p in historical_runs})

    libraries = ['native/build/libprag_crypto.a', 'build/compute/native/libprag_crypto.a',
                 'experiments/batch111/build/native/libprag_crypto.a']
    library = sha(libraries[0])
    if len({sha(p) for p in libraries}) != 1:
        raise RuntimeError('Worker/probe/current native libraries differ')
    files.update(libraries)
    registration = bind('records/batch111_run_registration.json')
    assert registration['native_library_sha256'] == library
    # Earlier successful runs remain valid for their original source snapshot.
    # They are never selected as current-build evidence merely by filename date.
    previous = 'records/batch111_before_callback_fix.tar.gz'
    files.add(previous)
    with tarfile.open(ROOT / previous) as tar:
        prior_manifest = json.load(tar.extractfile('artifacts/bfv_native_manifest.json'))
        for path, expected in prior_manifest['files'].items():
            if hashlib.sha256(tar.extractfile(path).read()).hexdigest() != expected:
                raise RuntimeError('Pre-fix archive identity changed: ' + path)
    if read('records/batch111_actual_context.json') != read('artifacts/bfv_context.json'):
        raise RuntimeError('Actual BFV context changed')
    policy = bind('records/private_model_device_policy.json')
    gate = policy['native_compilation_and_runtime_gate_check']
    assert gate['worker_sha256'] == sha('build/compute/prag_worker')
    assert gate['public_stderr'] == 'BACKEND_NOT_READY' and gate['rejected_before_input_or_sockets']
    assert gate['production_library_excludes_development_decode_symbol']
    build_paths = libraries + ['build/compute/prag_worker', 'native/build/prag_bfv_context',
        'records/batch111_actual_context.json', 'records/batch111_native_configure.log',
        'records/batch111_native_build.log', 'records/batch111_compute_configure.log',
        'records/batch111_compute_build.log', 'compute/model.hpp', 'compute/numerics.hpp',
        'compute/batch_numerics.hpp', 'compute/CMakeLists.txt', 'native/CMakeLists.txt']
    build = {'scope': 'current builds and closed gate only', 'native_version': '1.0.11',
        'worker_crypto_library_matches_mtls_library': True, 'actual_context_matches_locked': True,
        'model_executed': False, 'full_R2': False, 'jobs_per_build': 1, 'current_gate_probe': gate,
        'source_and_binary_sha256': {p: sha(p) for p in build_paths}}
    write('records/compute_batch111_build.json', build)
    bind('records/compute_batch111_build.json')
    verify(build['source_and_binary_sha256'])

    rms = []
    for path in registration['rms']:
        record = read(path)
        assert record['production_Numerics_rms_many']
        evidence(record, 'closed')
        assert record['result'] == 'matched_CUDA_R1' and record['native_version'] == '1.0.11'
        assert record['source_and_binary_sha256'][libraries[2]] == library
        name = path; bind(name)
        rms.append({'record': name, **{k: record[k] for k in ('dimension', 'batch', 'case', 'application_bytes',
            'elapsed_seconds', 'reference_failed', 'tensor', 'weight_sha256')}, 'phases_inclusive': phases(record)})
    required = {(d, b, c) for d in (128, 1024, 2048) for b, c in ((1, 'normal'), (16, 'normal'), (16, 'boundaries'))}
    assert {(r['dimension'], r['batch'], r['case']) for r in rms} == required and len(rms) == 9

    primitive_path = registration['primitives']
    primitive = bind(primitive_path); evidence(primitive, 'closed')
    assert len(primitive['cases']) == 29 and all(c['matched'] for c in primitive['cases'])
    failure_path = registration['failures']
    failures = bind(failure_path)
    evidence(failures, lambda p: 'closed' if p.endswith(('_4.C.ledger', '_4.S.ledger')) else 'aborted')
    assert len(failures['cases']) == 26 and all(c['expected_rejection'] for c in failures['cases'])
    assert {(c['party'], c['mode']) for c in failures['cases']} == {(p, m) for p in ('C', 'S') for m in range(13)}
    attention_paths = registration['attention']
    attention = []
    for path in attention_paths:
        record = bind(path); complete = record['result'] == 'matched_CUDA_R1'
        evidence(record, 'closed' if complete else 'aborted')
        attention.append({'record': path, 'complete': complete, 'rows': record['rows'], 'keys': record['keys'],
            'case': record['case'], 'application_bytes': record['application_bytes'],
            'seconds': record.get('seconds', record.get('elapsed_seconds')), 'reference_failed': record['reference_failed'],
            'phases_inclusive': phases(record)})
    assert len(attention) == 4 and all(r['complete'] for r in attention)
    assert {(r['rows'], r['keys'], r['case']) for r in attention} == {
        (4,16,'normal'), (1,1280,'normal'), (16,1280,'normal'), (16,1280,'mask_boundaries')}

    baseline = bind('records/batch110_baseline_profile.json')
    verify(baseline['source_and_binary_sha256'])
    for case in baseline['cases']:
        evidence(case, 'aborted')
    baseline_summary = [{k: c[k] for k in ('dimension', 'batch', 'elapsed_seconds', 'application_bytes', 'completed')}
                        for c in baseline['cases']]
    candidate = []
    for path in sorted(ROOT.glob('records/batch110_matrix_candidate_*.json')):
        r = bind(str(path.relative_to(ROOT))); evidence(r, 'closed')
        assert r['candidate_only'] and not r['production_enabled']
        candidate.append({'record': str(path.relative_to(ROOT)), 'rows': r['rows'], 'inner': r['inner'],
            'columns': r['columns'], 'application_bytes': r['cases'][0]['application_bytes'],
            'seconds': r['cases'][0]['seconds'], 'candidate_only': True})

    stage = {'kind': 'verified_reusable_numeric_kernel_stage_not_full_R2', 'native_version': '1.0.11',
        'native_library_sha256': library, 'worker_sha256': sha('build/compute/prag_worker'),
        'archived_v110_manifest_files_verified': len(old_manifest['files']), 'unchanged_BFV_OT_source_files': unchanged,
        'historical_runtime_identity': 'v110 and nested v109 runs retain original source and library identities; not relabeled current',
        'rms_runs': rms, 'primitive_record': primitive_path, 'primitive_cases': 29,
        'failure_record': failure_path, 'failure_modes': 13, 'attention_runs': attention,
        'pre_callback_fix_archive': previous, 'pre_callback_fix_manifest_files_verified': len(prior_manifest['files']),
        'old_scalar_actual_profiles': baseline_summary, 'old_scalar_unreached': baseline['unreached_phases'],
        'prior_candidate_matrix_preprocessing': candidate, 'matrix_preprocessing_integrated': True, 'matrix_preprocessing_authorization': 'DEC-015',
        'profile_units': 'C-side application bytes count both directions once; excludes TLS/TCP/IP and harness reconstruction',
        'profile_aggregation': 'Each phase includes nested phases. Do not sum parent and children. Repeated same-name NR phases are aggregated.',
        'material_units': 'batch_input_OT and ot_C_to_S/ot_S_to_C count choices, not RTT. Base OT setup is distinct. BatchOps GC counters in primitive/candidate probes do not count internal BFV GC; use BFV decoder coordinate reservations and its pinned circuit bound.',
        'full_layer_R2': False, 'full_Decode_R2': False, 'full_model_R2': False, 'private_GPU_backend': False,
        'CPU_model_executed': False, 'full_system_production_enabled': False, 'formal_acceptance_defined': False,
        'proof_scope': 'local derivation, source review and finite actual runs; not independent audit',
        'unmeasured_current_scope': ['full layer/Decode', 'SwiGLU and RoPE batching', 'L2', 'private retrieval',
                                    'full fixed-budget query and all session overhead'],
        'proof_document': 'records/batch111_implementation.md'}
    write('records/batch111_stage.json', stage); bind('records/batch111_stage.json')
    from tools.build_batch111_cost import main as build_cost
    build_cost()
    bind('records/batch111_cost_scope.json')
    files.add('tools/build_batch111_cost.py')

    bound = old_bound.copy()
    bound.update(native_interface_version='prag-native-v1.0.11',
        execution_provenance='BFV source and bound unchanged from v109/v110; matrix triple composition and current batch runs separately bound',
        historical_source_archive=archive)
    write('artifacts/bfv_correctness_bound.json', bound)
    files.add('artifacts/bfv_correctness_bound.json')
    files.update(['records/batch111_implementation.md', 'records/matrix_triple_bfv_candidate.md', 'records/batch110_stage.json', 'tools/record_batch111_stage.py',
        'records/crypto_pack109_implementation.md', 'records/crypto-bfv-correctness.md', 'records/crypto-dependencies.json',
        'artifacts/bfv_context.json', 'tools/build_batch111_evidence.py', 'tools/build_crypto.sh',
        'experiments/batch111/CMakeLists.txt', 'native/src/batch.cpp', 'native/src/batch_matrix.inc', 'native/include/prag/batch.hpp',
        'experiments/batch110_baseline/run.py'])
    manifest = {'artifact': 'ART-002-native-materialization', 'native_interface_version': 'prag-native-v1.0.11',
        'native_primitive_enabled': True, 'full_system_production_enabled': False,
        'current_batch_stage': 'records/batch111_stage.json', 'current_library_sha256': library,
        'BFV_source_identity_unchanged': True, 'historical_v110_archive': archive,
        'historical_v110_runtime_records': historical_runs, 'historical_runs_are_current_library_runs': False,
        'hash_algorithm': 'SHA-256', 'files': {p: sha(p) for p in sorted(files)}}
    write('artifacts/bfv_native_manifest.json', manifest)
    print(json.dumps({'native_version': '1.0.11', 'manifest_files': len(files), 'complete_RMS_runs': 9,
        'complete_attention_runs': 4, 'incomplete_attention_runs': 0, 'full_R2': False}))


if __name__ == '__main__':
    main()
