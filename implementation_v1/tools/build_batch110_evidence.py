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

    def one(pattern):
        paths = list(ROOT.glob(pattern))
        if len(paths) != 1:
            raise RuntimeError('Expected unique evidence: ' + pattern)
        return str(paths[0].relative_to(ROOT))

    def phases(record):
        path = next(e['path'] for e in record['evidence'] if e['path'].endswith('.C.trace.jsonl'))
        rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
        phases = {}
        for row in rows:
            if row['event'] == 'begin':
                continue
            group = phases.setdefault(row['phase'], {'occurrences': 0, 'seconds': 0.0, 'application_bytes': 0,
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
    archive = 'records/v109_source_snapshot.tar.gz'
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
                     p.startswith(('native/src/', 'native/include/prag/', 'third_party/'))]
        verify({p: old_manifest['files'][p] for p in unchanged})
        historical_runs = [p for p in old_manifest['files'] if p.startswith('records/') and p.endswith('.json')]
        verify({p: old_manifest['files'][p] for p in historical_runs})

    libraries = ['native/build/libprag_crypto.a', 'build/compute/native/libprag_crypto.a',
                 'experiments/batch110/build/native/libprag_crypto.a']
    library = sha(libraries[0])
    if len({sha(p) for p in libraries}) != 1:
        raise RuntimeError('Worker/probe/current native libraries differ')
    files.update(libraries)
    if read('records/batch110_actual_context.json') != read('artifacts/bfv_context.json'):
        raise RuntimeError('Actual BFV context changed')
    policy = bind('records/private_model_device_policy.json')
    gate = policy['native_compilation_and_runtime_gate_check']
    assert gate['worker_sha256'] == sha('build/compute/prag_worker')
    assert gate['public_stderr'] == 'BACKEND_NOT_READY' and gate['rejected_before_input_or_sockets']
    assert gate['production_library_excludes_development_decode_symbol']
    build_paths = libraries + ['build/compute/prag_worker', 'native/build/prag_bfv_context',
        'records/batch110_actual_context.json', 'records/batch110_native_configure.log',
        'records/batch110_native_build.log', 'records/batch110_compute_configure.log',
        'records/batch110_compute_build.log', 'compute/model.hpp', 'compute/numerics.hpp',
        'compute/batch_numerics.hpp', 'compute/CMakeLists.txt', 'native/CMakeLists.txt']
    build = {'scope': 'current builds and closed gate only', 'native_version': '1.0.10',
        'worker_crypto_library_matches_mtls_library': True, 'actual_context_matches_locked': True,
        'model_executed': False, 'full_R2': False, 'jobs_per_build': 1, 'current_gate_probe': gate,
        'source_and_binary_sha256': {p: sha(p) for p in build_paths}}
    write('records/compute_batch110_build.json', build)
    bind('records/compute_batch110_build.json')
    verify(build['source_and_binary_sha256'])

    rms = []
    for path in sorted(ROOT.glob('records/batch110_rms_*.json')):
        record = read(path)
        if not record.get('production_Numerics_rms_many'):
            continue
        evidence(record, 'closed')
        assert record['result'] == 'matched_CUDA_R1' and record['native_version'] == '1.0.10'
        assert record['source_and_binary_sha256'][libraries[2]] == library
        name = str(path.relative_to(ROOT)); bind(name)
        rms.append({'record': name, **{k: record[k] for k in ('dimension', 'batch', 'case', 'application_bytes',
            'elapsed_seconds', 'reference_failed', 'tensor', 'weight_sha256')}, 'phases_inclusive': phases(record)})
    required = {(d, b, c) for d in (128, 1024, 2048) for b, c in ((1, 'normal'), (16, 'normal'), (16, 'boundaries'))}
    assert {(r['dimension'], r['batch'], r['case']) for r in rms} == required and len(rms) == 9

    primitive_path = one('records/batch110_primitives_bd8599f79e5c489891c29b4dc40e6259.json')
    primitive = bind(primitive_path); evidence(primitive, 'closed')
    assert len(primitive['cases']) == 27 and all(c['matched'] for c in primitive['cases'])
    failure_path = one('records/batch110_failures_d2b850d83a14478c917f9f751950b90a.json')
    failures = bind(failure_path)
    evidence(failures, lambda p: 'closed' if p.endswith(('_4.C.ledger', '_4.S.ledger')) else 'aborted')
    assert len(failures['cases']) == 16 and all(c['expected_rejection'] for c in failures['cases'])
    attention_paths = [one('records/batch110_attention_' + suffix + '.json') for suffix in (
        '4_16_normal_de63f9c6bc984c92b92ee92df1dc05e9',
        '4_16_mask_boundaries_d787d1f652a74bcd9bdfd1d9c65a2abe',
        '1_1280_normal_373f55b84fd348df9864aaa0c9e3859f',
        '16_1280_mask_boundaries_dd511c3e8cb442b6ba539456339fb9a9')]
    attention = []
    for path in attention_paths:
        record = bind(path); complete = record['result'] == 'matched_CUDA_R1'
        evidence(record, 'closed' if complete else 'aborted')
        attention.append({'record': path, 'complete': complete, 'rows': record['rows'], 'keys': record['keys'],
            'case': record['case'], 'application_bytes': record['application_bytes'],
            'seconds': record.get('seconds', record.get('elapsed_seconds')), 'reference_failed': record['reference_failed'],
            'phases_inclusive': phases(record)})
    assert sum(r['complete'] for r in attention) == 3

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

    stage = {'kind': 'verified_reusable_numeric_kernel_stage_not_full_R2', 'native_version': '1.0.10',
        'native_library_sha256': library, 'worker_sha256': sha('build/compute/prag_worker'),
        'archived_v109_manifest_files_verified': len(old_manifest['files']), 'unchanged_BFV_OT_source_files': unchanged,
        'historical_runtime_identity': 'v109 runs remain v109; archive holds matching sources/binaries',
        'rms_runs': rms, 'primitive_record': primitive_path, 'primitive_cases': 27,
        'failure_record': failure_path, 'failure_modes': 8, 'attention_runs': attention,
        'old_scalar_actual_profiles': baseline_summary, 'old_scalar_unreached': baseline['unreached_phases'],
        'candidate_matrix_preprocessing': candidate, 'candidate_production_enabled': False,
        'profile_units': 'C-side application bytes count both directions once; excludes TLS/TCP/IP and harness reconstruction',
        'profile_aggregation': 'Each phase includes nested phases. Do not sum parent and children. Repeated same-name NR phases are aggregated.',
        'material_units': 'batch_input_OT and ot_C_to_S/ot_S_to_C count choices, not RTT. Base OT setup is distinct. BFV candidate BatchOps GC counters do not count internal BFV GC.',
        'full_layer_R2': False, 'full_Decode_R2': False, 'full_model_R2': False, 'private_GPU_backend': False,
        'CPU_model_executed': False, 'full_system_production_enabled': False, 'formal_acceptance_defined': False,
        'proof_scope': 'local derivation, source review and finite actual runs; not independent audit',
        'unmeasured_current_scope': ['full layer/Decode', 'SwiGLU and RoPE batching', 'L2', 'private retrieval',
                                    'full fixed-budget query and all session overhead'],
        'proof_document': 'records/batch110_implementation.md'}
    write('records/batch110_stage.json', stage); bind('records/batch110_stage.json')

    bound = old_bound.copy()
    bound.update(native_interface_version='prag-native-v1.0.10',
        execution_provenance='BFV mathematical/source bound unchanged from archived v109; historical BFV/Token runtime records are v109, current batch runtime records separately bound',
        historical_source_archive=archive)
    write('artifacts/bfv_correctness_bound.json', bound)
    files.add('artifacts/bfv_correctness_bound.json')
    files.update(['records/batch110_implementation.md', 'records/matrix_triple_bfv_candidate.md',
        'records/crypto_pack109_implementation.md', 'records/crypto-bfv-correctness.md', 'records/crypto-dependencies.json',
        'artifacts/bfv_context.json', 'tools/build_batch110_evidence.py', 'tools/build_crypto.sh',
        'experiments/batch110/CMakeLists.txt', 'native/src/batch.cpp', 'native/include/prag/batch.hpp',
        'experiments/batch110_baseline/run.py'])
    manifest = {'artifact': 'ART-002-native-materialization', 'native_interface_version': 'prag-native-v1.0.10',
        'native_primitive_enabled': True, 'full_system_production_enabled': False,
        'current_batch_stage': 'records/batch110_stage.json', 'current_library_sha256': library,
        'BFV_source_identity_unchanged': True, 'historical_v109_archive': archive,
        'historical_v109_runtime_records': historical_runs, 'historical_runs_are_current_library_runs': False,
        'hash_algorithm': 'SHA-256', 'files': {p: sha(p) for p in sorted(files)}}
    write('artifacts/bfv_native_manifest.json', manifest)
    print(json.dumps({'native_version': '1.0.10', 'manifest_files': len(files), 'complete_RMS_runs': 9,
        'complete_attention_runs': 3, 'incomplete_attention_runs': 1, 'full_R2': False}))


if __name__ == '__main__':
    main()
