"""Materialize the current hybrid build identity without relabeling old runs.

No model is executed. Historical protocol bounds apply only after their exact
source/context prerequisites are verified. Active long runs retain independent
registrations and are never treated as completed by this command.
"""
from pathlib import Path
import hashlib
import json
import tarfile

from tools.build_hybrid112_evidence import main as build_evidence, read, sha

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    build_evidence()
    baseline = 'records/v111_source_snapshot.tar.gz'
    with tarfile.open(ROOT / baseline) as tar:
        old = json.load(tar.extractfile('artifacts/bfv_native_manifest.json'))
        for path, expected in old['files'].items():
            stream = tar.extractfile(path)
            require(stream is not None and hashlib.sha256(stream.read()).hexdigest() == expected,
                    'BASELINE_ARCHIVE_CHANGED: ' + path)
        bound = json.load(tar.extractfile('artifacts/bfv_correctness_bound.json'))
        # Retain the existing artifact's version/provenance: it is a conditional
        # bound from the baseline, not a claim that old runs used the new binary.
        require(read('artifacts/bfv_correctness_bound.json') == bound, 'BASELINE_BOUND_CHANGED')
    native_paths = [p for p in old['files'] if p.startswith(
        ('native/src/', 'native/include/prag/', 'third_party/'))]
    changed = sorted(p for p in native_paths if sha(p) != old['files'][p])
    require(changed == ['native/include/prag/batch.hpp', 'native/src/batch.cpp'],
            'UNREVIEWED_CRYPTO_SOURCE_CHANGE')
    for path, digest in bound['source_sha256'].items():
        require(sha(path) == digest, 'BOUND_PREREQUISITE_CHANGED: ' + path)
    require(read('records/hybrid112_actual_context.json') == read('artifacts/bfv_context.json'),
            'LOCKED_CONTEXT_CHANGED')
    cache = (ROOT / 'build/compute112release/CMakeCache.txt').read_text()
    for flag in ('PRAG_BUILD_HYBRID_DEV', 'PRAG_BUILD_ISOLATED_DEVELOPMENT', 'PRAG_BUILD_MODEL_DEV'):
        require(flag + ':BOOL=OFF' in cache, 'DEVELOPMENT_BUILD_FLAG: ' + flag)
    production = read('records/hybrid112_production_build.json')
    for path, digest in production['source_and_binary_sha256'].items():
        require(sha(path) == digest, 'CURRENT_BUILD_SOURCE_CHANGED: ' + path)
    require('VERSION 1.0.12' in (ROOT / 'native/CMakeLists.txt').read_text(), 'NATIVE_VERSION_CHANGED')
    library = sha('build/compute112release/native/libprag_crypto.a')
    require(library == sha('build/compute112m3/native/libprag_crypto.a'), 'PROTOCOL_LIBRARY_CHANGED')
    hybrid = read('artifacts/hybrid112_evidence.json')
    stage = {
        'kind': 'current_hybrid_build_and_finite_evidence_not_full_R2',
        'native_interface_version': 'prag-native-v1.0.12',
        'actual_context_matches_locked': True,
        'baseline_archive': baseline, 'baseline_archive_sha256': sha(baseline),
        'baseline_manifest_files_verified': len(old['files']),
        'unchanged_BFV_OT_GC_dependency_sources': sorted(set(native_paths) - set(changed)),
        'changed_batch_interface_sources': changed,
        'change_scope': 'GC opcode 14: bound public table, unsigned secret-position selection and validity; BFV/OT and matrix triple algebra unchanged',
        'bound_artifact_keeps_original_version': bound['native_interface_version'],
        'conditional_bound_applies_to_current_BFV_source': True,
        'current_crypto_library_sha256': library,
        'current_worker_sha256': sha('build/compute112release/prag_worker'),
        'production_build': 'records/hybrid112_production_build.json',
        'finite_evidence': 'artifacts/hybrid112_evidence.json',
        'historical_runs_are_current_library_runs': False,
        'M1_complete_finite_shapes': hybrid['M1_complete_finite_shapes'],
        'M2_complete': hybrid['M2_complete'], 'M3_complete': hybrid['M3_complete'],
        'full_R2_complete': False, 'production_enabled': False,
        'formal_acceptance_defined': False,
        'resource_policy': hybrid['resource_policy'],
        'scope_limit': 'compiled graph plus finite separately archived executions; no exhaustive model-domain or whole-session security claim',
    }
    (ROOT / 'records/hybrid112_materialization.json').write_text(json.dumps(stage, ensure_ascii=False, indent=2) + '\n')
    files = set(native_paths)
    files.update(production['source_and_binary_sha256'])
    for folder in ('compute', 'private_rag'):
        files.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.hpp', '.cpp', '.inc', '.cu', '.sql', '.txt'))
    files.update([
        baseline, 'native/CMakeLists.txt', 'records/hybrid112_materialization.json',
        'records/hybrid112_actual_context.json', 'build/compute112release/native/prag_bfv_context',
        'build/compute112release/CMakeCache.txt', 'records/hybrid112_production_build.json',
        'artifacts/hybrid112_evidence.json', 'artifacts/bfv_context.json',
        'artifacts/bfv_correctness_bound.json', 'records/hybrid112_implementation.md',
        'records/module_contracts.md', 'records/crypto-dependencies.json',
        'tools/build_hybrid112_manifest.py', 'tools/build_hybrid112_evidence.py',
        'tools/build_hybrid.py', 'tools/build_crypto.sh',
        'tools/hybrid112_status.py', 'tools/hybrid112_schedule.py',
        'records/hybrid112_public_schedule.json',
        'tools/check_m2_completion.py', 'records/hybrid112_M2_closure.md',
        'records/hybrid112_M2_closure.json',
        'experiments/hybrid112/registered_build.py',
        'experiments/hybrid112/run_layer.py', 'experiments/hybrid112/run_decode.py',
        'experiments/hybrid112/run_m3.py',
    ])
    for record in hybrid['completed'] + hybrid['pending']:
        for key in ('record', 'registration', 'execution_source_archive'):
            if key in record:
                files.add(record[key])
    manifest = {
        'artifact': 'ART-002-native-materialization',
        'native_interface_version': 'prag-native-v1.0.12',
        'native_primitive_enabled': True, 'full_system_production_enabled': False,
        'current_hybrid_stage': 'records/hybrid112_materialization.json',
        'current_library_sha256': library, 'BFV_source_identity_unchanged': True,
        'historical_v111_archive': baseline, 'historical_runs_are_current_library_runs': False,
        'hash_algorithm': 'SHA-256', 'files': {p: sha(p) for p in sorted(files)},
    }
    (ROOT / 'artifacts/bfv_native_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'native_version': '1.0.12', 'manifest_files': len(files),
                     'actual_context_matches_locked': True, 'full_R2_complete': False}))


if __name__ == '__main__':
    main()
