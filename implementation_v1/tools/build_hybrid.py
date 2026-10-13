"""Build into a fresh directory, preserving active runs and historical binaries.

The default builds the closed production worker and BFV context exporter. The
explicit development option builds isolated probes too, without running them.
No weights, constants, configuration or recorded model result are overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', help='new direct child of build/; existing directories are rejected')
    parser.add_argument('--development', action='store_true', help='also compile isolated hybrid probes; execute no model')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    name = args.name or 'hybrid-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[a-z][a-z0-9_-]{0,80}', name):
        parser.error('name must be a simple lowercase directory name')
    for program in ('cmake', 'nvcc', 'nm'):
        if shutil.which(program) is None:
            parser.error('missing build dependency: ' + program)
    if not (ROOT / 'third_party/SEAL/build/cmake/SEALConfig.cmake').is_file():
        parser.error('locked SEAL build missing; build the pinned dependency first')
    (ROOT / 'build').mkdir(exist_ok=True)
    build = ROOT / 'build' / name
    # No existing build may be selected, even if its process appears idle.
    build.mkdir(parents=False, exist_ok=False)
    record = {'build_directory': str(build.relative_to(ROOT)), 'development': args.development,
              'model_executed': False, 'production_enabled': False, 'jobs': args.jobs,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'result': 'incomplete'}
    inputs = {Path(__file__).resolve()}
    for folder in ('compute', 'native/src', 'native/include/prag'):
        inputs.update(p for p in (ROOT / folder).rglob('*') if p.is_file()
                      and p.suffix in ('.hpp', '.cpp', '.inc', '.cu', '.txt'))
    inputs.add(ROOT / 'native/CMakeLists.txt')
    if args.development:
        for folder in ('experiments/hybrid112', 'experiments/batch110_baseline'):
            inputs.update(p for p in (ROOT / folder).rglob('*') if p.is_file()
                          and p.suffix in ('.hpp', '.cpp', '.inc', '.cu', '.txt'))
    for dependency, folder in [('SEAL', 'native/src'), ('EzPC', 'SCI/src')]:
        repo = ROOT / 'third_party' / dependency
        tracked = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z', folder],
                                 capture_output=True, check=True).stdout
        inputs.update(repo / path.decode() for path in tracked.split(b'\0') if path)
    inputs.add(ROOT / 'third_party/SEAL/build/native/src/seal/util/config.h')
    inputs.update((ROOT / 'third_party/SEAL/build').rglob('libseal*.a'))
    before = {str(path.relative_to(ROOT)): sha(path) for path in sorted(inputs)}
    record['source_and_dependency_sha256'] = before
    configure = ['cmake', '-S', str(ROOT / 'compute'), '-B', str(build),
                 '-DCMAKE_BUILD_TYPE=Release', '-DPRAG_BUILD_ISOLATED_DEVELOPMENT=OFF',
                 '-DPRAG_BUILD_MODEL_DEV=OFF',
                 '-DPRAG_BUILD_HYBRID_DEV=' + ('ON' if args.development else 'OFF')]
    host_compiler = Path('/usr/lib/nvidia-cuda-toolkit/bin/g++')
    if host_compiler.is_file():
        configure.append('-DCMAKE_CUDA_HOST_COMPILER=' + str(host_compiler))
    # CMake checks pinned dependency commits and clean tracked dependency files.
    try:
        with (build / 'configure.log').open('w') as log:
            subprocess.run(configure, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        with (build / 'build.log').open('w') as log:
            subprocess.run(['cmake', '--build', str(build), '-j' + str(args.jobs)],
                           cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        context = build / 'actual_bfv_context.json'
        subprocess.run([str(build / 'native/prag_bfv_context'), str(context)], check=True,
                       capture_output=True, text=True)
        if json.loads(context.read_text()) != json.loads((ROOT / 'artifacts/bfv_context.json').read_text()):
            raise RuntimeError('BFV_CONTEXT_MISMATCH')
        # Exactly the normal argc, invalid unopened descriptors/paths. The
        # source-controlled gate must reject before any input interpretation.
        record['gate_cases'] = []
        for party in ('C', 'S'):
            for mode in ('prepare', 'query'):
                gate = subprocess.run([str(build / 'prag_worker'), party, mode, *(['unused'] * 19)],
                                      capture_output=True, text=True, timeout=10, check=False)
                record['gate_cases'].append({'party': party, 'mode': mode, 'exit_code': gate.returncode,
                                            'stdout_empty': not gate.stdout, 'stderr': gate.stderr.strip()})
                if gate.returncode != 2 or gate.stdout or gate.stderr.strip() != 'BACKEND_NOT_READY':
                    raise RuntimeError('CLOSED_PRODUCTION_GATE_EXPECTED')
        crypto = build / 'native/libprag_crypto.a'
        symbols = subprocess.run(['nm', '-C', str(crypto)], capture_output=True, text=True, check=True).stdout
        if 'development_bfv59_decode' in symbols:
            raise RuntimeError('DEVELOPMENT_DECODE_IN_PRODUCTION_LIBRARY')
        if any(sha(ROOT / path) != digest for path, digest in before.items()):
            raise RuntimeError('SOURCE_CHANGED_DURING_BUILD')
        outputs = [build / 'prag_worker', crypto, build / 'libprag_local_cuda.a',
                   build / 'native/prag_bfv_context', context]
        if args.development:
            outputs.extend(build / name for name in ('prag_hybrid_local', 'prag_hybrid_layer',
                           'prag_hybrid_decode', 'prag_hybrid_postprocess', 'prag_hybrid_m3'))
        record.update(result='compiled_context_matched_gate_closed',
                      actual_context_matches_locked=True, rejects_before_input=True,
                      production_library_excludes_development_decode=True,
                      output_sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs})
    except BaseException as exc:
        record['failure_type'] = type(exc).__name__
        raise
    finally:
        record['configure_argv'] = configure
        record['tool_sha256'] = before[str(Path(__file__).resolve().relative_to(ROOT))]
        (build / 'build_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'build_directory': record['build_directory'], 'result': record['result'],
                          'model_executed': False, 'production_enabled': False}), flush=True)


if __name__ == '__main__':
    main()
