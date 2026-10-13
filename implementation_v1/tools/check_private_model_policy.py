"""Verify the current Python/native closed GPU model gates without any model.

The native call uses valid argument count and role but no real input, file,
certificate or socket. It must reject before interpreting those arguments.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from private_rag.activation import report, require_private_backend
from private_rag.errors import ErrorCode, PublicError

ROOT = Path(__file__).resolve().parents[1]


def main():
    policy_path = ROOT / 'records/private_model_device_policy.json'
    policy = json.loads(policy_path.read_text())
    activation = report()
    if activation['production_enabled'] or activation['cpu_model_fallback']:
        raise RuntimeError('Expected an unavailable private GPU backend')
    try:
        require_private_backend()
    except PublicError as error:
        if error.code != ErrorCode.BACKEND_NOT_READY:
            raise
    else:
        raise RuntimeError('Python model gate did not reject')
    worker = ROOT / 'build/compute/prag_worker'
    result = subprocess.run([str(worker), 'C', 'query'] + ['unused'] * 19,
                            capture_output=True, text=True, timeout=5)
    if (result.returncode, result.stdout, result.stderr.strip()) != (2, '', 'BACKEND_NOT_READY'):
        raise RuntimeError('Native model gate did not reject before input parsing')
    library = ROOT / 'native/build/libprag_crypto.a'
    symbols = subprocess.run(['nm', '-C', str(library)],
                             capture_output=True, text=True, check=True).stdout
    if 'development_bfv59_decode' in symbols:
        raise RuntimeError('Production library contains the development decode entry')
    policy['activation'] = activation
    policy['native_compilation_and_runtime_gate_check'] = {
        'build': 'existing_binary_runtime_gate_checked', 'build_model_executed': False,
        'worker_path': str(worker.relative_to(ROOT)), 'worker_bytes': worker.stat().st_size,
        'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
        'exit_code': result.returncode, 'public_stderr': result.stderr.strip(),
        'rejected_before_input_or_sockets': True,
        'production_library_excludes_development_decode_symbol': True,
        'probe': str(Path(__file__).resolve().relative_to(ROOT)),
    }
    for path in policy['source_sha256']:
        policy['source_sha256'][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    print(json.dumps({'production_enabled': False, 'model_executed': False,
                      **policy['native_compilation_and_runtime_gate_check']}))


if __name__ == '__main__':
    main()
