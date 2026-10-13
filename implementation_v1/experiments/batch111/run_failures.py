"""Public protocol-rejection diagnostics; never reconstructs secret inputs."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import uuid
from private_rag.pki import create_local_pki

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    prefix = ROOT / ('records/batch111_failures_' + uuid.uuid4().hex)
    with tempfile.TemporaryDirectory() as temp:
        cert = Path(temp) / 'pki'
        create_local_pki(cert)
        cp = subprocess.run([str(ROOT / 'experiments/batch111/build/prag_batch111_failure'),
                             str(prefix), str(cert), 'unused'], capture_output=True, text=True, timeout=280)
    result = {'exit_code': cp.returncode, 'cases': [json.loads(line) for line in cp.stdout.splitlines()],
              'stderr': cp.stderr, 'full_model_R2': False}
    result['files'] = {p: sha(p) for p in ('experiments/batch111/run_failures.py',
        'experiments/batch111/failure_probe.cpp', 'experiments/batch111/build/prag_batch111_failure',
        'experiments/batch111/build/native/libprag_crypto.a', 'native/src/batch.cpp',
        'native/src/batch_matrix.inc', 'native/include/prag/batch.hpp')}
    result['ledgers'] = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
                         for p in ROOT.glob(str(prefix.relative_to(ROOT)) + '*.ledger')]
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'record': str(prefix.relative_to(ROOT)) + '.json', 'exit_code': cp.returncode,
                      'party_results': len(result['cases']), 'stderr': cp.stderr}), flush=True)
    assert cp.returncode == 0 and len(result['cases']) == 26 and all(c['expected_rejection'] for c in result['cases'])


if __name__ == '__main__':
    main()
