"""Isolated candidate only: full-ring BFV matrix triples, synthetic CUDA oracle."""
import hashlib
import json
import struct
import subprocess
import tempfile
import uuid
from pathlib import Path

import torch
from private_rag.pki import create_local_pki
from references.resource_policy import require_model_device

ROOT = Path(__file__).resolve().parents[2]
MASK = (1 << 59) - 1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(rows, inner, columns):
    device = require_model_device('R1')
    x = ((torch.arange(rows * inner, device=device) % 37) - 18).reshape(rows, inner) * 257
    y = ((torch.arange(inner * columns, device=device) % 41) - 20).reshape(inner, columns) * 113
    # Each integer product and the absolute sum are < 2^53; FP64 dot is exact.
    assert inner * 18 * 257 * 20 * 113 < 1 << 53
    wanted = (x.to(torch.float64) @ y.to(torch.float64)).to(torch.int64) & MASK
    words = torch.cat([x.flatten(), y.flatten()]) & MASK
    fixture_bytes = struct.pack('<11Q', 1, 101, rows, columns, inner, 0, 0, words.numel(), 0, wanted.numel(), 0)
    fixture_bytes += words.cpu().numpy().astype('<u8').tobytes()
    fixture_bytes += wanted.cpu().numpy().astype('<u8').tobytes()
    prefix = ROOT / f'records/batch110_matrix_candidate_{rows}_{inner}_{columns}_{uuid.uuid4().hex}'
    source_paths = ['experiments/batch110/candidate_matrix_probe.cpp', 'experiments/batch110/matrix_triple_candidate.hpp',
                    'experiments/batch110/run_candidate.py', 'experiments/batch110/build/prag_batch110_candidate_matrix',
                    'experiments/batch110/build/native/libprag_crypto.a', 'native/src/bfv.cpp',
                    'native/src/bfv_matrix.inc', 'native/src/bfv_tile_decoder.inc', 'native/src/secure_ops.cpp']
    identity = {p: sha(ROOT / p) for p in source_paths}
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        create_local_pki(temp / 'pki')
        fixture = temp / 'fixture.bin'
        fixture.write_bytes(fixture_bytes)
        result = subprocess.run([str(ROOT / source_paths[3]), str(prefix), str(temp / 'pki'), str(fixture)],
                                capture_output=True, text=True, timeout=330)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = {'result': 'incomplete_or_failed', 'stdout': result.stdout}
    report.update(exit_code=result.returncode, stderr=result.stderr, rows=rows, inner=inner, columns=columns,
                  candidate_only=True, production_enabled=False, full_model_R2=False, private_GPU_backend=False,
                  oracle='CUDA exact integer via bounded FP64', device=torch.cuda.get_device_name(device),
                  fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(), source_sha256=identity,
                  BFV_secret_key_owner='C', preparation='two cross products using existing BFV private_matrix',
                  deadline_seconds=300)
    assert identity == {p: sha(ROOT / p) for p in source_paths}, 'source changed during execution'
    report['ledgers'] = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in
                         ROOT.glob(str(prefix.relative_to(ROOT)) + '*.ledger')]
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'record': str(prefix.relative_to(ROOT)) + '.json', **{k: v for k, v in report.items()
          if k not in ('source_sha256', 'ledgers', 'stdout')}}), flush=True)
    return result.returncode == 0


if __name__ == '__main__':
    import sys
    if not run(*(int(v) for v in sys.argv[1:])):
        raise SystemExit(1)
