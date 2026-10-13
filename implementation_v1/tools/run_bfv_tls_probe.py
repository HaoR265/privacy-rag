"""Run a bounded public BFV diagnostic using the production library and mTLS.

This entry creates no model, management session or persistent key. It is an
implementation debugging helper, not a formal test or production RAG entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import uuid

from private_rag.pki import create_local_pki

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', default='native/build')
    parser.add_argument('--target', default='prag_production_bfv_packed_probe',
                        choices=['prag_production_bfv_packed_probe',
                                 'prag_production_bfv_merged_probe'])
    parser.add_argument('--record', required=True,
                        help='New record filename in records/, without directory')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+\.json', args.record):
        parser.error('record must be a simple JSON filename')
    destination = ROOT / 'records' / args.record
    if destination.exists():
        parser.error('refusing to overwrite a previous execution record')
    build = (ROOT / args.build_dir).resolve()
    build.relative_to(ROOT)
    binary = build / args.target
    sources = [ROOT / 'native/CMakeLists.txt', ROOT / 'native/src/bfv.cpp',
               ROOT / 'native/include/prag/bfv.hpp',
               ROOT / 'native/include/prag/secure_ops.hpp',
               Path(__file__).resolve(), build / 'libprag_crypto.a', binary]
    probe_source = ('dev_bfv_packed_probe.cpp' if 'packed' in args.target
                    else 'dev_bfv_merged_probe.cpp')
    sources.append(ROOT / 'native/src' / probe_source)
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    prefix = ROOT / 'records' / ('bfv-tls-' + uuid.uuid4().hex)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='prag-bfv-tls-') as temporary:
        pki = Path(temporary) / 'pki'
        create_local_pki(pki)
        process = subprocess.Popen(['nice', '-n', '10', str(binary), str(prefix), str(pki)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=180)
        except BaseException:
            # Stop only this diagnostic's dedicated process group, including
            # its forked peer; never leave a peer blocked on a closed channel.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise
    if process.returncode:
        raise RuntimeError(f'BFV diagnostic exit {process.returncode}: {stderr}')
    result = json.loads(stdout)
    if (result['isolated_development'] or not result['native_TLS13_mTLS']
            or result['full_R2'] or result['GPU_used']):
        raise RuntimeError('Unexpected diagnostic execution scope')
    after = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if before != after:
        raise RuntimeError('Diagnostic source or binary changed during execution')
    ledgers = []
    for party in ('C', 'S'):
        path = Path(str(prefix) + f'.{party}.ledger')
        lines = path.read_text().splitlines()
        if lines[-1] != 'terminal=closed all_reserved=burned':
            raise RuntimeError('Diagnostic ledger did not close and burn')
        counts = {}
        for line in lines:
            match = re.fullmatch(r'\d+ reserved (\S+) \d+ ring59', line)
            if match:
                name = match.group(1)
                counts[name] = counts.get(name, 0) + 1
        ledgers.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                        'terminal': lines[-1], 'reservation_counts': counts})
    if ledgers[0]['reservation_counts'] != ledgers[1]['reservation_counts']:
        raise RuntimeError('Diagnostic reservation schedules differ')
    result.update(scope='Public synthetic two-party BFV diagnostic only; no model or formal acceptance',
                  elapsed_seconds=time.monotonic() - started,
                  exit_code=process.returncode, temporary_pki_removed=not pki.exists(),
                  ledger_checks=ledgers, source_and_binary_sha256=after,
                  source_and_binary_unchanged_during_run=True,
                  child_process_waited_by_probe=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
