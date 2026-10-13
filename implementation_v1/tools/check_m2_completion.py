"""Recheck one completed M2 execution without running a model.

Requires the original archive, ledgers and traces, which are not distributed
with a source-only checkout. This verifies recorded evidence; it is neither a
new R2 execution nor a general security proof or formal acceptance suite.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from tools.build_hybrid112_evidence import (
    M2, ROOT, read, sha, verify_archive, verify_finished_record,
)
from tools.hybrid112_schedule import check_final_trace, plan

ARCHIVE = 'records/hybrid112_M2_running_source_snapshot.tar.gz'


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def inspect_ledger(path, session, configuration):
    """Require a single bound, contiguous reservation stream and final close."""
    expected = f'format=1 session={session} configuration={configuration}\n'
    counts = Counter()
    entries = 0
    with path.open() as stream:
        require(stream.readline() == expected, 'LEDGER_BINDING')
        for line in stream:
            if line == 'terminal=closed all_reserved=burned\n':
                require(stream.read() == '', 'LEDGER_CONTENT_AFTER_CLOSE')
                require(entries > 0, 'LEDGER_NO_RESERVATIONS')
                return {'reservations': entries, 'first_sequence': 0,
                        'last_sequence': entries - 1, 'kinds': dict(counts),
                        'contiguous_unique_sequence': True,
                        'terminal': 'closed', 'all_reserved': 'burned'}
            match = re.fullmatch(r'(0|[1-9][0-9]*) reserved ([A-Za-z0-9_]+) (0|[1-9][0-9]*) ring59\n', line)
            require(match is not None, 'LEDGER_BAD_RESERVATION')
            sequence, kind, size = match.groups()
            require(int(sequence) == entries, 'LEDGER_REUSED_OR_MISSING_SEQUENCE')
            if entries == 0:
                require((kind, size) == ('session', '0'), 'LEDGER_FIRST_RESERVATION')
            counts[kind] += 1
            entries += 1
    raise RuntimeError('LEDGER_MISSING_CLOSE')


def main():
    result = read(M2 + '.json')
    registration = read(M2 + '.registration.json')
    require(all(result.get(k) == v for k, v in registration.items()), 'REGISTRATION_CHANGED')
    identities = registration['source_and_binary_sha256']
    require(hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
            == result['configuration'], 'SOURCE_CONFIGURATION_BINDING')
    verify_archive(registration, ARCHIVE)
    verify_finished_record(result)
    require(result['exit_codes'] == [0, 0] and result['stderr'] == {'C': '', 'S': ''}, 'PARTY_EXIT')
    require(result['result'] == 'matched_CUDA_R1'
            and result['final_token_and_KV_matched'] is True
            and result['failure_matched'] is True
            and result['numeric_failure'] is False, 'FINAL_COMPARISON')
    require(result['output_words'] == 1 + 28 * 2 * 8 * 128, 'FINAL_OUTPUT_SHAPE')
    require((result['layers'], result['hidden'], result['rows']) == (28, 2048, 1), 'MODEL_SHAPE')
    require(result['positions'] == [1024] and result['validity'] == [1]
            and result['fixed_prefix_slots'] == 1024
            and result['prefix_was_computed_by_R2'] is False
            and result['full_model_R2'] is False, 'DEVELOPMENT_BOUNDARY')
    ledgers = []
    for party in ('C', 'S'):
        path = M2 + '.' + party + '.ledger'
        ledgers.append({'party': party, 'path': path, 'sha256': sha(path),
                        **inspect_ledger(ROOT / path, result['session'], result['configuration'])})
    require(ledgers[0]['sha256'] == ledgers[1]['sha256'], 'PARTY_RESERVATION_MISMATCH')
    traces = check_final_trace(M2, plan('generation', 1, decode=True), token_trace=False)
    # Runtime PID provenance comes from the original controller start event.
    starts = [json.loads(line) for line in (ROOT / 'records/hybrid112_decode_live.log').read_text().splitlines()
              if line.startswith('{')]
    start = next(event for event in starts if event.get('event') == 'R2_decode_started'
                 and event.get('record') == M2)
    workers = start['workers']
    absent = all(not Path('/proc', str(pid)).exists() for pid in workers)
    # Directory identity was observed in these workers' command lines while
    # running. It is an observation, not a field invented in their final record.
    temporary = Path('/tmp/prag-hybrid112-7v7lam0t')
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(arg.endswith(b'/run_decode.py') for arg in args) or (
                args and args[0].endswith(b'/prag_hybrid_decode')
                and any(result['session'].encode() in arg for arg in args)):
            active.append(int(proc.name))
    require(absent and not temporary.exists() and not active, 'M2_CLEANUP_NOT_CONFIRMED')
    audited = ['compute/model.hpp', 'compute/artifacts.hpp', 'compute/local_shares.cu',
               'experiments/hybrid112/decode_worker.cpp', 'experiments/hybrid112/run_decode.py',
               'native/src/channel.cpp', 'native/src/secure_ops.cpp']
    report = {
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'completed M2 evidence and resource closure; no new model execution',
        'record': M2 + '.json', 'record_sha256': sha(M2 + '.json'),
        'execution_source_archive': ARCHIVE, 'archive_sha256': sha(ARCHIVE),
        'archived_identities_verified': len(identities),
        'audited_execution_source_sha256': {p: identities[p] for p in audited},
        'manual_source_audit': 'records/hybrid112_M2_closure.md',
        'final_comparison': {k: result[k] for k in ('final_token_and_KV_matched', 'failure_matched',
                             'numeric_failure', 'output_words', 'expected_sha256', 'device')},
        'ledgers': ledgers, 'public_trace_checks': traces,
        'cleanup': {'worker_pids': workers, 'workers_absent': absent,
                    'matching_active_processes': active,
                    'observed_temporary_directory': str(temporary),
                    'temporary_directory_absent': not temporary.exists(),
                    'oracle_persistence': 'archived controller uses RAM and memfd only; no raw oracle file written',
                    'GPU_release_basis': 'worker and controller exit; cache.clear precedes close in archived worker',
                    'ledger_evidence_retained': True,
                    'device_forensic_erasure_proven': False},
        'verification_boundary': 'public-input controller compares final shares only after endpoints exit; no oracle data or reconstruction enters R2',
        'prefix_was_computed_by_R2': False, 'full_R2_complete': False,
        'production_enabled': False, 'formal_acceptance_defined': False,
        'checker_sha256': sha('tools/check_m2_completion.py'),
    }
    (ROOT / 'records/hybrid112_M2_closure.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'M2_final_evidence_verified': True, 'ledger_reservations_per_party': ledgers[0]['reservations'],
                      'cleanup_confirmed': True, 'new_model_execution': False, 'full_R2_complete': False}))


if __name__ == '__main__':
    main()
