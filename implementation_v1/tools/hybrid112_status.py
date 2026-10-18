"""Read public diagnostic progress once; never open shares or resume sessions.

A recent trace is evidence of progress, not proof that a worker is still alive.
Only a finished result with closed ledgers can establish a completed milestone.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from tools.build_hybrid112_evidence import M2, M3, ROOT
from tools.public_trace import inspect_trace


def read(path):
    return json.loads(path.read_text())


def trace_status(path):
    if not path.exists():
        return {'trace_available': False}
    diagnostic = inspect_trace(path)
    completed = diagnostic['valid_prefix_ended_phases']
    stack = diagnostic['unclosed_phases_at_valid_prefix_boundary']
    token_phases = {'Token_lookup': 1, 'Token_lookup_batch_4': 4}
    tokens_traced = any(name in completed or name in stack for name in token_phases)
    unknown_token_phases = sorted(name for name in set(completed) | set(stack)
                                  if name.startswith('Token_lookup') and name not in token_phases)
    return {'trace_available': True, 'trace': str(path),
            'last_write_utc': datetime.fromtimestamp(diagnostic['initial_file_mtime_ns'] / 1e9, timezone.utc).isoformat(),
            'complete_transformer_layers': completed.get('Transformer_layer', 0),
            'complete_private_token_lookups': ('unknown_token_phase_profile' if unknown_token_phases else
                sum(completed.get(name, 0) * slots for name, slots in token_phases.items())
                if tokens_traced else 'not_traced_by_this_execution'),
            'completed_token_phase_counts': {name: completed.get(name, 0) for name in token_phases},
            'unknown_token_phases': unknown_token_phases,
            'unclosed_phases_at_valid_prefix_boundary': stack,
            'partial_last_line': diagnostic['partial_last_line'],
            'public_aborted_phases': diagnostic['observed_abort_events'],
            'structural_issues': diagnostic['structural_issues'],
            'structure_balanced': diagnostic['structure_balanced'],
            'snapshot_sha256': diagnostic['snapshot_sha256'],
            'file_changed_during_read': diagnostic['file_changed_during_read'],
            'scope': diagnostic['scope'],
            'worker_liveness': 'not_inferred_from_trace'}


def run_status(prefix):
    terminal = ROOT / (prefix + '.json')
    result = {'prefix': prefix, 'terminal_result_available': terminal.exists()}
    if terminal.exists():
        record = read(terminal)
        result['terminal_result'] = record.get('result', 'not_recorded')
    result['parties'] = {party: trace_status(ROOT / (prefix + '.' + party + '.trace.jsonl'))
                         for party in ('C', 'S')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', action='append',
                        help='record prefix relative to implementation root, or absolute; repeatable')
    args = parser.parse_args()
    if args.prefix:
        print(json.dumps({'observed_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'read-only public snapshots; no liveness or protocol completion proof',
            'runs': [run_status(prefix) for prefix in args.prefix]}, ensure_ascii=False, indent=2))
        return
    status = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'read-only public progress snapshot, not model output or liveness proof',
              'M2': run_status(M2), 'M3': {'prefix': M3, 'stages': []}}
    m3_final = ROOT / (M3 + '.json')
    status['M3']['terminal_result_available'] = m3_final.exists()
    if m3_final.exists():
        status['M3']['terminal_result'] = read(m3_final).get('result', 'not_recorded')
    for path in sorted((ROOT / 'records').glob(Path(M3).name + '.*.registration.json')):
        registration = read(path)
        prefix = str(path.relative_to(ROOT)).removesuffix('.registration.json')
        status['M3']['stages'].append({**run_status(prefix),
            'mode': registration['mode'], 'fixed_slots': registration['rows']})
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
