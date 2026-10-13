"""Read public diagnostic progress once; never open shares or resume sessions.

A recent trace is evidence of progress, not proof that a worker is still alive.
Only a finished result with closed ledgers can establish a completed milestone.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

from tools.build_hybrid112_evidence import M2, M3, ROOT


def read(path):
    return json.loads(path.read_text())


def trace_status(path):
    if not path.exists():
        return {'trace_available': False}
    stack, completed, aborted = [], Counter(), Counter()
    partial_tail = False
    # A writer may be in the middle of the last public JSON line. Do not turn
    # that condition into a completed phase or a claimed protocol error.
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                partial_tail = True
                break
            row = json.loads(line)
            name = row['phase']
            if row['event'] == 'begin':
                stack.append(name)
            elif row['event'] in ('end', 'aborted'):
                if not stack or stack.pop() != name:
                    raise RuntimeError('UNBALANCED_PUBLIC_TRACE')
                (completed if row['event'] == 'end' else aborted)[name] += 1
            else:
                raise RuntimeError('UNKNOWN_PUBLIC_TRACE_EVENT')
    return {'trace_available': True, 'trace': str(path.relative_to(ROOT)),
            'last_write_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            'complete_transformer_layers': completed['Transformer_layer'],
            'complete_private_token_lookups': (completed['Token_lookup']
                if 'Token_lookup' in completed or 'Token_lookup' in stack
                else 'not_traced_by_this_execution'),
            'public_active_phases': stack, 'partial_last_line': partial_tail,
            'public_aborted_phases': dict(aborted),
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
