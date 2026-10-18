"""Read public phase diagnostics, including interrupted or malformed traces.

This is a diagnostic reader, never a completion/ledger verifier. Costs are
inclusive per path in the structurally valid prefix. Do not sum parent and
child groups, or the same traffic measured by both parties. A phase end is
not proof that its model output or protocol result is correct.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


METRICS = ('seconds', 'application_bytes', 'frames_sent', 'frames_received',
           'direction_runs')


def _metrics(row):
    values = {}
    for key in METRICS:
        value = row.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value < 0 or (isinstance(value, int) and value > 2**64 - 1)
                or (isinstance(value, float) and not math.isfinite(value))
                or (key != 'seconds' and not isinstance(value, int))):
            return None
        values[key] = value
    return values


def inspect_trace(path):
    """Read a bounded file snapshot; stop structural inference at first error.

    In particular, an out-of-order aborted ancestor is reported, not silently
    repaired into valid nesting. Subsequent abort events may be counted as raw
    observations, but cannot complete phases or contribute to trusted costs.
    No trace, ledger, process or live session is modified.
    """
    path = Path(path)
    digest = hashlib.sha256()
    stack, issues, groups = [], [], {}
    completed, aborted_observed = Counter(), Counter()
    valid_prefix = True
    lines = 0
    partial = False
    snapshot_bytes = 0
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        size = before.st_size
        remaining = size
        while remaining:
            raw = stream.readline(remaining)
            if not raw:
                issues.append({'line': lines + 1, 'code': 'FILE_SHRANK_DURING_READ'})
                valid_prefix = False
                break
            remaining -= len(raw)
            snapshot_bytes += len(raw)
            digest.update(raw)
            if not raw.endswith(b'\n'):
                partial = True
                break
            lines += 1
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeError):
                if valid_prefix:
                    issues.append({'line': lines, 'code': 'INVALID_JSON'})
                valid_prefix = False
                continue
            if not isinstance(row, dict):
                if valid_prefix:
                    issues.append({'line': lines, 'code': 'INVALID_EVENT_OBJECT'})
                valid_prefix = False
                continue
            phase, event = row.get('phase'), row.get('event')
            if not isinstance(phase, str) or not phase or len(phase) > 160 or '/' in phase:
                if valid_prefix:
                    issues.append({'line': lines, 'code': 'INVALID_PHASE_NAME'})
                valid_prefix = False
                continue
            if event == 'aborted':
                aborted_observed[phase] += 1
            if not valid_prefix:
                continue
            if event == 'begin':
                stack.append({'phase': phase, 'children': Counter(), 'children_complete': True})
                continue
            if event not in ('end', 'aborted'):
                issues.append({'line': lines, 'code': 'UNKNOWN_EVENT'})
                valid_prefix = False
                continue
            if not stack or stack[-1]['phase'] != phase:
                issues.append({'line': lines, 'code': 'NON_LIFO_TERMINAL',
                               'event': event, 'phase': phase,
                               'expected_phase': stack[-1]['phase'] if stack else None})
                valid_prefix = False
                continue
            name = '/'.join(item['phase'] for item in stack)
            item = stack.pop()
            values = _metrics(row)
            if event == 'end':
                completed[phase] += 1
            if stack:
                if values is None or event != 'end':
                    stack[-1]['children_complete'] = False
                else:
                    stack[-1]['children'].update(values)
            group = groups.setdefault(name, {'ended': 0, 'aborted': 0,
                'measured_ended': 0, 'missing_metrics': 0,
                'ended_inclusive': dict.fromkeys(METRICS, 0),
                'aborted_inclusive': dict.fromkeys(METRICS, 0),
                'exclusive_available_occurrences': 0,
                'ended_exclusive': dict.fromkeys(METRICS, 0)})
            group['ended' if event == 'end' else 'aborted'] += 1
            if values is None:
                group['missing_metrics'] += 1
                continue
            target = group['ended_inclusive' if event == 'end' else 'aborted_inclusive']
            for key, value in values.items():
                target[key] += value
            if event != 'end':
                continue
            group['measured_ended'] += 1
            # Printed wall seconds are rounded. Do not invent an exclusive
            # duration when the rounded child sum exceeds the parent value.
            exclusive = {key: value - item['children'][key] for key, value in values.items()}
            if item['children_complete'] and all(value >= 0 for value in exclusive.values()):
                group['exclusive_available_occurrences'] += 1
                for key, value in exclusive.items():
                    group['ended_exclusive'][key] += value
        after = os.fstat(stream.fileno())
    return {
        'trace': str(path), 'snapshot_sha256': digest.hexdigest(),
        'snapshot_bytes': snapshot_bytes, 'complete_json_lines': lines,
        'initial_file_mtime_ns': before.st_mtime_ns,
        'file_changed_during_read': (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                                   != (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        'snapshot_boundary': 'initial file length, sha256 of bytes actually read; not an atomic filesystem snapshot',
        'partial_last_line': partial, 'structural_issues': issues,
        'valid_prefix_ended_phases': dict(completed),
        'observed_abort_events': dict(aborted_observed),
        'unclosed_phases_at_valid_prefix_boundary': [item['phase'] for item in stack],
        'structure_balanced': valid_prefix and not partial and not stack,
        'phase_costs_in_valid_prefix': groups,
        'scope': 'public diagnostics only; ended phases do not certify results or session closure',
        'cost_units': 'per-party inclusive wall seconds and bidirectional application bytes; not CPU time or wire bytes',
        'worker_liveness': 'not_inferred_from_trace',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps({'traces': [inspect_trace(path) for path in args.trace]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
