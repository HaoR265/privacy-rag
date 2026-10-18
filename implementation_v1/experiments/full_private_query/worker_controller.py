"""Isolated scheduler for the full SessionRunner ABI; no share reconstruction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import time
import uuid

from private_rag.compute_input import WorkerBinding, sealed_input
from private_rag.compute_process import _validate_snapshot, _decode_result
from private_rag.delivery import PreparationShare
from private_rag.lifecycle import Lifecycle
from private_rag.memfd import create_memfd
from model_tools.reference_run_record import write_new
from tools.hybrid112_schedule import check_final_trace, full_session_plan

ROOT = Path(__file__).resolve().parents[2]


def check_cancel(stop):
    if stop.is_set():
        raise InterruptedError('DEVELOPMENT_RUN_CANCELLED')


def inspect_ledger(path, session, configuration):
    """Read only public reservations, requiring binding, order and final marker."""
    result = {'available': path.is_file(), 'closed_and_burned': False,
              'aborted_and_burned': False, 'reservation_sequence_valid': False}
    if not result['available']:
        return result
    digest = hashlib.sha256()
    sequence = 0
    terminal = None
    valid = True
    with path.open('rb') as stream:
        header = stream.readline()
        expected = f'format=1 session={uuid.UUID(session).hex} configuration={configuration}\n'.encode()
        valid = header == expected
        for raw in stream:
            if terminal is not None or not raw.endswith(b'\n'):
                valid = False
            if raw.startswith(b'terminal='):
                terminal = raw
                continue
            fields = raw.split()
            try:
                row_valid = (len(fields) == 5 and int(fields[0]) == sequence and
                             fields[1] == b'reserved' and int(fields[3]) >= 0 and fields[4] == b'ring59')
            except ValueError:
                row_valid = False
            valid = valid and row_valid
            digest.update(raw)
            sequence += 1
    result.update(reservations=sequence, reservation_sha256=digest.hexdigest(),
                  reservation_sequence_valid=valid and sequence > 0,
                  closed_and_burned=valid and terminal == b'terminal=closed all_reserved=burned\n',
                  aborted_and_burned=valid and terminal == b'terminal=aborted all_reserved=burned\n')
    return result


def shutdown_workers(workers, pairs, acknowledgements):
    """Break ACK/lane waits first, then reap; hard termination is not clean abort."""
    for fd in acknowledgements:
        try:
            os.close(fd)
        except OSError:
            pass
    acknowledgements.clear()
    for pair in pairs:
        for channel in pair:
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            channel.close()
    forced = []
    deadline = time.monotonic() + 30
    for child in workers:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.terminate()
            forced.append({'pid': child.pid, 'signal': 'SIGTERM'})
    deadline = time.monotonic() + 10
    for child in workers:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            forced.append({'pid': child.pid, 'signal': 'SIGKILL'})
            child.wait()
    return forced


def native_stage(repo, root, certificates, binary, configuration, output, mode,
                 expected_session, stop, *, document=None, question=None, collection=None):
    """Fresh claim -> both computing ACKs -> both releasing ACKs -> final outputs.

    The controller is a trusted local developer orchestrator. Separate child
    workers receive only their own sealed inputs; outputs are never added or
    XORed together. Preparation shares enter the existing D/S storage APIs.
    """
    owner = str(uuid.uuid4())
    life = Lifecycle(repo)
    sid = expected_session
    workers, pairs, streams, descriptors, ack_writes = [], [], [], [], []
    result = {'mode': mode, 'result': 'incomplete', 'production_gate_enabled': False}
    started = time.monotonic()
    prefix = None
    bindings = None
    try:
        check_cancel(stop)
        accepted = repo.claim_accepted_compute_session(owner)
        sid = accepted['session_id']
        if sid != expected_session or accepted['kind'] != mode or accepted['configuration'] != configuration:
            raise RuntimeError('ACCEPTED_SESSION_BINDING_MISMATCH')
        prefix = output / sid
        bindings = [WorkerBinding(sid, configuration, accepted['snapshot_sha256'], p, mode) for p in ('C', 'S')]
        records = [_validate_snapshot(bindings[0], accepted['snapshot'], collection, ()),
                   _validate_snapshot(bindings[1], accepted['snapshot'], None,
                                      () if mode == 'prepare' else tuple(repo.snapshot_server_records(sid)))]
        client_role = 'D' if mode == 'prepare' else 'U'
        if accepted['peer_certificate_sha256'] != certificates[client_role]['certificate_sha256']:
            raise RuntimeError('ACCEPTED_PEER_MISMATCH')
        result.update(session=sid, snapshot_sha256=accepted['snapshot_sha256'], configuration=configuration,
                      record_count=len(records[0]), expected_schedule=full_session_plan(mode))
        write_new(Path(str(prefix) + '.registration.json'), result)
        life.reserve(sid, 0, 'session_randomness', [1])
        pairs = [socket.socketpair() for _ in range(3)]
        event_reads, outputs = [], []
        for index, party in enumerate(('C', 'S')):
            check_cancel(stop)
            own_role = client_role if party == 'C' else 'S-compute'
            peer_role = 'S-compute' if party == 'C' else client_role
            sides = [pairs[0][index], pairs[1 if index == 0 else 2][index], pairs[2 if index == 0 else 1][index]]
            inp = sealed_input(bindings[index], records[index],
                               document=document if index == 0 else None,
                               question=question if index == 0 else None)
            descriptors.append(inp)
            out = create_memfd('private-rag-own-final-output')
            descriptors.append(out)
            outputs.append(out)
            event_read, event_write = os.pipe()
            ack_read, ack_write = os.pipe()
            descriptors.extend([event_read, event_write, ack_read])
            ack_writes.append(ack_write)
            event_reads.append(event_read)
            trace = str(prefix) + '.' + party
            stdout = Path(trace + '.stdout').open('x')
            stderr = Path(trace + '.stderr').open('x')
            streams.extend([stdout, stderr])
            pki = root / 'pki'
            args = [str(binary), party, mode, *[str(s.fileno()) for s in sides], str(inp), str(out),
                    uuid.UUID(sid).hex, configuration, accepted['snapshot_sha256'],
                    certificates[peer_role]['certificate_sha256'], str(pki / 'ca.pem'),
                    str(pki / (own_role + '.pem')), str(pki / (own_role + '.key')), 'localhost',
                    str(ROOT / 'artifacts/models'), str(ROOT / 'artifacts/constants/constants_v1.bin'),
                    str(ROOT / 'artifacts/template/template_v1.bin'), trace + '.ledger',
                    str(event_write), str(ack_read), trace]
            inherited = [s.fileno() for s in sides] + [inp, out, event_write, ack_read]
            workers.append(subprocess.Popen(args, pass_fds=inherited, stdout=stdout, stderr=stderr, cwd=ROOT))
            for fd in (inp, event_write, ack_read):
                os.close(fd)
                descriptors.remove(fd)
        # Keep parent lane descriptors solely to interrupt stuck peer waits.
        # They carry no application traffic and are all shut down on failure.
        seen = [0, 0]
        acknowledged = 0
        event_eof = set()
        while any(w.poll() is None for w in workers):
            check_cancel(stop)
            repo.check_session_cancellation(sid, owner)
            if any(w.poll() not in (None, 0) for w in workers):
                raise RuntimeError('NATIVE_WORKER_FAILED')
            readable, _, _ = select.select([fd for fd in event_reads if fd not in event_eof], [], [], 0.5)
            for fd in readable:
                index = event_reads.index(fd)
                value = os.read(fd, 1)
                if not value:
                    event_eof.add(fd)
                    if seen[index] != 2:
                        raise RuntimeError('LIFECYCLE_EVENT_EOF')
                    continue
                if value[0] != seen[index] + 1 or value[0] > 2:
                    raise RuntimeError('LIFECYCLE_EVENT_ORDER')
                seen[index] = value[0]
            if seen == [acknowledged + 1] * 2:
                acknowledged += 1
                life.transition(sid, 'computing' if acknowledged == 1 else 'releasing')
                for fd in ack_writes:
                    if os.write(fd, bytes([acknowledged])) != 1:
                        raise RuntimeError('LIFECYCLE_ACK_WRITE')
        if acknowledged != 2 or any(w.returncode != 0 for w in workers):
            raise RuntimeError('NATIVE_SESSION_INCOMPLETE')
        for stream in streams:
            stream.close()
        ledger = {party: inspect_ledger(Path(str(prefix) + '.' + party + '.ledger'), sid, configuration)
                  for party in ('C', 'S')}
        result['ledger_checks'] = ledger
        if (not all(v['closed_and_burned'] for v in ledger.values()) or
                ledger['C']['reservation_sha256'] != ledger['S']['reservation_sha256']):
            raise RuntimeError('LEDGER_COMPLETION_MISMATCH')
        result['trace_checks'] = check_final_trace(str(prefix), full_session_plan(mode), token_trace=True)
        finals = []
        for index, fd in enumerate(outputs):
            expected = (8201 if index == 0 else 8200) if mode == 'prepare' else (2312 if index == 0 else 0)
            if os.fstat(fd).st_size != expected:
                raise RuntimeError('FINAL_OUTPUT_SHAPE')
            raw = os.pread(fd, expected + 1, 0)
            finals.append(_decode_result(bindings[index], raw))
        result['parties'] = {party: json.loads(Path(str(prefix) + '.' + party + '.stdout').read_text())
                             for party in ('C', 'S')}
        if any(v.get('result') != 'completed_full_private_session' or v.get('mode') != mode
               for v in result['parties'].values()):
            raise RuntimeError('WORKER_COMPLETION_RECORD_MISMATCH')
        if mode == 'prepare':
            server = finals[1]
            repo.complete_native_preparation(sid, PreparationShare(sid, 'S', server.embedding, server.eligible))
        else:
            life.finish(sid, acknowledged=True)
        result['result'] = 'completed_private_session'
        return bindings[0], finals[0], result
    except BaseException as error:
        result.update(result='incomplete', exception_type=type(error).__name__)
        raise
    finally:
        result['forced_termination'] = shutdown_workers(workers, pairs, ack_writes)
        result['exit_codes'] = [w.returncode for w in workers]
        for fd in descriptors:
            try:
                os.close(fd)
            except OSError:
                pass
        for stream in streams:
            stream.close()
        if sid is None:
            sid = repo.claimed_session(owner)
        if sid is not None:
            prefix = output / sid
            result['ledger_after_cleanup'] = {
                p: inspect_ledger(Path(str(prefix) + '.' + p + '.ledger'), sid, configuration) for p in ('C', 'S')}
            result['normal_abort_confirmed'] = (result['result'] != 'completed_private_session' and
                len(workers) == 2 and not result['forced_termination'] and
                all(v['aborted_and_burned'] for v in result['ledger_after_cleanup'].values()))
            if result['result'] != 'completed_private_session':
                try:
                    life.recover_after_process_death(sid, operator_confirmed_worker_and_lanes_stopped=True)
                    result['management_recovered_after_workers_stopped'] = True
                except BaseException as error:
                    result['management_recovered_after_workers_stopped'] = False
                    result['recovery_exception_type'] = type(error).__name__
                    write_new(Path(str(prefix) + '.cleanup-failure.json'), result)
                    raise
            result['elapsed_seconds'] = time.monotonic() - started
            write_new(Path(str(prefix) + '.result.json'), result)
