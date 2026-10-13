"""Isolated, actual 257/257/160-slot private Embedding and delivery integration.

Runs production protocol kernels, PostgreSQL management, TLS clients, encrypted
provider storage and HPKE. It has no general production activation switch and
never calls Generation. Both parties see only their own sealed inputs. The
separate development controller compares final outputs after one-use sessions
close; no reconstructed embedding or oracle value is supplied to any R2 stage.
No crypto checkpoint, resume path, shortened budget, or computation timeout.
"""
from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid

from private_rag.client import TLSFiles, ProviderClient, UserClient
from private_rag.compute_input import WorkerBinding
from private_rag.compute_process import (_validate_snapshot, LocalPreparationResult,
                                        retain_provider_preparation, load_provider_preparation)
from private_rag.database import Repository
from private_rag.delivery import PreparationShare, DeliveryBinding, canonical_json
from private_rag.lifecycle import Lifecycle
from private_rag.management import create_app
from private_rag.memfd import create_memfd, seal_memfd
from private_rag.model_input import InputPreparer
from private_rag.pki import create_local_pki
from private_rag.storage import ObjectStore
from private_rag.tls_server import serve, tls_context
from private_rag.vault import ProviderVault, RecipientKeyStore
from experiments.hybrid112.registered_build import registered_build, add_sources, freeze, copy_binary

ROOT = Path(__file__).resolve().parents[2]
MASK = (1 << 59) - 1
REFERENCE = ROOT / 'records/hybrid112_m3_CUDA_reference.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def event(name, **fields):
    print(json.dumps({'event': name, **fields}, ensure_ascii=False), flush=True)


def seal(data):
    fd = create_memfd('prag112-M3-own-input', allow_sealing=True)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise RuntimeError('INPUT_WRITE')
            view = view[n:]
        os.lseek(fd, 0, 0)
        seal_memfd(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def source_registration(build_directory):
    _, build_identities = registered_build(build_directory, 'prag_hybrid_m3')
    paths = set()
    for folder in ('compute', 'native/src', 'native/include/prag', 'private_rag',
                   'references', 'experiments/hybrid112'):
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.hpp', '.cpp', '.cu', '.inc', '.txt', '.sql'):
                paths.add(str(p.relative_to(ROOT)))
    paths.update(['experiments/batch110_baseline/meter.hpp', 'native/CMakeLists.txt',
                  str(build_directory / 'prag_hybrid_m3'), str(build_directory / 'native/libprag_crypto.a'),
                  str(build_directory / 'libprag_local_cuda.a'), 'artifacts/constants/constants_v1.bin',
                  'artifacts/constants/constants_v1.json', 'artifacts/constants/numeric_profile_v1_qk.json',
                  'artifacts/models/embedding/quantized_f12/manifest.json',
                  'artifacts/models/embedding/checkpoint/tokenizer.json',
                  'artifacts/models/generation/checkpoint/tokenizer.json',
                  'artifacts/template/template_v1.json', 'pyproject.toml', str(REFERENCE.relative_to(ROOT))])
    return add_sources(build_identities, sorted(paths))


def native_stage(repo, root, certificates, binary, cfg, run_prefix, slots, mode, collection=None):
    """Diagnostic scheduler with actual durable accept/claim/release transitions."""
    owner = str(uuid.uuid4())
    accepted = repo.claim_accepted_compute_session(owner)
    sid = accepted['session_id']
    life = Lifecycle(repo)
    prefix = Path(str(run_prefix) + '.' + sid)
    bindings = [WorkerBinding(sid, cfg, accepted['snapshot_sha256'], party,
                              'prepare' if mode == 'prepare' else 'query') for party in ('C', 'S')]
    server_records = () if mode == 'prepare' else tuple(repo.snapshot_server_records(sid))
    records = [_validate_snapshot(bindings[0], accepted['snapshot'], collection, ()),
               _validate_snapshot(bindings[1], accepted['snapshot'], None, server_records)]
    count = len(records[0])
    reg = {'session': sid, 'configuration': cfg, 'snapshot': accepted['snapshot'],
           'snapshot_sha256': accepted['snapshot_sha256'], 'mode': mode, 'rows': len(slots.tokens),
           'records': count, 'fractional_bits': {'tokens': 0, 'embedding': 20, 'validity': 0},
           'secret_inputs': 'C owner slots and own U records; S own records only',
           'output': 'own shares; only D receives preparation failure', 'layers': 28,
           'independent_development_entry': True, 'production_gate_enabled': False}
    save(Path(str(prefix) + '.registration.json'), reg)
    life.reserve(sid, 0, 'session_randomness', [1])
    life.transition(sid, 'computing')
    pairs = [socket.socketpair() for _ in range(3)]
    coordinator, lifecycle = socket.socketpair()
    fds, streams, workers = [], [], []
    result = {}
    released = False
    started = time.monotonic()
    try:
        for index, role in enumerate(('C', 'S')):
            side = [pairs[0][index], pairs[1 if index == 0 else 2][index], pairs[2 if index == 0 else 1][index]]
            header = b'PR112IN1' + bytes([index]) + bytes(7) + uuid.UUID(sid).bytes + bytes.fromhex(cfg) + bytes.fromhex(accepted['snapshot_sha256'])
            owned = struct.pack(f'<{len(slots.tokens)}Q', *slots.tokens) + bytes(slots.validity) if role == 'C' else b''
            inp = seal(header + owned + b''.join(record.encode() for record in records[index]))
            out = create_memfd('prag112-M3-own-output')
            fds.extend([inp, out])
            peer = 'S-compute' if role == 'C' else ('D' if mode == 'prepare' else 'U')
            stdout, stderr = (root / (role + '.stdout')).open('w'), (root / (role + '.stderr')).open('w')
            streams.extend([stdout, stderr])
            descriptors = [s.fileno() for s in side] + [inp, out]
            if role == 'S':
                descriptors.append(lifecycle.fileno())
            cmd = [str(binary), role, *map(str, descriptors[:3]), str(inp), str(out), str(root / 'pki'),
                   certificates[peer]['certificate_sha256'], uuid.UUID(sid).hex, cfg,
                   accepted['snapshot_sha256'], str(ROOT), 'embedding', mode, str(len(slots.tokens)),
                   str(count), str(prefix), str(lifecycle.fileno() if role == 'S' else -1)]
            workers.append(subprocess.Popen(cmd, pass_fds=descriptors, stdout=stdout, stderr=stderr))
            # No parent input copies or descriptors survive launch unnecessarily.
            os.close(inp)
            fds[-2] = -1
        for pair in pairs:
            for sock in pair:
                sock.close()
        lifecycle.close()
        event('M3_private_stage_started', mode=mode, session=sid, workers=[w.pid for w in workers],
              rows=len(slots.tokens), layers=28, records=count, record=str(prefix.relative_to(ROOT)))
        while any(w.poll() is None for w in workers):
            repo.check_session_cancellation(sid, owner)
            if any(w.poll() not in (None, 0) for w in workers):
                raise RuntimeError('NATIVE_STAGE_ABORTED')
            ready, _, _ = select.select([coordinator] if not released else [], [], [], 2)
            if ready:
                if coordinator.recv(1) != b'R':
                    raise RuntimeError('LIFECYCLE_RELEASE')
                life.transition(sid, 'releasing')
                coordinator.sendall(b'A')
                released = True
        if not released or any(w.returncode != 0 for w in workers):
            raise RuntimeError('INCOMPLETE_STAGE')
        finals = []
        words = 1025 if mode == 'prepare' else count + 2 * 513
        for index, fd in enumerate((fds[1], fds[3])):
            os.lseek(fd, 0, 0)
            size = words * 8 + (int(index == 0) if mode == 'prepare' else 1)
            with os.fdopen(os.dup(fd), 'rb') as source:
                raw = source.read(size + 1)
            if len(raw) != size:
                raise RuntimeError('OUTPUT_SHAPE')
            values = struct.unpack(f'<{words}Q', raw[:words * 8])
            if any(value > MASK for value in values) or (len(raw) > words * 8 and raw[-1] > 1):
                raise RuntimeError('OUTPUT_CANONICAL')
            finals.append((values, raw[-1] if len(raw) > words * 8 else None))
        if mode == 'prepare':
            repo.complete_native_preparation(sid, PreparationShare(sid, 'S', finals[1][0][:-1], finals[1][0][-1]))
        else:
            life.finish(sid, acknowledged=True)
        for stream in streams:
            stream.close()
        result = {**reg, 'result': 'completed_actual_private_stage', 'elapsed_seconds': time.monotonic() - started,
                  'exit_codes': [w.returncode for w in workers],
                  'parties': {r: json.loads((root / (r + '.stdout')).read_text()) for r in ('C', 'S')}}
        # No private failure value is placed into S/management diagnostics.
        result['ledgers_closed_and_burned'] = all('terminal=closed all_reserved=burned' in
            Path(str(prefix) + '.' + role + '.ledger').read_text() for role in ('C', 'S'))
        if not result['ledgers_closed_and_burned']:
            raise RuntimeError('LEDGER_NOT_CLOSED')
        return bindings, finals, result
    except BaseException as exc:
        result = {**reg, 'result': 'aborted', 'exception_type': type(exc).__name__}
        raise
    finally:
        for w in workers:
            if w.poll() is None:
                w.terminate()
        for w in workers:
            if w.poll() is None:
                w.wait(timeout=30)
        for pair in pairs:
            for sock in pair:
                sock.close()
        coordinator.close()
        lifecycle.close()
        for fd in fds:
            if fd >= 0:
                os.close(fd)
        for stream in streams:
            stream.close()
        if result.get('result') != 'completed_actual_private_stage':
            life.recover_after_process_death(sid, operator_confirmed_worker_and_lanes_stopped=True)
            result['stderr'] = {r: (root / (r + '.stderr')).read_text() if (root / (r + '.stderr')).exists() else '' for r in ('C', 'S')}
        result['evidence'] = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in
                              sorted(prefix.parent.glob(prefix.name + '.*')) if p.suffix != '.json']
        save(Path(str(prefix) + '.json'), result)


def compare_final(finals, expected, expected_failed, mode):
    import torch
    from references.resource_policy import require_model_device
    device = require_model_device('R1')
    actual = (torch.tensor(finals[0][0], device=device, dtype=torch.int64) +
              torch.tensor(finals[1][0], device=device, dtype=torch.int64)) & MASK
    wanted = torch.tensor(expected, device=device, dtype=torch.int64) & MASK
    failed = finals[0][1] if mode == 'prepare' else finals[0][1] ^ finals[1][1]
    answer = {'final_integers_match': bool(torch.equal(actual, wanted)),
              'failure_matches': failed == expected_failed, 'numeric_failure': bool(failed),
              'words': len(expected)}
    if not answer['final_integers_match'] or not answer['failure_matches']:
        raise RuntimeError('CUDA_ORACLE_MISMATCH')
    return answer


async def exercise(repo, root, cfg, prefix, binary, reference, binary_sha256):
    certificates = create_local_pki(root / 'pki')
    user, provider_id = str(uuid.uuid4()), str(uuid.uuid4())
    repo.enroll_account(user, 'U', certificates['U']['certificate_sha256'])
    repo.enroll_account(provider_id, 'D', certificates['D']['certificate_sha256'])
    app = create_app(repo)
    fetched = []
    async def counted_app(scope, receive, send):
        if scope.get('path', '').startswith('/v1/sync/packages/'):
            fetched.append(scope['path'])
        await app(scope, receive, send)
    pki = root / 'pki'
    tls = tls_context(str(pki / 'S-management.pem'), str(pki / 'S-management.key'), str(pki / 'ca.pem'), server=True)
    listener = await serve(counted_app, '127.0.0.1', 0, tls)
    url = f'https://localhost:{listener.sockets[0].getsockname()[1]}'
    results = {'stages': [], 'generation_executed': False, 'production_gate_enabled': False,
               'full_query_completed': False, 'formal_acceptance': False}
    def clients():
        def files(role):
            return TLSFiles(pki / (role + '.pem'), pki / (role + '.key'), pki / 'ca.pem', certificates['S-management']['certificate_sha256'])
        def diagnostic_admission():
            # Isolated Repository instance only; the management production
            # activation function remains closed and is never replaced.
            if sha(binary) != binary_sha256:
                raise RuntimeError('REGISTERED_BINARY_CHANGED')
        prep = InputPreparer(ROOT / 'artifacts/models', ROOT / 'artifacts/template/template_v1.json')
        chunks = [prep.document_chunks(t, 'registered-public-M3')[0] for t in reference['documents']]
        question = prep.question(reference['question'])
        vault = ProviderVault(provider_id, 'isolated-M3')
        keys = RecipientKeyStore(user)
        try:
            vault.initialize_new()
            with ProviderClient(url, files('D'), timeout=30) as provider, UserClient(url, files('U'), user, cfg, timeout=30) as recipient:
                key_version = recipient.register_new_key()
                if key_version != 1:
                    raise RuntimeError('FRESH_KEY_VERSION')
                opaque_order = {}
                for index, chunk in enumerate(chunks):
                    document = provider.create_document('public-M3-development', f'public document {index}')
                    version = provider.create_version(document, 1, cfg)
                    provider.authorize(document, user, True)
                    # D owner plaintext is saved only in its authenticated vault.
                    body = canonical_json({'text': chunk.text, 'tokens': chunk.body.tokens, 'validity': chunk.body.validity})
                    vault.save(root / f'document-{index}.vault', document, body)
                    if vault.load(root / f'document-{index}.vault', document) != body:
                        raise RuntimeError('VAULT_ROUNDTRIP')
                    sid = repo.start_preparation(provider_id, document, version, 0, activation_check=diagnostic_admission)
                    bindings, final, stage = native_stage(repo, root, certificates, binary, cfg, prefix, chunk.embedding, 'prepare')
                    stage['oracle'] = compare_final(final, reference['vectors'][index] + [reference['eligible'][index]], reference['failures'][index], 'prepare')
                    local = LocalPreparationResult('C', final[0][0][:-1], final[0][0][-1], bool(final[0][1]))
                    retained = root / f'preparation-{index}.vault'
                    dshare = retain_provider_preparation(bindings[0], local, vault, retained)
                    if load_provider_preparation(bindings[0], vault, retained) != dshare:
                        raise RuntimeError('PREPARATION_VAULT_ROUNDTRIP')
                    status = provider.session_status(sid)
                    if status['state'] != 'completed' or status['release_acknowledged'] is not True:
                        raise RuntimeError('PREPARATION_NOT_ACKNOWLEDGED')
                    plan = provider.delivery_plan(document, version, user)
                    binding = DeliveryBinding(**plan['bindings'][0])
                    provider.stage_prepared_block(document, binding, plan['public_key_hex'], chunk.body.tokens, chunk.body.validity, dshare)
                    provider.publish_version(document, version)
                    opaque_order[binding.opaque_slot] = index
                    results['stages'].append(stage)
                    save(Path(str(prefix) + '.progress.json'), results)
                    event('M3_private_document_delivered', document_index=index, session=sid, oracle=stage['oracle'])
                collection = recipient.synchronize()
                if len(collection.records) != 2 or len(fetched) != 2:
                    raise RuntimeError('FULL_SYNC_SHAPE')
                order = [opaque_order[e['binding']['opaque_slot']] for e in collection.manifest['entries']]
                order_path = root / 'oracle-public-order.json'
                save(order_path, order)
                expected_path = root / 'oracle-public-evidence.bin'
                subprocess.run([sys.executable, str(ROOT / 'experiments/hybrid112/m3_reference.py'), 'retrieval', str(expected_path),
                                '--reference', str(REFERENCE), '--order', str(order_path)], check=True)
                raw = expected_path.read_bytes()
                repo.accept_query(user, collection.manifest['manifest_sha256'], cfg, activation_check=diagnostic_admission)
                bindings, final, stage = native_stage(repo, root, certificates, binary, cfg, prefix, question.embedding, 'retrieve', collection)
                expected = struct.unpack(f'<{len(raw) // 8}Q', raw)
                stage['oracle'] = compare_final(final, expected, reference['failures'][2], 'retrieve')
                results['stages'].append(stage)
                results.update(result='M3_matched_CUDA_R1', full_sync_N=2, full_package_fetches=len(fetched),
                               actual_private_embedding_slots=[257, 257, 160], layers_each=28,
                               protected_provider_storage_verified=True, actual_HPKE_delivery=True,
                               immutable_authorized_snapshot=True)
        finally:
            for backend, service, name in [(keys._backend(), keys.SERVICE, f'{user}:1'),
                                           (vault._backend(), vault.SERVICE, f'{provider_id}:isolated-M3')]:
                if backend.get_password(service, name) is not None:
                    backend.delete_password(service, name)
                if backend.get_password(service, name) is not None:
                    raise RuntimeError('TEMPORARY_KEY_CLEANUP')
            results['temporary_OS_keys_removed'] = True
    try:
        await asyncio.to_thread(clients)
    finally:
        listener.close()
        await listener.wait_closed()
        results['temporary_listener_stopped'] = True
        save(Path(str(prefix) + '.progress.json'), results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory', required=True,
                        help='fresh registered development build from tools.build_hybrid')
    args = parser.parse_args()
    os.chdir(ROOT)
    build_directory = (ROOT / args.build_directory).resolve()
    if not build_directory.is_relative_to((ROOT / 'build').resolve()):
        parser.error('build directory must belong to this project build/ directory')
    build_directory = build_directory.relative_to(ROOT)
    reference = json.loads(REFERENCE.read_text())
    if reference['rows'] != [257, 257, 160] or any(reference['failures']):
        raise RuntimeError('PUBLIC_REFERENCE_NUMERIC_FAILURE')
    identities = source_registration(build_directory)
    binary_sha256 = identities[str(build_directory / 'prag_hybrid_m3')]
    cfg = hashlib.sha256(canonical_json(identities)).hexdigest()
    prefix = ROOT / f'records/hybrid112_M3_{uuid.uuid4().hex}'
    save(Path(str(prefix) + '.registration.json'), {'source_and_binary_sha256': identities, 'configuration': cfg,
         'public_sample_reference_sha256': sha(REFERENCE), 'private_embedding_budgets': [257, 257, 160],
         'full_model_shapes_unchanged': True, 'no_Generation_run': True, 'production_gate_enabled': False})
    archive = freeze(prefix, identities)
    pg = Path('/usr/lib/postgresql/18/bin')
    result = {'result': 'not_completed', 'configuration': cfg, 'source_archive_sha256': sha(archive)}
    try:
        with tempfile.TemporaryDirectory(prefix='prag-M3-') as td:
            root = Path(td)
            binary = root / 'prag_hybrid_m3'
            copy_binary(ROOT / build_directory / 'prag_hybrid_m3', binary, binary_sha256)
            data = root / 'pg'
            subprocess.run([str(pg / 'initdb'), '-D', str(data), '-A', 'trust', '--no-locale', '-E', 'UTF8'], check=True, capture_output=True)
            subprocess.run([str(pg / 'pg_ctl'), '-D', str(data), '-l', str(root / 'pg.log'), '-o', f"-k {root} -h '' -p 55483", '-w', 'start'], check=True, capture_output=True)
            try:
                repo = Repository(f'host={root} port=55483 dbname=postgres', ObjectStore(root / 'objects'))
                repo.initialize()
                result.update(asyncio.run(exercise(repo, root, cfg, prefix, binary, reference, binary_sha256)))
            finally:
                subprocess.run([str(pg / 'pg_ctl'), '-D', str(data), '-m', 'fast', '-w', 'stop'], check=True, capture_output=True)
                result['temporary_postgresql_stopped'] = True
    finally:
        save(Path(str(prefix) + '.json'), result)
        event('M3_controller_finished', record=str(prefix.relative_to(ROOT)) + '.json', result=result['result'])


if __name__ == '__main__':
    main()
