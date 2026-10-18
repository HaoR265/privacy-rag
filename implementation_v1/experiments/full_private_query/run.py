"""Full fixed-budget private RAG development invocation, not a production switch.

Two explicitly public single-chunk documents are privately prepared, retained,
HPKE delivered and fully synchronized. A fresh SessionRunner query releases
only the client's final answer. Only after R2 stops, a separate RTX2060 R1
process computes the same public input in opaque manifest order for comparison.
No historical result, intermediate reconstruction, resume or shortened budget.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from experiments.full_private_query.worker_controller import native_stage, check_cancel
from experiments.full_private_query.gpu_binding import operator_device_binding
from experiments.hybrid112.registered_build import registered_build, add_sources, freeze, copy_binary, sha
from experiments.hybrid112.sleep_guard import prevent_sleep
from model_tools.reference_run_record import read_public_input, write_new, utc_now
from private_rag.client import TLSFiles, ProviderClient, UserClient
from private_rag.compute_process import retain_provider_preparation, load_provider_preparation
from private_rag.database import Repository
from private_rag.delivery import DeliveryBinding, canonical_json
from private_rag.management import create_app
from private_rag.model_input import InputPreparer
from private_rag.pki import create_local_pki
from private_rag.storage import ObjectStore
from private_rag.tls_server import serve, tls_context
from private_rag.vault import ProviderVault, RecipientKeyStore

ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE = 'prag_full_private_session'


def source_registration(build_directory):
    binary, identities = registered_build(build_directory, EXECUTABLE)
    sources = set()
    for folder in ('private_rag', 'references', 'model_tools', 'experiments/full_private_query'):
        sources.update(str(path.relative_to(ROOT)) for path in (ROOT / folder).rglob('*')
                       if path.is_file() and path.suffix in ('.py', '.sql', '.hpp', '.cpp'))
    sources.update(['experiments/hybrid112/registered_build.py', 'experiments/hybrid112/sleep_guard.py',
                    'tools/hybrid112_schedule.py', 'tools/hybrid112_status.py', 'tools/public_trace.py',
                    'tools/check_cuda_environment.py', 'artifacts/template/template_v1.json',
                    'artifacts/template/template_v1.bin', 'artifacts/constants/constants_v1.bin',
                    'artifacts/constants/constants_v1.json', 'artifacts/constants/numeric_profile_v1_qk.json',
                    'pyproject.toml', 'requirements.lock'])
    for model in ('embedding', 'generation'):
        sources.add(f'artifacts/models/{model}/quantized_f12/manifest.json')
        for path in (ROOT / 'artifacts/models' / model / 'checkpoint').glob('*.json'):
            sources.add(str(path.relative_to(ROOT)))
    return binary, add_sources(identities, sorted(sources))


def run_child(command, stop, stdout, stderr):
    """Own subprocess, with cooperative SIGTERM and explicit forced-kill record."""
    check_cancel(stop)
    with stdout.open('x') as out, stderr.open('x') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, cwd=ROOT)
        try:
            while child.poll() is None:
                check_cancel(stop)
                stop.wait(0.25)
            if child.returncode:
                raise RuntimeError('DEVELOPMENT_CHILD_FAILED_' + str(child.returncode))
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    write_new(stdout.with_suffix(stdout.suffix + '.forced-termination.json'),
                              {'pid': child.pid, 'signal': 'SIGKILL', 'clean_abort_asserted': False})


async def exercise(repo, runtime, output, configuration, binary, binary_sha, public, stop, result):
    certificates = create_local_pki(runtime / 'pki')
    user, provider_id = str(uuid.uuid4()), str(uuid.uuid4())
    repo.enroll_account(user, 'U', certificates['U']['certificate_sha256'])
    repo.enroll_account(provider_id, 'D', certificates['D']['certificate_sha256'])
    app, fetched = create_app(repo), []

    async def counted_app(scope, receive, send):
        if scope.get('path', '').startswith('/v1/sync/packages/'):
            fetched.append(scope['path'])
        await app(scope, receive, send)

    pki = runtime / 'pki'
    listener = await serve(counted_app, '127.0.0.1', 0, tls_context(
        str(pki / 'S-management.pem'), str(pki / 'S-management.key'), str(pki / 'ca.pem'), server=True))
    url = f'https://localhost:{listener.sockets[0].getsockname()[1]}'

    def clients():
        def files(role):
            return TLSFiles(pki / (role + '.pem'), pki / (role + '.key'), pki / 'ca.pem',
                            certificates['S-management']['certificate_sha256'])

        def development_admission():
            check_cancel(stop)
            if sha(binary) != binary_sha:
                raise RuntimeError('REGISTERED_BINARY_CHANGED')
            # Only this temporary Repository receives this callback. The
            # production activation function and official worker are unchanged.

        prep = InputPreparer(ROOT / 'artifacts/models', ROOT / 'artifacts/template/template_v1.json')
        chunk_lists = [prep.document_chunks(text, f'public-full-query-{i}')
                       for i, text in enumerate(public['documents'])]
        if len(chunk_lists) != 2 or any(len(chunks) != 1 for chunks in chunk_lists):
            raise ValueError('EXACTLY_TWO_NONEMPTY_SINGLE_CHUNK_DOCUMENTS_REQUIRED')
        chunks = [items[0] for items in chunk_lists]
        question = prep.question(public['question'])
        key_id = 'full-private-query-' + uuid.uuid4().hex
        vault = ProviderVault(provider_id, key_id)
        keys = RecipientKeyStore(user)
        owned_keys = ((keys, keys.SERVICE, f'{user}:1'),
                      (vault, vault.SERVICE, f'{provider_id}:{key_id}'))
        cleanup_errors = []
        try:
            check_cancel(stop)
            # Persist exact, fresh names BEFORE either OS key can be created.
            # This is private recovery metadata, never key material; do not
            # discover or delete other runs' keys by enumerating a backend.
            recovery_path = output / 'owned-key-recovery.json'
            write_new(recovery_path, {'user_id': user, 'provider_id': provider_id,
                'provider_key_id': key_id, 'runtime_directory': str(runtime),
                'keys': [{'service': service, 'name': name}
                         for _, service, name in owned_keys],
                'scope': 'Exact OS key names owned by this temporary run; no private key values'})
            result['owned_key_recovery_record'] = str(recovery_path)
            vault.initialize_new()
            with ProviderClient(url, files('D'), timeout=30) as provider, \
                    UserClient(url, files('U'), user, configuration, timeout=30) as recipient:
                if recipient.register_new_key() != 1:
                    raise RuntimeError('FRESH_RECIPIENT_KEY_REQUIRED')
                opaque_order = {}
                for index, chunk in enumerate(chunks):
                    check_cancel(stop)
                    document = provider.create_document('public-full-private-query', f'public document {index}')
                    version = provider.create_version(document, 1, configuration)
                    provider.authorize(document, user, True)
                    body = canonical_json({'text': chunk.text, 'tokens': chunk.body.tokens, 'validity': chunk.body.validity})
                    vault_path = runtime / f'document-{index}.vault'
                    vault.save(vault_path, document, body)
                    if vault.load(vault_path, document) != body:
                        raise RuntimeError('PROVIDER_VAULT_ROUNDTRIP')
                    session = repo.start_preparation(provider_id, document, version, 0,
                                                     activation_check=development_admission)
                    binding, local, stage = native_stage(repo, runtime, certificates, binary, configuration,
                        output, 'prepare', session, stop, document=chunk.embedding)
                    result['stages'].append(stage)
                    retained = runtime / f'preparation-{index}.vault'
                    share = retain_provider_preparation(binding, local, vault, retained)
                    if load_provider_preparation(binding, vault, retained) != share:
                        raise RuntimeError('PREPARATION_VAULT_ROUNDTRIP')
                    status = provider.session_status(session)
                    if status['state'] != 'completed' or status['release_acknowledged'] is not True:
                        raise RuntimeError('PREPARATION_RELEASE_NOT_ACKNOWLEDGED')
                    plan = provider.delivery_plan(document, version, user)
                    if len(plan['bindings']) != 1:
                        raise RuntimeError('SINGLE_CHUNK_DELIVERY_PLAN_REQUIRED')
                    delivery = DeliveryBinding(**plan['bindings'][0])
                    provider.stage_prepared_block(document, delivery, plan['public_key_hex'],
                                                  chunk.body.tokens, chunk.body.validity, share)
                    provider.publish_version(document, version)
                    opaque_order[delivery.opaque_slot] = index
                    write_new(output / f'document-{index}.completed.json',
                              {'session': session, 'private_preparation_completed': True,
                               'protected_vault_roundtrip': True, 'HPKE_delivery_published': True})
                    # No preparation vector, eligibility or failure bit is
                    # logged or used as a reference. Both outputs remain shares.
                    del local, share, body
                check_cancel(stop)
                collection = recipient.synchronize()
                if (len(collection.records) != 2 or collection.manifest['N'] != 2 or
                        len(fetched) != 2 or len(set(fetched)) != 2):
                    raise RuntimeError('COMPLETE_AUTHORIZED_SYNC_REQUIRED')
                order = [opaque_order[e['binding']['opaque_slot']] for e in collection.manifest['entries']]
                if sorted(order) != [0, 1]:
                    raise RuntimeError('OPAQUE_MANIFEST_ORDER_MISMATCH')
                # Only public source documents are reordered. No embeddings,
                # retrieval indices, evidence or private generation state leave R2.
                ordered_public = {'question': public['question'],
                                  'documents': [public['documents'][i] for i in order]}
                write_new(output / 'reference_input.json', ordered_public)
                write_new(output / 'manifest-order.json', {'source_document_indices': order,
                    'manifest_sha256': collection.manifest['manifest_sha256']})
                session = repo.accept_query(user, collection.manifest['manifest_sha256'], configuration,
                                            activation_check=development_admission)
                if isinstance(session, dict):
                    if session.get('status') != 'accepted':
                        raise RuntimeError('QUERY_NOT_ACCEPTED')
                    session = session['snapshot']['session_id']
                _, answer, stage = native_stage(repo, runtime, certificates, binary, configuration,
                    output, 'query', session, stop, question=question, collection=collection)
                result['stages'].append(stage)
                client_final = {'tokens': list(answer.tokens), 'validity': list(answer.validity), 'status': answer.status}
                write_new(output / 'client-final-answer.json', client_final)
                result.update(full_private_query_completed=True, full_sync_N=2, full_package_fetches=2,
                              protected_provider_storage_verified=True, HPKE_delivery_completed=True,
                              private_embedding_slots=[257, 257, 160], layers_each=28,
                              generation_prefill_slots=1024, generation_decode_calls=255,
                              final_answer_slots=256, server_query_output_bytes=0)
                return client_final
        finally:
            # Names are freshly allocated to this run. Try every owned key even
            # if one cleanup fails; never enumerate or delete other user keys.
            for store, service, name in owned_keys:
                try:
                    backend = store._backend()
                    if backend.get_password(service, name) is not None:
                        backend.delete_password(service, name)
                    if backend.get_password(service, name) is not None:
                        raise RuntimeError('OWNED_TEMPORARY_KEY_REMAINS')
                except Exception as error:
                    cleanup_errors.append({'service': service, 'name': name,
                                           'exception_type': type(error).__name__})
            result['temporary_OS_keys_removed'] = not cleanup_errors
            if cleanup_errors:
                result['key_cleanup_errors'] = cleanup_errors
                result['runtime_retained_for_safe_cleanup'] = str(runtime)
                result['runtime_retention_reason'] = 'Owned OS key deletion could not be confirmed; preserve private recovery context'
                raise RuntimeError('TEMPORARY_KEY_CLEANUP_FAILED')

    client_task = asyncio.create_task(asyncio.to_thread(clients))
    try:
        # Signals set stop instead of cancelling the loop: management TLS stays
        # alive until the client thread closes workers and releases its keys.
        return await asyncio.shield(client_task)
    except BaseException:
        stop.set()
        try:
            await asyncio.shield(client_task)
        except BaseException:
            pass
        raise
    finally:
        listener.close()
        await listener.wait_closed()
        result['temporary_listener_stopped'] = True


def run(args, stop):
    if Path.cwd().resolve() != ROOT:
        raise ValueError('RUN_FROM_IMPLEMENTATION_V1_DIRECTORY')
    public, raw, source = read_public_input(args.public_input_json)
    if len(public['documents']) != 2:
        raise ValueError('EXACTLY_TWO_PUBLIC_DOCUMENTS_REQUIRED')
    output = Path(os.path.abspath(args.output_dir))
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    result = {'result': 'incomplete', 'created_at': utc_now(), 'stages': [],
              'full_private_query_completed': False, 'R1_final_answer_matched': False,
              'production_gate_enabled': False, 'formal_acceptance': False,
              'scope': 'Local public-input full private RAG developer integration; not a security proof'}
    runtime = None
    pg_started = False
    repo = None
    pg = args.postgresql_bin.resolve()
    try:
        (output / 'public_input.json').write_bytes(raw)
        binary, identities = source_registration(args.build_directory)
        configuration = hashlib.sha256(canonical_json(identities)).hexdigest()
        binary_sha = identities[str(binary.relative_to(ROOT))]
        result['configuration'] = configuration
        write_new(output / 'start.json', {**result, 'source_and_binary_sha256': identities,
                  'operator_gpu_binding': args.operator_gpu_binding,
                  'public_input_source': source, 'model_CPU_fallback': False,
                  'independent_reference_runs_after_R2_workers_stop': True,
                  'budgets': {'document_slots': [257, 257], 'query_slots': 160,
                              'query_body_slots': 128, 'prefill_slots': 1024, 'decode_calls': 255}})
        archive = freeze(output / 'registered', identities)
        result['source_archive_sha256'] = sha(archive)
        run_child([sys.executable, '-m', 'tools.check_cuda_environment', '--output', str(output / 'cuda-preflight.json')],
                  stop, output / 'cuda-preflight.stdout', output / 'cuda-preflight.stderr')
        if json.loads((output / 'cuda-preflight.json').read_text()).get('gpu_ready') is not True:
            raise RuntimeError('RTX2060_REQUIRED')
        result['CUDA_preflight_passed'] = True
        runtime = Path(tempfile.mkdtemp(prefix='prag-full-'))
        os.chmod(runtime, 0o700)
        private_binary = copy_binary(binary, runtime / EXECUTABLE, binary_sha)
        data = runtime / 'pg'
        check_cancel(stop)
        subprocess.run([str(pg / 'initdb'), '-D', str(data), '-A', 'trust', '--no-locale', '-E', 'UTF8'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        check_cancel(stop)
        # Mark ownership before startup: a failed pg_ctl can still leave a server.
        pg_started = True
        subprocess.run([str(pg / 'pg_ctl'), '-D', str(data), '-l', str(runtime / 'pg.log'),
                        '-o', f"-k {runtime} -h '' -p 55483", '-w', 'start'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        repo = Repository(f'host={runtime} port=55483 dbname=postgres', ObjectStore(runtime / 'objects'))
        repo.initialize()
        answer = asyncio.run(exercise(repo, runtime, output, configuration, private_binary, binary_sha, public, stop, result))
        check_cancel(stop)
        # These files/values cannot reach R2: all native sessions have closed.
        run_child([sys.executable, '-m', 'model_tools.run_full_r1_rag',
                   '--public-input-json', str(output / 'reference_input.json'),
                   '--output-dir', str(output / 'independent-r1'), '--acknowledge-plaintext-reference'],
                  stop, output / 'reference.stdout', output / 'reference.stderr')
        reference = json.loads((output / 'independent-r1/result.json').read_text())
        if (reference.get('fixed_generation_executed') is not True or
                reference.get('fixed_generation_prefill_calls') != 1 or
                reference.get('fixed_generation_decode_calls') != 255 or
                reference.get('model_CPU_fallback') is not False or
                reference.get('start_to_finish_identity_unchanged') is not True or
                reference.get('answer_slots') != 256 or
                len(reference.get('tokens', [])) != 256 or len(reference.get('validity', [])) != 256 or
                type(reference.get('status')) is not int or reference['status'] not in range(4)):
            raise RuntimeError('INCOMPLETE_CUDA_REFERENCE_NOT_COMPARABLE')
        comparison = {key + '_match': answer[key] == reference[key] for key in ('tokens', 'validity', 'status')}
        comparison['scope'] = 'Client permitted final answer only; no intermediate reconstruction'
        write_new(output / 'final-comparison.json', comparison)
        if not all(comparison[k + '_match'] for k in ('tokens', 'validity', 'status')):
            raise RuntimeError('FINAL_CLIENT_ANSWER_DIFFERS_FROM_CUDA_R1')
        if any(sha(ROOT / name) != expected for name, expected in identities.items()):
            raise RuntimeError('REGISTERED_SOURCE_OR_ARTIFACT_CHANGED_DURING_RUN')
        result.update(R1_final_answer_matched=True, source_identities_unchanged=True,
                      result='complete_private_query_and_independent_final_comparison')
    except BaseException as error:
        result.update(result='incomplete', exception_type=type(error).__name__)
        raise
    finally:
        try:
            if repo is not None:
                try:
                    with repo.transaction() as connection:
                        states = connection.execute('SELECT kind,state,release_acknowledged FROM sessions ORDER BY started_at').fetchall()
                        reserved = connection.execute("SELECT count(*) AS n FROM materials WHERE state='reserved'").fetchone()['n']
                        references = connection.execute('SELECT count(*) AS n FROM session_references').fetchone()['n']
                    result['management_cleanup'] = {'sessions': states, 'reserved_materials': reserved,
                                                    'live_snapshot_references': references}
                    if (reserved != 0 or references != 0 or
                            any(s['state'] not in ('completed', 'aborted') for s in states)):
                        result.update(result='incomplete', management_cleanup_incomplete=True)
                    if result['full_private_query_completed'] and (len(states) != 3 or
                            any(s['state'] != 'completed' or not s['release_acknowledged'] for s in states)):
                        result.update(result='incomplete', management_completion_mismatch=True)
                except Exception as error:
                    result.update(result='incomplete', management_inspection_exception_type=type(error).__name__)
            if pg_started:
                stopped = subprocess.run([str(pg / 'pg_ctl'), '-D', str(runtime / 'pg'), '-m', 'fast', '-w', 'stop'],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                probe = subprocess.run([str(pg / 'pg_ctl'), '-D', str(runtime / 'pg'), 'status'],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                result['temporary_postgresql_stopped'] = stopped.returncode == 0 and probe.returncode == 3
                if not result['temporary_postgresql_stopped']:
                    result.update(result='incomplete', runtime_retained_for_safe_cleanup=str(runtime))
                    raise RuntimeError('POSTGRESQL_STOP_NOT_CONFIRMED_RUNTIME_RETAINED')
            if runtime is not None:
                if result.get('key_cleanup_errors'):
                    # PostgreSQL and the TLS listener are already stopped;
                    # keep only this run's private recovery context on disk.
                    result.update(result='incomplete', temporary_runtime_removed=False,
                                  runtime_retained_for_safe_cleanup=str(runtime))
                else:
                    shutil.rmtree(runtime)
                    result['temporary_runtime_removed'] = not runtime.exists()
        except BaseException as error:
            result.update(result='incomplete', cleanup_exception_type=type(error).__name__)
            raise
        finally:
            result['finished_at'] = utc_now()
            write_new(output / 'result.json', result)
            if sys.exc_info()[0] is None and result['result'] == 'incomplete':
                raise RuntimeError('DEVELOPMENT_RUN_NOT_COMPLETE_SEE_RESULT')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-input-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New private directory; parent must already exist')
    parser.add_argument('--build-directory', type=Path, required=True, help='Registered isolated development build containing ' + EXECUTABLE)
    parser.add_argument('--postgresql-bin', type=Path, default=Path('/usr/lib/postgresql/18/bin'))
    parser.add_argument('--gpu-uuid', help='Explicit RTX2060 UUID for enabled native GPU operators')
    parser.add_argument('--gpu-pci', help='PCI bus address matching --gpu-uuid')
    parser.add_argument('--acknowledge-public-input-and-plaintext-reference', action='store_true', required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    def cancelled(signum, frame):
        stop.set()
    previous = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGINT, signal.SIGTERM)}
    old_umask = os.umask(0o077)
    try:
        build = json.loads((args.build_directory / 'build_record.json').read_text())
        with operator_device_binding(build, args.gpu_uuid, args.gpu_pci) as binding:
            args.operator_gpu_binding = binding
            with prevent_sleep():
                run(args, stop)
    finally:
        os.umask(old_umask)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
