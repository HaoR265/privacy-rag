#!/usr/bin/env python3
"""Isolated public lifecycle/persistence debugging; never enables private model.

Events and local shares below are explicitly synthetic. This checks the actual
PostgreSQL transactions and OS vault, not native event emission or R2 inference.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from private_rag.compute_input import WorkerBinding
from private_rag.compute_process import LocalPreparationResult, retain_provider_preparation, load_provider_preparation
from private_rag.database import Repository
from private_rag.delivery import PreparationShare, canonical_json, digest
from private_rag.errors import ErrorCode, PublicError
from private_rag.lifecycle import Lifecycle
from private_rag.scheduler import ComputePorts, ComputeScheduler, SchedulerLifecycleEvents, connect_worker_sockets
from private_rag.storage import ObjectStore
from private_rag.vault import ProviderVault


def expect(code, call):
    try: call()
    except PublicError as exc: assert exc.code==code and str(exc)==code.value
    else: raise AssertionError('required fixed public rejection absent')


def main():
    project=Path(__file__).resolve().parents[1]
    results={'scope':'public_synthetic_scheduler_lifecycle_debug','formal_acceptance':False,
             'private_model_executed':False,'native_events_executed':False,'fake_backend_used':False}
    pg=Path('/usr/lib/postgresql/18/bin')
    with tempfile.TemporaryDirectory(prefix='prag-scheduler-') as temporary:
        root=Path(temporary); data=root/'pg'
        subprocess.run([str(pg/'initdb'),'-D',str(data),'-A','trust','--no-locale','-E','UTF8'],check=True,capture_output=True)
        subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-l',str(root/'pg.log'),'-o',f"-k {root} -h '' -p 55485",'-w','start'],check=True,capture_output=True)
        try:
            repo=Repository(f'host={root} port=55485 dbname=postgres',ObjectStore(root/'objects'));repo.initialize()
            life=Lifecycle(repo); owner=str(uuid.uuid4()); fingerprint=hashlib.sha256(b'public scheduler debug account').hexdigest()
            repo.enroll_account(owner,'D',fingerprint)
            config=hashlib.sha256(b'public synthetic configuration only').hexdigest()
            document=repo.create_document(owner,'public-debug','synthetic lifecycle')
            version=repo.create_version(owner,document,1,config)
            session=repo.start_preparation(owner,document,version,0)
            descriptor=repo.preparation_descriptor(owner,session)
            admission=repo.accepted_compute_session()
            assert admission['peer_certificate_sha256']==fingerprint and admission['configuration']==config
            assert admission['snapshot_sha256']==descriptor['snapshot_sha256']==digest(canonical_json(admission['snapshot']))
            expect(ErrorCode.NOT_AUTHORIZED,lambda:repo.preparation_descriptor(str(uuid.uuid4()),session))
            expect(ErrorCode.BUSY,lambda:repo.start_preparation(owner,document,version,0))
            # The gate must precede resolver, database claiming and listeners.
            def unused_settings(_): raise AssertionError('closed gate called resolver')
            expect(ErrorCode.BACKEND_NOT_READY,lambda:ComputeScheduler(repo,unused_settings).serve_one())
            expect(ErrorCode.BACKEND_NOT_READY,lambda:connect_worker_sockets('invalid.invalid',ComputePorts()))
            assert repo.accepted_compute_session()['session_id']==session
            events=SchedulerLifecycleEvents(repo,session)
            share=PreparationShare(session,'S',(7,)*1024,9)
            expect(ErrorCode.BAD_STATE,lambda:repo.complete_native_preparation(session,share))
            expect(ErrorCode.BAD_STATE,lambda:events('RELEASING'))
            expect(ErrorCode.REPLAY_OR_ORDER,lambda:events('arbitrary'))
            material=life.reserve(session,0,'session_randomness',[1])
            events('AUTHENTICATED')
            expect(ErrorCode.BAD_STATE,lambda:events('AUTHENTICATED'))
            expect(ErrorCode.BAD_STATE,lambda:life.finish(session,acknowledged=True))
            events('RELEASING')
            repo.complete_native_preparation(session,share)
            with repo.transaction() as c:
                state=c.execute('SELECT state,release_acknowledged FROM sessions WHERE session_id=%s',(session,)).fetchone()
                saved=c.execute('SELECT server_object FROM preparations WHERE preparation_id=%s',(session,)).fetchone()
                assert state=={'state':'completed','release_acknowledged':True}
                assert c.execute('SELECT state FROM materials WHERE material_id=%s',(material,)).fetchone()['state']=='burned'
            assert struct.unpack('<1025Q',repo.objects.get(saved['server_object'],8200))==(7,)*1024+(9,)
            expect(ErrorCode.BAD_STATE,lambda:repo.complete_native_preparation(session,share))
            # A separately completed preparation may replace a lost D-local
            # copy without reusing the first session or its one-use materials.
            second=repo.start_preparation(owner,document,version,0)
            second_events=SchedulerLifecycleEvents(repo,second)
            second_events('AUTHENTICATED'); second_events('RELEASING')
            repo.complete_native_preparation(second,PreparationShare(second,'S',(11,)*1024,13))
            repo.publish_version(owner,document,version)  # DISTINCT block coverage
            third=repo.start_preparation(owner,document,version,0)
            uncertain=life.reserve(third,0,'session_randomness',[1])
            life.finish(third,acknowledged=False,public_fault='DISCONNECTED')
            with repo.transaction() as c:
                assert c.execute('SELECT count(*) AS n FROM preparations WHERE document_id=%s',(document,)).fetchone()['n']==2
                assert c.execute('SELECT state FROM materials WHERE material_id=%s',(uncertain,)).fetchone()['state']=='burned'
                assert c.execute('SELECT state FROM sessions WHERE session_id=%s',(third,)).fetchone()['state']=='aborted'
            results['postgres_transactions']='accepted snapshot/certificate pin, ordered persisted events, no early completion, own-share object retained before atomic completion, rerun independent preparation, abort burns reservations'
            results['closed_gate']='scheduler and C connector reject before resolver/listeners/session state changes'
            cli=subprocess.run([sys.executable,'-m','private_rag.cli','compute-once','--dsn','unavailable',
                 '--objects',str(root/'must-not-create-objects'),'--root',str(project),'--cert','unavailable',
                 '--key','unavailable','--ca','unavailable','--peer-certificates',str(root/'unavailable-peers'),
                 '--ledger-directory',str(root/'must-not-create-ledger')],capture_output=True,text=True)
            assert cli.returncode==2 and json.loads(cli.stdout)=={'error':'BACKEND_NOT_READY'} and not cli.stderr
            assert not (root/'must-not-create-objects').exists() and not (root/'must-not-create-ledger').exists()
            results['cli_gate']='fixed error exit 2 before stores/database/listeners'
        finally:
            subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-m','fast','-w','stop'],check=True,capture_output=True)
            results['temporary_postgresql_stopped']=True
        # Fresh project key only: actual OS key store and AES-256-GCM, using
        # synthetic D-local output; never passes the D failure bit to S.
        vault=ProviderVault(str(uuid.uuid4()),'scheduler-debug-'+str(uuid.uuid4()))
        backend=vault._backend(); name=f'{vault.provider_id}:{vault.key_id}'; created=False
        try:
            vault.initialize_new(); created=True
            binding=WorkerBinding(session,config,descriptor['snapshot_sha256'],'C','prepare')
            local=LocalPreparationResult('C',(17,)*1024,19,False)
            retained=retain_provider_preparation(binding,local,vault,root/'preparation.vault')
            assert load_provider_preparation(binding,vault,root/'preparation.vault')==retained
            expect(ErrorCode.MATERIALS_NOT_READY,lambda:retain_provider_preparation(binding,dataclasses.replace(local,client_failed=True),vault,root/'must-not-exist.vault'))
            assert not (root/'must-not-exist.vault').exists()
            expect(ErrorCode.VERSION_MISMATCH,lambda:load_provider_preparation(dataclasses.replace(binding,configuration='0'*64),vault,root/'preparation.vault'))
            results['D_local_retention']='actual OS key/AES256GCM roundtrip, binding rejects mismatch, secret failure cannot become deliverable share'
        finally:
            if created:
                backend.delete_password(vault.SERVICE,name)
                assert backend.get_password(vault.SERVICE,name) is None
                results['temporary_provider_key_deleted']=True
    (project/'records/scheduler_debug.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
