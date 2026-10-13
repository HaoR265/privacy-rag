#!/usr/bin/env python3
"""Real mTLS/PG public cancellation debugging; no model or fake backend."""
from __future__ import annotations
import asyncio
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from psycopg.types.json import Jsonb
from private_rag.client import TLSFiles, UserClient, ProviderClient
from private_rag.database import Repository
from private_rag.errors import ErrorCode, PublicError
from private_rag.lifecycle import Lifecycle
from private_rag.management import create_app
from private_rag.pki import create_local_pki
from private_rag.storage import ObjectStore
from private_rag.tls_server import serve, tls_context


def expect(code,call):
    try: call()
    except PublicError as exc: assert exc.code==code and str(exc)==code.value
    else: raise AssertionError('required public rejection absent')


async def exercise(repo,root):
    pki=root/'pki'; certificates=create_local_pki(pki)
    owner,user=str(uuid.uuid4()),str(uuid.uuid4())
    repo.enroll_account(owner,'D',certificates['D']['certificate_sha256'])
    repo.enroll_account(user,'U',certificates['U']['certificate_sha256'])
    paths=[]; app=create_app(repo)
    async def counted(scope,receive,send):
        paths.append((scope.get('method'),scope.get('path')))
        await app(scope,receive,send)
    tls=tls_context(str(pki/'S-management.pem'),str(pki/'S-management.key'),str(pki/'ca.pem'),server=True)
    listener=await serve(counted,'127.0.0.1',0,tls)
    url=f'https://localhost:{listener.sockets[0].getsockname()[1]}'
    result={}
    def clients():
        def files(role):
            return TLSFiles(pki/(role+'.pem'),pki/(role+'.key'),pki/'ca.pem',certificates['S-management']['certificate_sha256'])
        config='1'*64  # Public metadata only; no configuration validation bypass.
        with ProviderClient(url,files('D')) as provider, UserClient(url,files('U'),user,config) as recipient:
            document=provider.create_document('public-cancel-debug','public fixture')
            version=provider.create_version(document,1,config)
            # Production admission remains closed. Its predetermined session
            # identity permits one public cancel request without replaying POST.
            expect(ErrorCode.BACKEND_NOT_READY,lambda:provider.start_preparation(document,version,0))
            with repo.transaction() as c:
                assert c.execute('SELECT count(*) AS n FROM sessions').fetchone()['n']==0
                assert c.execute('SELECT count(*) AS n FROM session_cancel_intents').fetchone()['n']==1
            assert [method for method,path in paths if path.endswith('/preparations')]==['POST']
            assert len([1 for method,path in paths if path.endswith('/cancel')])==1
            result['closed_gate_and_identity']='one failed admission POST, one identity-bound public cancel; no session or model'
            life=Lifecycle(repo)
            # Explicit public fixture enters only the internal repository; the
            # management API has no facility to manufacture model completion.
            session=repo.start_preparation(owner,document,version,0)
            expect(ErrorCode.NOT_AUTHORIZED,lambda:recipient.cancel_session(session))
            assert provider.cancel_session(session)=={'session_id':session,'state':'aborted','cancel_requested':True}
            assert provider.session_status(session)['public_fault']=='DISCONNECTED'
            result['unclaimed']='own mTLS cancel aborts; foreign certificate rejected'
            session=repo.start_preparation(owner,document,version,0)
            worker=str(uuid.uuid4()); assert repo.claim_accepted_compute_session(worker)['session_id']==session
            life.transition(session,'computing')
            assert provider.cancel_session(session)=={'session_id':session,'state':'computing','cancel_requested':True}
            expect(ErrorCode.BUSY,lambda:repo.start_preparation(owner,document,version,0))
            expect(ErrorCode.DISCONNECTED,lambda:repo.check_session_cancellation(session,worker))
            # This fixture launched no child/fds. Only the coordinator's
            # post-resource-cleanup transaction is allowed to free the slot.
            life.finish(session,acknowledged=False,public_fault='DISCONNECTED')
            assert provider.session_status(session)['state']=='aborted'
            result['claimed']='cancel remains computing/global BUSY until coordinator terminal transaction'
            # Public U-owned fixture verifies the same existing authenticated
            # cancellation API; it contains no query text or secret shares.
            query=str(uuid.uuid4())
            with repo.transaction() as c:
                c.execute("INSERT INTO sessions(session_id,account_id,kind,state,snapshot) VALUES(%s,%s,'query','accepted',%s)",(query,user,Jsonb({'model_numeric_version':config})))
            expect(ErrorCode.NOT_AUTHORIZED,lambda:provider.cancel_session(query))
            assert recipient.cancel_session(query)['state']=='aborted'
            assert recipient.session_status(query)['public_fault']=='DISCONNECTED'
            result['U_own_session']='same cancel route authorized by actual U certificate only'
            with repo.transaction() as c:
                assert c.execute("SELECT count(*) AS n FROM sessions WHERE state IN ('accepted','computing','releasing')").fetchone()['n']==0
    try: await asyncio.to_thread(clients)
    finally:
        listener.close(); await listener.wait_closed()
        await asyncio.sleep(0.05)
    result['temporary_listener_stopped']=True
    return result


def main():
    pg=Path('/usr/lib/postgresql/18/bin')
    results={'scope':'isolated_real_mtls_public_cancel_debug','formal_acceptance':False,
             'private_model_executed':False,'production_gate_bypassed':False,'fake_backend_used':False}
    with tempfile.TemporaryDirectory(prefix='prag-cancel-mtls-') as temporary:
        root=Path(temporary); data=root/'pg'
        subprocess.run([str(pg/'initdb'),'-D',str(data),'-A','trust','--no-locale','-E','UTF8'],check=True,capture_output=True)
        subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-l',str(root/'pg.log'),'-o',f"-k {root} -h '' -p 55487",'-w','start'],check=True,capture_output=True)
        try:
            repo=Repository(f'host={root} port=55487 dbname=postgres',ObjectStore(root/'objects')); repo.initialize()
            results.update(asyncio.run(exercise(repo,root)))
        finally:
            subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-m','fast','-w','stop'],check=True,capture_output=True)
            results['temporary_postgresql_stopped']=True
    target=Path(__file__).resolve().parents[1]/'records/cancellation_mtls_debug.json'
    target.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
