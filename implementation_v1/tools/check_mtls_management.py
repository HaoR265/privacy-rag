#!/usr/bin/env python3
"""Real localhost TLS 1.3 + mTLS + isolated PostgreSQL integration debugging."""
from __future__ import annotations
import asyncio
import json
import secrets
import ssl
import subprocess
import tempfile
import uuid
from pathlib import Path
import httpx
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from private_rag.pki import create_local_pki
from private_rag.tls_server import tls_context,serve
from private_rag.database import Repository
from private_rag.storage import ObjectStore
from private_rag.management import create_app
from private_rag.client import TLSFiles, UserClient, ProviderClient
from private_rag.application import UserApplication, ProviderApplication, _wait_for_server_completion
from private_rag.compute_process import WorkerSettings
from private_rag.delivery import DeliveryBinding, PreparationShare, MASK
from private_rag.lifecycle import Lifecycle
from private_rag.errors import ErrorCode, PublicError
from private_rag.vault import RecipientKeyStore, ProviderVault

async def exercise(repo,root):
    pki=root/'pki'; certificates=create_local_pki(pki)
    user,owner=str(uuid.uuid4()),str(uuid.uuid4())
    repo.enroll_account(user,'U',certificates['U']['certificate_sha256'])
    repo.enroll_account(owner,'D',certificates['D']['certificate_sha256'])
    def context(role):
        return tls_context(str(pki/(role+'.pem')),str(pki/(role+'.key')),str(pki/'ca.pem'),server=role.startswith('S-'))
    app=create_app(repo); package_fetches=[]
    async def counted_app(scope,receive,send):
        if scope.get('path','').startswith('/v1/sync/packages/'):
            package_fetches.append(scope['path'])
        await app(scope,receive,send)
    listener=await serve(counted_app,'127.0.0.1',0,context('S-management'))
    port=listener.sockets[0].getsockname()[1]; url=f'https://localhost:{port}'
    results={}
    try:
        async with httpx.AsyncClient(verify=context('U'),trust_env=False,timeout=10) as client:
            key=X25519PrivateKey.generate()
            response=await client.post(url+'/v1/receiver-keys',json={'public_key_hex':key.public_key().public_bytes_raw().hex()})
            assert response.status_code==200 and response.json()['key_version']==1,response.text
            response=await client.get(url+'/v1/sync',headers={'x-user-id':owner})
            assert response.status_code==200,response.text
            manifest=response.json(); assert manifest['N']==0 and manifest['user_id']==user
            response=await client.post(url+'/v1/queries/accept',json={'manifest_sha256':manifest['manifest_sha256'],'model_numeric_version':'0'*64})
            assert response.status_code==200 and response.json()['status']=='no_authorized_material'
            response=await client.post(url+'/v1/documents',json={'organization_id':'debug','internal_title':'public'},headers={'x-user-id':owner})
            assert response.status_code==403
        async with httpx.AsyncClient(verify=context('D'),trust_env=False,timeout=10) as client:
            response=await client.get(url+'/v1/sync',headers={'x-user-id':user})
            assert response.status_code==403
            response=await client.post(url+'/v1/documents',json={'organization_id':'debug','internal_title':'public'})
            assert response.status_code==200
        def production_clients():
            # Public synthetic preparation is inserted only here through the
            # isolated debug repository. The production client has no such API.
            def files(role):
                return TLSFiles(pki/(role+'.pem'),pki/(role+'.key'),pki/'ca.pem',certificates['S-management']['certificate_sha256'])
            keys=RecipientKeyStore(user); temporary_versions=[]
            project=Path(__file__).resolve().parents[1]
            config=(project/'artifacts/configuration.sha256').read_text().strip()
            results['configuration_sha256']=config
            def compute_settings(role):
                return WorkerSettings(project,pki/'ca.pem',pki/(role+'.pem'),pki/(role+'.key'),pki/'S-compute.pem',
                                      'localhost',certificates['S-compute']['certificate_sha256'],root/'must-not-create-ledger')
            with UserClient(url,files('U'),user,config,timeout=10) as recipient, ProviderClient(url,files('D'),timeout=10) as provider:
                try:
                    assert recipient.register_new_key()==2; temporary_versions.append(2)
                    assert recipient.synchronize().records==()
                    assert recipient.accept_query()['status']=='no_authorized_material'
                    application=UserApplication(recipient,compute_settings('U'),'127.0.0.1')
                    empty_answer=application.ask('公开调试问题')
                    assert empty_answer.status=='no_authorized_material' and empty_answer.text=='当前没有可用于回答的授权资料'
                    document=provider.create_document('public-debug','client synthetic input')
                    version=provider.create_version(document,2,config)
                    provider_application=ProviderApplication(provider,compute_settings('D'),config,'127.0.0.1')
                    chunk=provider_application.prepare_document('公开合成资料块',document,version)[0]
                    try:
                        provider_application.prepare_chunk(document,version,0,chunk,vault=ProviderVault(owner,'must-not-create-key'),
                            retained_chunk_path=root/'must-not-create-chunk',retained_preparation_path=root/'must-not-create-preparation')
                    except PublicError as exc: assert exc.code==ErrorCode.BACKEND_NOT_READY
                    else: raise AssertionError('D application bypassed private backend gate')
                    assert not (root/'must-not-create-chunk').exists() and not (root/'must-not-create-preparation').exists()
                    try: provider.start_preparation(document,version,0)
                    except PublicError as exc: assert exc.code==ErrorCode.BACKEND_NOT_READY and str(exc)=='BACKEND_NOT_READY'
                    else: raise AssertionError('production preparation gate did not reject missing backend')
                    results['preparation_admission_gate']='real mTLS route rejects BACKEND_NOT_READY before session insertion'
                    assert provider.authorize(document,user,True)==1
                    plan=provider.delivery_plan(document,version,user)
                    life=Lifecycle(repo)
                    for index,binding in enumerate(plan['bindings']):
                        preparation=str(uuid.uuid4()); dz=tuple(secrets.randbits(59) for _ in range(1024)); de=secrets.randbits(59)
                        dshare=PreparationShare(preparation,'D',dz,de)
                        sshare=PreparationShare(preparation,'S',tuple((-x)&MASK for x in dz),(1-de)&MASK)
                        session=repo.start_preparation(owner,document,version,index)
                        life.transition(session,'computing'); life.transition(session,'releasing'); life.finish(session,acknowledged=True)
                        repo.retain_completed_preparation(session,sshare)
                        _wait_for_server_completion(provider,session)
                        assert provider.session_status(session)['total_bytes'] is None
                        try: recipient.session_status(session)
                        except PublicError as exc: assert exc.code==ErrorCode.NOT_AUTHORIZED
                        else: raise AssertionError('another account could read provider session')
                        provider.stage_prepared_block(document,DeliveryBinding(**binding),plan['public_key_hex'],tuple([42+index]+[0]*255),tuple([1]+[0]*255),dshare)
                    provider.publish_version(document,version)
                    first=recipient.synchronize(); second=recipient.synchronize()
                    assert len(first.records)==2 and first.records==second.records
                    assert len(package_fetches)==4 and len(set(package_fetches))==2
                    try: recipient.accept_query()
                    except PublicError as exc: assert exc.code==ErrorCode.BACKEND_NOT_READY and str(exc)=='BACKEND_NOT_READY'
                    else: raise AssertionError('production activation gate did not reject missing backend')
                    try: application.ask('公开调试问题')
                    except PublicError as exc: assert exc.code==ErrorCode.BACKEND_NOT_READY
                    else: raise AssertionError('U application bypassed private backend gate')
                    assert len(package_fetches)==6 and not (root/'must-not-create-ledger').exists()
                    results['owner_local_application']='actual tokenizer preprocessing and mTLS N0 response; N>0 fresh full sync then closed gate; D selected chunk rejected before local files, acceptance, sockets or worker'
                    provider.authorize(document,user,False)
                    assert recipient.synchronize().records==()
                    assert len(first.records)==2  # immutable held snapshot
                    results['production_clients']='OS keystore registration; D metadata/ACL/reshare/HPKE/stage/publish; complete U sync twice; revoke; backend gate fixed error'
                    results['full_sync_package_fetches']=len(package_fetches)
                    results['full_sync_N']=2
                    results['preparation_source']='public_synthetic_debug_only_not_private_embedding'
                    results['preparation_persist_ack']='D reads own completed+release_acknowledged via mTLS; foreign account rejected; unknown total_bytes remains null'
                finally:
                    backend=keys._backend()
                    # The only new version this isolated, random user can own.
                    for version in (2,):
                        name=f'{user}:{version}'
                        if backend.get_password(keys.SERVICE,name) is not None:
                            backend.delete_password(keys.SERVICE,name)
                        assert backend.get_password(keys.SERVICE,name) is None
                    results['temporary_client_hpke_key_deleted']=True
        await asyncio.to_thread(production_clients)
        # TLS requires a client certificate before any HTTP spoof header is read.
        anonymous=ssl.create_default_context(cafile=str(pki/'ca.pem'))
        anonymous.minimum_version=anonymous.maximum_version=ssl.TLSVersion.TLSv1_3
        anonymous.set_ecdh_curve('X25519')
        async with httpx.AsyncClient(verify=anonymous,trust_env=False,timeout=10) as client:
            try:
                await client.get(url+'/v1/sync',headers={'x-user-id':user,'x-client-cert':certificates['U']['certificate_sha256']})
            except httpx.TransportError:
                results['missing_client_certificate']='TLS connection rejected'
            else:
                raise AssertionError('TLS admitted unauthenticated client')
        # Independent untrusted project CA cannot authenticate to this server.
        alien=root/'alien-pki'; create_local_pki(alien)
        untrusted=tls_context(str(alien/'U.pem'),str(alien/'U.key'),str(pki/'ca.pem'),server=False)
        async with httpx.AsyncClient(verify=untrusted,trust_env=False,timeout=10) as client:
            try: await client.get(url+'/v1/status')
            except httpx.TransportError: results['untrusted_client_CA']='TLS connection rejected'
            else: raise AssertionError('TLS admitted untrusted client')
        # An actual socket exposes the negotiated version (management HTTP API
        # never reports key material, question text or secret intermediates).
        reader,writer=await asyncio.open_connection('127.0.0.1',port,ssl=context('U'),server_hostname='localhost')
        results['negotiated_tls']=writer.get_extra_info('ssl_object').version()
        writer.close(); await writer.wait_closed()
        assert results['negotiated_tls']=='TLSv1.3'
        results['authenticated_routes']='U key registration/full sync/N0; D metadata registration; role and header spoof rejection'
    finally:
        listener.close(); await listener.wait_closed()
        # All client contexts have closed their own connections by this point.
        await asyncio.sleep(0.05)
        results['temporary_listener_stopped']=True
    return results

def main():
    pg=Path('/usr/lib/postgresql/18/bin')
    results={'scope':'isolated_real_transport_and_management_debug','formal_acceptance':False,'private_computation_executed':False}
    with tempfile.TemporaryDirectory(prefix='prag-mtls-') as temporary:
        root=Path(temporary);data=root/'pg'
        subprocess.run([str(pg/'initdb'),'-D',str(data),'-A','trust','--no-locale','-E','UTF8'],check=True,capture_output=True)
        subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-l',str(root/'pg.log'),'-o',f"-k {root} -h '' -p 55484",'-w','start'],check=True,capture_output=True)
        try:
            repo=Repository(f'host={root} port=55484 dbname=postgres',ObjectStore(root/'objects'));repo.initialize()
            results.update(asyncio.run(exercise(repo,root)))
        finally:
            subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-m','fast','-w','stop'],check=True,capture_output=True)
            results['temporary_postgresql_stopped']=True
    target=Path(__file__).resolve().parents[1]/'records'/'management_mtls_debug.json'
    target.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
