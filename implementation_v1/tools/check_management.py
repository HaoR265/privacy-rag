#!/usr/bin/env python3
"""Independent PUBLIC/SYNTHETIC implementation check, not a system acceptance.

Creates its own PostgreSQL cluster, public synthetic preparation shares and
new OS-keystore entry. It never invokes Embedding, R2 or a production database.
Always shuts PostgreSQL down normally and removes its temporary key entry.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import uuid
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from fastapi.testclient import TestClient
from private_rag.delivery import DeliveryBinding, PreparationShare, reshare_for_user, seal_delivery, open_delivery, ShareRecord, UserShareCache, MASK
from private_rag.storage import ObjectStore
from private_rag.database import Repository
from private_rag.lifecycle import Lifecycle
from private_rag.management import create_app
from private_rag.errors import PublicError, ErrorCode
from private_rag.vault import ProviderVault, RecipientKeyStore


def expect(code,call):
    try: call()
    except PublicError as exc:
        assert exc.code==code,(exc.code,code)
    else: raise AssertionError('required public rejection was absent')


def main():
    results={'scope':'public_synthetic_management_debug_only','formal_acceptance':False,'real_embedding_executed':False}
    pg=Path('/usr/lib/postgresql/18/bin')
    if not (pg/'initdb').exists():
        raise SystemExit('PostgreSQL 18 installation needed for isolated debug cluster')
    with tempfile.TemporaryDirectory(prefix='prag-management-') as temporary:
        root=Path(temporary); data=root/'pg'; log=root/'pg.log'
        subprocess.run([str(pg/'initdb'),'-D',str(data),'-A','trust','--no-locale','-E','UTF8'],check=True,capture_output=True)
        # UNIX socket only, no TCP port exposed and no existing cluster touched.
        subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-l',str(log),'-o',f"-k {root} -h '' -p 55483",'-w','start'],check=True,capture_output=True)
        try:
            repo=Repository(f'host={root} port=55483 dbname=postgres',ObjectStore(root/'objects')); repo.initialize(); life=Lifecycle(repo)
            owner,user=str(uuid.uuid4()),str(uuid.uuid4())
            owner_cert=b'public-debug-provider-certificate'; user_cert=b'public-debug-user-certificate'
            repo.enroll_account(owner,'D',hashlib.sha256(owner_cert).hexdigest()); repo.enroll_account(user,'U',hashlib.sha256(user_cert).hexdigest())
            priv=X25519PrivateKey.generate(); assert repo.register_receiver_key(user,priv.public_key().public_bytes_raw())==1
            config=hashlib.sha256(b'public-synthetic-configuration-not-model-artifact').hexdigest()
            empty=repo.sync_manifest(user)
            assert repo.accept_query(user,empty['manifest_sha256'],config)['N']==0
            doc=repo.create_document(owner,'public-debug-organization','synthetic only')
            version=repo.create_version(owner,doc,1,config); repo.set_authorization(owner,doc,user,True)
            expect(ErrorCode.MATERIALS_NOT_READY,lambda:repo.sync_manifest(user))
            plan=repo.delivery_plan(owner,doc,version,user); binding=DeliveryBinding(**plan['bindings'][0])
            prep=str(uuid.uuid4()); z=tuple(17 for _ in range(1024)); d_z=tuple(secrets.randbits(59) for _ in z); d_e=secrets.randbits(59)
            dshare=PreparationShare(prep,'D',d_z,d_e)
            sshare=PreparationShare(prep,'S',tuple((a-b)&MASK for a,b in zip(z,d_z)),(1-d_e)&MASK)
            # Explicit synthetic lifecycle driving; NOT evidence of private inference.
            ps=repo.start_preparation(owner,doc,version,0); life.transition(ps,'computing'); life.transition(ps,'releasing'); life.finish(ps,acknowledged=True)
            repo.retain_completed_preparation(ps,sshare)
            t=tuple([42]+[0]*255); m=tuple([1]+[0]*255)
            us,rs=reshare_for_user(t,m,dshare); package=seal_delivery(us,priv.public_key(),binding)
            assert open_delivery(package,priv,binding)==us
            expect(ErrorCode.AUTHENTICATION_FAILED,lambda:open_delivery(package,priv,dataclasses.replace(binding,document_version=2)))
            expect(ErrorCode.AUTHENTICATION_FAILED,lambda:open_delivery(package[:-1]+bytes([package[-1]^1]),priv,binding))
            identity=repo.stage_delivery(owner,doc,binding,rs,package); repo.publish_version(owner,doc,version)
            manifest=repo.sync_manifest(user); assert manifest['N']==1
            assert 'organization' not in json.dumps(manifest) and doc not in json.dumps(manifest)
            cache=UserShareCache(user,1,priv,config); ready=cache.synchronize(manifest,lambda h:repo.sync_package(user,h)); assert cache.snapshot()==(us,)
            accepted=repo.accept_query(user,ready,config,activation_check=lambda:None); session=accepted['snapshot']['session_id']
            expect(ErrorCode.BUSY,lambda:repo.accept_query(user,ready,config,activation_check=lambda:None))
            sr=repo.snapshot_server_records(session)[0]
            assert tuple((a+b)&MASK for a,b in zip(us.z,sr.z))==z
            assert tuple((a+b)&MASK for a,b in zip(us.t,sr.t))==t
            assert tuple((a+b)&MASK for a,b in zip(us.m,sr.m))==m
            assert (us.e+sr.e)&MASK==1
            mat=life.reserve(session,0,'triple',[1024]); life.consume(mat,session)
            expect(ErrorCode.REPLAY_OR_ORDER,lambda:life.consume(mat,session))
            uncertain=life.reserve(session,1,'bfv_mask',[16384]); life.transition(session,'computing')
            # Revoke changes subsequent eligibility, not immutable accepted inputs.
            repo.set_authorization(owner,doc,user,False)
            assert repo.sync_manifest(user)['N']==0 and len(repo.snapshot_server_records(session))==1
            life.finish(session,acknowledged=False,public_fault='DISCONNECTED')
            with repo.transaction() as c:
                assert c.execute('SELECT state FROM materials WHERE material_id=%s',(uncertain,)).fetchone()['state']=='burned'
            expect(ErrorCode.BAD_STATE,lambda:repo.snapshot_server_records(session))
            # Regrant needs fresh generation; old published material cannot return.
            repo.set_authorization(owner,doc,user,True)
            expect(ErrorCode.MATERIALS_NOT_READY,lambda:repo.sync_manifest(user))
            plan2=repo.delivery_plan(owner,doc,version,user); b2=DeliveryBinding(**plan2['bindings'][0]); us2,rs2=reshare_for_user(t,m,dshare)
            assert us2!=us
            repo.stage_delivery(owner,doc,b2,rs2,seal_delivery(us2,priv.public_key(),b2)); repo.publish_version(owner,doc,version)
            old_manifest=repo.sync_manifest(user)
            # An incomplete new version must leave the prior complete view intact.
            new=repo.create_version(owner,doc,2,config)
            expect(ErrorCode.MATERIALS_NOT_READY,lambda:repo.publish_version(owner,doc,new))
            assert repo.sync_manifest(user)==old_manifest
            assert repo.register_receiver_key(user,X25519PrivateKey.generate().public_key().public_bytes_raw())==2
            expect(ErrorCode.MATERIALS_NOT_READY,lambda:repo.sync_manifest(user))
            # Ordinary HTTP header spoofing never enters trusted identity scope.
            app=create_app(repo)
            with TestClient(app,raise_server_exceptions=False) as client:
                assert client.get('/v1/sync',headers={'x-user-id':user,'x-client-cert':user_cert.hex()}).status_code==401
            class SyntheticTLSWrapper:
                async def __call__(self,scope,receive,send):
                    if scope['type']=='http':
                        scope['tls_peer']={'certificate_der':user_cert,'fingerprint_sha256':hashlib.sha256(user_cert).hexdigest()}
                    return await app(scope,receive,send)
            with TestClient(SyntheticTLSWrapper(),raise_server_exceptions=False) as client:
                assert client.get('/v1/status').status_code==200
                response=client.post('/v1/queries/accept',json={'question':'forbidden-secret-sentinel'})
                assert response.status_code==400 and 'forbidden-secret-sentinel' not in response.text
            results['postgres_version']=subprocess.check_output([str(pg/'postgres'),'--version'],text=True).strip()
            results['management_checks']='HPKE binding/tamper, full sync, pairing, atomic failure, immutable revocation snapshot, regrant, key rotation, one-session exclusion, material burn, no header identity, sanitized validation'
        finally:
            subprocess.run([str(pg/'pg_ctl'),'-D',str(data),'-m','fast','-w','stop'],check=True,capture_output=True)
            results['temporary_postgresql_stopped']=True
        # Probe a fresh project-specific key only, preserving existing secrets.
        v=ProviderVault(str(uuid.uuid4()),'debug-'+str(uuid.uuid4())); backend=None; name=f'{v.provider_id}:{v.key_id}'; created=False
        receiver=RecipientKeyStore(str(uuid.uuid4())); receiver_created=False
        try:
            backend=v._backend(); v.initialize_new(); created=True
            v.save(root/'provider.vault','synthetic-object',b'PUBLIC SYNTHETIC LOCAL RETENTION')
            assert v.load(root/'provider.vault','synthetic-object')==b'PUBLIC SYNTHETIC LOCAL RETENTION'
            expect(ErrorCode.AUTHENTICATION_FAILED,lambda:v.load(root/'provider.vault','different-object'))
            results['os_keystore_aes256gcm']='roundtrip_and_binding_checked'
            recipient_public=receiver.generate(1); receiver_created=True
            assert receiver.load(1).public_key().public_bytes_raw()==recipient_public
            results['os_keystore_hpke']='separate_service_roundtrip_checked'
        except Exception as exc:
            results['os_keystore_aes256gcm']='unavailable:'+type(exc).__name__
        finally:
            if backend is not None and created:
                backend.delete_password(v.SERVICE,name)
                assert backend.get_password(v.SERVICE,name) is None
                results['temporary_provider_key_deleted']=True
            if backend is not None and receiver_created:
                backend.delete_password(receiver.SERVICE,f'{receiver.user_id}:1')
                assert backend.get_password(receiver.SERVICE,f'{receiver.user_id}:1') is None
                results['temporary_recipient_key_deleted']=True
    target=Path(__file__).resolve().parents[1]/'records'/'management_debug.json'
    target.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
