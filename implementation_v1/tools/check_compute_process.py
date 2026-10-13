#!/usr/bin/env python3
"""Public binding/gate/resource debugging only; no fake or enabled backend."""
from __future__ import annotations
import dataclasses
import json
import os
from pathlib import Path
import socket
import tempfile
import uuid
from private_rag.client import SynchronizedCollection
from private_rag.compute_input import WorkerBinding
from private_rag.compute_process import (ConnectedWorkerSockets, WorkerSettings,
    _validate_snapshot, validate_public_configuration, run_worker)
from private_rag.delivery import DeliveryBinding, ShareRecord, canonical_json, digest
from private_rag.errors import ErrorCode, PublicError
from private_rag.pki import create_local_pki


def expect(code, call):
    try: call()
    except PublicError as exc: assert exc.code==code and str(exc)==code.value
    else: raise AssertionError('missing fixed public rejection')


def main():
    root=Path(__file__).resolve().parents[1]
    config=(root/'artifacts/configuration.sha256').read_text().strip()
    user,session=str(uuid.uuid4()),str(uuid.uuid4())
    binding=DeliveryBinding(user,1,str(uuid.uuid4()),1,1,config)
    manifest={'format_version':1,'user_id':user,'authorization_version':1,'N':1,
              'entries':[{'binding':binding.public_dict(),'package_sha256':'0'*64}]}
    manifest['manifest_sha256']=digest(canonical_json(manifest))
    snapshot={**manifest,'session_id':session,'model_numeric_version':config,'protocol_version':'prag-v1',
              'fixed_shapes':{'t':256,'m':256,'z':1024,'e':1,'prefill':1024,'output':256}}
    worker=WorkerBinding(session,config,digest(canonical_json(snapshot)),'C','query')
    # Public synthetic one-party record, never a replacement compute backend.
    record=ShareRecord('U',(0,)*256,(0,)*256,(0,)*1024,0)
    collection=SynchronizedCollection(canonical_json(manifest),(record,))
    assert _validate_snapshot(worker,snapshot,collection,())==(record,)
    expect(ErrorCode.VERSION_MISMATCH,lambda:_validate_snapshot(dataclasses.replace(worker,snapshot_sha256='0'*64),snapshot,collection,()))
    expect(ErrorCode.BAD_STATE,lambda:_validate_snapshot(worker,snapshot,collection,(dataclasses.replace(record,party='S'),)))
    expect(ErrorCode.BAD_STATE,lambda:_validate_snapshot(worker,snapshot,SynchronizedCollection(canonical_json(manifest),(dataclasses.replace(record,party='S'),)),()))
    with tempfile.TemporaryDirectory(prefix='prag-worker-binding-') as temporary:
        folder=Path(temporary); pki=create_local_pki(folder/'pki')
        settings=WorkerSettings(root,folder/'pki/ca.pem',folder/'pki/U.pem',folder/'pki/U.key',
                                folder/'pki/S-compute.pem','localhost',pki['S-compute']['certificate_sha256'],folder/'ledger')
        assert validate_public_configuration(worker,settings)['ring_bits']==59
        expect(ErrorCode.AUTHENTICATION_FAILED,lambda:validate_public_configuration(worker,dataclasses.replace(settings,peer_certificate_sha256='0'*64)))
        peers=[]; owned=[]
        try:
            for _ in range(3):
                local,remote=socket.socketpair(); owned.append(local.detach());peers.append(remote)
            expect(ErrorCode.BACKEND_NOT_READY,lambda:run_worker(worker,settings,ConnectedWorkerSockets(*owned),snapshot=snapshot,on_event=lambda event:None,collection=collection))
            for fd in owned:
                try: os.fstat(fd)
                except OSError: pass
                else: raise AssertionError('gate rejection leaked owned descriptor')
            assert not (folder/'ledger').exists()
        finally:
            for peer in peers: peer.close()
    result={'scope':'public_compute_process_boundary_debug','formal_acceptance':False,'private_model_executed':False,'configuration_sha256':config,
            'configuration_actual_files':'client scope hashes verified','peer_certificate_pin':'actual generated Ed25519 certificate verified',
            'one_party_boundary':'wrong party and mixed-party inputs rejected','snapshot_binding':'canonical hash mismatch rejected',
            'activation_gate':'BACKEND_NOT_READY before memfd/ledger/worker creation','owned_socket_fds':'closed on gate rejection',
            'fake_backend_used':False,'temporary_certificates_removed':True}
    (root/'records/compute_process_debug.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
