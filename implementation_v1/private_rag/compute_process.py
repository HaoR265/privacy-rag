"""One party's gated native-worker launcher and local pipe result boundary.

This does not allocate computation sessions or implement the server listener.
The scheduler must supply one authenticated, frozen public snapshot and three
connected sockets for that same peer. This module never receives both shares.
"""
from __future__ import annotations
import hashlib
import json
import os
import selectors
import socket
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID
from typing import Callable
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .activation import require_private_backend
from .client import SynchronizedCollection
from .compute_input import WorkerBinding, sealed_input
from .configuration import check_configuration
from .deployment import verify_deployment
from .delivery import DeliveryBinding, ShareRecord, MASK, canonical_json, digest
from .delivery import PreparationShare
from .errors import ErrorCode, PublicError
from .model_input import LocalTokenSlots, QuestionInput
from .native_ledger import verify_closed_ledger
from .worker_cleanup import cleanup_worker, close_owned, duplicate_lanes, defer_cancellation_signals
from .tls_server import tls_context
from .vault import ProviderVault


@dataclass(frozen=True)
class ConnectedWorkerSockets:
    """Own socket objects until run_worker copies fds into its private owner.

    No bare descriptor transfer: closing the outer objects remains idempotent
    even after their numeric descriptors have been reused elsewhere.
    """
    compute: socket.socket
    heartbeat_send: socket.socket
    heartbeat_receive: socket.socket

    def channels(self):
        return self.compute,self.heartbeat_send,self.heartbeat_receive

    def acquire(self, owned: set[int]) -> tuple[int,int,int]:
        with defer_cancellation_signals():
            channels=self.channels()
            if any(not isinstance(channel,socket.socket) for channel in channels):
                raise PublicError(ErrorCode.BAD_FORMAT)
            descriptors=tuple(channel.fileno() for channel in channels)
            if any(fd<3 for fd in descriptors) or len(set(descriptors))!=3:
                raise PublicError(ErrorCode.BAD_FORMAT)
            copies=duplicate_lanes(descriptors,owned)
            self.close()
            return copies

    def close(self):
        original_failure=sys.exc_info()[0] is not None
        first_error=None
        with defer_cancellation_signals():
            for channel in self.channels():
                if isinstance(channel,socket.socket):
                    try: channel.close()
                    except OSError as error:
                        if first_error is None: first_error=error
            if first_error is not None and not original_failure:
                raise first_error

    def __enter__(self): return self
    def __exit__(self,*unused): self.close()


@dataclass(frozen=True)
class WorkerSettings:
    root: Path
    ca_certificate: Path
    certificate: Path
    private_key: Path
    peer_certificate: Path
    peer_name: str
    peer_certificate_sha256: str
    ledger_directory: Path


@dataclass(frozen=True,repr=False)
class LocalAnswer:
    tokens: tuple[int,...]
    validity: tuple[bool,...]
    status: int


@dataclass(frozen=True,repr=False)
class LocalPreparationResult:
    party: str
    embedding: tuple[int,...]
    eligible: int
    # Only C/D may receive the private preparation failure bit.
    client_failed: bool | None


def retain_provider_preparation(binding: WorkerBinding,result: LocalPreparationResult,
                                vault: ProviderVault,path: Path) -> PreparationShare:
    """D local: persist successful own shares encrypted before any delivery.

    A private numeric failure never becomes a publishable PreparationShare and
    is not reported to S by this function. The caller retains the source/body
    chunk in its own vault independently under the same document version.
    """
    if binding.party!='C' or binding.mode!='prepare' or result.party!='C' or result.client_failed is not False:
        raise PublicError(ErrorCode.MATERIALS_NOT_READY)
    binding.encode(0)
    share=PreparationShare(binding.session_id,'D',result.embedding,result.eligible)
    payload=b'PRAGDP01'+UUID(binding.session_id).bytes+bytes.fromhex(binding.configuration)+bytes.fromhex(binding.snapshot_sha256)+struct.pack('<1025Q',*(share.z+(share.e,)))
    vault.save(path,binding.session_id,payload)
    return share


def load_provider_preparation(binding: WorkerBinding,vault: ProviderVault,path: Path) -> PreparationShare:
    if binding.party!='C' or binding.mode!='prepare': raise PublicError(ErrorCode.BAD_STATE)
    binding.encode(0)
    payload=vault.load(path,binding.session_id)
    prefix=b'PRAGDP01'+UUID(binding.session_id).bytes+bytes.fromhex(binding.configuration)+bytes.fromhex(binding.snapshot_sha256)
    if len(payload)!=8288 or payload[:88]!=prefix: raise PublicError(ErrorCode.VERSION_MISMATCH)
    values=struct.unpack('<1025Q',payload[88:])
    return PreparationShare(binding.session_id,'D',values[:1024],values[1024])


def _sha(value):
    if not isinstance(value,str) or len(value)!=64:
        raise PublicError(ErrorCode.BAD_FORMAT)
    try:
        if bytes.fromhex(value).hex()!=value: raise ValueError()
    except ValueError: raise PublicError(ErrorCode.BAD_FORMAT) from None


def _validate_snapshot(binding: WorkerBinding, snapshot: dict,
                       collection: SynchronizedCollection | None,
                       server_records: tuple[ShareRecord,...]) -> tuple[ShareRecord,...]:
    """Public field/one-party boundary; does not enable or run a backend."""
    try:
        if str(UUID(binding.session_id))!=binding.session_id: raise ValueError()
    except (ValueError,TypeError,AttributeError): raise PublicError(ErrorCode.BAD_FORMAT) from None
    _sha(binding.configuration); _sha(binding.snapshot_sha256)
    if binding.party not in ('C','S') or binding.mode not in ('prepare','query'):
        raise PublicError(ErrorCode.BAD_FORMAT)
    if not isinstance(snapshot,dict) or digest(canonical_json(snapshot))!=binding.snapshot_sha256 or snapshot.get('model_numeric_version')!=binding.configuration:
        raise PublicError(ErrorCode.VERSION_MISMATCH)
    if binding.mode=='prepare':
        if collection is not None or server_records: raise PublicError(ErrorCode.BAD_SHAPE)
        # Preparation snapshots have no query manifest; their fixed body is
        # from Repository.start_preparation's owner/version/block acceptance.
        if set(snapshot)!={'document_id','document_version','block_index','model_numeric_version'}:
            raise PublicError(ErrorCode.BAD_FORMAT)
        if type(snapshot['document_version']) is not int or snapshot['document_version']<1 or type(snapshot['block_index']) is not int or snapshot['block_index']<0:
            raise PublicError(ErrorCode.BAD_FORMAT)
        try:
            if str(UUID(snapshot['document_id']))!=snapshot['document_id']: raise ValueError()
        except (ValueError,TypeError,AttributeError): raise PublicError(ErrorCode.BAD_FORMAT) from None
        return ()
    fields={'format_version','user_id','authorization_version','N','entries','manifest_sha256'}
    if set(snapshot)!=fields|{'session_id','model_numeric_version','protocol_version','fixed_shapes'} or snapshot.get('session_id')!=binding.session_id or snapshot.get('protocol_version')!='prag-v1':
        raise PublicError(ErrorCode.VERSION_MISMATCH)
    if snapshot['fixed_shapes']!={'t':256,'m':256,'z':1024,'e':1,'prefill':1024,'output':256}:
        raise PublicError(ErrorCode.BAD_SHAPE)
    manifest={k:snapshot[k] for k in fields}
    if type(manifest['N']) is not int or manifest['N']<1 or not isinstance(manifest['entries'],list) or manifest['N']!=len(manifest['entries']):
        raise PublicError(ErrorCode.BAD_SHAPE)
    if type(manifest['format_version']) is not int or manifest['format_version']!=1 or type(manifest['authorization_version']) is not int or manifest['authorization_version']<0:
        raise PublicError(ErrorCode.BAD_FORMAT)
    if digest(canonical_json({k:v for k,v in manifest.items() if k!='manifest_sha256'}))!=manifest['manifest_sha256']:
        raise PublicError(ErrorCode.VERSION_MISMATCH)
    slots=[]
    for entry in manifest['entries']:
        if not isinstance(entry,dict) or set(entry)!={'binding','package_sha256'}: raise PublicError(ErrorCode.BAD_FORMAT)
        _sha(entry['package_sha256'])
        try: record_binding=DeliveryBinding(**entry['binding'])
        except TypeError: raise PublicError(ErrorCode.BAD_FORMAT) from None
        if record_binding.user_id!=manifest['user_id'] or record_binding.model_numeric_version!=binding.configuration:
            raise PublicError(ErrorCode.VERSION_MISMATCH)
        slots.append(record_binding.opaque_slot)
    if slots!=sorted(slots) or len(set(slots))!=len(slots): raise PublicError(ErrorCode.BAD_FORMAT)
    if binding.party=='C':
        if collection is None or server_records: raise PublicError(ErrorCode.BAD_STATE)
        if collection.manifest!=manifest: raise PublicError(ErrorCode.VERSION_MISMATCH)
        records=collection.records
    else:
        if collection is not None: raise PublicError(ErrorCode.BAD_STATE)
        records=server_records
    if not isinstance(records,tuple) or len(records)!=manifest['N']:
        raise PublicError(ErrorCode.BAD_SHAPE)
    if any(not isinstance(record,ShareRecord) or record.party!=('U' if binding.party=='C' else 'S') for record in records):
        raise PublicError(ErrorCode.BAD_STATE)
    return records


def validate_public_configuration(binding: WorkerBinding, settings: WorkerSettings) -> dict:
    """Hash and credential checks only; never bypasses the run activation gate."""
    _sha(binding.configuration); _sha(settings.peer_certificate_sha256)
    if binding.party not in ('C','S') or not settings.peer_name or len(settings.peer_name)>253 or '\x00' in settings.peer_name:
        raise PublicError(ErrorCode.BAD_FORMAT)
    try:
        config=check_configuration(Path(settings.root),binding.configuration,server_weights=binding.party=='S')
        if config.get('protocol')!='prag-v1' or config.get('ring_bits')!=59:
            raise PublicError(ErrorCode.VERSION_MISMATCH)
        for path in (settings.certificate,settings.peer_certificate):
            cert=x509.load_pem_x509_certificate(Path(path).read_bytes())
            if not isinstance(cert.public_key(),Ed25519PublicKey): raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
            if path==settings.peer_certificate and hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()!=settings.peer_certificate_sha256:
                raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
        key_path=Path(settings.private_key)
        if key_path.is_symlink() or not key_path.is_file() or stat.S_IMODE(key_path.stat().st_mode)&0o077:
            raise PublicError(ErrorCode.BAD_STATE)
        # Validate local certificate/key correspondence and CA availability.
        tls_context(str(settings.certificate),str(settings.private_key),str(settings.ca_certificate),server=binding.party=='S')
        return config
    except PublicError: raise
    except (OSError,ValueError,TypeError,KeyError):
        raise PublicError(ErrorCode.MATERIALS_NOT_READY) from None


def _validate_sockets(descriptors: tuple[int,int,int]):
    if any(type(fd) is not int or fd<3 for fd in descriptors) or len(set(descriptors))!=3:
        raise PublicError(ErrorCode.BAD_FORMAT)
    for fd in descriptors:
        with defer_cancellation_signals():
            channel=socket.socket(fileno=fd)
            try:
                if channel.family not in (socket.AF_INET,socket.AF_INET6) or channel.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE)!=socket.SOCK_STREAM:
                    raise PublicError(ErrorCode.BAD_FORMAT)
                channel.getpeername()
            finally: channel.detach()


def _close(fd):
    if type(fd) is int and fd>=3:
        try: os.close(fd)
        except OSError: pass


def _decode_result(binding: WorkerBinding, data: bytes):
    if binding.mode=='prepare':
        expected=8201 if binding.party=='C' else 8200
        if len(data)!=expected: raise PublicError(ErrorCode.BAD_SHAPE)
        values=struct.unpack('<1025Q',data[:8200])
        if any(v>MASK for v in values) or (binding.party=='C' and data[-1]>1):
            raise PublicError(ErrorCode.BAD_FORMAT)
        return LocalPreparationResult(binding.party,values[:1024],values[1024],bool(data[-1]) if binding.party=='C' else None)
    if binding.party=='S':
        if data: raise PublicError(ErrorCode.BAD_SHAPE)
        return None
    if len(data)!=2312: raise PublicError(ErrorCode.BAD_SHAPE)
    tokens=struct.unpack('<256Q',data[:2048]); validity=data[2048:2304]; status=struct.unpack('<Q',data[2304:])[0]
    if any(t>151642 for t in tokens) or any(v>1 for v in validity) or status>3:
        raise PublicError(ErrorCode.BAD_FORMAT)
    if any(t!=0 for t,v in zip(tokens,validity) if not v) or (status in (1,2) and any(validity)):
        raise PublicError(ErrorCode.BAD_FORMAT)
    return LocalAnswer(tokens,tuple(bool(v) for v in validity),status)


def _collect_worker_output(process,expected: int,read_fd: int,event_read: int,ack_write: int,
                           on_event: Callable[[str],None],check_cancelled: Callable[[],None] | None=None) -> bytes:
    """Drain both actual pipes concurrently; no backend or elapsed-time gate."""
    selector=selectors.DefaultSelector()
    try:
        selector.register(read_fd,selectors.EVENT_READ); selector.register(event_read,selectors.EVENT_READ)
        result=bytearray(); eof=False; event_eof=False; next_event=1
        while not eof or not event_eof or process.poll() is None:
            if check_cancelled is not None: check_cancelled()
            if process.poll() not in (None,0): raise PublicError(ErrorCode.BAD_STATE)
            if eof and event_eof:
                try: process.wait(timeout=0.25)
                except subprocess.TimeoutExpired: pass
                continue
            for key,_ in selector.select(timeout=0.25):
                if key.fd==event_read:
                    event=os.read(event_read,1)
                    if not event:
                        event_eof=True; selector.unregister(event_read)
                        if next_event!=3: raise PublicError(ErrorCode.BAD_STATE)
                    else:
                        if next_event>2 or event!=bytes([next_event]): raise PublicError(ErrorCode.REPLAY_OR_ORDER)
                        on_event('AUTHENTICATED' if next_event==1 else 'RELEASING')
                        if os.write(ack_write,event)!=1: raise PublicError(ErrorCode.DISCONNECTED)
                        next_event+=1
                    continue
                part=os.read(key.fd,expected+1-len(result))
                if not part:
                    eof=True; selector.unregister(read_fd)
                    if len(result)!=expected: raise PublicError(ErrorCode.BAD_SHAPE)
                else:
                    result.extend(part)
                    if len(result)>expected: raise PublicError(ErrorCode.BAD_SHAPE)
        if process.returncode!=0 or next_event!=3: raise PublicError(ErrorCode.BAD_STATE)
        return bytes(result)
    finally:
        selector.close()


def run_worker(binding: WorkerBinding, settings: WorkerSettings, sockets: ConnectedWorkerSockets,
               *, snapshot: dict, on_event: Callable[[str],None], collection: SynchronizedCollection | None=None,
               server_records: tuple[ShareRecord,...]=(), question: QuestionInput | None=None,
               document: LocalTokenSlots | None=None,check_cancelled: Callable[[],None] | None=None):
    """Run this party once, consuming socket ownership even on gate rejection.

    No two-party orchestration, reconstruction, retry, reference mode or caller
    supplied executable is available. stdout/stderr cannot escape to a log.
    """
    owned=set(); process=None; cancellation_lanes=()
    try:
        require_private_backend()  # MUST precede all secret/memfd/process work.
        descriptors=sockets.acquire(owned)
        if not callable(on_event): raise PublicError(ErrorCode.BAD_STATE)
        if check_cancelled is not None and not callable(check_cancelled): raise PublicError(ErrorCode.BAD_STATE)
        if check_cancelled is not None: check_cancelled()
        records=_validate_snapshot(binding,snapshot,collection,server_records)
        validate_public_configuration(binding,settings)
        _validate_sockets(descriptors)
        cancellation_lanes=duplicate_lanes(descriptors,owned)
        root=Path(settings.root).resolve()
        deployment=verify_deployment(root,binding.configuration)
        worker=deployment.worker
        ledger=Path(settings.ledger_directory).resolve()
        ledger.mkdir(mode=0o700,parents=True,exist_ok=True)
        if stat.S_IMODE(ledger.stat().st_mode)&0o077: raise PublicError(ErrorCode.BAD_STATE)
        ledger_path=ledger/(binding.session_id+'.'+binding.party+'.ledger')
        if ledger_path.exists(): raise PublicError(ErrorCode.REPLAY_OR_ORDER)
        with defer_cancellation_signals():
            input_fd=sealed_input(binding,records,question=question,document=document); owned.add(input_fd)
            read_fd,write_fd=os.pipe2(os.O_CLOEXEC); owned.update((read_fd,write_fd))
            event_read,event_write=os.pipe2(os.O_CLOEXEC); owned.update((event_read,event_write))
            ack_read,ack_write=os.pipe2(os.O_CLOEXEC); owned.update((ack_read,ack_write))
        argv=[str(worker),binding.party,binding.mode,*(str(fd) for fd in descriptors),str(input_fd),str(write_fd),
              UUID(binding.session_id).hex,binding.configuration,binding.snapshot_sha256,settings.peer_certificate_sha256,
              str(Path(settings.ca_certificate).resolve()),str(Path(settings.certificate).resolve()),str(Path(settings.private_key).resolve()),settings.peer_name,
              str(root/'artifacts/models'),str(root/'artifacts/constants/constants_v1.bin'),str(root/'artifacts/template/template_v1.bin'),str(ledger_path),
              str(event_write),str(ack_read)]
        if len(argv)!=22: raise PublicError(ErrorCode.BAD_STATE)
        with defer_cancellation_signals():
            process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                     close_fds=True,pass_fds=(*descriptors,input_fd,write_fd,event_write,ack_read),cwd=root,start_new_session=True)
        # Only the child retains its inherited channels and writer; the parent
        # drains the pipe while running (8201 bytes can exceed pipe capacity).
        for fd in (*descriptors,input_fd,write_fd,event_write,ack_read): close_owned(owned,fd)
        expected=(8201 if binding.party=='C' else 8200) if binding.mode=='prepare' else (2312 if binding.party=='C' else 0)
        result=_collect_worker_output(process,expected,read_fd,event_read,ack_write,on_event,check_cancelled)
        verify_closed_ledger(ledger_path,binding.session_id,binding.configuration)
        return _decode_result(binding,result)
    except PublicError: raise
    except Exception:
        raise PublicError(ErrorCode.BAD_STATE) from None
    finally:
        # A live child sees transport/ACK failure before any forced signal.
        # Normal collector completion has already reaped the child: only close.
        try:
            cleanup_worker(process,owned,cancellation_lanes)
        finally:
            sockets.close()
