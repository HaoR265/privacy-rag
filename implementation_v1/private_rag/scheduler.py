"""Single accepted-session, three fixed-lane native compute scheduling.

The raw TCP listeners reveal no data and read no admission header. Native TLS
authenticates all three connections against the accepted account fingerprint
and binds the frozen public snapshot before acknowledging AUTHENTICATED.
"""
from __future__ import annotations
import selectors
import math
import socket
import time
from dataclasses import dataclass
from uuid import uuid4
from typing import Callable
import psycopg
from .activation import require_private_backend
from .compute_input import WorkerBinding
from .compute_process import (ConnectedWorkerSockets, WorkerSettings,
                              LocalPreparationResult, run_worker)
from .database import Repository, LOCK_ID
from .delivery import PreparationShare
from .errors import ErrorCode, PublicError
from .lifecycle import Lifecycle


@dataclass(frozen=True)
class ComputePorts:
    compute: int=9443
    client_outgoing_heartbeat: int=9444
    server_outgoing_heartbeat: int=9445

    def values(self):
        ports=(self.compute,self.client_outgoing_heartbeat,self.server_outgoing_heartbeat)
        if any(type(p) is not int or not 1<=p<=65535 for p in ports) or len(set(ports))!=3:
            raise PublicError(ErrorCode.BAD_FORMAT)
        return ports


def connect_worker_sockets(host: str,ports: ComputePorts,*,public_timeout: float=30) -> ConnectedWorkerSockets:
    """C opens fixed physical lanes. run_worker subsequently owns these fds."""
    require_private_backend()
    opened=[]
    try:
        if type(public_timeout) not in (int,float) or not math.isfinite(public_timeout) or public_timeout<=0:
            raise PublicError(ErrorCode.BAD_FORMAT)
        for port in ports.values():
            channel=socket.create_connection((host,port),timeout=public_timeout)
            channel.settimeout(None); opened.append(channel)
        fds=[channel.detach() for channel in opened]
        return ConnectedWorkerSockets(*fds)  # C: compute,C_out,S_out
    except (OSError,ValueError,TypeError):
        raise PublicError(ErrorCode.DISCONNECTED) from None
    finally:
        for channel in opened: channel.close()


class SchedulerLifecycleEvents:
    """Persist native events before the process wrapper sends their ACKs."""
    def __init__(self,repository: Repository,session: str):
        self.lifecycle=Lifecycle(repository); self.session=session

    def __call__(self,event: str):
        if event=='AUTHENTICATED': self.lifecycle.transition(self.session,'computing')
        elif event=='RELEASING': self.lifecycle.transition(self.session,'releasing')
        else: raise PublicError(ErrorCode.REPLAY_OR_ORDER)


class ComputeScheduler:
    def __init__(self,repository: Repository,settings_for_session: Callable[[dict],WorkerSettings],
                 *,host: str='127.0.0.1',ports: ComputePorts=ComputePorts()):
        self.repository=repository; self.settings_for_session=settings_for_session
        self.host=host; self.ports=ports

    def serve_one(self,*,admission_timeout: float=30) -> str:
        """Own exactly one accepted session; any public failure aborts it once.

        Start this before C's application. The fixed listeners are ready while
        the coordinator waits for acceptance; it accepts no socket until then.
        A dedicated PostgreSQL session lock prevents multiple coordinators.
        There is no implicit restart/recovery of an old active worker.
        """
        require_private_backend()
        listeners=[]; channels={}; selector=None; session=None
        worker_owner=str(uuid4()); claim_attempted=False
        life=Lifecycle(self.repository)
        lock=None
        def abort_after_resources_closed(fault):
            nonlocal session
            # run_worker already reaped any child before raising. During
            # admission, close every accepted/listening lane before releasing
            # the global active slot in the database.
            for channel in channels.values(): channel.close()
            for listener in listeners: listener.close()
            try:
                if session is None and claim_attempted:
                    session=self.repository.claimed_session(worker_owner)
                if session is not None:
                    life.finish(session,acknowledged=False,public_fault=fault)
            except Exception:
                pass  # DB outage retains active state; no unsafe resume
        try:
            if type(admission_timeout) not in (int,float) or not math.isfinite(admission_timeout) or admission_timeout<=0:
                raise PublicError(ErrorCode.BAD_FORMAT)
            ports=self.ports.values()
            lock=psycopg.connect(self.repository.dsn,autocommit=True)
            if not lock.execute('SELECT pg_try_advisory_lock(%s)',(LOCK_ID+1,)).fetchone()[0]:
                raise PublicError(ErrorCode.BUSY)
            selector=selectors.DefaultSelector()
            for lane,port in enumerate(ports):
                listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                listeners.append(listener)
                listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                listener.bind((self.host,port)); listener.listen(1); listener.setblocking(False)
                selector.register(listener,selectors.EVENT_READ,lane)
            while True:
                try:
                    claim_attempted=True
                    admission=self.repository.claim_accepted_compute_session(worker_owner)
                    break
                except PublicError as exc:
                    if exc.code!=ErrorCode.MATERIALS_NOT_READY: raise
                    # Public database polling only; no peer input is read and
                    # no old computing/releasing session is resumed.
                    time.sleep(0.25)
            session=admission['session_id']
            def check_cancelled(): self.repository.check_session_cancellation(session,worker_owner)
            check_cancelled()
            settings=self.settings_for_session(admission)
            if settings.peer_certificate_sha256!=admission['peer_certificate_sha256']:
                raise PublicError(ErrorCode.NOT_AUTHORIZED)
            binding=WorkerBinding(session,admission['configuration'],admission['snapshot_sha256'],'S',admission['kind'])
            # Load only the pinned S records; no latest ACL/version lookup can
            # replace inputs of a session already accepted before revocation.
            records=tuple(self.repository.snapshot_server_records(session)) if binding.mode=='query' else ()
            deadline=time.monotonic()+admission_timeout
            while len(channels)!=3:
                check_cancelled()
                remaining=deadline-time.monotonic()
                if remaining<=0: raise PublicError(ErrorCode.DISCONNECTED)
                for event,_ in selector.select(min(remaining,0.25)):
                    listener=event.fileobj; lane=event.data
                    channel,_=listener.accept(); channel.setblocking(True)
                    channels[lane]=channel
                    selector.unregister(listener);listener.close()
            lifecycle_event=SchedulerLifecycleEvents(self.repository,session)
            fds={lane:channel.detach() for lane,channel in channels.items()}
            # S sends its own heartbeat on S_out and reads C_out.
            result=run_worker(binding,settings,ConnectedWorkerSockets(fds[0],fds[2],fds[1]),
                              snapshot=admission['snapshot'],on_event=lifecycle_event,
                              server_records=records,check_cancelled=check_cancelled)
            if binding.mode=='prepare':
                if not isinstance(result,LocalPreparationResult) or result.party!='S' or result.client_failed is not None:
                    raise PublicError(ErrorCode.BAD_STATE)
                # fsync object first, then one DB transaction completes/registers
                # the preparation; the private failure bit never reaches S.
                self.repository.complete_native_preparation(session,PreparationShare(session,'S',result.embedding,result.eligible))
            else:
                if result is not None: raise PublicError(ErrorCode.BAD_STATE)
                # run_worker returns only after exit0, both state-event ACKs,
                # and native's final client release ACK; S receives no answer.
                life.finish(session,acknowledged=True)
            return session
        except PublicError as exc:
            abort_after_resources_closed(exc.code.value)
            raise
        except Exception:
            abort_after_resources_closed(ErrorCode.BAD_STATE.value)
            raise PublicError(ErrorCode.BAD_STATE) from None
        except BaseException:
            abort_after_resources_closed(ErrorCode.DISCONNECTED.value)
            raise
        finally:
            if selector is not None: selector.close()
            for channel in channels.values(): channel.close()
            for listener in listeners: listener.close()
            if lock is not None: lock.close()
