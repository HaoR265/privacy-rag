"""Durable one-use reservation, global session terminal cleanup (CRY-009)."""
from __future__ import annotations
import uuid
from psycopg.types.json import Jsonb
from .database import Repository, ACTIVE_STATES, LOCK_ID, _session_identity
from .errors import PublicError, ErrorCode

class Lifecycle:
    def __init__(self, repository: Repository):
        self.repository=repository

    def transition(self,session: str,state: str):
        predecessors={'computing':['accepted'],'releasing':['computing']}
        if state not in predecessors: raise PublicError(ErrorCode.BAD_STATE)
        with self.repository.transaction() as c:
            s=c.execute('SELECT state,cancel_requested FROM sessions WHERE session_id=%s',(session,)).fetchone()
            if not s or s['state'] not in predecessors[state]: raise PublicError(ErrorCode.BAD_STATE)
            if s['cancel_requested']: raise PublicError(ErrorCode.DISCONNECTED)
            c.execute('UPDATE sessions SET state=%s WHERE session_id=%s',(state,session))

    def reserve(self,session: str,operation_id: int,kind: str,shape: list[int],ring_bits: int=59) -> str:
        if kind not in ('triple','ot_prg','bfv_mask','enc0','session_randomness') or ring_bits!=59 or type(operation_id) is not int or operation_id<0 or any(type(x) is not int or x<1 for x in shape):
            raise PublicError(ErrorCode.BAD_FORMAT)
        material=str(uuid.uuid4())
        with self.repository.transaction() as c:
            s=c.execute('SELECT state,cancel_requested FROM sessions WHERE session_id=%s',(session,)).fetchone()
            if not s or s['state'] not in ('accepted','computing') or s['cancel_requested']: raise PublicError(ErrorCode.BAD_STATE)
            prior=c.execute('SELECT 1 FROM materials WHERE session_id=%s AND operation_id=%s AND kind=%s',(session,operation_id,kind)).fetchone()
            if prior: raise PublicError(ErrorCode.REPLAY_OR_ORDER)
            c.execute("INSERT INTO materials(material_id,session_id,operation_id,kind,descriptor,state) VALUES(%s,%s,%s,%s,%s,'reserved')",(material,session,operation_id,kind,Jsonb({'shape':shape,'ring_bits':ring_bits})))
        return material

    def consume(self,material: str,session: str):
        with self.repository.transaction() as c:
            r=c.execute("UPDATE materials m SET state='consumed',finished_at=now() FROM sessions s WHERE m.material_id=%s AND m.session_id=%s AND m.state='reserved' AND s.session_id=m.session_id AND s.state IN ('accepted','computing') AND NOT s.cancel_requested RETURNING material_id",(material,session)).fetchone()
            if not r: raise PublicError(ErrorCode.REPLAY_OR_ORDER)

    def finish(self,session: str,*,acknowledged: bool,public_fault: str | None=None,total_bytes: int | None=None):
        if type(acknowledged) is not bool or (total_bytes is not None and (type(total_bytes) is not int or total_bytes<0)):
            raise PublicError(ErrorCode.BAD_FORMAT)
        allowed_faults={e.value for e in ErrorCode}
        if public_fault is not None and public_fault not in allowed_faults: raise PublicError(ErrorCode.BAD_FORMAT)
        with self.repository.transaction() as c:
            s=c.execute('SELECT state,kind,cancel_requested FROM sessions WHERE session_id=%s',(session,)).fetchone()
            if not s or s['state'] not in ACTIVE_STATES: raise PublicError(ErrorCode.BAD_STATE)
            if acknowledged and (public_fault is not None or s['state']!='releasing' or s['cancel_requested']):
                raise PublicError(ErrorCode.BAD_STATE)
            terminal='completed' if acknowledged else 'aborted'
            c.execute('UPDATE sessions SET state=%s,ended_at=now(),public_fault=%s,total_bytes=%s,release_acknowledged=%s WHERE session_id=%s',(terminal,public_fault,total_bytes,acknowledged,session))
            c.execute("UPDATE materials SET state='burned',finished_at=now() WHERE session_id=%s AND state='reserved'",(session,))
            c.execute('DELETE FROM session_references WHERE session_id=%s',(session,))

    def recover_after_process_death(self,session_id: str,*,operator_confirmed_worker_and_lanes_stopped: bool) -> int:
        """Operator-only recovery after proving old child AND all lanes stopped.

        Losing the PostgreSQL owner lock alone does not prove child termination.
        The required explicit assertion records the caller's responsibility;
        it is not a process-liveness detector. Target one known old session,
        never a newly accepted session. This resumes/replays no computation.
        """
        _session_identity(session_id)
        if operator_confirmed_worker_and_lanes_stopped is not True:
            raise PublicError(ErrorCode.BAD_STATE)
        with self.repository.transaction() as c:
            if not c.execute('SELECT pg_try_advisory_xact_lock(%s) AS acquired',(LOCK_ID+1,)).fetchone()['acquired']:
                raise PublicError(ErrorCode.BUSY)
            sessions=c.execute("SELECT session_id FROM sessions WHERE session_id=%s AND state IN ('accepted','computing','releasing')",(session_id,)).fetchall()
            for s in sessions:
                session=s['session_id']
                c.execute("UPDATE sessions SET state='aborted',ended_at=now(),public_fault='DISCONNECTED' WHERE session_id=%s",(session,))
                c.execute("UPDATE materials SET state='burned',finished_at=now() WHERE session_id=%s AND state='reserved'",(session,))
                c.execute('DELETE FROM session_references WHERE session_id=%s',(session,))
            return len(sessions)
