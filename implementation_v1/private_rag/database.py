"""PostgreSQL transaction boundary for authorization, publication and snapshots.

No query text, answer, reconstructed vector or hit exists in this API/schema.
Preparation completion is an internal compute-coordinator operation, never an
HTTP upload. All normal reads use the user's complete authorized collection.
"""
from __future__ import annotations
import json
import struct
import uuid
from contextlib import contextmanager
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from .delivery import DeliveryBinding, PreparationShare, ServerReshare, ShareRecord, apply_reshare, canonical_json, digest, RECORD_BYTES, HPKE_BYTES
from .storage import ObjectStore
from .errors import PublicError, ErrorCode

LOCK_ID = 7258371401893
ACTIVE_STATES = ('accepted','computing','releasing')


def _fail(code):
    raise PublicError(code)


def _session_identity(value: str | None) -> str:
    if value is None: return str(uuid.uuid4())
    try:
        if str(uuid.UUID(value))!=value: raise ValueError()
    except (ValueError,TypeError,AttributeError): _fail(ErrorCode.BAD_FORMAT)
    return value


class Repository:
    def __init__(self, dsn: str, objects: ObjectStore):
        self.dsn, self.objects = dsn, objects

    @contextmanager
    def transaction(self):
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            # All authorization mutations and acceptance serialize against this
            # short transaction lock; long computation never holds a DB lock.
            conn.execute('SELECT pg_advisory_xact_lock(%s)', (LOCK_ID,))
            yield conn

    def initialize(self):
        with self.transaction() as c:
            c.execute((Path(__file__).parent/'schema'/'001_management.sql').read_text())

    def enroll_account(self, account_id: str, role: str, certificate_sha256: str):
        """Operator-only enrollment; deliberately absent from HTTP routes."""
        if role not in ('D','U') or len(certificate_sha256) != 64:
            _fail(ErrorCode.BAD_FORMAT)
        with self.transaction() as c:
            c.execute('INSERT INTO accounts(account_id,role,certificate_sha256) VALUES(%s,%s,%s)', (account_id,role,certificate_sha256))

    def authenticate(self, fingerprint: str) -> dict:
        with psycopg.connect(self.dsn,row_factory=dict_row) as c:
            a = c.execute('SELECT account_id,role FROM accounts WHERE certificate_sha256=%s AND active', (fingerprint,)).fetchone()
        if not a: _fail(ErrorCode.AUTHENTICATION_FAILED)
        return {'account_id':str(a['account_id']), 'role':a['role']}

    @staticmethod
    def _account(c, account_id, role=None):
        a=c.execute('SELECT * FROM accounts WHERE account_id=%s AND active', (account_id,)).fetchone()
        if not a or (role and a['role']!=role): _fail(ErrorCode.NOT_AUTHORIZED)
        return a

    @staticmethod
    def _owner(c, owner, document):
        d=c.execute('SELECT * FROM documents WHERE document_id=%s AND owner_id=%s', (document,owner)).fetchone()
        if not d: _fail(ErrorCode.NOT_AUTHORIZED)
        Repository._account(c,owner,'D')
        return d

    def receiver_key_state(self, user_id: str) -> dict:
        """U can recover a lost registration response using public key state."""
        with self.transaction() as c:
            a=self._account(c,user_id,'U')
            key=c.execute('SELECT public_key FROM recipient_keys WHERE user_id=%s AND key_version=%s AND active',(user_id,a['key_version'])).fetchone()
            return {'user_id':user_id,'key_version':a['key_version'],'public_key_hex':bytes(key['public_key']).hex() if key else None}

    def register_receiver_key(self, user_id: str, public_key: bytes, expected_previous_version: int | None=None) -> int:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
        try: X25519PublicKey.from_public_bytes(public_key)
        except ValueError: _fail(ErrorCode.BAD_FORMAT)
        with self.transaction() as c:
            a=self._account(c,user_id,'U')
            if expected_previous_version is not None and (type(expected_previous_version) is not int or expected_previous_version!=a['key_version']):
                _fail(ErrorCode.VERSION_MISMATCH)
            if c.execute('SELECT 1 FROM recipient_keys WHERE user_id=%s AND public_key=%s',(user_id,public_key)).fetchone():
                _fail(ErrorCode.REPLAY_OR_ORDER)
            version=a['key_version']+1
            c.execute('UPDATE recipient_keys SET active=false WHERE user_id=%s', (user_id,))
            c.execute('INSERT INTO recipient_keys VALUES(%s,%s,%s,true)', (user_id,version,public_key))
            c.execute('UPDATE accounts SET key_version=%s,authorization_version=authorization_version+1 WHERE account_id=%s', (version,user_id))
        return version

    def create_document(self, owner: str, organization: str, title: str) -> str:
        document=str(uuid.uuid4())
        with self.transaction() as c:
            self._account(c,owner,'D')
            c.execute('INSERT INTO documents(document_id,owner_id,organization_id,internal_title) VALUES(%s,%s,%s,%s)', (document,owner,organization,title))
        return document

    def create_version(self, owner: str, document: str, block_count: int, configuration: str) -> int:
        if type(block_count) is not int or block_count<1 or len(configuration)!=64: _fail(ErrorCode.BAD_FORMAT)
        with self.transaction() as c:
            self._owner(c,owner,document)
            version=c.execute('SELECT coalesce(max(version),0)+1 AS v FROM document_versions WHERE document_id=%s', (document,)).fetchone()['v']
            c.execute("INSERT INTO document_versions VALUES(%s,%s,%s,%s,'staging')", (document,version,block_count,configuration))
        return version

    def set_authorization(self, owner: str, document: str, user: str, allowed: bool) -> int:
        if type(allowed) is not bool: _fail(ErrorCode.BAD_FORMAT)
        with self.transaction() as c:
            self._owner(c,owner,document); self._account(c,user,'U')
            prior=c.execute('SELECT * FROM acl WHERE document_id=%s AND user_id=%s', (document,user)).fetchone()
            if prior and prior['allowed']==allowed: return prior['generation']
            generation=prior['generation']+1 if prior else 1
            c.execute('INSERT INTO acl VALUES(%s,%s,%s,%s) ON CONFLICT(document_id,user_id) DO UPDATE SET allowed=EXCLUDED.allowed,generation=EXCLUDED.generation', (document,user,allowed,generation))
            c.execute('UPDATE accounts SET authorization_version=authorization_version+1 WHERE account_id=%s',(user,))
            return generation

    def delivery_plan(self, owner: str, document: str, version: int, user: str) -> dict:
        """D receives registered public recipient key and per-block bindings."""
        with self.transaction() as c:
            self._owner(c,owner,document); a=self._account(c,user,'U')
            acl=c.execute('SELECT * FROM acl WHERE document_id=%s AND user_id=%s AND allowed',(document,user)).fetchone()
            v=c.execute('SELECT * FROM document_versions WHERE document_id=%s AND version=%s',(document,version)).fetchone()
            k=c.execute('SELECT public_key FROM recipient_keys WHERE user_id=%s AND key_version=%s AND active',(user,a['key_version'])).fetchone()
            if not acl: _fail(ErrorCode.NOT_AUTHORIZED)
            if not v or not k: _fail(ErrorCode.MATERIALS_NOT_READY)
            bindings=[]
            for i in range(v['block_count']):
                c.execute('INSERT INTO user_slots VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',(document,user,i,str(uuid.uuid4())))
                slot=c.execute('SELECT opaque_slot FROM user_slots WHERE document_id=%s AND user_id=%s AND block_index=%s',(document,user,i)).fetchone()['opaque_slot']
                generation=c.execute('SELECT coalesce(max(share_generation),0)+1 AS g FROM deliveries WHERE user_id=%s AND opaque_slot=%s',(user,slot)).fetchone()['g']
                bindings.append(DeliveryBinding(user,a['key_version'],str(slot),version,generation,v['model_numeric_version']).public_dict())
            return {'public_key_hex':bytes(k['public_key']).hex(),'bindings':bindings}

    def start_preparation(self, owner: str, document: str, version: int, block_index: int, *, activation_check=None, session_id: str | None=None) -> str:
        session=_session_identity(session_id)
        with self.transaction() as c:
            self._owner(c,owner,document)
            v=c.execute('SELECT * FROM document_versions WHERE document_id=%s AND version=%s',(document,version)).fetchone()
            if not v or type(block_index) is not int or not 0<=block_index<v['block_count']: _fail(ErrorCode.BAD_SHAPE)
            if activation_check is not None: activation_check()
            if c.execute('SELECT 1 FROM session_cancel_intents WHERE account_id=%s AND session_id=%s',(owner,session)).fetchone(): _fail(ErrorCode.DISCONNECTED)
            if c.execute('SELECT 1 FROM sessions WHERE session_id=%s',(session,)).fetchone(): _fail(ErrorCode.REPLAY_OR_ORDER)
            if c.execute("SELECT 1 FROM sessions WHERE state IN ('accepted','computing','releasing')").fetchone(): _fail(ErrorCode.BUSY)
            snapshot={'document_id':document,'document_version':version,'block_index':block_index,'model_numeric_version':v['model_numeric_version']}
            c.execute("INSERT INTO sessions(session_id,account_id,kind,state,snapshot) VALUES(%s,%s,'prepare','accepted',%s)",(session,owner,Jsonb(snapshot)))
            return session

    def accepted_compute_session(self) -> dict:
        """Internal scheduler: exactly the one globally accepted snapshot."""
        with self.transaction() as c:
            s=c.execute("SELECT s.*,a.certificate_sha256,a.role FROM sessions s JOIN accounts a ON a.account_id=s.account_id WHERE s.state='accepted' AND a.active").fetchone()
            if not s: _fail(ErrorCode.MATERIALS_NOT_READY)
            if (s['kind']=='query' and s['role']!='U') or (s['kind']=='prepare' and s['role']!='D'):
                _fail(ErrorCode.BAD_STATE)
            snapshot=s['snapshot']
            return {'session_id':str(s['session_id']),'account_id':str(s['account_id']),
                    'kind':s['kind'],'peer_certificate_sha256':s['certificate_sha256'],
                    'snapshot':snapshot,'snapshot_sha256':digest(canonical_json(snapshot)),
                    'configuration':snapshot['model_numeric_version']}

    def claim_accepted_compute_session(self,worker_owner: str) -> dict:
        """Claim under the management lock before opening/using any worker fd."""
        _session_identity(worker_owner)
        with self.transaction() as c:
            s=c.execute("SELECT s.*,a.certificate_sha256,a.role FROM sessions s JOIN accounts a ON a.account_id=s.account_id WHERE s.state='accepted' AND NOT s.worker_claimed AND NOT s.cancel_requested AND a.active").fetchone()
            if not s: _fail(ErrorCode.MATERIALS_NOT_READY)
            if (s['kind']=='query' and s['role']!='U') or (s['kind']=='prepare' and s['role']!='D'): _fail(ErrorCode.BAD_STATE)
            c.execute('UPDATE sessions SET worker_claimed=true,worker_owner=%s WHERE session_id=%s',(worker_owner,s['session_id']))
            snapshot=s['snapshot']
            return {'session_id':str(s['session_id']),'account_id':str(s['account_id']),
                    'kind':s['kind'],'peer_certificate_sha256':s['certificate_sha256'],
                    'snapshot':snapshot,'snapshot_sha256':digest(canonical_json(snapshot)),
                    'configuration':snapshot['model_numeric_version']}

    def claimed_session(self,worker_owner: str) -> str | None:
        """Coordinator exception recovery for interruption just after commit."""
        with self.transaction() as c:
            row=c.execute("SELECT session_id FROM sessions WHERE worker_owner=%s AND state IN ('accepted','computing','releasing')",(worker_owner,)).fetchone()
            return str(row['session_id']) if row else None

    def check_session_cancellation(self,session: str,worker_owner: str):
        with self.transaction() as c:
            s=c.execute('SELECT state,cancel_requested,worker_owner FROM sessions WHERE session_id=%s',(session,)).fetchone()
            if not s or str(s['worker_owner'])!=worker_owner: _fail(ErrorCode.BAD_STATE)
            if s['cancel_requested']: _fail(ErrorCode.DISCONNECTED)
            if s['state'] not in ACTIVE_STATES: _fail(ErrorCode.BAD_STATE)

    def request_session_cancel(self,account: str,session: str) -> dict:
        """Only public disconnect/cancel; never receives a numeric failure bit.

        Unclaimed sessions can end immediately. Claimed sessions keep the
        global active slot until their coordinator has reaped its worker.
        """
        with self.transaction() as c:
            self._account(c,account)
            s=c.execute('SELECT state,worker_claimed,account_id FROM sessions WHERE session_id=%s',(session,)).fetchone()
            if not s:
                # Cancellation may overtake a submitted request still queued at
                # HTTP ingress. Bind this durable intent to this account only.
                c.execute('INSERT INTO session_cancel_intents(account_id,session_id) VALUES(%s,%s) ON CONFLICT DO NOTHING',(account,session))
                return {'session_id':session,'state':'aborted','cancel_requested':True}
            if str(s['account_id'])!=account: _fail(ErrorCode.NOT_AUTHORIZED)
            if s['state'] in ACTIVE_STATES:
                if s['worker_claimed']:
                    c.execute('UPDATE sessions SET cancel_requested=true WHERE session_id=%s',(session,))
                else:
                    c.execute("UPDATE sessions SET state='aborted',ended_at=now(),public_fault='DISCONNECTED',cancel_requested=true WHERE session_id=%s",(session,))
                    c.execute("UPDATE materials SET state='burned',finished_at=now() WHERE session_id=%s AND state='reserved'",(session,))
                    c.execute('DELETE FROM session_references WHERE session_id=%s',(session,))
            row=c.execute('SELECT state,cancel_requested FROM sessions WHERE session_id=%s',(session,)).fetchone()
            return {'session_id':session,**row}

    def preparation_descriptor(self,owner: str,session_id: str) -> dict:
        with self.transaction() as c:
            self._account(c,owner,'D')
            s=c.execute("SELECT snapshot FROM sessions WHERE session_id=%s AND account_id=%s AND kind='prepare'",(session_id,owner)).fetchone()
            if not s: _fail(ErrorCode.NOT_AUTHORIZED)
            return {'session_id':session_id,'snapshot':s['snapshot'],'snapshot_sha256':digest(canonical_json(s['snapshot']))}

    def complete_native_preparation(self, session_id: str, share: PreparationShare):
        """Internal only, after native final ACK; persist before atomic completion.

        S never receives a private numeric failure bit. Only D may turn a
        successful local result into a deliverable preparation share.
        """
        if share.party!='S' or share.preparation_id!=session_id: _fail(ErrorCode.BAD_STATE)
        payload=struct.pack('<1025Q',*(share.z+(share.e,)))
        obj=self.objects.put(payload)
        with self.transaction() as c:
            s=c.execute("SELECT snapshot FROM sessions WHERE session_id=%s AND kind='prepare' AND state='releasing' AND NOT cancel_requested",(session_id,)).fetchone()
            if not s: _fail(ErrorCode.BAD_STATE)
            v=s['snapshot']
            c.execute('INSERT INTO preparations VALUES(%s,%s,%s,%s,%s,%s)',(session_id,v['document_id'],v['document_version'],v['block_index'],session_id,obj))
            c.execute("UPDATE sessions SET state='completed',ended_at=now(),release_acknowledged=true WHERE session_id=%s",(session_id,))
            c.execute("UPDATE materials SET state='burned',finished_at=now() WHERE session_id=%s AND state='reserved'",(session_id,))
            c.execute('DELETE FROM session_references WHERE session_id=%s',(session_id,))

    def retain_completed_preparation(self, session_id: str, share: PreparationShare):
        """Internal legacy/debug coordinator import after complete preparation.

        Production scheduler uses complete_native_preparation after final ACK.
        This method does not execute or certify Embedding;
        no public endpoint can manufacture rows.
        """
        if share.party!='S': _fail(ErrorCode.BAD_FORMAT)
        payload=struct.pack('<1025Q',*(share.z+(share.e,)))
        obj=self.objects.put(payload)
        with self.transaction() as c:
            s=c.execute("SELECT * FROM sessions WHERE session_id=%s AND kind='prepare' AND state='completed'",(session_id,)).fetchone()
            if not s: _fail(ErrorCode.BAD_STATE)
            v=s['snapshot']
            c.execute('INSERT INTO preparations VALUES(%s,%s,%s,%s,%s,%s)',(share.preparation_id,v['document_id'],v['document_version'],v['block_index'],session_id,obj))

    def stage_delivery(self, owner: str, document: str, binding: DeliveryBinding, message: ServerReshare, package: bytes) -> str:
        if len(package)!=HPKE_BYTES: _fail(ErrorCode.BAD_SHAPE)
        with self.transaction() as c:
            self._owner(c,owner,document)
            a=self._account(c,binding.user_id,'U')
            acl=c.execute('SELECT * FROM acl WHERE document_id=%s AND user_id=%s AND allowed',(document,binding.user_id)).fetchone()
            p=c.execute('SELECT p.*,v.model_numeric_version FROM preparations p JOIN document_versions v ON v.document_id=p.document_id AND v.version=p.document_version WHERE p.preparation_id=%s AND p.document_id=%s AND p.document_version=%s',(message.preparation_id,document,binding.document_version)).fetchone()
            if not acl: _fail(ErrorCode.NOT_AUTHORIZED)
            if not p: _fail(ErrorCode.MATERIALS_NOT_READY)
            slot=c.execute('SELECT opaque_slot FROM user_slots WHERE document_id=%s AND user_id=%s AND block_index=%s',(document,binding.user_id,p['block_index'])).fetchone()
            if not slot or str(slot['opaque_slot'])!=binding.opaque_slot or a['key_version']!=binding.key_version or p['model_numeric_version']!=binding.model_numeric_version: _fail(ErrorCode.VERSION_MISMATCH)
            last=c.execute('SELECT coalesce(max(share_generation),0) AS g FROM deliveries WHERE user_id=%s AND opaque_slot=%s',(binding.user_id,binding.opaque_slot)).fetchone()['g']
            if binding.share_generation!=last+1: _fail(ErrorCode.REPLAY_OR_ORDER)
            vals=struct.unpack('<1025Q',self.objects.get(p['server_object'],8200))
            server=apply_reshare(PreparationShare(message.preparation_id,'S',vals[:1024],vals[-1]),message)
            sh=self.objects.put(server.encode()); ph=self.objects.put(package)
            identity=str(uuid.uuid4())
            c.execute('INSERT INTO deliveries VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',(identity,document,binding.document_version,binding.user_id,binding.opaque_slot,p['block_index'],binding.key_version,binding.share_generation,acl['generation'],Jsonb(binding.public_dict()),sh,ph))
            return identity

    def publish_version(self, owner: str, document: str, version: int):
        with self.transaction() as c:
            d=self._owner(c,owner,document)
            if d['current_version'] is not None and version<d['current_version']: _fail(ErrorCode.VERSION_MISMATCH)
            v=c.execute('SELECT * FROM document_versions WHERE document_id=%s AND version=%s',(document,version)).fetchone()
            if not v: _fail(ErrorCode.MATERIALS_NOT_READY)
            prepared=c.execute('SELECT DISTINCT block_index FROM preparations WHERE document_id=%s AND document_version=%s ORDER BY block_index',(document,version)).fetchall()
            if [p['block_index'] for p in prepared]!=list(range(v['block_count'])): _fail(ErrorCode.MATERIALS_NOT_READY)
            users=c.execute("SELECT acl.*,a.key_version FROM acl JOIN accounts a ON a.account_id=acl.user_id WHERE acl.document_id=%s AND acl.allowed AND a.active",(document,)).fetchall()
            selected=[]
            for u in users:
                rows=c.execute('SELECT DISTINCT ON (block_index) * FROM deliveries WHERE document_id=%s AND document_version=%s AND user_id=%s AND key_version=%s AND acl_generation=%s ORDER BY block_index,share_generation DESC',(document,version,u['user_id'],u['key_version'],u['generation'])).fetchall()
                if [r['block_index'] for r in rows]!=list(range(v['block_count'])): _fail(ErrorCode.MATERIALS_NOT_READY)
                for r in rows:
                    b=DeliveryBinding(**r['binding'])
                    if b.model_numeric_version!=v['model_numeric_version']: _fail(ErrorCode.VERSION_MISMATCH)
                    self.objects.get(r['server_object'],RECORD_BYTES); self.objects.get(r['package_object'],HPKE_BYTES)
                selected.extend(rows)
            c.execute('DELETE FROM published_deliveries WHERE document_id=%s',(document,))
            for r in selected:
                c.execute('INSERT INTO published_deliveries VALUES(%s,%s,%s,%s)',(document,r['user_id'],r['block_index'],r['delivery_id']))
            c.execute("UPDATE document_versions SET state='published' WHERE document_id=%s AND version=%s",(document,version))
            c.execute('UPDATE documents SET current_version=%s WHERE document_id=%s',(version,document))
            for u in users: c.execute('UPDATE accounts SET authorization_version=authorization_version+1 WHERE account_id=%s',(u['user_id'],))

    def _manifest(self, c, user: str) -> tuple[dict,list[dict]]:
        a=self._account(c,user,'U')
        # ACL-authorized unpublished or stale-key scope is NOT silently omitted.
        scopes=c.execute('SELECT d.document_id,d.current_version,v.block_count FROM acl x JOIN documents d USING(document_id) LEFT JOIN document_versions v ON v.document_id=d.document_id AND v.version=d.current_version WHERE x.user_id=%s AND x.allowed',(user,)).fetchall()
        rows=c.execute('SELECT r.* FROM published_deliveries p JOIN deliveries r USING(delivery_id) JOIN acl x ON x.document_id=p.document_id AND x.user_id=p.user_id WHERE p.user_id=%s AND x.allowed AND r.acl_generation=x.generation AND r.key_version=%s ORDER BY r.opaque_slot',(user,a['key_version'])).fetchall()
        if any(s['current_version'] is None for s in scopes) or len(rows)!=sum(s['block_count'] or 0 for s in scopes): _fail(ErrorCode.MATERIALS_NOT_READY)
        entries=[{'binding':r['binding'],'package_sha256':r['package_object']} for r in rows]
        manifest={'format_version':1,'user_id':user,'authorization_version':a['authorization_version'],'N':len(entries),'entries':entries}
        manifest['manifest_sha256']=digest(canonical_json(manifest))
        return manifest,rows

    def sync_manifest(self,user: str) -> dict:
        with self.transaction() as c: return self._manifest(c,user)[0]

    def sync_package(self,user: str, package_sha256: str) -> bytes:
        with self.transaction() as c:
            _,rows=self._manifest(c,user)
            if package_sha256 not in {r['package_object'] for r in rows}: _fail(ErrorCode.NOT_AUTHORIZED)
        return self.objects.get(package_sha256,HPKE_BYTES)

    def accept_query(self,user: str,manifest_sha256: str, configuration: str, *, activation_check=None, session_id: str | None=None) -> dict:
        session=_session_identity(session_id)
        with self.transaction() as c:
            manifest,rows=self._manifest(c,user)
            if manifest['manifest_sha256']!=manifest_sha256: _fail(ErrorCode.VERSION_MISMATCH)
            if not rows: return {'status':'no_authorized_material','message':'当前没有可用于回答的授权资料','N':0}
            if any(r['binding']['model_numeric_version']!=configuration for r in rows): _fail(ErrorCode.VERSION_MISMATCH)
            # Gate the latest snapshot under the same lock as ACL/publication.
            if activation_check is None: _fail(ErrorCode.BACKEND_NOT_READY)
            activation_check()
            if c.execute('SELECT 1 FROM session_cancel_intents WHERE account_id=%s AND session_id=%s',(user,session)).fetchone(): _fail(ErrorCode.DISCONNECTED)
            for r in rows: self.objects.get(r['server_object'],RECORD_BYTES); self.objects.get(r['package_object'],HPKE_BYTES)
            if c.execute('SELECT 1 FROM sessions WHERE session_id=%s',(session,)).fetchone(): _fail(ErrorCode.REPLAY_OR_ORDER)
            if c.execute("SELECT 1 FROM sessions WHERE state IN ('accepted','computing','releasing')").fetchone(): _fail(ErrorCode.BUSY)
            snapshot={**manifest,'session_id':session,'model_numeric_version':configuration,'protocol_version':'prag-v1','fixed_shapes':{'t':256,'m':256,'z':1024,'e':1,'prefill':1024,'output':256}}
            c.execute("INSERT INTO sessions(session_id,account_id,kind,state,snapshot) VALUES(%s,%s,'query','accepted',%s)",(session,user,Jsonb(snapshot)))
            for r in rows: c.execute('INSERT INTO session_references VALUES(%s,%s)',(session,r['delivery_id']))
            c.execute('UPDATE accounts SET query_count=query_count+1 WHERE account_id=%s',(user,))
            return {'status':'accepted','snapshot':snapshot}

    def snapshot_server_records(self,session_id: str) -> list[ShareRecord]:
        """Compute process only; no HTTP route."""
        with self.transaction() as c:
            s=c.execute("SELECT state FROM sessions WHERE session_id=%s AND kind='query'",(session_id,)).fetchone()
            if not s or s['state'] not in ACTIVE_STATES: _fail(ErrorCode.BAD_STATE)
            rows=c.execute('SELECT d.server_object FROM session_references r JOIN deliveries d USING(delivery_id) WHERE r.session_id=%s ORDER BY d.opaque_slot',(session_id,)).fetchall()
        return [ShareRecord.decode(self.objects.get(r['server_object'],RECORD_BYTES),'S') for r in rows]

    def session_status(self,user: str,session_id: str) -> dict:
        with self.transaction() as c:
            s=c.execute('SELECT session_id,state,started_at,ended_at,public_fault,total_bytes,release_acknowledged FROM sessions WHERE session_id=%s AND account_id=%s',(session_id,user)).fetchone()
            if not s: _fail(ErrorCode.NOT_AUTHORIZED)
            return s
