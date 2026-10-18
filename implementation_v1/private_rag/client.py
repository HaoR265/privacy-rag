"""mTLS management clients; questions and reconstructed data have no API here.

Each synchronization fetches the complete manifest and every HPKE package. A
failed synchronization never replaces the last complete collection and makes
it ineligible for accepting a new query. Native computation consumes the local
immutable one-party records separately from this management connection.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4
import httpx
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
from .delivery import (DeliveryBinding, PreparationShare, ShareRecord, UserShareCache,
                       HPKE_BYTES, canonical_json, reshare_for_user, seal_delivery)
from .errors import ErrorCode, PublicError
from .tls_server import tls_context
from .vault import RecipientKeyStore


def _uuid(value: str) -> str:
    try:
        if str(UUID(value))!=value: raise ValueError()
    except (ValueError,TypeError,AttributeError):
        raise PublicError(ErrorCode.BAD_FORMAT) from None
    return value


def _sha(value: str) -> str:
    try:
        if len(value)!=64 or bytes.fromhex(value).hex()!=value: raise ValueError()
    except (ValueError,TypeError):
        raise PublicError(ErrorCode.BAD_FORMAT) from None
    return value


@dataclass(frozen=True)
class TLSFiles:
    certificate: Path
    private_key: Path
    ca_certificate: Path
    # Optional leaf pin supplements the required internal CA/hostname checks.
    server_certificate_sha256: str | None=None


class ManagementClient:
    def __init__(self, base_url: str, tls: TLSFiles, *, timeout: float=180):
        parsed=urlsplit(base_url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('','/'):
            raise PublicError(ErrorCode.BAD_FORMAT)
        self._pin=_sha(tls.server_certificate_sha256) if tls.server_certificate_sha256 is not None else None
        self._http=httpx.Client(base_url=base_url.rstrip('/'),
            verify=tls_context(str(tls.certificate),str(tls.private_key),str(tls.ca_certificate),server=False),
            trust_env=False,follow_redirects=False,timeout=timeout)

    def close(self):
        self._http.close()

    def __enter__(self): return self
    def __exit__(self,*unused): self.close()

    def _request(self, method: str, path: str, body=None, *, binary=False, request_timeout: float | None=None):
        # No retry of state-changing calls; an uncertain completion is resolved
        # through the corresponding public state, never by silently replaying.
        try:
            options={} if request_timeout is None else {'timeout':request_timeout}
            with self._http.stream(method,path,json=body,**options) as response:
                stream=response.extensions.get('network_stream')
                tls=stream.get_extra_info('ssl_object') if stream is not None else None
                if tls is None or tls.version()!='TLSv1.3':
                    raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
                # httpcore may expose CPython's underlying _SSLSocket, whose
                # getpeercert argument is positional-only.
                der=tls.getpeercert(True)
                if not isinstance(x509.load_der_x509_certificate(der).public_key(),Ed25519PublicKey):
                    raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
                if self._pin is not None and hashlib.sha256(der).hexdigest()!=self._pin:
                    raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
                data=bytearray()
                limit=HPKE_BYTES if binary and response.status_code==200 else 64*1024*1024
                for chunk in response.iter_bytes():
                    if len(data)+len(chunk)>limit: raise PublicError(ErrorCode.BAD_SHAPE)
                    data.extend(chunk)
                if response.status_code!=200:
                    try:
                        error=json.loads(data)
                        if set(error)!={'error'}: raise ValueError()
                        code=ErrorCode(error['error'])
                    except (ValueError,TypeError,UnicodeError):
                        code=ErrorCode.BAD_STATE
                    raise PublicError(code)
                if binary:
                    if len(data)!=HPKE_BYTES: raise PublicError(ErrorCode.BAD_SHAPE)
                    return bytes(data)
                value=json.loads(data)
                if not isinstance(value,dict): raise PublicError(ErrorCode.BAD_FORMAT)
                return value
        except PublicError:
            raise
        except httpx.TransportError:
            raise PublicError(ErrorCode.DISCONNECTED) from None
        except (ValueError,TypeError,UnicodeError):
            raise PublicError(ErrorCode.BAD_FORMAT) from None

    def status(self) -> dict:
        return self._request('GET','/v1/status')

    def cancel_session(self,session_id: str) -> dict:
        """Public explicit/disconnect cancellation, never a numeric outcome."""
        result=self._request('POST','/v1/sessions/'+_uuid(session_id)+'/cancel',request_timeout=5)
        if (set(result)!={'session_id','state','cancel_requested'} or result['session_id']!=session_id
            or result['state'] not in ('accepted','computing','releasing','completed','aborted')
            or type(result['cancel_requested']) is not bool):
            raise PublicError(ErrorCode.BAD_FORMAT)
        return result

    def cancel_after_public_failure(self,session_id: str):
        """Single bounded best effort; original failure remains authoritative."""
        try: self.cancel_session(session_id)
        except Exception: pass

    def session_status(self, session_id: str) -> dict:
        result=self._request('GET','/v1/sessions/'+_uuid(session_id))
        if (set(result)!={'session_id','state','started_at','ended_at','public_fault','total_bytes','release_acknowledged'}
            or result['session_id']!=session_id or result['state'] not in ('accepted','computing','releasing','completed','aborted')
            or type(result['release_acknowledged']) is not bool or not isinstance(result['started_at'],str)
            or (result['ended_at'] is not None and not isinstance(result['ended_at'],str))
            or (result['total_bytes'] is not None and (type(result['total_bytes']) is not int or result['total_bytes']<0))
            or (result['public_fault'] is not None and (not isinstance(result['public_fault'],str)
                or result['public_fault'] not in {code.value for code in ErrorCode}))):
            raise PublicError(ErrorCode.BAD_FORMAT)
        if result['state']=='completed':
            if not result['release_acknowledged'] or result['public_fault'] is not None or result['ended_at'] is None:
                raise PublicError(ErrorCode.BAD_STATE)
        elif result['state']=='aborted':
            if result['release_acknowledged'] or result['ended_at'] is None:
                raise PublicError(ErrorCode.BAD_STATE)
        elif result['release_acknowledged'] or result['ended_at'] is not None or result['public_fault'] is not None:
            raise PublicError(ErrorCode.BAD_STATE)
        return result


@dataclass(frozen=True,repr=False)
class SynchronizedCollection:
    manifest_json: bytes
    records: tuple[ShareRecord,...]

    @property
    def manifest(self) -> dict:
        return json.loads(self.manifest_json)


class UserClient(ManagementClient):
    def __init__(self, base_url: str, tls: TLSFiles, user_id: str, configuration: str, *, timeout: float=180):
        self.user_id=_uuid(user_id); self.configuration=_sha(configuration)
        self.keys=RecipientKeyStore(self.user_id)
        self._collection: SynchronizedCollection | None=None
        self._ready=False
        super().__init__(base_url,tls,timeout=timeout)

    def receiver_key_state(self) -> dict:
        state=self._request('GET','/v1/receiver-keys')
        if set(state)!={'user_id','key_version','public_key_hex'} or state['user_id']!=self.user_id:
            raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
        version=state['key_version']; public=state['public_key_hex']
        if type(version) is not int or not 0<=version<(1<<63): raise PublicError(ErrorCode.BAD_FORMAT)
        if version==0:
            if public is not None: raise PublicError(ErrorCode.BAD_FORMAT)
        else:
            _sha(public)
        return state

    def register_new_key(self) -> int:
        """Persist a fresh OS-keystore key before registering its public part.

        Existing unregistered material at the next version is recoverable after
        interruption. A lost HTTP response is checked against public server
        state without replaying the registration or deleting the private key.
        """
        self._ready=False
        previous=self.receiver_key_state()['key_version']; version=previous+1
        try:
            key=self.keys.load(version)
            public=key.public_key().public_bytes_raw()
        except PublicError as exc:
            if exc.code!=ErrorCode.MATERIALS_NOT_READY: raise
            public=self.keys.generate(version)
        try:
            result=self._request('POST','/v1/receiver-keys',{'public_key_hex':public.hex(),'expected_previous_version':previous})
        except PublicError as exc:
            if exc.code!=ErrorCode.DISCONNECTED: raise
            state=self.receiver_key_state()
            if state['key_version']!=version or state['public_key_hex']!=public.hex():
                raise PublicError(ErrorCode.DISCONNECTED) from None
            result={'key_version':version}
        if result!={'key_version':version}: raise PublicError(ErrorCode.VERSION_MISMATCH)
        return version

    def synchronize(self) -> SynchronizedCollection:
        self._ready=False
        state=self.receiver_key_state(); version=state['key_version']
        if version==0: raise PublicError(ErrorCode.MATERIALS_NOT_READY)
        key=self.keys.load(version)
        if key.public_key().public_bytes_raw().hex()!=state['public_key_hex']:
            raise PublicError(ErrorCode.VERSION_MISMATCH)
        manifest=self._request('GET','/v1/sync')
        # A fresh verifier deliberately fetches ALL current entries even if
        # earlier local ciphertexts remain cached. There is no delta endpoint.
        cache=UserShareCache(self.user_id,version,key,self.configuration)
        cache.synchronize(manifest,lambda sha:self._request('GET','/v1/sync/packages/'+_sha(sha),binary=True))
        collection=SynchronizedCollection(canonical_json(manifest),cache.snapshot())
        self._collection=collection; self._ready=True
        return collection

    def synchronized_collection(self) -> SynchronizedCollection:
        if not self._ready or self._collection is None: raise PublicError(ErrorCode.MATERIALS_NOT_READY)
        return self._collection

    def accept_query(self) -> dict:
        """Accept only public snapshot metadata; question shares go to compute."""
        collection=self.synchronized_collection(); manifest=collection.manifest
        requested_session=str(uuid4())
        try:
            result=self._request('POST','/v1/queries/accept',{'manifest_sha256':manifest['manifest_sha256'],'model_numeric_version':self.configuration,'session_id':requested_session})
            if result.get('status')=='no_authorized_material':
                if manifest['N']!=0 or result!={'status':'no_authorized_material','message':'当前没有可用于回答的授权资料','N':0}:
                    raise PublicError(ErrorCode.VERSION_MISMATCH)
            elif result.get('status')=='accepted':
                snapshot=result.get('snapshot',{})
                if not isinstance(snapshot,dict): raise PublicError(ErrorCode.BAD_FORMAT)
                if any(snapshot.get(k)!=v for k,v in manifest.items()) or snapshot.get('model_numeric_version')!=self.configuration:
                    raise PublicError(ErrorCode.VERSION_MISMATCH)
                _uuid(snapshot.get('session_id'))
                if snapshot['session_id']!=requested_session: raise PublicError(ErrorCode.VERSION_MISMATCH)
                if snapshot.get('protocol_version')!='prag-v1' or snapshot.get('fixed_shapes')!={'t':256,'m':256,'z':1024,'e':1,'prefill':1024,'output':256}:
                    raise PublicError(ErrorCode.VERSION_MISMATCH)
            else: raise PublicError(ErrorCode.BAD_FORMAT)
            return result
        except BaseException:
            self.cancel_after_public_failure(requested_session)
            raise
        finally:
            # Every new query requires another complete synchronization. The
            # immutable collection remains available to an already accepted run.
            self._ready=False


class ProviderClient(ManagementClient):
    def create_document(self, organization_id: str, internal_title: str) -> str:
        result=self._request('POST','/v1/documents',{'organization_id':organization_id,'internal_title':internal_title})
        return _uuid(result.get('document_id'))

    def create_version(self, document: str, block_count: int, configuration: str) -> int:
        result=self._request('POST',f'/v1/documents/{_uuid(document)}/versions',{'block_count':block_count,'model_numeric_version':_sha(configuration)})
        version=result.get('document_version')
        if type(version) is not int or version<1: raise PublicError(ErrorCode.BAD_FORMAT)
        return version

    def start_preparation(self, document: str, version: int, block_index: int) -> dict:
        if type(version) is not int or version<1 or type(block_index) is not int or block_index<0:
            raise PublicError(ErrorCode.BAD_FORMAT)
        requested_session=str(uuid4())
        try:
            result=self._request('POST',f'/v1/documents/{_uuid(document)}/versions/{version}/preparations',{'block_index':block_index,'session_id':requested_session})
            if set(result)!={'session_id','snapshot','snapshot_sha256'} or not isinstance(result['snapshot'],dict): raise PublicError(ErrorCode.BAD_FORMAT)
            if result['session_id']!=requested_session: raise PublicError(ErrorCode.VERSION_MISMATCH)
            _uuid(result['session_id']); _sha(result['snapshot_sha256'])
            from .delivery import digest
            expected={'document_id':document,'document_version':version,'block_index':block_index,
                      'model_numeric_version':result['snapshot'].get('model_numeric_version')}
            if result['snapshot']!=expected or digest(canonical_json(expected))!=result['snapshot_sha256']:
                raise PublicError(ErrorCode.VERSION_MISMATCH)
            _sha(expected['model_numeric_version'])
            return result
        except BaseException:
            self.cancel_after_public_failure(requested_session)
            raise

    def authorize(self, document: str, recipient: str, allowed: bool) -> int:
        result=self._request('PUT',f'/v1/documents/{_uuid(document)}/authorization',{'user_id':_uuid(recipient),'allowed':allowed})
        generation=result.get('authorization_generation')
        if type(generation) is not int or generation<1: raise PublicError(ErrorCode.BAD_FORMAT)
        return generation

    def delivery_plan(self, document: str, version: int, recipient: str) -> dict:
        if type(version) is not int or version<1: raise PublicError(ErrorCode.BAD_FORMAT)
        result=self._request('GET',f'/v1/documents/{_uuid(document)}/versions/{version}/delivery-plan/{_uuid(recipient)}')
        if set(result)!={'public_key_hex','bindings'}: raise PublicError(ErrorCode.BAD_FORMAT)
        _sha(result['public_key_hex'])
        if not isinstance(result['bindings'],list) or not result['bindings']: raise PublicError(ErrorCode.BAD_SHAPE)
        try: bindings=[DeliveryBinding(**x) for x in result['bindings']]
        except TypeError: raise PublicError(ErrorCode.BAD_FORMAT) from None
        if any(b.user_id!=recipient or b.document_version!=version for b in bindings): raise PublicError(ErrorCode.VERSION_MISMATCH)
        if len({b.opaque_slot for b in bindings})!=len(bindings): raise PublicError(ErrorCode.BAD_SHAPE)
        return result

    def stage_prepared_block(self, document: str, binding: DeliveryBinding, recipient_public_key_hex: str,
                             tokens: tuple[int,...], validity: tuple[int,...], preparation: PreparationShare) -> str:
        """Only a completed private-preparation D share may enter this call.

        The service independently requires its matching retained S preparation;
        this API cannot create or mark a preparation complete. Raw tokens stay
        at D: only the independent S shares/deltas and U HPKE package are sent.
        """
        public=X25519PublicKey.from_public_bytes(bytes.fromhex(_sha(recipient_public_key_hex)))
        user_share,server_delta=reshare_for_user(tokens,validity,preparation)
        package=seal_delivery(user_share,public,binding)
        result=self._request('POST',f'/v1/documents/{_uuid(document)}/deliveries',{
            'binding':binding.public_dict(),'preparation_id':preparation.preparation_id,
            'reshare_hex':server_delta.encode().hex(),'package_hex':package.hex()})
        return _uuid(result.get('delivery_id'))

    def publish_version(self, document: str, version: int) -> None:
        if type(version) is not int or version<1: raise PublicError(ErrorCode.BAD_FORMAT)
        result=self._request('POST',f'/v1/documents/{_uuid(document)}/versions/{version}/publish')
        if result!={'status':'published','document_version':version}: raise PublicError(ErrorCode.BAD_FORMAT)
