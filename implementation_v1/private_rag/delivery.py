"""D→U delivery and D/S independent resharing (CRY-014, DAT-001/002).

Only one party's additive share is represented per object. No reconstruction
entrypoint exists here. Exact layout is artifacts/interface_delivery.json.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import secrets
import struct
import uuid
from dataclasses import dataclass
from typing import Literal
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.hpke import Suite, KEM, KDF, AEAD
from .errors import PublicError, ErrorCode

MODULUS = 1 << 59
MASK = MODULUS - 1
WORDS = 1537
RECORD_BYTES = WORDS * 8
HPKE_BYTES = RECORD_BYTES + 32 + 16
INFO_DOMAIN = b'private-rag/delivery/v1\x00'
SUITE = Suite(KEM.X25519, KDF.HKDF_SHA256, AEAD.AES_128_GCM)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _words(values: tuple[int, ...], length: int) -> None:
    if not isinstance(values, tuple) or len(values) != length:
        raise PublicError(ErrorCode.BAD_SHAPE)
    if any(type(x) is not int or not 0 <= x < MODULUS for x in values):
        raise PublicError(ErrorCode.BAD_FORMAT)


@dataclass(frozen=True)
class DeliveryBinding:
    user_id: str
    key_version: int
    opaque_slot: str
    document_version: int
    share_generation: int
    model_numeric_version: str
    protocol_version: str = 'prag-v1'
    format_version: int = 1

    def __post_init__(self):
        try:
            for v in (self.user_id, self.opaque_slot):
                if str(uuid.UUID(v)) != v:
                    raise ValueError()
            for v in (self.key_version, self.document_version, self.share_generation):
                if type(v) is not int or not 1 <= v < (1 << 63):
                    raise ValueError()
            if type(self.format_version) is not int or self.format_version != 1 or self.protocol_version != 'prag-v1':
                raise ValueError()
            if len(self.model_numeric_version) != 64 or bytes.fromhex(self.model_numeric_version).hex() != self.model_numeric_version:
                raise ValueError()
        except (ValueError, TypeError):
            raise PublicError(ErrorCode.BAD_FORMAT) from None

    def public_dict(self) -> dict:
        return dataclasses.asdict(self)

    def encode(self) -> bytes:
        return canonical_json(self.public_dict())

    @classmethod
    def decode(cls, data: bytes) -> DeliveryBinding:
        try:
            obj = cls(**json.loads(data))
            if obj.encode() != data:
                raise ValueError()
            return obj
        except (TypeError, ValueError, UnicodeError):
            raise PublicError(ErrorCode.BAD_FORMAT) from None


@dataclass(frozen=True, repr=False)
class ShareRecord:
    party: Literal['U', 'S']
    t: tuple[int, ...]
    m: tuple[int, ...]
    z: tuple[int, ...]
    e: int

    def __post_init__(self):
        if self.party not in ('U', 'S'):
            raise PublicError(ErrorCode.BAD_FORMAT)
        _words(self.t, 256)
        _words(self.m, 256)
        _words(self.z, 1024)
        _words((self.e,), 1)

    def encode(self) -> bytes:
        return struct.pack('<1537Q', *(self.t + self.m + self.z + (self.e,)))

    @classmethod
    def decode(cls, data: bytes, party: Literal['U', 'S']) -> ShareRecord:
        if len(data) != RECORD_BYTES:
            raise PublicError(ErrorCode.BAD_SHAPE)
        values = struct.unpack('<1537Q', data)
        return cls(party, values[:256], values[256:512], values[512:1536], values[-1])


@dataclass(frozen=True, repr=False)
class PreparationShare:
    """z/e already emitted by the fixed private Embedding computation."""
    preparation_id: str
    party: Literal['D', 'S']
    z: tuple[int, ...]
    e: int

    def __post_init__(self):
        if self.party not in ('D', 'S'):
            raise PublicError(ErrorCode.BAD_FORMAT)
        try:
            if str(uuid.UUID(self.preparation_id))!=self.preparation_id: raise ValueError()
        except (ValueError, TypeError):
            raise PublicError(ErrorCode.BAD_FORMAT) from None
        _words(self.z, 1024)
        _words((self.e,), 1)


@dataclass(frozen=True, repr=False)
class ServerReshare:
    """The only resharing plaintext D sends S; never contains U's z/e share."""
    preparation_id: str
    t: tuple[int, ...]
    m: tuple[int, ...]
    delta_z: tuple[int, ...]
    delta_e: int

    def __post_init__(self):
        try:
            if str(uuid.UUID(self.preparation_id)) != self.preparation_id: raise ValueError()
        except (ValueError, TypeError):
            raise PublicError(ErrorCode.BAD_FORMAT) from None
        _words(self.t, 256)
        _words(self.m, 256)
        _words(self.delta_z, 1024)
        _words((self.delta_e,), 1)

    def encode(self) -> bytes:
        return struct.pack('<1537Q', *(self.t + self.m + self.delta_z + (self.delta_e,)))

    @classmethod
    def decode(cls, data: bytes, preparation_id: str) -> ServerReshare:
        values = ShareRecord.decode(data, 'S')
        return cls(preparation_id, values.t, values.m, values.z, values.e)


def reshare_for_user(t: tuple[int, ...], m: tuple[int, ...], preparation: PreparationShare) -> tuple[ShareRecord, ServerReshare]:
    """D local: fresh independent 59-bit CSPRNG words for every invocation."""
    if preparation.party != 'D':
        raise PublicError(ErrorCode.BAD_FORMAT)
    _words(t, 256)
    _words(m, 256)
    if any(v not in (0, 1) for v in m) or any(v > 151642 for v in t):
        raise PublicError(ErrorCode.BAD_FORMAT)
    r_t = tuple(secrets.randbits(59) for _ in t)
    r_m = tuple(secrets.randbits(59) for _ in m)
    delta = tuple(secrets.randbits(59) for _ in range(1025))
    u = ShareRecord('U', r_t, r_m, tuple((a+d)&MASK for a,d in zip(preparation.z,delta)), (preparation.e+delta[-1])&MASK)
    s = ServerReshare(preparation.preparation_id, tuple((a-r)&MASK for a,r in zip(t,r_t)), tuple((a-r)&MASK for a,r in zip(m,r_m)), delta[:1024], delta[-1])
    return u, s


def apply_reshare(preparation: PreparationShare, message: ServerReshare) -> ShareRecord:
    if preparation.party != 'S' or preparation.preparation_id != message.preparation_id:
        raise PublicError(ErrorCode.VERSION_MISMATCH)
    return ShareRecord('S', message.t, message.m, tuple((b-d)&MASK for b,d in zip(preparation.z,message.delta_z)), (preparation.e-message.delta_e)&MASK)


def seal_delivery(record: ShareRecord, public_key: X25519PublicKey, binding: DeliveryBinding) -> bytes:
    if record.party != 'U':
        raise PublicError(ErrorCode.BAD_FORMAT)
    # cryptography49's single-shot API authenticates auxiliary data in info;
    # its AEAD AAD is empty. All required metadata is context-bound here.
    return SUITE.encrypt(record.encode(), public_key, info=INFO_DOMAIN + binding.encode())


def open_delivery(package: bytes, private_key: X25519PrivateKey, binding: DeliveryBinding) -> ShareRecord:
    if len(package) != HPKE_BYTES:
        raise PublicError(ErrorCode.BAD_SHAPE)
    try:
        plaintext = SUITE.decrypt(package, private_key, info=INFO_DOMAIN + binding.encode())
    except (InvalidTag, ValueError):
        raise PublicError(ErrorCode.AUTHENTICATION_FAILED) from None
    return ShareRecord.decode(plaintext, 'U')


class UserShareCache:
    """U local complete-manifest verification; fetch decisions use only versions.

    An update becomes visible only after every current package authenticates.
    A caller holding the returned immutable snapshot may finish after revocation.
    """
    def __init__(self, user_id: str, key_version: int, private_key: X25519PrivateKey, configuration: str):
        self.user_id,self.key_version,self.private_key,self.configuration=user_id,key_version,private_key,configuration
        self._packages: dict[str,ShareRecord]={}
        self._manifest: dict | None=None

    def synchronize(self, manifest: dict, fetch_package) -> str:
        expected={'format_version','user_id','authorization_version','N','entries','manifest_sha256'}
        if not isinstance(manifest,dict) or set(manifest)!=expected or manifest['user_id']!=self.user_id or type(manifest['format_version']) is not int or manifest['format_version']!=1:
            raise PublicError(ErrorCode.VERSION_MISMATCH)
        if type(manifest['N']) is not int or manifest['N']<0 or type(manifest['authorization_version']) is not int or manifest['authorization_version']<0 or not isinstance(manifest['entries'],list):
            raise PublicError(ErrorCode.BAD_FORMAT)
        body={k:v for k,v in manifest.items() if k!='manifest_sha256'}
        if digest(canonical_json(body))!=manifest['manifest_sha256'] or manifest['N']!=len(manifest['entries']):
            raise PublicError(ErrorCode.BAD_FORMAT)
        fresh={}; slots=[]
        for entry in manifest['entries']:
            if not isinstance(entry,dict) or set(entry)!={'binding','package_sha256'}: raise PublicError(ErrorCode.BAD_FORMAT)
            try: binding=DeliveryBinding(**entry['binding'])
            except TypeError: raise PublicError(ErrorCode.BAD_FORMAT) from None
            if binding.user_id!=self.user_id or binding.key_version!=self.key_version or binding.model_numeric_version!=self.configuration:
                raise PublicError(ErrorCode.VERSION_MISMATCH)
            sha=entry['package_sha256']; slots.append(binding.opaque_slot)
            try:
                if len(sha)!=64 or bytes.fromhex(sha).hex()!=sha: raise ValueError()
            except (ValueError,TypeError): raise PublicError(ErrorCode.BAD_FORMAT) from None
            # Cached plaintext shares still need the same previously verified
            # binding, not merely the same ciphertext digest with new metadata.
            prior=self._packages.get(sha)
            old_binding=None
            if self._manifest:
                old_binding=next((x['binding'] for x in self._manifest['entries'] if x['package_sha256']==sha),None)
            if prior is not None and old_binding==entry['binding']:
                fresh[sha]=prior
            else:
                package=fetch_package(sha)
                if digest(package)!=sha: raise PublicError(ErrorCode.AUTHENTICATION_FAILED)
                fresh[sha]=open_delivery(package,self.private_key,binding)
        if slots!=sorted(slots) or len(set(slots))!=len(slots): raise PublicError(ErrorCode.BAD_FORMAT)
        self._packages=fresh
        self._manifest=json.loads(canonical_json(manifest))
        return manifest['manifest_sha256']

    def snapshot(self) -> tuple[ShareRecord,...]:
        if self._manifest is None: raise PublicError(ErrorCode.MATERIALS_NOT_READY)
        return tuple(self._packages[x['package_sha256']] for x in self._manifest['entries'])
