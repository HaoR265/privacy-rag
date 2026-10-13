"""D local retained files: AES-256-GCM; key material only in OS key store.

Python immutable byte copies cannot provide a physical secure-erasure guarantee.
No plaintext-key-file fallback and no default key are supplied.
"""
from __future__ import annotations
import base64
import os
import secrets
import struct
import tempfile
from pathlib import Path
import keyring
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from .delivery import canonical_json
from .errors import PublicError, ErrorCode

class ProviderVault:
    SERVICE = 'private-rag-v1/provider-aes256gcm'
    MAGIC = b'PRDV\x00\x01'

    def __init__(self, provider_id: str, key_id: str):
        self.provider_id, self.key_id = provider_id, key_id

    def _backend(self):
        backend = keyring.get_keyring()
        # Accept OS-backed modules, never keyrings.alt's plaintext/encrypted files.
        module = type(backend).__module__
        if module not in ('keyring.backends.SecretService', 'keyring.backends.kwallet', 'keyring.backends.macOS', 'keyring.backends.Windows'):
            raise PublicError(ErrorCode.BACKEND_NOT_READY)
        return backend

    def initialize_new(self) -> None:
        backend = self._backend()
        name = f'{self.provider_id}:{self.key_id}'
        if backend.get_password(self.SERVICE, name) is not None:
            raise PublicError(ErrorCode.BAD_STATE)
        backend.set_password(self.SERVICE, name, base64.b64encode(secrets.token_bytes(32)).decode('ascii'))

    def _key(self) -> bytes:
        try:
            value = self._backend().get_password(self.SERVICE, f'{self.provider_id}:{self.key_id}')
            key = base64.b64decode(value, validate=True) if value else b''
            if len(key) != 32:
                raise ValueError()
            return key
        except PublicError:
            raise
        except Exception:
            raise PublicError(ErrorCode.BACKEND_NOT_READY) from None

    def _aad(self, object_id: str) -> bytes:
        return canonical_json({'format_version': 1, 'provider_id': self.provider_id, 'key_id': self.key_id, 'object_id': object_id})

    def save(self, path: Path, object_id: str, plaintext: bytes) -> None:
        nonce = secrets.token_bytes(12)
        data = self.MAGIC + nonce + AESGCM(self._key()).encrypt(nonce, plaintext, self._aad(object_id))
        path = Path(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.vault-')
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
            dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def load(self, path: Path, object_id: str) -> bytes:
        data = Path(path).read_bytes()
        if len(data) < 34 or data[:6] != self.MAGIC:
            raise PublicError(ErrorCode.BAD_FORMAT)
        try:
            return AESGCM(self._key()).decrypt(data[6:18], data[18:], self._aad(object_id))
        except InvalidTag:
            raise PublicError(ErrorCode.AUTHENTICATION_FAILED) from None


class RecipientKeyStore:
    """U long-term HPKE keys in an OS backend, isolated from D-vault keys."""
    SERVICE = 'private-rag-v1/recipient-hpke-x25519'

    def __init__(self, user_id: str):
        self.user_id=user_id

    def _backend(self):
        return ProviderVault(self.user_id,'unused')._backend()

    def generate(self,key_version: int) -> bytes:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        if type(key_version) is not int or key_version<1: raise PublicError(ErrorCode.BAD_FORMAT)
        name=f'{self.user_id}:{key_version}'; backend=self._backend()
        if backend.get_password(self.SERVICE,name) is not None: raise PublicError(ErrorCode.BAD_STATE)
        key=X25519PrivateKey.generate()
        backend.set_password(self.SERVICE,name,base64.b64encode(key.private_bytes_raw()).decode('ascii'))
        return key.public_key().public_bytes_raw()

    def load(self,key_version: int):
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        try:
            value=self._backend().get_password(self.SERVICE,f'{self.user_id}:{key_version}')
            if value is None: raise PublicError(ErrorCode.MATERIALS_NOT_READY)
            return X25519PrivateKey.from_private_bytes(base64.b64decode(value,validate=True))
        except PublicError: raise
        except Exception: raise PublicError(ErrorCode.BACKEND_NOT_READY) from None
