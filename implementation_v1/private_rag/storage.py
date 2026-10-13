"""Immutable, durable content-addressed objects; PostgreSQL owns publication."""
from __future__ import annotations
import os
import re
import tempfile
from pathlib import Path
from .delivery import digest
from .errors import PublicError, ErrorCode

class ObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _path(self, sha256: str) -> Path:
        if not re.fullmatch('[0-9a-f]{64}', sha256):
            raise PublicError(ErrorCode.BAD_FORMAT)
        return self.root / sha256

    def put(self, payload: bytes) -> str:
        name = digest(payload)
        path = self._path(name)
        if path.exists():
            self.get(name)
            return name
        fd, temporary = tempfile.mkstemp(prefix='.staging-', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            # Content-addressed concurrent writers are identical; hardlink never
            # overwrites another existing immutable object.
            try:
                os.link(temporary, path)
            except FileExistsError:
                self.get(name)
            dfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            os.unlink(temporary)
        return name

    def get(self, sha256: str, expected_length: int | None = None) -> bytes:
        try:
            fd = os.open(self._path(sha256), os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                data = stream.read()
        except OSError:
            raise PublicError(ErrorCode.MATERIALS_NOT_READY) from None
        if digest(data) != sha256 or (expected_length is not None and len(data) != expected_length):
            raise PublicError(ErrorCode.MATERIALS_NOT_READY)
        return data
