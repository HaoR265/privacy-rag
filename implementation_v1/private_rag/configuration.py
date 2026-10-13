"""Immutable artifact binding, separate from permission to enable computation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .errors import ErrorCode, PublicError

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def canonical(value) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')

def check_configuration(root: Path, expected_digest: str, *, server_weights: bool) -> dict:
    """Check artifact identity before snapshot acceptance/worker startup.

    No model execution occurs here. A correctly hashed configuration can still
    fail the independent NUM-004 activation check.
    """
    root=root.resolve(); path=root/'artifacts/configuration.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_digest:
        raise PublicError(ErrorCode.VERSION_MISMATCH)
    config=json.loads(raw)
    if config.get('format_version')!='private-rag-configuration-v1' or canonical(config)!=raw:
        raise PublicError(ErrorCode.BAD_FORMAT)
    for artifact in config['files']:
        if artifact['scope']=='server_weights' and not server_weights: continue
        target=root/artifact['path']
        if not target.resolve().is_relative_to(root) or not target.is_file() or target.is_symlink():
            raise PublicError(ErrorCode.MATERIALS_NOT_READY)
        if target.stat().st_size!=artifact['bytes'] or sha256_file(target)!=artifact['sha256']:
            raise PublicError(ErrorCode.VERSION_MISMATCH)
    return config
