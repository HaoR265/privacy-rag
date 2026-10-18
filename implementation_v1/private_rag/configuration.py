"""Immutable artifact binding, separate from permission to enable computation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
from .errors import ErrorCode, PublicError

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def canonical(value) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')

# These are the fixed implementation's semantics, not a user-selected profile.
# A future authorized profile change must update this binding with its artifacts.
FIXED_METADATA = {
    "format_version": "private-rag-configuration-v1",
    "protocol": "prag-v1", "ring_bits": 59,
    "numeric_version": "v1-f12-coordinated-20261013",
    "model_revisions": {
        "embedding": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "generation": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    },
    "budgets": {"cache": 1280, "decode": 255, "document_embedding": 257,
                "evidence": [256, 256], "outputs": 256, "prefill": 1024,
                "query_embedding": 160, "question": 128},
}


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_configuration_metadata(config: dict):
    """Check fixed meanings and all entries before any role-based file skip.

    This is not activation, a source/binary binding, or a completeness proof for
    an arbitrary artifact list. Deployment trust separately pins the digest.
    """
    if type(config) is not dict or set(config) != set(FIXED_METADATA) | {"template_sha256", "files"}:
        raise PublicError(ErrorCode.BAD_FORMAT)
    if canonical({key: config[key] for key in FIXED_METADATA}) != canonical(FIXED_METADATA):
        raise PublicError(ErrorCode.VERSION_MISMATCH)
    if not _digest(config["template_sha256"]) or type(config["files"]) is not list or not config["files"]:
        raise PublicError(ErrorCode.BAD_FORMAT)
    seen = set()
    template = None
    for entry in config["files"]:
        if type(entry) is not dict or set(entry) != {"path", "scope", "bytes", "sha256"}:
            raise PublicError(ErrorCode.BAD_FORMAT)
        path = entry["path"]
        if (not isinstance(path, str) or not path or "\\" in path or "\x00" in path or
            PurePosixPath(path).is_absolute() or PurePosixPath(path).as_posix() != path or
            ".." in PurePosixPath(path).parts or not path.startswith("artifacts/") or path in seen):
            raise PublicError(ErrorCode.BAD_FORMAT)
        seen.add(path)
        if (entry["scope"] not in ("public", "server_manifest", "server_weights") or
            type(entry["bytes"]) is not int or entry["bytes"] < 0 or not _digest(entry["sha256"])):
            raise PublicError(ErrorCode.BAD_FORMAT)
        if path == "artifacts/template/template_v1.json":
            template = entry
    if template is None or template["scope"] != "public" or template["sha256"] != config["template_sha256"]:
        raise PublicError(ErrorCode.VERSION_MISMATCH)


def check_configuration(root: Path, expected_digest: str, *, server_weights: bool) -> dict:
    """Check artifact identity before snapshot acceptance/worker startup.

    No model execution occurs here. A correctly hashed configuration can still
    fail the independent NUM-004 activation check.
    """
    root=root.resolve(); path=root/'artifacts/configuration.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_digest:
        raise PublicError(ErrorCode.VERSION_MISMATCH)
    try:
        config=json.loads(raw)
        if canonical(config)!=raw:
            raise PublicError(ErrorCode.BAD_FORMAT)
    except (ValueError, TypeError):
        raise PublicError(ErrorCode.BAD_FORMAT) from None
    validate_configuration_metadata(config)
    for artifact in config['files']:
        if artifact['scope']=='server_weights' and not server_weights: continue
        target=root/artifact['path']
        if not target.resolve().is_relative_to(root) or not target.is_file() or target.is_symlink():
            raise PublicError(ErrorCode.MATERIALS_NOT_READY)
        if target.stat().st_size!=artifact['bytes'] or sha256_file(target)!=artifact['sha256']:
            raise PublicError(ErrorCode.VERSION_MISMATCH)
    return config
