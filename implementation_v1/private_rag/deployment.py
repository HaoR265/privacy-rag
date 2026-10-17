"""Verify a code-pinned deployment before creating private worker inputs.

Artifact configuration and build identity are distinct. This is identity
validation, not approval of a BFV argument or permission to enable a backend.
Stat checks detect observed replacement/mutation; they do not seal a later exec.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from . import deployment_trust
from .configuration import canonical
from .errors import ErrorCode, PublicError
from .worker_cleanup import defer_cancellation_signals

WORKER_PATH = 'build/compute/prag_worker'
DEPLOYMENT_PATH = 'artifacts/deployment.json'
TRUST_PATH = 'private_rag/deployment_trust.py'
KINDS = frozenset({'worker', 'source', 'link_dependency', 'runtime_dependency', 'bfv_applicability'})


@dataclass(frozen=True)
class VerifiedDeployment:
    worker: Path
    configuration_sha256: str
    build_identity_sha256: str
    deployment_sha256: str
    files: tuple[dict, ...]


def _fail(code=ErrorCode.BAD_FORMAT):
    raise PublicError(code)


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        _fail()
    return value


def _stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _path(root: Path, name: str, kind: str) -> Path:
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        _fail()
    candidate = Path(name)
    if '..' in candidate.parts or candidate.as_posix() != name:
        _fail()
    if candidate.is_absolute():
        if kind not in {'link_dependency', 'runtime_dependency'} or not any(
                candidate.is_relative_to(base) for base in (Path('/usr/lib'), Path('/usr/lib64'))):
            _fail()
        path = candidate
    else:
        path = root / candidate
    # Reject symbolic path components, including a final-link substitution.
    # System records must use the actual absolute library path, not an alias.
    for component in (path, *path.parents):
        if component == root or component == component.parent:
            break
        if component.is_symlink():
            _fail(ErrorCode.MATERIALS_NOT_READY)
    return path


def _stable_file(path: Path, *, capture=False):
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or (capture and before.st_size > 8 * 1024 * 1024):
        _fail(ErrorCode.MATERIALS_NOT_READY)
    hasher = hashlib.sha256()
    content = bytearray() if capture else None
    fd = None
    try:
        # Block cancellation only while installing owned-fd state. The long
        # hash read remains interruptible, and every opened fd reaches finally.
        with defer_cancellation_signals():
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        opened = os.fstat(fd)
        if _stat_identity(before) != _stat_identity(opened):
            _fail(ErrorCode.VERSION_MISMATCH)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            if capture:
                content.extend(chunk)
                if len(content) > 8 * 1024 * 1024:
                    _fail(ErrorCode.MATERIALS_NOT_READY)
        after = os.fstat(fd)
        named_after = path.stat(follow_symlinks=False)
        if not (_stat_identity(before) == _stat_identity(after) == _stat_identity(named_after)):
            _fail(ErrorCode.VERSION_MISMATCH)
    finally:
        with defer_cancellation_signals():
            if fd is not None:
                closing, fd = fd, None
                os.close(closing)
    return {'path': str(path), 'sha256': hasher.hexdigest(), 'bytes': before.st_size,
            'stat_before': _stat_identity(before), 'stat_after': _stat_identity(named_after)}, bytes(content) if capture else None


def verify_deployment(root: Path, configuration_sha256: str) -> VerifiedDeployment:
    """No caller-provided trust digest or enable switch; None always rejects."""
    trusted = deployment_trust.TRUSTED_DEPLOYMENT_SHA256
    if trusted is None:
        _fail(ErrorCode.BACKEND_NOT_READY)
    try:
        _sha(trusted); _sha(configuration_sha256)
        root = Path(root).resolve(strict=True)
        observed, raw = _stable_file(_path(root, DEPLOYMENT_PATH, 'source'), capture=True)
        if observed['sha256'] != trusted:
            _fail(ErrorCode.VERSION_MISMATCH)
        record = json.loads(raw)
        if not isinstance(record, dict) or set(record) != {
                'format_version', 'configuration_sha256', 'build_identity_sha256', 'build'}:
            _fail()
        if canonical(record) != raw or record['format_version'] != 'private-rag-deployment-v1':
            _fail()
        if _sha(record['configuration_sha256']) != configuration_sha256:
            _fail(ErrorCode.VERSION_MISMATCH)
        build = record['build']
        if not isinstance(build, dict) or set(build) != {'worker_abi', 'files'} or build['worker_abi'] != 'private-rag-worker-v1':
            _fail()
        if _sha(record['build_identity_sha256']) != hashlib.sha256(canonical(build)).hexdigest():
            _fail(ErrorCode.VERSION_MISMATCH)
        files = build['files']
        if not isinstance(files, list) or not files:
            _fail()
        names, kinds, observations = [], set(), [observed]
        worker = None
        for item in files:
            if not isinstance(item, dict) or set(item) != {'path', 'kind', 'bytes', 'sha256'}:
                _fail()
            name, kind = item['path'], item['kind']
            if not isinstance(kind, str) or kind not in KINDS or type(item['bytes']) is not int or item['bytes'] < 0:
                _fail()
            # Both parties verify build identities. Model artifacts, including
            # S-only weight payloads, belong exclusively to the role-sensitive
            # configuration checker and must never be opened by this path.
            if isinstance(name, str) and (name == 'artifacts/models' or name.startswith('artifacts/models/')):
                _fail()
            path = _path(root, name, kind)
            if name in {DEPLOYMENT_PATH, 'artifacts/configuration.json', TRUST_PATH}:
                _fail()  # no deployment/configuration/trust-root self-reference
            if name == WORKER_PATH:
                if kind != 'worker' or worker is not None:
                    _fail()
                worker = path
            elif kind == 'worker':
                _fail()
            _sha(item['sha256'])
            actual, _ = _stable_file(path)
            if actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']:
                _fail(ErrorCode.VERSION_MISMATCH)
            names.append(name); kinds.add(kind); observations.append({**actual, 'kind': kind})
        if names != sorted(set(names)) or kinds != KINDS or worker is None:
            _fail()
        if not os.access(worker, os.X_OK):
            _fail(ErrorCode.BACKEND_NOT_READY)
        return VerifiedDeployment(worker, configuration_sha256, record['build_identity_sha256'], trusted, tuple(observations))
    except PublicError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise PublicError(ErrorCode.MATERIALS_NOT_READY) from None
