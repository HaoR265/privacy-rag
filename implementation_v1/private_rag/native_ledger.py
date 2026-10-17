"""Validate this party's completed native ledger; never read shares or peer files.

The format is emitted by native/src/secure_ops.cpp. This is a local lifecycle
check, not proof of the arithmetic schedule or of another party's execution.
"""
from __future__ import annotations
import os
import re
import stat
from pathlib import Path
from uuid import UUID
from .errors import ErrorCode, PublicError

_RESERVATION = re.compile(rb'(0|[1-9][0-9]*) reserved ([A-Za-z0-9_]+) (0|[1-9][0-9]*) ring59\n')
_MAX_U64 = (1 << 64) - 1


def verify_closed_ledger(path: Path, session_id: str, configuration: str) -> int:
    """Stream a private regular ledger and return its reservation count.

    Failures have one fixed public error. Unknown material names are accepted:
    the native writer owns those names, and this validator does not invent a
    second protocol allowlist. A completed ledger is immutable after child exit.
    """
    fd = None
    try:
        session_hex = UUID(session_id).hex
        if str(UUID(session_id)) != session_id or re.fullmatch('[0-9a-f]{64}', configuration) is None:
            raise ValueError()
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o077 or before.st_nlink != 1):
            raise ValueError()
        header = f'format=1 session={session_hex} configuration={configuration}\n'.encode('ascii')
        expected = 0
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            if stream.readline(4097) != header:
                raise ValueError()
            while True:
                # All writer kind names are short ASCII identifiers; bounded
                # reads prevent an invalid file consuming unbounded memory.
                row = stream.readline(4097)
                if row == b'terminal=closed all_reserved=burned\n':
                    if expected == 0 or stream.read(1):
                        raise ValueError()
                    break
                match = _RESERVATION.fullmatch(row)
                if match is None or len(row) > 4096:
                    raise ValueError()
                sequence, kind, count = match.groups()
                sequence, count = int(sequence), int(count)
                if sequence != expected or sequence > _MAX_U64 or count > _MAX_U64:
                    raise ValueError()
                if expected == 0 and (kind != b'session' or count != 0):
                    raise ValueError()
                if expected > 0 and kind == b'session':
                    raise ValueError()
                expected += 1
            after = os.fstat(stream.fileno())
            current = os.lstat(path)
            fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink', 'st_uid')
            if any(getattr(before, key) != getattr(after, key) or
                   getattr(after, key) != getattr(current, key) for key in fields):
                raise ValueError()
        return expected
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):
        raise PublicError(ErrorCode.BAD_STATE) from None
    finally:
        if fd is not None:
            os.close(fd)
