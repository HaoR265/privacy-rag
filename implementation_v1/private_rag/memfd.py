"""Linux anonymous sealed memory files, including Python builds without os.memfd_create."""
import ctypes
import fcntl
import os


def create_memfd(name: str, *, allow_sealing: bool = False) -> int:
    # Linux ABI flags; the libc route creates the same kernel object and never
    # falls back to a pathname or an unsealed temporary file.
    flags = 1 | (2 if allow_sealing else 0)  # MFD_CLOEXEC | MFD_ALLOW_SEALING
    if hasattr(os, 'memfd_create'):
        return os.memfd_create(name, flags)
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(name.encode('utf-8'), flags)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def seal_memfd(fd: int) -> None:
    # Linux fcntl ABI, including builds that omit the Python named constants.
    fcntl.fcntl(fd, 1033, 15)  # F_ADD_SEALS, WRITE|GROW|SHRINK|SEAL
    if fcntl.fcntl(fd, 1034) & 15 != 15:  # F_GET_SEALS
        raise RuntimeError('MEMFD_SEAL_FAILED')
