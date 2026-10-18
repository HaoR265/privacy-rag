"""CODE-P04: exceptions carry public codes only, never input values."""
from enum import Enum


class ErrorCode(str, Enum):
    BAD_FORMAT = "BAD_FORMAT"
    BAD_SHAPE = "BAD_SHAPE"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    MATERIALS_NOT_READY = "MATERIALS_NOT_READY"
    BUSY = "BUSY"
    BAD_STATE = "BAD_STATE"
    REPLAY_OR_ORDER = "REPLAY_OR_ORDER"
    BACKEND_NOT_READY = "BACKEND_NOT_READY"
    DISCONNECTED = "DISCONNECTED"


class PublicError(Exception):
    def __init__(self, code: ErrorCode):
        self.code = code
        super().__init__(code.value)
