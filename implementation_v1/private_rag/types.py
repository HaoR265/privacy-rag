"""CRY-001/CODE-P01: explicit local-share types and checked public descriptors."""
from dataclasses import dataclass, field
from math import prod
from enum import Enum
from uuid import UUID
from .errors import ErrorCode, PublicError

RING_BITS = 59
MODULUS = 1 << RING_BITS
MASK = MODULUS - 1


class Party(str, Enum):
    C = "C"
    S = "S"


@dataclass(frozen=True)
class Descriptor:
    session_id: str
    module_id: str
    operation_id: int
    shape: tuple[int, ...]
    representation: str
    model_version: str
    protocol_version: str = "prag-v1"
    fractional_bits: int | None = None
    ring_bits: int | None = None
    signed_interpretation: str | None = None
    bit_width: int | None = None

    def __post_init__(self):
        try:
            UUID(self.session_id)
        except (ValueError, TypeError, AttributeError):
            raise PublicError(ErrorCode.BAD_FORMAT) from None
        if type(self.operation_id) is not int or self.operation_id < 0:
            raise PublicError(ErrorCode.BAD_FORMAT)
        if not isinstance(self.shape, tuple) or any(type(d) is not int or d < 0 for d in self.shape):
            raise PublicError(ErrorCode.BAD_SHAPE)
        if not self.module_id or not self.model_version or self.protocol_version != "prag-v1":
            raise PublicError(ErrorCode.VERSION_MISMATCH)
        if self.representation in {"arithmetic_share", "private_weight"}:
            if self.ring_bits != 59 or self.signed_interpretation != "twos_complement" or type(self.fractional_bits) is not int or not 0 <= self.fractional_bits <= 58:
                raise PublicError(ErrorCode.BAD_FORMAT)
        elif self.representation == "boolean_share":
            if type(self.bit_width) is not int or self.bit_width < 1 or self.fractional_bits is not None:
                raise PublicError(ErrorCode.BAD_FORMAT)
        elif self.representation != "public_value":
            raise PublicError(ErrorCode.BAD_FORMAT)


@dataclass(frozen=True, repr=False)
class ArithmeticShare:
    descriptor: Descriptor
    party: Party
    values: tuple[int, ...] = field(repr=False)

    def __post_init__(self):
        if self.descriptor.representation != "arithmetic_share" or not isinstance(self.party, Party):
            raise PublicError(ErrorCode.BAD_FORMAT)
        if len(self.values) != prod(self.descriptor.shape):
            raise PublicError(ErrorCode.BAD_SHAPE)
        if any(type(x) is not int or not 0 <= x < MODULUS for x in self.values):
            raise PublicError(ErrorCode.BAD_FORMAT)

    def __repr__(self):
        return "ArithmeticShare(<private>)"


@dataclass(frozen=True, repr=False)
class BooleanShare:
    descriptor: Descriptor
    party: Party
    bits: tuple[bool, ...] = field(repr=False)

    def __post_init__(self):
        d = self.descriptor
        if d.representation != "boolean_share" or not isinstance(self.party, Party):
            raise PublicError(ErrorCode.BAD_FORMAT)
        if len(self.bits) != prod(d.shape) * d.bit_width or any(type(x) is not bool for x in self.bits):
            raise PublicError(ErrorCode.BAD_SHAPE)


@dataclass(frozen=True, repr=False)
class SecretToken:
    share: ArithmeticShare

    def __post_init__(self):
        if self.share.descriptor.fractional_bits != 0:
            raise PublicError(ErrorCode.BAD_FORMAT)


@dataclass(frozen=True, repr=False)
class PrivateWeight:
    descriptor: Descriptor
    values: tuple[int, ...] = field(repr=False)
    owner: Party = Party.S

    def __post_init__(self):
        if self.owner != Party.S or self.descriptor.representation != "private_weight":
            raise PublicError(ErrorCode.BAD_FORMAT)
        if len(self.values) != prod(self.descriptor.shape):
            raise PublicError(ErrorCode.BAD_SHAPE)


@dataclass(frozen=True)
class Public:
    value: int | str | tuple
