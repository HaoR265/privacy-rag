"""ART-003/RUN-002: fixed frames; no compression or automatic retry.

The sequence belongs to one directional connection. Payload size and frame count
are public and supplied by the operation plan, never derived from secret values.
"""
import json
import struct
from dataclasses import dataclass
from .errors import ErrorCode, PublicError

HEADER_BYTES = 4096
PAYLOAD_BYTES = 65536
PREFIX = struct.Struct("!4sHHQII")
FRAME_BYTES = PREFIX.size + HEADER_BYTES + PAYLOAD_BYTES
REQUIRED = {"session_id", "stage", "operation_sequence", "public_configuration", "chunk_index", "chunk_count", "payload_bytes", "format_version"}


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PublicError(ErrorCode.BAD_FORMAT)
        result[key] = value
    return result


@dataclass(frozen=True)
class Frame:
    header: dict
    payload: bytes

    def encode(self) -> bytes:
        h = self.header
        if set(h) != REQUIRED or h["format_version"] != 1:
            raise PublicError(ErrorCode.BAD_FORMAT)
        for name in ("operation_sequence", "chunk_index", "chunk_count", "payload_bytes"):
            if type(h[name]) is not int or h[name] < 0:
                raise PublicError(ErrorCode.BAD_FORMAT)
        if not 0 <= h["chunk_index"] < h["chunk_count"] or h["payload_bytes"] != len(self.payload) or len(self.payload) > PAYLOAD_BYTES:
            raise PublicError(ErrorCode.BAD_SHAPE)
        data = canonical_json(h)
        if len(data) > HEADER_BYTES:
            raise PublicError(ErrorCode.BAD_FORMAT)
        return PREFIX.pack(b"PRG1", 1, 0, h["operation_sequence"], len(data), len(self.payload)) + data.ljust(HEADER_BYTES, b"\0") + self.payload.ljust(PAYLOAD_BYTES, b"\0")

    @classmethod
    def decode(cls, raw: bytes):
        if len(raw) != FRAME_BYTES:
            raise PublicError(ErrorCode.BAD_SHAPE)
        magic, version, flags, seq, hn, pn = PREFIX.unpack_from(raw)
        if (magic, version, flags) != (b"PRG1", 1, 0) or not 0 < hn <= HEADER_BYTES or pn > PAYLOAD_BYTES:
            raise PublicError(ErrorCode.BAD_FORMAT)
        hstart = PREFIX.size
        pstart = hstart + HEADER_BYTES
        if any(raw[hstart+hn:pstart]) or any(raw[pstart+pn:]):
            raise PublicError(ErrorCode.BAD_FORMAT)
        try:
            header = json.loads(raw[hstart:hstart+hn], object_pairs_hook=_unique)
            frame = cls(header, raw[pstart:pstart+pn])
            if header["operation_sequence"] != seq or frame.encode() != raw:
                raise PublicError(ErrorCode.BAD_FORMAT)
            return frame
        except (ValueError, KeyError, TypeError, UnicodeError):
            raise PublicError(ErrorCode.BAD_FORMAT) from None


class Receiver:
    def __init__(self, session_id: str, configuration: str):
        self.session_id, self.configuration = session_id, configuration
        self.next_sequence = 0

    def accept(self, raw: bytes, *, stage: str, chunk_index: int, chunk_count: int, payload_bytes: int) -> bytes:
        frame = Frame.decode(raw)
        expected = {"session_id": self.session_id, "public_configuration": self.configuration, "stage": stage,
                    "operation_sequence": self.next_sequence, "chunk_index": chunk_index, "chunk_count": chunk_count,
                    "payload_bytes": payload_bytes, "format_version": 1}
        if frame.header != expected:
            raise PublicError(ErrorCode.REPLAY_OR_ORDER)
        self.next_sequence += 1
        return frame.payload


def encode_ring(values) -> bytes:
    if any(type(x) is not int or not 0 <= x < (1 << 59) for x in values):
        raise PublicError(ErrorCode.BAD_FORMAT)
    return struct.pack(f"<{len(values)}Q", *values)


def decode_ring(raw: bytes, count: int) -> tuple[int, ...]:
    if len(raw) != 8 * count:
        raise PublicError(ErrorCode.BAD_SHAPE)
    values = struct.unpack(f"<{count}Q", raw)
    if any(x >= (1 << 59) for x in values):
        raise PublicError(ErrorCode.BAD_FORMAT)
    return values
