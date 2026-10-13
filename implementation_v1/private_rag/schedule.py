"""Public fixed execution profile. It does not assert an implemented transport plan."""
from dataclasses import dataclass, asdict
import hashlib
from .wire import canonical_json
from .errors import PublicError, ErrorCode

@dataclass(frozen=True)
class Stage:
    name: str
    rows: int
    key_prefix: int
    layers: int
    output_positions: int
    cache_write: int | None


def query_schedule(n: int):
    if type(n) is not int or n < 0:
        raise PublicError(ErrorCode.BAD_SHAPE)
    if n == 0:
        yield Stage("no_authorized_material", 0, 0, 0, 0, None)
        return
    yield Stage("query_embedding", 160, 160, 28, 1, None)
    yield Stage("full_scan_retrieval", n, 1024, 0, 2, None)
    yield Stage("secret_evidence_extraction", n, 256, 0, 2, None)
    yield Stage("generation_prefill", 1024, 1024, 28, 1, 0)
    for t in range(255):
        yield Stage("generation_decode", 1, 1025+t, 28, 1, 1024+t)
    yield Stage("final_release", 256, 0, 0, 256, None)


def schedule_hash(n: int) -> str:
    return hashlib.sha256(canonical_json([asdict(x) for x in query_schedule(n)])).hexdigest()
