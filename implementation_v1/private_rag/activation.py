"""CRY-015/NUM-004: production activation is a checked build capability.

No environment variable or CLI flag can turn reference calculations into R2.
The graph and scheduler are wired, but full-model numeric and execution evidence
are still missing. Primitive-level correctness does not activate the full graph.
"""
from dataclasses import dataclass, asdict
from .errors import ErrorCode, PublicError

@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    evidence: str | None

# These entries must be updated in source only together with reviewed artifacts
# and an actual implementation. A user-supplied JSON `approved=true` is insufficient.
PRODUCTION_CAPABILITIES = (
    Capability("bfv_session_failure_bound", True, "artifacts/bfv_correctness_bound.json"),
    Capability("private_model_rtx2060_backend", False, "records/private_model_device_policy.json"),
    Capability("model_numeric_range_evidence", False, "artifacts/constants/range_evidence_v1_coordinated.json"),
    Capability("complete_private_model_execution", False, "records/implementation_progress.md"),
    Capability("compute_mtls_framing_and_cleanup", False, "records/crypto-debug-run.json"),
)

def report():
    return {"production_enabled": all(x.available for x in PRODUCTION_CAPABILITIES),
            "capabilities": [asdict(x) for x in PRODUCTION_CAPABILITIES],
            "reference_fallback": False,
            "required_model_device": "NVIDIA GeForce RTX 2060",
            "cpu_model_fallback": False}

def require_private_backend():
    if not all(x.available for x in PRODUCTION_CAPABILITIES):
        raise PublicError(ErrorCode.BACKEND_NOT_READY)
