"""Exclusive, public-input run records; does not import or execute a model."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
from pathlib import Path

from model_tools.reference_provenance import identify

ENTRYPOINT = "model_tools/run_full_r1_rag.py"
ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    """Never replace an existing output, including a pre-existing symlink."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes a fully written file atomically,
        # and fails if the destination exists. No replace/rename overwrite.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_public_input(path):
    """Require explicit plaintext input, not a historical result or fallback."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    public = json.loads(raw.decode("utf-8"))
    if not isinstance(public, dict) or set(public) != {"question", "documents"}:
        raise ValueError("PUBLIC_INPUT_REQUIRES_ONLY_QUESTION_AND_DOCUMENTS")
    if not isinstance(public["question"], str) or not isinstance(public["documents"], list):
        raise ValueError("PUBLIC_INPUT_INVALID_TYPES")
    if any(not isinstance(document, str) for document in public["documents"]):
        raise ValueError("PUBLIC_INPUT_DOCUMENTS_MUST_BE_STRINGS")
    # Empty input follows the pipeline's existing no-material semantics. Token
    # budgets, control strings and normalization remain InputPreparer's job.
    return public, raw, {"path": str(path), "sha256": sha256(raw), "bytes": len(raw)}


def capture_identity():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("RUN_FROM_IMPLEMENTATION_V1_DIRECTORY")
    identity = identify(ENTRYPOINT)
    for relative in ("model_tools/reference_run_record.py", "pyproject.toml", "requirements.lock"):
        identity["files_sha256"][relative] = sha256(Path(relative).read_bytes())
    identity["snapshot_sha256"] = sha256(json.dumps(
        identity["files_sha256"], sort_keys=True, separators=(",", ":")
    ).encode())
    identity["weight_validation_scope"] = (
        "Quantized and checkpoint manifest identities are captured here; this is not a full "
        "tensor scan. CUDAIntegerQwen verifies each quantized tensor hash when loading it."
    )
    return identity


def preserve_inputs(directory, raw_input, identity):
    """Keep exact input and weight inventories alongside their captured hashes."""
    with (directory / "public_input.json").open("xb") as stream:
        stream.write(raw_input)
        stream.flush()
        os.fsync(stream.fileno())
    for kind in ("embedding", "generation"):
        source = f"artifacts/models/{kind}/quantized_f12/manifest.json"
        raw = Path(source).read_bytes()
        if sha256(raw) != identity["files_sha256"][source]:
            raise RuntimeError("REFERENCE_WEIGHT_MANIFEST_CHANGED_DURING_PREPARATION")
        with (directory / f"{kind}.weights.json").open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())


def compare_identity(identity):
    """A failure to inspect provenance is a mismatch, never a successful check."""
    try:
        current = capture_identity()
    except Exception as error:
        return {"unchanged": False, "inspection_error": type(error).__name__ + ": " + str(error)}
    before, after = identity["files_sha256"], current["files_sha256"]
    changed = [path for path in sorted(before.keys() | after.keys()) if before.get(path) != after.get(path)]
    return {"unchanged": not changed, "changed_files": changed,
            "finish_snapshot_sha256": current["snapshot_sha256"]}
