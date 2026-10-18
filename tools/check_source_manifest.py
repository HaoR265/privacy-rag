"""Verify the distributed file identities without building or running a model."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts:
            failures.append(name + ": invalid path")
            continue
        path = root / name
        if not path.is_file() or path.is_symlink():
            failures.append(name + ": missing or symbolic link")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name + ": SHA-256 mismatch")
    print(json.dumps({"verified": not failures,
                      "files_checked": len(manifest["files"]),
                      "failures": failures,
                      "model_executed": False}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
