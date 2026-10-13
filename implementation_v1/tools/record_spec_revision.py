"""Record current document identities and the actual read-only package check."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT.parent / "项目文档/第一版机读规范"


def main():
    run = subprocess.run([sys.executable, str(SPEC / "校验规范.py")], capture_output=True, text=True, check=True)
    check = json.loads(run.stdout)
    if check["errors"] or check["system_tests_executed"]:
        raise ValueError("SPEC_CHECK_FAILED")
    original = json.loads((ROOT / "records/spec_read_manifest.json").read_text())
    before = {f["name"]: f["sha256"] for f in original["files"]}
    files = []
    for path in sorted(SPEC.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            name = path.relative_to(SPEC).as_posix()
            raw = path.read_bytes()
            files.append({"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                          "original_read_sha256": before.get(name)})
    result = {"scope": "document revision identities, distinct from model/cryptographic execution",
              "approved_numeric_decisions": ["DEC-011", "DEC-012", "DEC-013"],
              "approved_protocol_layout_scope": "DEC-014: evidenced multi-target packing and public block homomorphic accumulation within unchanged parameters, roles, privacy and budgets",
              "approved_matrix_preprocessing_scope": "DEC-015: BFV protected random cross terms for matrix Beaver triples; scalar/vector OT, C key ownership, parameters, models and fixed budgets unchanged",
              "revision_policy": "Edit current source text; justified reasonable numeric adjustments are delegated within unchanged security, model, cryptographic and fixed-budget constraints.",
              "numeric_profile": json.loads((ROOT / "artifacts/constants/numeric_profile_v1_qk.json").read_text())["profile_id"],
              "machine_package_consistency": check, "machine_files": files,
              "formal_tests_and_acceptance_defined": False, "full_R2_executed": False}
    (ROOT / "records/spec_revision_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "records/spec_consistency.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"machine_files": len(files), "rules_checked": check["rules"], "errors": check["errors"], "full_R2_executed": False}))


if __name__ == "__main__":
    main()
