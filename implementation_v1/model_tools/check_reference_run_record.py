"""Development checks for output isolation/provenance only; no model or protocol.

Temporary public inputs and injected failures check runner bookkeeping. They do
not establish CUDA availability, model correctness, or formal acceptance.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from model_tools.reference_run_record import ROOT, read_public_input, write_new
from model_tools.run_full_r1_rag import run


def main():
    checks = []

    def rejected(name, function, expected):
        try:
            function()
        except expected:
            checks.append(name)
        else:
            raise AssertionError(name)

    def execute(source, directory, **kwargs):
        with redirect_stdout(io.StringIO()):
            return run(source, directory, acknowledge_plaintext_reference=True, **kwargs)

    with tempfile.TemporaryDirectory(prefix="reference-record-development-") as temporary:
        root = Path(temporary)
        source = root / "input.json"
        public = {"question": "公开资料使用什么方案？", "documents": ["公开合成资料使用 HPKE。"]}
        source.write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")
        directory = root / "new-run"
        result = execute(source, directory, prepare_only=True)
        assert result["status"] == "prepared_only" and result["model_executed"] is False
        assert "torch" not in sys.modules and "references.pipeline" not in sys.modules
        assert not (directory / "result.json").exists()
        assert json.loads((directory / "public_input.json").read_text()) == public
        start = json.loads((directory / "start.json").read_text())
        assert start["public_input_source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert start["start_provenance"]["files_sha256"]["model_tools/run_full_r1_rag.py"]
        for kind in ("embedding", "generation"):
            inventory = directory / f"{kind}.weights.json"
            manifest = f"artifacts/models/{kind}/quantized_f12/manifest.json"
            assert hashlib.sha256(inventory.read_bytes()).hexdigest() == start["start_provenance"]["files_sha256"][manifest]
        checks.append("prepare_only_records_real_input_sources_and_manifests_without_model_import")
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        rejected("existing_run_rejected", lambda: execute(source, directory, prepare_only=True), FileExistsError)
        assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
        checks.append("existing_run_bytes_unchanged")
        rejected("existing_input_file_rejected_as_output", lambda: execute(source, source, prepare_only=True), FileExistsError)
        empty = root / "empty-existing"
        empty.mkdir()
        rejected("empty_existing_directory_rejected", lambda: execute(source, empty, prepare_only=True), FileExistsError)
        dangling = root / "dangling-output"
        dangling.symlink_to(root / "missing-target")
        rejected("dangling_output_symlink_rejected", lambda: execute(source, dangling, prepare_only=True), FileExistsError)
        assert not (root / "missing-target").exists()
        rejected("missing_plaintext_acknowledgement_rejected", lambda: run(source, root / "unacknowledged", prepare_only=True), ValueError)
        assert not (root / "unacknowledged").exists()
        for index, bad in enumerate(({"public_input": public}, {"question": "x"},
                                     {"question": "x", "documents": [3]},
                                     {"question": "x", "documents": [], "extra": True})):
            bad_source = root / f"invalid-{index}.json"
            bad_source.write_text(json.dumps(bad))
            rejected(f"explicit_input_schema_case_{index}_rejected", lambda: read_public_input(bad_source), ValueError)
        existing = root / "preserved.json"
        existing.write_text("original")
        rejected("record_file_no_replace", lambda: write_new(existing, {"new": True}), FileExistsError)
        assert existing.read_text() == "original" and not list(root.glob(".pending-*"))
        with patch("model_tools.run_full_r1_rag.compare_identity", return_value={"unchanged": False, "changed_files": ["injected-change"]}):
            rejected("provenance_change_rejected", lambda: execute(source, root / "changed", prepare_only=True), RuntimeError)
        failed = json.loads((root / "changed/failure.json").read_text())
        assert failed["identity_check"]["unchanged"] is False
        assert not (root / "changed/prepared.json").exists() and not (root / "changed/result.json").exists()
        with patch("model_tools.run_full_r1_rag.capture_identity", side_effect=KeyboardInterrupt):
            rejected("interruption_recorded", lambda: execute(source, root / "interrupted", prepare_only=True), KeyboardInterrupt)
        assert json.loads((root / "interrupted/failure.json").read_text())["status"] == "interrupted"
        # This is deliberately injected before Torch/model import, not a claim
        # that a real GPU preflight was attempted or passed in this check.
        with patch("references.resource_policy.require_model_device", side_effect=RuntimeError("DEVELOPMENT_INJECTED_CUDA_REJECTION")):
            rejected("injected_device_failure_recorded_without_model", lambda: execute(source, root / "device-failure"), RuntimeError)
        failure = json.loads((root / "device-failure/failure.json").read_text())
        assert failure["stage"] == "device_preflight" and not failure["model_call_started"]
        assert "torch" not in sys.modules and "references.pipeline" not in sys.modules
        cli = subprocess.run([sys.executable, "-m", "model_tools.run_full_r1_rag"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
        assert cli.returncode == 2 and "--output-dir" in cli.stderr and "--public-input-json" in cli.stderr
        checks.append("CLI_requires_explicit_input_output_and_acknowledgement")
        cli_prepared = root / "cli-prepared"
        cli = subprocess.run([sys.executable, "-m", "model_tools.run_full_r1_rag",
                              "--public-input-json", str(source), "--output-dir", str(cli_prepared),
                              "--acknowledge-plaintext-reference", "--prepare-only"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
        assert cli.returncode == 0, cli.stderr
        assert json.loads((cli_prepared / "prepared.json").read_text())["model_executed"] is False
        assert not (cli_prepared / "result.json").exists()
        checks.append("CLI_prepare_only_creates_independent_preparation_record")
    print(json.dumps({"scope": "Runner bookkeeping development checks only; no model, CUDA computation or R2",
                      "checks_passed": len(checks), "checks": checks, "model_executed": False,
                      "GPU_preflight_executed": False, "protocol_executed": False,
                      "temporary_outputs_removed": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
