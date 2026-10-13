"""Implementation check of actual artifact identities and tokenizer rejection.

Only an isolated copy is mutated. This is not a model run or acceptance suite.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from private_rag.configuration import check_configuration
from private_rag.errors import ErrorCode, PublicError


def main():
    expected = (ROOT / "artifacts/configuration.sha256").read_text().strip()
    started = time.monotonic()
    config = check_configuration(ROOT, expected, server_weights=True)
    full_seconds = time.monotonic() - started
    target = "artifacts/models/generation/checkpoint/tokenizer.json"
    public = [f for f in config["files"] if f["scope"] != "server_weights"]
    if target not in {f["path"] for f in public}:
        raise RuntimeError("TOKENIZER_NOT_BOUND")
    with tempfile.TemporaryDirectory(prefix="prag-config-check-") as directory:
        copy_root = Path(directory)
        (copy_root / "artifacts").mkdir()
        shutil.copyfile(ROOT / "artifacts/configuration.json", copy_root / "artifacts/configuration.json")
        for item in public:
            source = ROOT / item["path"]
            destination = copy_root / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        check_configuration(copy_root, expected, server_weights=False)
        # Keep length unchanged, so rejection requires actual hash checking.
        with (copy_root / target).open("r+b") as file:
            original = file.read(1)
            file.seek(0)
            file.write(bytes([original[0] ^ 1]))
        try:
            check_configuration(copy_root, expected, server_weights=False)
        except PublicError as error:
            if error.code != ErrorCode.VERSION_MISMATCH:
                raise
        else:
            raise RuntimeError("MUTATED_TOKENIZER_ACCEPTED")
    result = {
        "scope": "actual identity check, no model execution or formal acceptance",
        "configuration_sha256": expected,
        "checked_files_server": len(config["files"]),
        "checked_files_client": len(public),
        "server_hash_seconds": full_seconds,
        "server_all_files_valid": True,
        "client_without_server_weights_valid": True,
        "same_size_mutated_tokenizer_rejected": True,
        "temporary_copy_removed": not copy_root.exists(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_R2_executed": False,
    }
    (ROOT / "records/configuration_bindings_debug.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
