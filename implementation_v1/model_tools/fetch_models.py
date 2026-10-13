"""Fetch the two MOD-001 revisions and hash real files; never uses moving main."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "embedding": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "generation": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(root: Path, metadata_only: bool = False) -> None:
    for name, (repo, revision) in MODELS.items():
        dest = root / name / "checkpoint"
        snapshot_download(repo, revision=revision, local_dir=dest, max_workers=2,
                          allow_patterns=["*.json", "*.txt", "LICENSE"] if metadata_only else
                          ["*.json", "*.txt", "*.safetensors", "LICENSE"])
        files = [{"path": p.relative_to(dest).as_posix(), "bytes": p.stat().st_size,
                  "sha256": sha256(p)} for p in sorted(dest.rglob("*"))
                 if p.is_file() and ".cache" not in p.parts]
        manifest = {"format_version": "model-checkpoint-v1", "model": repo,
                    "revision": revision, "weights_present": any(p["path"].endswith(".safetensors") for p in files),
                    "files": files}
        (dest.parent / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"model": name, "files": len(files), "weights_present": manifest["weights_present"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    fetch(args.root, args.metadata_only)
