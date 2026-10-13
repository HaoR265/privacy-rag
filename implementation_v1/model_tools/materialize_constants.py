"""Versioned public constants with 80/120 decimal digit floor agreement."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import mpmath as mp


def materialize(root: Path = Path("artifacts/constants")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    def produce(precision: int) -> dict:
        with mp.workdps(precision):
            lut = [[int(mp.floor(mp.exp(-mp.mpf(b) * mp.power(2, power)) * 2**24))
                    for b in range(256)] for power in (-12, -4, 4)]
            rope_cos, rope_sin = [], []
            theta = [mp.power(1000000, -mp.mpf(2*i)/128) for i in range(64)]
            for p in range(1280):
                rope_cos.append([int(mp.floor(mp.cos(p*t)*2**20)) for t in theta])
                rope_sin.append([int(mp.floor(mp.sin(p*t)*2**20)) for t in theta])
            return {"format_version": "public-numeric-constants-v1", "exp_f24": lut,
                    "rope_cos_f20": rope_cos, "rope_sin_f20": rope_sin,
                    "attention_scale_f20": int(mp.floor(2**20/mp.sqrt(128))),
                    "epsilon_f24": 16, "dummy_key_physical_index": 0}
    data = produce(80)
    if data != produce(120):
        raise RuntimeError("CONSTANT_FLOOR_PRECISION_DISAGREEMENT")
    path = root / "constants_v1.json"
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    (root / "manifest.json").write_text(json.dumps({"format_version": "constant-manifest-v1",
        "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "generation": "mpmath floor at 80 and 120 decimal digits; identical integer tables",
        "precision_agreement": True, "formal_interval_certificate": False}, indent=2) + "\n")


if __name__ == "__main__":
    materialize()
