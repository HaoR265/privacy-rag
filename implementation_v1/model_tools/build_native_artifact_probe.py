"""Generate public fixed-position expectations directly from pinned checkpoints.

Developer-only input for prag_dev_session_probe; does not run inference or
open any server shares. This is not a production verification/acceptance policy.
"""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from model_tools.quantize_weights import tensor_headers

FIELDS = ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
          "self_attn.o_proj.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
          "mlp.up_proj.weight", "mlp.down_proj.weight"]


def build():
    root = Path("artifacts/models")
    constants = json.loads(Path("artifacts/constants/constants_v1.json").read_text())
    words = [constants["attention_scale_f20"]]
    words += [x for table in constants["exp_f24"] for x in table]
    words += [constants[name][p][c] & ((1 << 59) - 1)
              for name in ("rope_cos_f20", "rope_sin_f20") for c in range(64) for p in range(1280)]
    digest = hashlib.sha256(struct.pack("<" + "Q" * len(words), *words)).hexdigest()
    lines = []
    for kind in ("embedding", "generation"):
        sources = {}
        for path in sorted((root / kind / "checkpoint").glob("*.safetensors")):
            base, header = tensor_headers(path)
            for name, entry in header.items():
                canonical = name if name.startswith("model.") else "model." + name
                sources[canonical] = (path, base, entry)
        names = [(-2, 0, "model.embed_tokens.weight"), (-1, 0, "model.norm.weight")]
        names += [(layer, field, f"model.layers.{layer}.{name}") for layer in range(28) for field, name in enumerate(FIELDS)]
        for layer, field, name in names:
            path, base, entry = sources[name]
            shape = entry["shape"]
            raw = np.memmap(path, dtype={"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[entry["dtype"]],
                            offset=base + entry["data_offsets"][0], mode="r", shape=tuple(shape))
            rows, cols = (shape[0], shape[1]) if len(shape) == 2 else (shape[0], 1)
            for row, col in ((0, 0), (rows // 2, cols // 2), (rows - 1, cols - 1)):
                source = np.asarray(raw[row, col] if len(shape) == 2 else raw[row])
                value = ((source.astype(np.uint32) << 16).view(np.float32)
                         if entry["dtype"] == "BF16" else source.astype(np.float32))
                expected = int(np.floor(np.float64(value) * 4096))
                lines.append(f"{kind} {layer} {field} {row} {col} {expected}")
            del raw
    output = Path("records/models_native_artifact_expected.txt")
    data = "PRDEVART1\n" + digest + "\n" + str(len(lines)) + "\n" + "\n".join(lines) + "\n"
    output.write_text(data)
    manifest = {"purpose": "isolated developer probe, not formal acceptance specification",
                "expected_source": "exact pinned original safetensors values, floor(x*4096)",
                "sample_locations": "first, middle and last scalar of each of all 310 canonical tensors per model",
                "total_weight_samples": len(lines), "constants_checked_words": len(words),
                "constants_words_sha256": digest, "fixture_sha256": hashlib.sha256(data.encode()).hexdigest(),
                "weight_reconstruction_over_protocol": False}
    Path("records/models_native_artifact_expected.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    build()
