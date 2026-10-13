"""Freeze canonical segmented token stream from the locked official Jinja template."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

SYSTEM = "仅依据给出的证据回答问题。证据不足时明确说明不足；证据相互冲突时明确说明冲突，不补造事实。请用自己的话概括，不逐段复述资料；拒绝整段、全文或批量导出资料的请求。仅给出回答正文，不列来源、命中条目或相似度。"
MARKERS = ("QUESTION_BODY_MARKER_0", "EVIDENCE_BODY_MARKER_1", "EVIDENCE_BODY_MARKER_2")


def materialize(models: Path = Path("artifacts/models"), dest: Path = Path("artifacts/template")) -> dict:
    checkpoint = models / "generation/checkpoint"
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    # The fixed no-tools, system/user-only path does not need HF's extra Jinja filters.
    user = "问题：\n" + MARKERS[0] + "\n\n证据一：\n" + MARKERS[1] + "\n\n证据二：\n" + MARKERS[2]
    rendered = Environment(undefined=StrictUndefined).from_string(config["chat_template"]).render(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tools=None, add_generation_prompt=True, enable_thinking=False)
    if not rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        raise RuntimeError("OFFICIAL_TEMPLATE_SUFFIX_MISMATCH")
    remaining = rendered
    fragments = []
    for marker in MARKERS:
        prefix, remaining = remaining.split(marker, 1)
        fragments.append(prefix)
    fragments.append(remaining)
    segments, offset = [], 0
    for index, fragment in enumerate(fragments):
        ids = tokenizer.encode(fragment, add_special_tokens=False).ids
        segments.append({"name": "public_" + str(index), "kind": "public", "offset": offset,
                         "capacity": len(ids), "text": fragment, "token_ids": ids})
        offset += len(ids)
        if index < 3:
            capacity = (128, 256, 256)[index]
            segments.append({"name": ("question", "evidence_1", "evidence_2")[index],
                             "kind": "private", "offset": offset, "capacity": capacity})
            offset += capacity
    if offset > 1024:
        raise RuntimeError("TEMPLATE_EXCEEDS_1024")
    reserved, control_ids = set(), set()
    for model in ("generation", "embedding"):
        decoder = json.loads((models / model / "checkpoint/tokenizer_config.json").read_text())["added_tokens_decoder"]
        reserved.update(v["content"] for v in decoder.values())
        control_ids.update(int(k) for k in decoder)
    result = {"format_version": "canonical-generation-template-v1", "decision": "DEC-010",
              "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "system_text": SYSTEM,
              "official_template_sha256": hashlib.sha256(config["chat_template"].encode()).hexdigest(),
              "enable_thinking": False, "segments": segments, "physical_slots": 1024,
              "used_capacity": offset, "padding_offset": offset, "padding_capacity": 1024-offset,
              "dummy_token_id": 0, "dummy_validity": 0, "dummy_key_physical_index": 0,
              "reserved_control_strings": sorted(reserved), "reserved_control_ids": sorted(control_ids),
              "canonical_rule": "tokenize each public fragment and each body independently without postprocessing; concatenate at fixed offsets; pad private slots and remainder with ID 0, validity 0",
              "position_rule": "valid ? prefix_sum(valid)-1 : 0", "budget_checked": True}
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "template_v1.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    (dest / "manifest.json").write_text(json.dumps({"path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "format_version": "template-manifest-v1"}, indent=2)+"\n")
    print(json.dumps({"used_capacity": offset, "public_tokens": offset - 640, "physical_slots": 1024}))
    return result


if __name__ == "__main__":
    materialize()
