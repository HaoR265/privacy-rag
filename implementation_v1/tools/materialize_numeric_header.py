"""Bind the compiled secure graph to the adopted, hashed numeric profile."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "artifacts/constants/numeric_profile_v1_qk.json"
raw = path.read_bytes()
profile = json.loads(raw)
lines = ["#pragma once", "#include <cstdint>", "#include <cstddef>", "#include <array>", "#include <stdexcept>", "namespace prag::model::numeric_profile {",
         f'inline constexpr char version[]="{profile["profile_id"]}";',
         f'inline constexpr char sha256[]="{hashlib.sha256(raw).hexdigest()}";',
         "struct QKBounds {std::uint64_t q_norm,k_norm,q_rope,k_rope;};",
         "struct DownBound {const char* sha256;std::uint64_t row_l1;};",
         "struct LinearBound {const char* tensor;const char* sha256;std::size_t rows,cols;std::uint64_t row_l1,input_abs_f12;};"]
joint = profile["swiglu"]["joint_and_down_proj_input_abs_f12"]
assert profile["profile_id"] == "v1-f12-coordinated-20261013"
assert profile["main_abs_f12"] == 2048 * 4096
assert profile["weight_abs_f12"] == 256 * 4096
assert joint == 2048**2 * 4096
lines.extend(["inline constexpr std::uint64_t main_real=2048;",
              f'inline constexpr std::uint64_t main_abs_f12={profile["main_abs_f12"]};',
              f'inline constexpr std::uint64_t weight_abs_f12={profile["weight_abs_f12"]};'])
lines.append(f"inline constexpr std::uint64_t swiglu_joint_abs_f12={joint};")
for field, name in (("sum_square_f24_limit_inclusive", "rms_sum_limit"),
                    ("coordinate_abs_f12_derived_cap", "rms_coordinate_cap")):
    lines.append(f"inline constexpr std::uint64_t {name}(std::size_t d) {{")
    for d in (128, 1024, 2048):
        value = profile["rms_domains"][str(d)][field]
        assert 0 < value < 2**56
        lines.append(f"  if(d=={d})return {value}ULL;")
    lines.extend(['  throw std::invalid_argument("BAD_SHAPE");', "}"])
lines.append(f'inline constexpr std::uint64_t l2_sum_limit={profile["l2_domain"]["sum_square_f24_limit_inclusive"]}ULL;')
lines.append(f'inline constexpr std::uint64_t l2_coordinate_cap={profile["l2_domain"]["coordinate_abs_f12"]}ULL;')
for model in profile["models"]:
    norm, rope = model["qk_norm_abs_candidate_f12"], model["rope_abs_candidate_f12"]
    values = [norm["q"], norm["k"], rope["q"], rope["k"]]
    assert all(type(v) is int and 0 < v < 2**56 for v in values)
    lines.append(f'inline constexpr QKBounds {model["model"]}' + "{" + ",".join(map(str, values)) + "};")
    bounds = sorted(model["down_proj_bounds"], key=lambda x: int(x["tensor"].split(".")[2]))
    assert len(bounds) == 28
    lines.append(f'inline constexpr std::array<DownBound,28> {model["model"]}_down' + "{{")
    for layer, bound in enumerate(bounds):
        assert bound["tensor"] == f"model.layers.{layer}.mlp.down_proj.weight"
        l1 = bound["maximum_row_l1_f12_integer"]
        assert l1 * joint == bound["full_dot_and_any_partial_sum_abs_f24_bound"] < 2**56
        digest = bound["sha256"]
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        lines.append(f'  {{"{digest}",{l1}}},')
    lines.append("}};")
    matrices = sorted(model["matrix_bounds"], key=lambda x: x["tensor"])
    assert len(matrices) == 197 and len({b["tensor"] for b in matrices}) == 197
    lines.append(f'inline constexpr std::array<LinearBound,{len(matrices)}> {model["model"]}_linear' + "{{")
    for bound in matrices:
        l1, input_cap = bound["maximum_row_l1_f12_integer"], bound["input_abs_f12"]
        assert 0 < l1 * input_cap == bound["full_dot_and_any_partial_sum_abs_f24_bound"] < 2**56
        assert input_cap == (joint if bound["tensor"].endswith(".down_proj.weight") else profile["main_abs_f12"])
        digest = bound["sha256"]
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        rows, cols = bound["shape"]
        lines.append(f'  {{"{bound["tensor"]}","{digest}",{rows},{cols},{l1},{input_cap}}},')
    lines.append("}};")
lines.append("}")
(ROOT / "compute/numeric_profile.hpp").write_text("\n".join(lines) + "\n")
print(json.dumps({"numeric_version": profile["profile_id"], "profile_sha256": hashlib.sha256(raw).hexdigest()}))
