"""Build public synthetic R1 expected values for an isolated native dev probe."""
import hashlib
import json
from pathlib import Path

import numpy as np

from references.r1_numeric import IntegerNumerics


def main():
    constants = Path("artifacts/constants/constants_v1.json")
    n = IntegerNumerics.load(constants)
    cases = {}
    for label, value in (("exp_zero", 0), ("exp_32", 32 << 12),
                         ("exp_tail", (32 << 12) + 1)):
        n.failed = False
        cases[label] = [int(n.exp_negative(value)), int(n.failed)]
    for label, value in (("silu_positive", 4096), ("silu_negative", -4096)):
        n.failed = False
        cases[label] = [int(n.silu(value)), int(n.failed)]
    for label, valid, allowed in (("softmax_valid", True, [True, True]),
                                  ("softmax_invalid", False, [False, True])):
        n.failed = False
        result = n.softmax(np.array([[4096, -4096]]), np.array([allowed]),
                           np.array([valid]))
        cases[label] = result[0].tolist() + [int(n.failed)]
    fixture = Path("records/models_numeric_operator_expected.txt")
    fixture.write_text("PRDEVNUM1\n" + "".join(
        f"{label} {len(values)} " + " ".join(map(str, values)) + "\n"
        for label, values in cases.items()))
    record = {
        "scope": "public synthetic developer values only; not acceptance or private inputs",
        "expected_source": "references/r1_numeric.py",
        "expected_source_sha256": hashlib.sha256(Path("references/r1_numeric.py").read_bytes()).hexdigest(),
        "constants_json_sha256": hashlib.sha256(constants.read_bytes()).hexdigest(),
        "constants_binary_sha256": hashlib.sha256(Path("artifacts/constants/constants_v1.bin").read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "exp_input_f12": [0, 32 << 12, (32 << 12) + 1],
        "silu_input_f12": [4096, -4096],
        "softmax_scores_f12": [4096, -4096],
        "softmax_allowed": {"valid_query": [True, True], "invalid_query": [False, True]},
        "cases": cases,
        "static_work_estimate": {
            "exp_calls": 9, "lut_tables_per_exp": 3, "entries_per_table": 256,
            "boolean_AND_per_equal59": 295, "cross_bit_OT_per_AND": 2,
            "lut_only_cross_bit_OT": 9 * 3 * 256 * 295 * 2,
            "note": "Each current cross-bit OT reserves a fresh ledger row and fsyncs; this estimate excludes other comparisons, floor, reciprocal and conversions. No full model is run."
        }
    }
    Path("records/models_numeric_operator_expected.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(cases))


if __name__ == "__main__":
    main()
