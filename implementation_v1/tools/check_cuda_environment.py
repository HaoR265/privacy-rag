"""Small RTX2060 availability check; never loads a model or falls back to CPU."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def inspect_cuda():
    report = {
        "scope": "Device open, CUDA initialization and tiny GPU arithmetic only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_loaded": False,
        "CPU_model_fallback_permitted": False,
        "full_R2_executed": False,
        "gpu_ready": False,
    }
    try:
        status = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        report["nvidia_smi"] = {"returncode": status.returncode, "stdout": status.stdout.strip(), "stderr": status.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        report["nvidia_smi"] = {"error": str(error)}
    nodes = {}
    for path in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            os.close(fd)
            nodes[path] = {"open": True}
        except OSError as error:
            nodes[path] = {"open": False, "errno": error.errno, "error": str(error)}
    report["device_nodes"] = nodes
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        result = lib.cuInit(0)
        label = ctypes.c_char_p()
        lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.cuGetErrorName(result, ctypes.byref(label))
        report["cuInit"] = {"code": result, "name": label.value.decode() if label.value else "unknown"}
        if result != 0:
            return report
        # Importing PyTorch and its tiny kernel is conditional on driver success.
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        report["torch"] = {"version": torch.__version__, "cuda_version": torch.version.cuda, "device_names": names}
        matches = [i for i, name in enumerate(names) if "RTX 2060" in name]
        if len(matches) != 1:
            report["error"] = "Exactly one visible RTX2060 is required; no other device is selected silently."
            return report
        device = torch.device("cuda", matches[0])
        left = torch.tensor([[1, -2, 3], [4, 5, -6]], dtype=torch.float64, device=device)
        right = torch.tensor([[7, 8], [-9, 10], [11, -12]], dtype=torch.float64, device=device)
        actual = left @ right
        expected = torch.tensor([[58, -48], [-83, 154]], dtype=torch.float64, device=device)
        integer = torch.tensor([-65537, -1, 0, 65537], dtype=torch.int64, device=device)
        high = torch.div(integer, 65536, rounding_mode="floor")
        low = integer - high * 65536
        torch.cuda.synchronize(device)
        correct = bool(torch.equal(actual, expected) and torch.equal(high * 65536 + low, integer))
        report["tiny_arithmetic"] = {"device": str(device), "fp64_matmul_and_int64_recomposition_correct": correct, "allocated_bytes": torch.cuda.memory_allocated(device)}
        report["gpu_ready"] = correct
    except (OSError, RuntimeError, AssertionError, ImportError) as error:
        report["error"] = str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "records/cuda_environment.json")
    args = parser.parse_args()
    report = inspect_cuda()
    raw = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(raw)
    print(raw, end="")
    return 0 if report["gpu_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
