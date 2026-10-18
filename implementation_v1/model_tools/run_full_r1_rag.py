"""Complete fixed-budget public CUDA R1 RAG developer invocation, never R2.

Run from implementation_v1. Every invocation requires an explicit public input
and a new output directory. --prepare-only checks record preparation without
importing Torch or executing any model; a prepared directory cannot be reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

from model_tools.reference_run_record import (
    ROOT, capture_identity, compare_identity, preserve_inputs, read_public_input,
    utc_now, write_new,
)


def run(public_input_json, output_dir, *, prepare_only=False, acknowledge_plaintext_reference=False):
    if not acknowledge_plaintext_reference:
        raise ValueError("PLAINTEXT_REFERENCE_ACKNOWLEDGEMENT_REQUIRED")
    if Path.cwd().resolve() != ROOT:
        raise ValueError("RUN_FROM_IMPLEMENTATION_V1_DIRECTORY")
    public, raw_input, input_source = read_public_input(public_input_json)
    # Do not resolve the final component: mkdir must reject even dangling
    # symlinks and existing empty directories, not follow them to a new target.
    directory = Path(os.path.abspath(output_dir))
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    start = time.monotonic()
    identity = None
    last_event = first_failure = None
    stage = "record_preparation"
    model_call_started = False
    progress_stream = None

    def progress(event):
        nonlocal first_failure, last_event
        event = {**event, "elapsed_seconds": time.monotonic() - start}
        last_event = event
        if event["event"] == "first_numeric_failure" and first_failure is None:
            first_failure = dict(event)
        progress_stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        progress_stream.flush()
        print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        progress_stream = (directory / "progress.jsonl").open("x", encoding="utf-8")
        identity = capture_identity()
        preserve_inputs(directory, raw_input, identity)
        registration = {
            "record_schema": "public-r1-run-v1", "created_at": utc_now(),
            "scope": "Independent public CUDA fixed-point reference; not R2 or formal acceptance",
            "mode": "prepare_only" if prepare_only else "execute",
            "pid": os.getpid(), "working_directory": str(ROOT), "output_directory": str(directory),
            "public_input_source": input_source, "input_snapshot": "public_input.json",
            "start_provenance": identity,
            "weight_manifest_snapshots": ["embedding.weights.json", "generation.weights.json"],
            "device_policy": "RTX2060 CUDA required; no CPU model fallback",
            "generation_budget": {"prefill_slots": 1024, "answer_slots": 256, "decode_calls": 255},
            "plaintext_reference_acknowledged": True,
        }
        write_new(directory / "start.json", registration)
        if prepare_only:
            identity_check = compare_identity(identity)
            if not identity_check["unchanged"]:
                raise RuntimeError("REFERENCE_SOURCE_OR_ARTIFACT_CHANGED")
            prepared = {"status": "prepared_only", "model_executed": False, "R2_executed": False,
                        "CUDA_checked": False, "tensor_files_verified": False,
                        "identity_check": identity_check, "created_at": utc_now()}
            write_new(directory / "prepared.json", prepared)
            print(json.dumps(prepared), flush=True)
            return prepared

        # Import model dependencies only after independent records exist; CUDA
        # failure now leaves a failure record rather than silently no evidence.
        stage = "device_preflight"
        from references.resource_policy import require_model_device
        device = require_model_device("R1")
        import torch
        from references.pipeline import run_reference
        torch.cuda.reset_peak_memory_stats(device)
        progress({"event": "CUDA_R1_RAG_start", "pid": os.getpid(), "device": str(device),
                  "device_name": torch.cuda.get_device_name(device)})
        stage = "reference_execution"
        model_call_started = True
        result = run_reference("R1", public["question"], public["documents"], progress=progress)
        torch.cuda.synchronize(device)
        stage = "result_registration"
        identity_check = compare_identity(identity)
        if not identity_check["unchanged"]:
            raise RuntimeError("REFERENCE_SOURCE_OR_ARTIFACT_CHANGED")
        generation_executed = bool(result.get("fixed_generation_executed"))
        result.update(
            scope=("Independent public CUDA R1 complete fixed-budget RAG developer run, not formal acceptance or R2"
                   if generation_executed else "Independent public CUDA R1 incomplete preparation result; fixed generation did not execute"),
            public_input=public, public_input_source=input_source,
            elapsed_seconds=time.monotonic() - start, first_failure=first_failure,
            start_provenance=identity, identity_check=identity_check,
            start_to_finish_identity_unchanged=True,
            fixed_generation_prefill_calls=1 if generation_executed else 0,
            fixed_generation_decode_calls=255 if generation_executed else 0,
            answer_slots=len(result.get("tokens", [])), valid_answer_tokens=sum(result.get("validity", [])),
            R2_executed=False, semantic_quality_assessed=False, pid=os.getpid(), device=str(device),
            device_name=torch.cuda.get_device_name(device), model_CPU_fallback=False,
            peak_allocated_device_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_device_bytes=torch.cuda.max_memory_reserved(device),
            runtime={"python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda},
            finished_at=utc_now(),
        )
        write_new(directory / "result.json", result)
        # The durable result is authoritative; stdout is only a convenience.
        return result
    except BaseException as error:
        # Includes KeyboardInterrupt and SIGTERM translated by main(). SIGKILL,
        # power loss or an unwritable filesystem cannot produce a final record.
        failure = {
            "scope": "Incomplete public R1 attempt; no complete result asserted",
            "status": "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
            "pid": os.getpid(), "elapsed_seconds": time.monotonic() - start,
            "stage": stage, "exception_type": type(error).__name__, "exception": str(error),
            "last_event": last_event, "public_input_source": input_source,
            "start_provenance": identity,
            "identity_check": compare_identity(identity) if identity is not None else None,
            "model_call_started": model_call_started, "fixed_generation_completion_confirmed": False,
            "model_CPU_fallback": False, "R2_executed": False, "finished_at": utc_now(),
        }
        try:
            write_new(directory / "failure.json", failure)
        except Exception as record_error:
            print("FAILURE_RECORD_WRITE_FAILED: " + str(record_error), file=sys.stderr)
        raise
    finally:
        if progress_stream is not None:
            progress_stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-input-json", type=Path, required=True,
                        help='Explicit UTF-8 object containing only "question" and "documents".')
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; parent must exist. Existing paths are always rejected.")
    parser.add_argument("--acknowledge-plaintext-reference", action="store_true", required=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Record input/provenance only; no CUDA check or model execution.")
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise InterruptedError("REFERENCE_TERMINATED_BY_SIGNAL_" + str(signum))

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args.public_input_json, args.output_dir, prepare_only=args.prepare_only,
            acknowledge_plaintext_reference=args.acknowledge_plaintext_reference)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
