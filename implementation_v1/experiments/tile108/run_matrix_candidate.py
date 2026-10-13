"""Public synthetic GPU R1 vs real production-mTLS matrix protocol diagnostics.

No model is executed. Expected output reconstruction belongs only to the probe.
Full-size shapes are isolated linear protocols, not an R2 Decode token.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid

import numpy as np
import torch
from private_rag.pki import create_local_pki
from references.resource_policy import require_model_device
from references.r1_cuda_numeric import CUDAIntegerNumerics

ROOT = Path(__file__).resolve().parents[2]
MASK = (1 << 59) - 1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(batch, inner, outputs, mode, path):
    device = require_model_device('R1')
    n = CUDAIntegerNumerics({'exp_f24': [], 'rope_cos_f20': [], 'rope_sin_f20': []}, device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    b = torch.arange(batch, device=device)[:, None]
    j = torch.arange(inner, device=device)[None, :]
    x = ((j * 13 + b * 19 + 7) % 255 - 127).expand(batch, inner).clone()
    if mode == 1:
        x.zero_(); x[:, 0] = (1 << 55) - 1
        if inner > 16:
            x[:, 16] = -3
    digest = hashlib.sha256()
    with path.open('wb') as stream:
        # Preserve row-major file order without retaining the full weight map.
        row_results = [[] for _ in range(batch)]
        for start_col in range(0, outputs, 4096):
            o = torch.arange(start_col, min(start_col + 4096, outputs), device=device)[:, None]
            w = (j * 7 + o * 11 + 3) % 63 - 31
            if mode == 1:
                w.zero_(); w[:, 0] = torch.where(o[:, 0] % 2 == 1, -1, 1)
                if inner > 16:
                    w[:, 16] = 2
            w = torch.where(o == 1, 0, w)
            # Mode1 has only columns 0 and16 active: every arbitrary partial
            # sum is <=2^55-1+6, and max weight row L1 is3. The split path
            # remains exact above binary64's single-integer precision.
            bound = (1 << 55) + 5 if mode == 1 else inner * 127 * 31
            if mode == 1:
                # The generic max(x)*row_L1 guard is intentionally too loose
                # for this sparse boundary. Evaluate its two public nonzero
                # columns separately on CUDA, then add exact int64 results.
                y = n.exact_matmul(x[:, :1], w[:, :1].T, (1 << 55), right_l1_bound=1)
                if inner > 16:
                    y = y + n.exact_matmul(x[:, 16:17], w[:, 16:17].T, 6, right_l1_bound=2)
            else:
                y = n.exact_matmul(x, w.T, bound, right_l1_bound=inner * 31)
            encoded = (y & MASK).cpu().numpy().astype('<u8', copy=False)
            for row in range(batch):
                row_results[row].append(encoded[row].tobytes())
        for pieces in row_results:
            for raw in pieces:
                stream.write(raw); digest.update(raw)
    torch.cuda.synchronize(device)
    result = {'device': str(device), 'device_name': torch.cuda.get_device_name(device),
              'CPU_reference_matmul': False, 'elapsed_seconds': time.monotonic() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'raw_partial_sum_abs_bound': bound, 'raw_output_fractional_bits': 24,
              'exact_GEMM_counts': n.gemm_counts, 'output_sha256': digest.hexdigest(),
              'synthetic_input_and_weights': True, 'model_checkpoint_used': False}
    del x, w, y, encoded, n
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['small', 'shapes'], default='small')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--build-dir', default='experiments/tile108/build')
    args = parser.parse_args()
    if not args.prefix.replace('_', '').replace('-', '').isalnum():
        parser.error('simple new record prefix required')
    cases = ([(3, 17, 7, 0), (3, 17, 7, 1), (16, 17, 33, 0)] if args.suite == 'small'
             else [(1, 2048, 2048, 0), (1, 2048, 6144, 0),
                   (1, 6144, 2048, 0), (1, 2048, 151936, 0)])
    build = (ROOT / args.build_dir).resolve(); build.relative_to(ROOT)
    binary = build / 'prag_production_matrix_probe'
    sources = [ROOT / p for p in ['experiments/tile108/bfv_candidate.cpp',
               'experiments/tile108/bfv_matrix.inc','experiments/tile108/tile_decoder.inc',
               'experiments/tile108/matrix_probe.cpp','experiments/tile108/run_matrix_candidate.py',
               'native/src/channel.cpp','native/include/prag/channel.hpp',
               'native/include/prag/secure_ops.hpp','native/include/prag/bfv.hpp',
               'references/r1_cuda_numeric.py','native/build-matrix107/libprag_crypto.a']]
    sources += [binary]
    identities = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    for batch, inner, outputs, mode in cases:
        name = f'{args.prefix}_{batch}x{inner}x{outputs}_m{mode}'
        destination = ROOT / 'records' / (name + '.json')
        if destination.exists():
            raise RuntimeError('refusing to overwrite ' + str(destination))
        print(json.dumps({'starting_case': name, 'reference_device': 'RTX2060',
                          'protocol_backend': 'SEAL/SCI CPU isolated matrix, not model'}), flush=True)
        with tempfile.TemporaryDirectory(prefix='prag-matrix-') as temp:
            temporary = Path(temp); pki = temporary / 'pki'; create_local_pki(pki)
            expected = temporary / 'expected.u64'
            ref = reference(batch, inner, outputs, mode, expected)
            prefix = ROOT / 'records' / (name + '_' + uuid.uuid4().hex)
            process = subprocess.Popen(['nice', '-n', '10', str(binary), str(prefix), str(pki),
                str(batch), str(inner), str(outputs), str(mode), str(expected)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=7200)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.communicate()
                raise
            if process.returncode:
                raise RuntimeError(f'{name} exit {process.returncode}: {stderr}')
            data = json.loads(stdout)
        if identities != {str(p.relative_to(ROOT)): sha(p) for p in sources}:
            raise RuntimeError('source changed during protocol diagnostic')
        ledgers = []
        for party in ['C', 'S']:
            path = Path(str(prefix) + f'.{party}.ledger')
            lines = path.read_text().splitlines()
            if lines[-1] != 'terminal=closed all_reserved=burned':
                raise RuntimeError('unclosed material ledger')
            counts = {}
            for line in lines:
                fields = line.split()
                if len(fields) == 5 and fields[1] == 'reserved':
                    counts[fields[2]] = counts.get(fields[2], 0) + 1
            ledgers.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'counts': counts})
        if ledgers[0]['counts'] != ledgers[1]['counts']:
            raise RuntimeError('different reservation schedule')
        counts = ledgers[0]['counts']
        assert counts['bfv_matrix_tile_mask_Enc0'] == data['returned_ciphertext_groups']
        assert counts['bfv_matrix_tile_GC'] == data['returned_ciphertext_groups']
        assert counts.get('bfv_matrix_target_GC', 0) == 0
        data.update(tile_GC_instances=counts['bfv_matrix_tile_GC'], GC_AND_gates=122*batch*outputs,
                    GC_client_OT_choices=2*batch*outputs, GC_OT_extension_batches=counts['bfv_matrix_tile_GC'],
                    ciphertext_rotations=0, ciphertext_modulus_switches_after_input=0, GC_division_rounds=0,
                    candidate_protocol=True, production_activated=False)
        data.update(scope='independent candidate matrix-level public synthetic R2 primitive vs separate GPU R1, not formal acceptance or a model',
                    R1_reference=ref, source_and_binary_sha256=identities, ledgers=ledgers,
                    exit_code=process.returncode, stderr=stderr, temporary_PKI_and_reference_removed=not temporary.exists(),
                    source_unchanged_during_run=True, child_waited=True)
        destination.write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps({'completed_case': name, **{k: data[k] for k in ['result', 'elapsed_seconds',
              'application_bytes', 'returned_ciphertext_groups', 'output_coordinates']}}), flush=True)


if __name__ == '__main__':
    main()
