"""Independent CUDA-only oracle for the registered public M3 development case.

No oracle output is supplied as an input to an R2 participant. The integration
controller uses it only after a stage has closed its one-use crypto session.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from private_rag.model_input import InputPreparer
from references.r1_cuda_model import CUDAIntegerQwen

ROOT = Path(__file__).resolve().parents[2]
MASK = (1 << 59) - 1
DOCUMENTS = ["图书馆周一至周五九点开放，十七点关闭。", "图书馆周末十点开放，十六点关闭。"]
QUESTION = "图书馆周六几点开放？"


@torch.inference_mode()
def embeddings(output):
    os.chdir(ROOT)
    preparer = InputPreparer(ROOT / 'artifacts/models', ROOT / 'artifacts/template/template_v1.json')
    chunks = [preparer.document_chunks(text, 'public-development')[0] for text in DOCUMENTS]
    question = preparer.question(QUESTION)
    model = CUDAIntegerQwen(ROOT / 'artifacts/models/embedding/quantized_f12')
    result = {'documents': DOCUMENTS, 'question': QUESTION, 'rows': [257, 257, 160],
              'device': torch.cuda.get_device_name(model.device), 'vectors': [], 'eligible': [], 'failures': []}
    try:
        for i, slots in enumerate([chunk.embedding for chunk in chunks] + [question.embedding]):
            vector, eligible = model.embedding(slots.tokens, slots.validity)
            # Device-to-host serialization of a public independent oracle only.
            result['vectors'].append(vector.tolist())
            result['eligible'].append(eligible)
            result['failures'].append(model.num.failed)
            print(json.dumps({'event': 'M3_CUDA_reference_embedding', 'sample': i,
                              'slots': len(slots.tokens), 'numeric_failure': model.num.failed}), flush=True)
        result['peak_allocated_device_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_device_bytes'] = torch.cuda.max_memory_reserved()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    finally:
        model.close()


@torch.inference_mode()
def retrieval(reference, order, output):
    from references.resource_policy import require_model_device
    device = require_model_device('R1')
    data = json.loads(reference.read_text())
    permutation = json.loads(order.read_text())
    if sorted(permutation) != [0, 1]:
        raise ValueError('ORACLE_ORDER')
    prep = InputPreparer(ROOT / 'artifacts/models', ROOT / 'artifacts/template/template_v1.json')
    chunks = [prep.document_chunks(text, 'public-development')[0] for text in DOCUMENTS]
    q = torch.tensor(data['vectors'][2], dtype=torch.int64, device=device)
    z = torch.tensor([data['vectors'][i] for i in permutation], dtype=torch.int64, device=device)
    valid = torch.tensor([data['eligible'][i] and data['eligible'][2] for i in permutation], device=device).bool()
    fine = (z * q).sum(1)
    coarse = ((z >> 12) * (q >> 12)).sum(1) << 24
    bound = 1024 * (2 * (1 << 21) * 4095 + 4095 ** 2)
    lower, upper = coarse - bound, coarse + bound
    # Fixed N=2: strict comparison keeps public earlier-index tie behavior.
    first = (valid[1] & ~valid[0]) | ((valid[1] == valid[0]) & (fine[1] > fine[0]))
    first_lower = (valid[1] & ~valid[0]) | ((valid[1] == valid[0]) & (lower[1] > lower[0]))
    pivot = torch.where(first_lower, 0, 1)
    keep = valid & (~valid[pivot] | (upper >= lower[pivot]))
    token_rows = torch.tensor([chunks[i].body.tokens for i in permutation], dtype=torch.int64, device=device)
    bit_rows = torch.tensor([chunks[i].body.validity for i in permutation], dtype=torch.int64, device=device)
    words = []
    for selected in (first.to(torch.int64), 1 - first.to(torch.int64)):
        words.extend([torch.where(valid[selected], token_rows[selected], 0),
                      torch.where(valid[selected], bit_rows[selected], 0),
                      valid[selected].to(torch.int64).reshape(1)])
    words.append(keep.to(torch.int64))
    raw = (torch.cat(words) & MASK).cpu().numpy().astype('<u8').tobytes()
    output.write_bytes(raw)
    print(json.dumps({'event': 'M3_CUDA_retrieval_reference', 'words': len(raw) // 8,
                      'sha256': hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['embeddings', 'retrieval'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--order', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.mode == 'embeddings':
        embeddings(args.output)
    else:
        retrieval(args.reference, args.order, args.output)
