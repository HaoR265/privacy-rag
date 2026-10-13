"""One public R0 1024-slot generation prefill observation; no decode or R1/R2."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from private_rag.model_input import InputPreparer
from references.pipeline import assemble_plaintext_prompt
from references.r0_model import FloatQwen
from model_tools.r0_activation_observer import ActivationObserver
from references.resource_policy import require_model_device


def main():
    require_model_device('R0')
    torch.set_num_threads(4)
    source=json.loads(Path('records/models_r0_full_rag_run.json').read_text())['public_input']
    prep=InputPreparer(Path('artifacts/models'),Path('artifacts/template/template_v1.json'))
    q=prep.question(source['question'])
    chunks=[c for i,d in enumerate(source['documents']) for c in prep.document_chunks(d,'public-reference-'+str(i))]
    started=time.monotonic()
    embedding=FloatQwen(Path('artifacts/models/embedding/checkpoint'))
    vectors=[embedding.embedding(c.embedding.tokens,c.embedding.validity).numpy() for c in chunks]
    qv=embedding.embedding(q.embedding.tokens,q.embedding.validity).numpy()
    scores=[float(np.dot(qv,z)) for z in vectors]
    ranked=sorted(range(len(chunks)),key=lambda i:(-scores[i],i))
    prompt,validity=assemble_plaintext_prompt(prep.template,q.body,[chunks[i].body for i in ranked[:2]])
    preparation_seconds=time.monotonic()-started
    del embedding
    model=FloatQwen(Path('artifacts/models/generation/checkpoint'))
    prefill_start=time.monotonic()
    with ActivationObserver(validity) as observer:
        output=model.forward(prompt,validity)
    elapsed=time.monotonic()-prefill_start
    if observer.layers!=28:raise RuntimeError('R0_PREFILL_OBSERVATION_INCOMPLETE')
    record={'scope':'Independent public R0 generation prefill only with actual weights and observer hooks; not complete generation',
            'public_input':source,'retrieval_order':ranked[:2],'retrieval_scores':scores,
            'prompt_shape':len(prompt),'prompt_valid_count':sum(validity),
            'prompt_sha256':hashlib.sha256(json.dumps([prompt,validity],separators=(',',':')).encode()).hexdigest(),
            'actual_layers':observer.layers,'prefill_calls':1,'decode_calls':0,
            'output_hidden_shape':list(output.shape),'all_hidden_finite':bool(torch.isfinite(output).all()),
            'preparation_seconds':preparation_seconds,'prefill_with_observation_seconds':elapsed,
            'device':str(model.device),'torch_threads':torch.get_num_threads(),
            'quantization_for_statistics':'floor(float32 observed value * 4096); not actual R1 propagated states',
            'R1_executed':False,'R2_executed':False,'statistics':observer.statistics}
    target=Path('records/models_r0_generation_prefill_observation.json')
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='statistics'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
