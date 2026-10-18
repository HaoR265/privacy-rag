"""Independent R0/R1 public-data RAG reference; never a private production route."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from private_rag.model_input import InputPreparer,public_prompt_layout
from references.resource_policy import require_model_device


def assemble_plaintext_prompt(template,question,evidence):
    tokens,validity=map(list,public_prompt_layout(template))
    slots={p["name"]:p for p in template["segments"] if p["kind"]=="private"}
    bodies={"question":question,"evidence_1":evidence[0],"evidence_2":evidence[1]}
    for name,body in bodies.items():
        if body is None:continue
        offset,capacity=slots[name]["offset"],slots[name]["capacity"]
        if len(body.tokens)!=capacity:raise ValueError("REFERENCE_BODY_SHAPE")
        tokens[offset:offset+capacity]=body.tokens
        validity[offset:offset+capacity]=body.validity
    return tokens,validity


def finalize_reference_answer(result, any_evidence, query_failed=False, dummy=0):
    """Apply final-release policy AFTER the complete independent generation run."""
    out=dict(result)
    if len(out["tokens"])!=256 or len(out["validity"])!=256:
        raise ValueError("REFERENCE_ANSWER_SHAPE")
    failed=bool(query_failed or out.get("numeric_failure",False))
    permitted=bool(any_evidence and not failed)
    if not permitted:
        out["tokens"]=[dummy]*256
        out["validity"]=[0]*256
    out["numeric_failure"]=failed
    out["length_limit"]=bool(out.get("length_limit",False) and permitted)
    out["insufficient_evidence"]=not bool(any_evidence)
    out["status"]=2 if failed else 1 if not any_evidence else 3 if out["length_limit"] else 0
    return out


def run_reference(profile,question,documents,models=Path("artifacts/models"),template=Path("artifacts/template/template_v1.json"),progress=None):
    """Arguments must be public/synthetic or separately authorized for full observation."""
    require_model_device(profile)
    prep=InputPreparer(models,template)
    q=prep.question(question)
    chunks=[chunk for i,doc in enumerate(documents) for chunk in prep.document_chunks(doc,"public-reference-"+str(i))]
    if not chunks:return {"profile":profile,"status":"no_authorized_material","fixed_generation_executed":False}
    embedding=generator=vector=None
    try:
        if profile=="R0":
            from references.r0_model import FloatQwen
            embedding=FloatQwen(models/"embedding/checkpoint")
            def vector(t,m):
                z=embedding.embedding(t,m).numpy()
                return z,int(np.any(z))
        elif profile=="R1":
            from references.r1_model import IntegerQwen
            embedding=IntegerQwen(models/"embedding/quantized_f12")
            embedding.progress=progress
            vector=embedding.embedding
        else:raise ValueError("ONLY_INDEPENDENT_R0_OR_R1")
        vectors=[]
        for index,chunk in enumerate(chunks):
            if progress is not None:progress({'event':'document_embedding_start','index':index,'physical_slots':len(chunk.embedding.tokens)})
            vectors.append(vector(chunk.embedding.tokens,chunk.embedding.validity))
            if progress is not None:progress({'event':'document_embedding_complete','index':index,'eligible':int(vectors[-1][1])})
            if profile=="R1" and embedding.num.failed:
                return {"profile":profile,"status":"preparation_numeric_failure",
                        "numeric_profile_version":embedding.profile_id,
                        "numeric_failure":True,"fixed_generation_executed":False,
                        "reason":"Failed preparation cannot be published or used as a document record"}
        if progress is not None:progress({'event':'query_embedding_start','physical_slots':len(q.embedding.tokens)})
        qv,qe=vector(q.embedding.tokens,q.embedding.validity)
        if progress is not None:progress({'event':'query_embedding_complete','eligible':int(qe)})
        query_failed=profile=="R1" and embedding.num.failed
        if profile=="R1":
            from references.r1_retrieval import retrieve
            retrieval=retrieve(qv,[z for z,e in vectors],qe,[e for z,e in vectors])
            selected=[chunks[i].body for i,v in zip(retrieval.top_indices,retrieval.top_validity) if v]
        else:
            scores=[float(np.dot(qv,z)) for z,e in vectors]
            ranked=sorted(range(len(chunks)),key=lambda i:(not bool(qe and vectors[i][1]),-scores[i],i))
            selected=[chunks[i].body for i in ranked[:2] if qe and vectors[i][1]]
        selected += [None]*(2-len(selected))
        prompt,validity=assemble_plaintext_prompt(prep.template,q.body,selected)
        # Release the Embedding checkpoint before allocating the generation model.
        # This is essential for the fixed two-model path on the 6 GB RTX 2060.
        if hasattr(embedding,'close'):embedding.close()
        vector=None;embedding=None
        import torch
        torch.cuda.empty_cache()
        if profile=="R0": generator=FloatQwen(models/"generation/checkpoint")
        else: generator=IntegerQwen(models/"generation/quantized_f12")
        if profile=='R1':generator.progress=progress
        generated=generator.generate(prompt,validity)
        result=finalize_reference_answer(generated,selected[0] is not None,query_failed,prep.dummy)
        result["profile"]=profile
        if profile=="R1":result["numeric_profile_version"]=generator.profile_id
        result["fixed_generation_executed"]=True
        return result
    finally:
        # Both preparation failures and CUDA/model exceptions release cached
        # checkpoints and break the development callback's model cycle.
        for model in (embedding,generator):
            if model is not None and hasattr(model,'close'):model.close()
        model=None;vector=None;embedding=None;generator=None
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()



if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["R0","R1"],required=True)
    parser.add_argument("--public-input-json",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--acknowledge-plaintext-reference",action="store_true",required=True)
    args=parser.parse_args()
    source=json.loads(args.public_input_json.read_text())
    start=time.monotonic()
    result=run_reference(args.profile,source["question"],source["documents"])
    result["elapsed_seconds"]=time.monotonic()-start
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
