"""Complete fixed-budget public CUDA R1 RAG developer invocation, never R2."""
import hashlib,json,time,os
from pathlib import Path
from references.pipeline import run_reference
from references.resource_policy import require_model_device
from model_tools.reference_provenance import identify,unchanged


def run():
    device=require_model_device('R1')
    identity=identify('model_tools/run_full_r1_rag.py')
    import torch
    source=json.loads(Path('records/models_r0_full_rag_run.json').read_text())
    public=source.get('public_input',source)
    if 'question' not in public:
        public={'question':'方案采用哪种加密交付方式？','documents':['资料使用 HPKE Base 模式交付给获得授权的用户。','交付方案使用 X25519、HKDF-SHA256 和 AES-128-GCM。']}
    destination=Path('records/models_coordinated_full_r1_cuda_rag_run.json')
    events_path=destination.with_suffix('.progress.jsonl')
    start=time.monotonic();events=[];first_failure=None
    def progress(event):
        nonlocal first_failure
        event={'elapsed_seconds':time.monotonic()-start,**event};events.append(event)
        if event['event']=='first_numeric_failure' and first_failure is None:first_failure=dict(event)
        with events_path.open('a') as stream:stream.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    events_path.write_text('');torch.cuda.reset_peak_memory_stats(device)
    progress({'event':'CUDA_R1_RAG_start','pid':os.getpid(),'device':str(device),
              'device_name':torch.cuda.get_device_name(device)})
    destination.with_suffix('.start.json').write_text(json.dumps({'pid':os.getpid(),'public_input':public,'start_provenance':identity},ensure_ascii=False,indent=2)+'\n')
    try:
        result=run_reference('R1',public['question'],public['documents'],progress=progress)
    except Exception as error:
        failure={'scope':'Incomplete CUDA R1 attempt, not a full result','pid':os.getpid(),'elapsed_seconds':time.monotonic()-start,
                 'exception_type':type(error).__name__,'exception':str(error),'last_event':events[-1] if events else None,
                 'start_provenance':identity,'start_to_finish_identity_unchanged':unchanged(identity,'model_tools/run_full_r1_rag.py'),
                 'fixed_generation_executed':False,'model_CPU_fallback':False}
        destination.with_suffix('.failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n')
        raise
    torch.cuda.synchronize(device)
    sources=['references/r1_model.py','references/r1_cuda_model.py','references/r1_cuda_numeric.py','references/resource_policy.py','references/pipeline.py']
    scope=('Independent public CUDA R1 complete fixed-budget RAG developer run, not formal acceptance or R2'
           if result.get('fixed_generation_executed') else
           'Independent public CUDA R1 incomplete preparation result; fixed generation did not execute, not formal acceptance or R2')
    result.update(scope=scope,
                  public_input=public,elapsed_seconds=time.monotonic()-start,first_failure=first_failure,
                  start_provenance=identity,start_to_finish_identity_unchanged=unchanged(identity,'model_tools/run_full_r1_rag.py'),
                  fixed_generation_prefill_calls=1 if result.get('fixed_generation_executed') else 0,
                  fixed_generation_decode_calls=255 if result.get('fixed_generation_executed') else 0,
                  answer_slots=len(result.get('tokens',[])),valid_answer_tokens=sum(result.get('validity',[])),
                  R2_executed=False,pid=os.getpid(),device=str(device),device_name=torch.cuda.get_device_name(device),
                  model_CPU_fallback=False,peak_allocated_device_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_device_bytes=torch.cuda.max_memory_reserved(device),
                  numeric_profile_sha256=hashlib.sha256(Path('artifacts/constants/numeric_profile_v1_qk.json').read_bytes()).hexdigest(),
                  source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['tokens','validity','public_input','source_hashes','start_provenance']},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
