"""Developer run checks on synthetic public input; no formal acceptance criteria."""
from pathlib import Path
import json
import numpy as np
from private_rag.model_input import InputPreparer,InputRejected,normalize_text
from references.r1_numeric import IntegerNumerics,logical_positions,last_valid


def run():
    p=InputPreparer(Path("artifacts/models"),Path("artifacts/template/template_v1.json"))
    q=p.question("隐私计算如何保护问题？")
    source=("隐私计算保护数据。\n"*100).encode()
    chunks=p.document_chunks(source,"public-synthetic-acl")
    assert len(q.body.tokens)==128 and len(q.embedding.tokens)==160
    for c in chunks:
        assert len(c.body.tokens)==256 and len(c.embedding.tokens)==257
        assert c.text==normalize_text(source)[c.character_start:c.character_end]
    rejected=[]
    for raw,code in [("<think>","RESERVED_CONTROL_STRING"),(b"\xff","INVALID_UTF8"),("词 "*1000,"TOKEN_BUDGET_EXCEEDED")]:
        try:p.question(raw)
        except InputRejected as error:
            assert str(error)==code
            rejected.append(code)
        else:raise AssertionError("LOCAL_INPUT_NOT_REJECTED")
    n=IntegerNumerics.load()
    assert n.l2(np.zeros(1024,dtype=np.int64))[1]==0
    assert n.l2(np.full(1024,256*4096,dtype=np.int64),record_valid=False)[1]==0
    assert not n.failed
    z,e=n.l2(np.full(1024,4096,dtype=np.int64))
    assert e and not n.failed
    assert logical_positions([1,1,0,0,1]).tolist()==[0,1,0,0,2]
    last,valid=last_valid(np.arange(10).reshape(5,2),[1,1,0,0,1])
    assert valid and last.tolist()==[8,9]
    report={"scope":"public synthetic developer checks; not acceptance specification","question_slots":[128,160],
            "document_slots":[256,257],"chunks":len(chunks),"local_rejections":rejected,
            "zero_and_invalid_l2_safe":True,"last_valid_with_holes":True,"r1_numeric_failure":n.failed,
            "r2_executed":False}
    Path("records/models_input_numeric_run.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":run()
