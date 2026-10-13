"""Small public CUDA/R1 operator equivalence checks; never a CPU model run."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import torch
from references.resource_policy import require_model_device
from references.r1_numeric import IntegerNumerics,MAIN,F24,floor_shift,logical_positions,last_valid
from references.r1_cuda_numeric import CUDAIntegerNumerics
from model_tools.reference_provenance import identify,unchanged


def main():
    device=require_model_device('R1')
    identity=identify('model_tools/check_r1_cuda_operators.py')
    reference=IntegerNumerics.load();gpu=CUDAIntegerNumerics.load(Path('artifacts/constants/constants_v1.json'),device)
    start=time.monotonic();results=[]
    def compare(label,cpu_fn,gpu_fn):
        reference.failed=False;gpu.failed=False
        expected=cpu_fn();actual=gpu_fn()
        expected=expected if isinstance(expected,tuple) else (expected,)
        actual=actual if isinstance(actual,tuple) else (actual,)
        assert len(expected)==len(actual)
        matches=[]
        for a,b in zip(expected,actual):
            b=b.detach().cpu().numpy() if isinstance(b,torch.Tensor) else b
            matches.append(np.array_equal(np.asarray(a),np.asarray(b)))
        if not all(matches) or reference.failed!=gpu.failed:raise AssertionError(label)
        results.append({'operator':label,'bitwise_integer_equal':True,'numeric_failure':gpu.failed,
            'output_shapes':[list(np.asarray(v).shape) for v in expected]})
    log_inputs=np.array(sorted(set([-5,0,1]+[x for bit in range(1,56) for x in [(1<<bit)-1,1<<bit,(1<<bit)+1]])),dtype=np.int64)
    compare('integer_log2_zero_negative_and_power_boundaries',lambda:np.array([int(x).bit_length()-1 if x>0 else -1 for x in log_inputs]),lambda:gpu._log2_floor(gpu.arr(log_inputs)))
    compare('signed_floor',lambda:floor_shift([-4097,-4096,-1,0,4097],12),lambda:gpu.floor_shift([-4097,-4096,-1,0,4097],12))
    compare('inverse_sqrt',lambda:reference.inverse_sqrt([F24,2*F24,4*F24-1],(-10,8)),lambda:gpu.inverse_sqrt([F24,2*F24,4*F24-1],(-10,8)))
    compare('reciprocal',lambda:reference.reciprocal([F24,3*F24,1280*F24]),lambda:gpu.reciprocal([F24,3*F24,1280*F24]))
    compare('exp_including_tail',lambda:reference.exp_negative([0,4096,32*4096,32*4096+1]),lambda:gpu.exp_negative([0,4096,32*4096,32*4096+1]))
    compare('silu_signed',lambda:reference.silu([-4096,4096]),lambda:gpu.silu([-4096,4096]))
    scores=np.array([[0,-4096],[1000,-1000]],dtype=np.int64);allowed=np.ones((2,2),dtype=bool);valid=np.array([1,0])
    compare('softmax_valid_and_invalid_query',lambda:reference.softmax(scores,allowed,valid),lambda:gpu.softmax(scores,allowed,valid))
    rng=np.random.default_rng(20261013)
    x=rng.integers(-4096,4096,(2,128),dtype=np.int64);gamma=np.full(128,4096,dtype=np.int64)
    compare('rms128',lambda:reference.rms_norm(x,gamma),lambda:gpu.rms_norm(x,gamma))
    invalid=x.copy();invalid[0,0]=MAIN+1
    compare('rms128_coordinate_replacement',lambda:reference.rms_norm(invalid,gamma),lambda:gpu.rms_norm(invalid,gamma))
    limit=2048*((1<<42)-16)-1;cap=math.isqrt(limit)
    sparse=np.zeros((2,2048),dtype=np.int64);sparse[0,0]=cap;sparse[1,:2]=cap
    compare('rms2048_sparse_and_energy_failure',lambda:reference.rms_norm(sparse,np.full(2048,4096)),lambda:gpu.rms_norm(sparse,np.full(2048,4096)))
    for count in (0,63,64):
        h=np.zeros(1024,dtype=np.int64);h[:count]=MAIN
        compare('l2_'+str(count)+'_maximum_coordinates',lambda h=h:reference.l2(h),lambda h=h:gpu.l2(h))
    x=rng.integers(-4096,4096,(2,3,128),dtype=np.int64);pos=[0,2,1279]
    compare('half_split_rope',lambda:reference.rope(x,pos),lambda:gpu.rope(x,pos))
    x=rng.integers(-100000,100000,(3,31),dtype=np.int64);weight=rng.integers(-65535,65535,(5,31),dtype=np.int64)
    l1=int(np.abs(weight).sum(axis=-1).max());bound=int(np.abs(x).max())*l1
    compare('direct_cuda_float64_integer_gemm',lambda:x@weight.T,lambda:gpu.exact_matmul(x,weight.T,bound,l1))
    x=rng.integers(-(1<<34),(1<<34),(3,31),dtype=np.int64);x[0,0]=-(1<<34);x[0,1]=(1<<34)-1
    bound=int(np.abs(x).max())*l1
    assert (1<<52)<=bound<(1<<56)
    compare('split_cuda_float64_integer_gemm',lambda:x@weight.T,lambda:gpu.exact_matmul(x,weight.T,bound,l1))
    p=reference.softmax(scores,allowed,valid);value=rng.integers(-MAIN,MAIN,(2,4),dtype=np.int64)
    compare('PV_probability_mass_bound',lambda:reference.probability_value(p,value),lambda:gpu.probability_value(p,value))
    holes=np.array([1,0,1,0,1],dtype=np.int64);hidden=np.arange(20,dtype=np.int64).reshape(5,4)
    compare('logical_positions',lambda:logical_positions(holes),lambda:gpu.logical_positions(holes))
    compare('last_valid_with_holes',lambda:last_valid(hidden,holes),lambda:gpu.last_valid(hidden,holes))
    torch.cuda.synchronize(device)
    record={'scope':'Small public CUDA exact integer operator developer checks; no CPU model, no R2, no formal acceptance',
        'device':str(device),'device_name':torch.cuda.get_device_name(device),'elapsed_seconds':time.monotonic()-start,
        'all_bitwise_equal':True,'checks':results,'gemm_counts':gpu.gemm_counts,
        'start_provenance':identity,'start_to_finish_identity_unchanged':unchanged(identity,'model_tools/check_r1_cuda_operators.py'),
        'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['references/r1_numeric.py','references/r1_cuda_numeric.py','references/resource_policy.py']}}
    Path('records/models_cuda_operator_development_run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['checks','source_sha256','start_provenance']},ensure_ascii=False))


if __name__=='__main__':main()
