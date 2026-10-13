"""Independent R1 exact-integer CUDA operators; never production R2.

All activation arithmetic is performed on CUDA. Binary64 GEMM is used only
where every integer product and arbitrary partial sum is exactly representable.
There is no CPU arithmetic fallback for a model tensor.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

F24=1<<24
MAIN=2048<<12
WEIGHT=256<<12
BUDGET=1<<56
EXACT=1<<52


class CUDAIntegerNumerics:
    def __init__(self,constants,device):
        self.device=torch.device(device)
        if self.device.type!='cuda':raise RuntimeError('R1_CPU_MODEL_INFERENCE_DISABLED')
        self.constants=constants
        self.exp_tables=self.arr(constants['exp_f24'])
        self.cos=self.arr(constants['rope_cos_f20'])
        self.sin=self.arr(constants['rope_sin_f20'])
        self.powers=self.arr([1<<i for i in range(57)])
        self._failed=torch.zeros((),dtype=torch.bool,device=self.device)
        self.first_failure=None
        self.first_sum_failure=None
        self.failure_callback=None
        self.gemm_counts={'direct_float64':0,'split_float64':0}

    @classmethod
    def load(cls,path,device):return cls(json.loads(Path(path).read_text()),device)

    @property
    def failed(self):return bool(self._failed.item())
    @failed.setter
    def failed(self,value):self._failed=torch.as_tensor(value,dtype=torch.bool,device=self.device)

    def arr(self,x):
        if isinstance(x,torch.Tensor):return x.to(device=self.device,dtype=torch.int64)
        return torch.as_tensor(x,dtype=torch.int64,device=self.device)

    def maximum_abs(self,x):
        x=self.arr(x)
        return int(torch.amax(torch.abs(x)).item()) if x.numel() else 0

    def _mark(self,bad):self._failed=self._failed|torch.any(bad)

    def checked(self,x,bound=MAIN,dummy=0,lower=None):
        x=self.arr(x);lower=-bound if lower is None else lower
        bad=(x<lower)|(x>bound);self._mark(bad)
        if self.failure_callback is not None and self.first_failure is None and bool(torch.any(bad).item()):
            import inspect
            frame=inspect.currentframe().f_back
            self.first_failure={'operator':'checked','shape':list(x.shape),'min_integer':int(x.min().item()),
                'max_integer':int(x.max().item()),'lower_allowed':int(lower),'upper_allowed':int(bound),
                'caller':frame.f_code.co_name,'source_line':frame.f_lineno}
            self.failure_callback(dict(self.first_failure))
        return torch.where(bad,dummy,x)

    def multiply(self,x,y):
        x,y=self.arr(x),self.arr(y)
        if self.maximum_abs(x)*self.maximum_abs(y)>=BUDGET:
            raise ArithmeticError('R1_CUDA_PRE_MULTIPLY_RANGE_NOT_ESTABLISHED')
        return x*y

    def floor_shift(self,x,shift):
        x=self.arr(x)
        if shift>=0:return torch.bitwise_right_shift(x,shift)
        if self.maximum_abs(x)*(1<<-shift)>=BUDGET:
            raise ArithmeticError('R1_CUDA_PRE_SHIFT_RANGE_NOT_ESTABLISHED')
        return torch.bitwise_left_shift(x,-shift)

    def shift_by_exponents(self,x,exponents):
        x,e=torch.broadcast_tensors(self.arr(x),self.arr(exponents))
        left=e.clamp_min(0);right=(-e).clamp_min(0)
        if bool(torch.any((left>55)|(right>62)).item()):raise ArithmeticError('R1_CUDA_SHIFT_EXPONENT')
        limit=torch.bitwise_right_shift(torch.full_like(left,BUDGET-1),left)
        if bool(torch.any(torch.abs(x)>limit).item()):raise ArithmeticError('R1_CUDA_PRE_SHIFT_RANGE_NOT_ESTABLISHED')
        return torch.where(e>=0,torch.bitwise_left_shift(x,left),torch.bitwise_right_shift(x,right))

    def _log2_floor(self,x):
        # Exact integer binary decomposition, without a 57-times expanded
        # temporary for a full [1024,6144] SwiGLU tensor. All callers supply x>0.
        remaining=x.clone();exponent=torch.zeros_like(x)
        for shift in (32,16,8,4,2,1):
            high=remaining>=(1<<shift)
            remaining=torch.where(high,torch.bitwise_right_shift(remaining,shift),remaining)
            exponent+=high.to(torch.int64)*shift
        return torch.where(x>0,exponent,-1)

    def _double_gemm(self,x,y,bound):
        if not 0<=bound<EXACT or self.maximum_abs(x)>=EXACT or self.maximum_abs(y)>=EXACT:
            raise ArithmeticError('R1_CUDA_BINARY64_DOT_BOUND_NOT_ESTABLISHED')
        out=torch.matmul(x.to(torch.float64),y.to(torch.float64))
        if out.device.type!='cuda' or out.dtype!=torch.float64:raise ArithmeticError('R1_CUDA_GEMM_BACKEND')
        integers=out.to(torch.int64)
        # This roundtrip is diagnostic only. Exactness comes from the absolute
        # product/partial-sum bound, not merely from the output being integral.
        if not bool(torch.all(out==integers.to(torch.float64)).item()):raise ArithmeticError('R1_CUDA_NONINTEGER_GEMM_RESULT')
        return integers

    def exact_matmul(self,x,y,bound,right_l1_bound=None):
        x,y=self.arr(x),self.arr(y)
        if x.shape[-1]!=y.shape[-2] or not 0<=bound<BUDGET:raise ArithmeticError('R1_CUDA_MATMUL_PRE_BOUND')
        if bound<EXACT:
            self.gemm_counts['direct_float64']+=1
            return self._double_gemm(x,y,bound)
        # floor decomposition is exact for negative as well as positive x.
        # The public right-column L1 bound covers each complete/arbitrary dot.
        l1=int(right_l1_bound) if right_l1_bound is not None else x.shape[-1]*self.maximum_abs(y)
        if l1<0:raise ArithmeticError('R1_CUDA_NEGATIVE_L1_BOUND')
        high=torch.bitwise_right_shift(x,16)
        low=x-torch.bitwise_left_shift(high,16)
        high_bound=self.maximum_abs(high)*l1;low_bound=self.maximum_abs(low)*l1
        if max(high_bound,low_bound)>=EXACT or (high_bound<<16)+low_bound>=BUDGET:
            raise ArithmeticError('R1_CUDA_SPLIT_GEMM_BOUND_NOT_ESTABLISHED')
        hi=self._double_gemm(high,y,high_bound);lo=self._double_gemm(low,y,low_bound)
        self.gemm_counts['split_float64']+=1
        return torch.bitwise_left_shift(hi,16)+lo

    def dot(self,x,y):
        x,y=self.arr(x),self.arr(y)
        bound=x.shape[-1]*self.maximum_abs(x)*self.maximum_abs(y)
        return self.exact_matmul(x,y,bound)

    def inverse_sqrt(self,a,bounds):
        a=self.arr(a);positive=a>0;self._mark(~positive)
        safe=torch.where(positive,a,F24);k=torch.div(self._log2_floor(safe)-24,2,rounding_mode='floor')
        allowed=(k>=bounds[0])&(k<=bounds[1]);self._mark(~allowed)
        k=torch.where(allowed,k,0);safe=torch.where(allowed,safe,F24)
        m=self.shift_by_exponents(safe,-2*k);y=torch.full_like(m,F24//2)
        for _ in range(6):
            y=self.checked(y,2*F24,F24//2,lower=0)
            u=self.floor_shift(self.multiply(y,y),24)
            v=self.floor_shift(self.multiply(m,u),24)
            w=self.floor_shift(self.multiply(y,3*F24-v),24)
            y=self.floor_shift(w,1)
        return self.checked(y,2*F24,F24//2,lower=0),k

    def reciprocal(self,a):
        a=self.arr(a);allowed=(a>=F24)&(a<=1280*F24);self._mark(~allowed)
        safe=torch.where(allowed,a,F24);k=self._log2_floor(safe)-24
        m=self.shift_by_exponents(safe,-k);r=torch.full_like(m,F24//2)
        for _ in range(6):
            r=self.checked(r,2*F24,F24//2,lower=0)
            v=self.floor_shift(self.multiply(m,r),24)
            r=self.floor_shift(self.multiply(r,2*F24-v),24)
        r=self.checked(r,2*F24,F24//2,lower=0)
        return self.shift_by_exponents(r,-k)

    def exp_negative(self,x):
        x=self.arr(x);self._mark(x<0);eligible=(x>=0)&(x<=32*4096)
        safe=torch.where(eligible,x,0)
        e0=self.exp_tables[0,safe%256];e1=self.exp_tables[1,(safe//256)%256];e2=self.exp_tables[2,safe//65536]
        e=self.floor_shift(self.multiply(self.floor_shift(self.multiply(e0,e1),24),e2),24)
        return torch.where(eligible,e,0)

    def silu(self,x):
        x=self.checked(x);e=self.exp_negative(torch.abs(x));r=self.reciprocal(F24+e)
        j=self.floor_shift(self.multiply(e,r),24);sigmoid=torch.where(x>=0,r,j)
        return self.checked(self.floor_shift(self.multiply(x,sigmoid),24))

    def square_sum_guard(self,x,coordinate_bound,sum_limit):
        original=self.arr(x)
        row_bad=torch.any((original < -coordinate_bound)|(original > coordinate_bound),dim=-1)
        safe=self.checked(original,coordinate_bound);squares=self.multiply(safe,safe)
        # Saturating binary reduction is equivalent at the row interface to the
        # fixed feature scan: all terms are nonnegative; a sum exceeds L iff at
        # least one scan addition would exceed L. Such rows are wholly discarded.
        # Each saturated addition is <=2*(L+1), strictly below 2^56 here.
        if 2*(sum_limit+1)>=BUDGET:raise ArithmeticError('R1_CUDA_SQUARE_REDUCTION_BOUND')
        partial=squares.clamp_max(sum_limit+1)
        while partial.shape[-1]>1:
            if partial.shape[-1]%2:partial=torch.cat((partial,torch.zeros_like(partial[...,:1])),dim=-1)
            partial=(partial[...,::2]+partial[...,1::2]).clamp_max(sum_limit+1)
        total=partial[...,0];over=total>sum_limit;row_bad=row_bad|over;self._mark(row_bad)
        if self.failure_callback is not None and self.first_sum_failure is None and bool(torch.any(over).item()):
            self.first_sum_failure={'operator':'pre_accumulation_square_sum_guard','shape':list(safe.shape),
                'sum_limit':int(sum_limit),'maximum_saturated_sum':int(total.max().item()),'schedule':'equivalent_saturating_CUDA_tree'}
            self.failure_callback(dict(self.first_sum_failure))
        return torch.where(row_bad[...,None],0,safe),torch.where(row_bad,0,total),row_bad

    def rms_norm(self,x,gamma,output_bound=MAIN):
        import math
        x,gamma=self.arr(x),self.arr(gamma);d=x.shape[-1]
        if d not in (128,1024,2048) or gamma.shape!=(d,) or self.maximum_abs(gamma)>WEIGHT:
            raise ValueError('R1_CUDA_INVALID_RMS_WEIGHT_OR_SHAPE')
        limit=d*((1<<42)-16)-1;cap=min(MAIN,math.isqrt(limit)) if d==128 else math.isqrt(limit)
        x,s,row_bad=self.square_sum_guard(x,cap,limit)
        a=torch.where(row_bad,F24,torch.div(s,d,rounding_mode='floor')+16)
        y,k=self.inverse_sqrt(a,(-10,8));b=self.shift_by_exponents(x,12-k[...,None]);b=self.checked(b,96*F24)
        h24=self.floor_shift(self.multiply(b,y[...,None]),24);h12=self.floor_shift(h24,12)
        return self.checked(self.floor_shift(self.multiply(h12,gamma),12),output_bound)

    def l2(self,hidden,record_valid=True,last_position_valid=True):
        hidden=self.arr(hidden)
        if hidden.shape!=(1024,):raise ValueError('R1_CUDA_INVALID_L2_SHAPE')
        hidden,s,row_bad=self.square_sum_guard(hidden,MAIN,(1<<52)-1)
        valid=self.arr(record_valid).bool()&self.arr(last_position_valid).bool()&(s>0)&~row_bad
        y,k=self.inverse_sqrt(torch.where(valid,s,F24),(-12,13));safe=torch.where(valid,hidden,0)
        b=self.checked(self.shift_by_exponents(safe,12-k),2*F24)
        z=self.checked(self.floor_shift(self.floor_shift(self.multiply(b,y),24),4),2<<20)
        valid=valid&~self._failed
        return torch.where(valid,z,0),valid.to(torch.int64)

    def softmax(self,scores,key_allowed,query_valid):
        scores=self.arr(scores);allowed=self.arr(key_allowed).bool().clone();valid=self.arr(query_valid).bool()
        if scores.shape!=allowed.shape or scores.shape[-1]>1280:raise ValueError('R1_CUDA_INVALID_SOFTMAX_SHAPE')
        invalid=~valid;allowed=torch.where(invalid[...,None],False,allowed);allowed[...,0]|=invalid
        work=torch.where(invalid[...,None],0,scores);missing=~torch.any(allowed,dim=-1);self._mark(missing)
        allowed[...,0]|=missing;work=torch.where(missing[...,None],0,work)
        maximum=torch.amax(torch.where(allowed,work,-(1<<57)),dim=-1,keepdim=True)
        e=torch.where(allowed,self.exp_negative(torch.where(allowed,maximum-work,0)),0)
        denominator=torch.sum(e,dim=-1,keepdim=True,dtype=torch.int64)
        return self.checked(self.floor_shift(self.multiply(e,self.reciprocal(denominator)),24),F24,lower=0)

    def rope(self,x,positions,input_bound=MAIN,output_bound=MAIN):
        x=self.checked(x,input_bound);positions=self.arr(positions)
        if bool(torch.any((positions<0)|(positions>=1280)).item()) or x.shape[-1]!=128:raise ValueError('R1_CUDA_INVALID_ROPE_SHAPE')
        c=self.cos[positions];s=self.sin[positions];a,b=x[...,:64],x[...,64:]
        first=self.floor_shift(self.multiply(a,c)-self.multiply(b,s),20)
        second=self.floor_shift(self.multiply(b,c)+self.multiply(a,s),20)
        return self.checked(torch.cat((first,second),dim=-1),output_bound)

    def probability_value(self,p,value):
        p=self.arr(p);value=self.checked(value)
        if p.shape[-1]!=value.shape[-2] or p.shape[-1]>1280:raise ValueError('R1_CUDA_INVALID_PV_SHAPE')
        if bool(torch.any((p<0)|(p>F24)).item()) or bool(torch.any(torch.sum(p,dim=-1)>=4*F24).item()):
            raise ArithmeticError('R1_CUDA_SOFTMAX_MASS_BOUND_NOT_ESTABLISHED')
        return self.exact_matmul(p,value,4*F24*self.maximum_abs(value))

    def linear(self,x,weight,raw=False,input_bound=MAIN,output_bound=MAIN,row_l1_bound=None):
        x=self.checked(x,input_bound)
        if row_l1_bound is None or input_bound*int(row_l1_bound)>=BUDGET or x.shape[-1]!=weight.shape[-1]:
            raise ArithmeticError('R1_CUDA_WEIGHT_L1_PRE_DOT_RANGE_NOT_ESTABLISHED')
        actual_bound=self.maximum_abs(x)*int(row_l1_bound)
        result=torch.empty(x.shape[:-1]+(weight.shape[0],),dtype=torch.int64,device=self.device)
        block=len(weight) if len(weight)<=8192 else 4096
        for first in range(0,len(weight),block):
            result[...,first:first+block]=self.exact_matmul(x,weight[first:first+block].T,actual_bound,row_l1_bound)
        return result if raw else self.checked(self.floor_shift(result,12),output_bound)

    def logical_positions(self,validity):
        validity=self.arr(validity);return torch.where(validity.bool(),torch.cumsum(validity,0)-1,0)

    def last_valid(self,hidden,validity):
        validity=self.arr(validity).bool()
        selectors=validity&(torch.flip(torch.cumsum(torch.flip(validity.to(torch.int64),[0]),0),[0])==1)
        return torch.sum(self.arr(hidden)*selectors[:,None],dim=0),torch.any(validity)

    def greedy(self,logits):
        logits=self.arr(logits).reshape(-1)
        if logits.shape!=(151936,):raise ValueError('R1_CUDA_INVALID_LOGIT_SHAPE')
        allowed=torch.cat((torch.arange(151644,device=self.device),self.arr([151645])))
        return allowed[torch.argmax(logits[allowed])]
