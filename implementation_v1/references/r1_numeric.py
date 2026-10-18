"""R1 ONLY: exact plaintext integer functional reference on authorized data.

No production module may import this package. Values are signed complete model
integers, never random shares. CPU int64 is safe only after pre-operation bounds.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

F24 = 1 << 24
MAIN = 2048 << 12
WEIGHT = 256 << 12
BUDGET = 1 << 56


def arr(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def maximum_abs(value) -> int:
    value = np.asarray(value)
    return max(abs(int(value.min(initial=0))), abs(int(value.max(initial=0))))


def multiply(x, y):
    if maximum_abs(x) * maximum_abs(y) >= BUDGET:
        raise ArithmeticError("R1_PRE_MULTIPLY_RANGE_NOT_ESTABLISHED")
    return arr(x) * arr(y)


def dot(x, y):
    x, y = arr(x), arr(y)
    bound=x.shape[-1] * maximum_abs(x) * maximum_abs(y)
    if x.shape[-1] != y.shape[-2] or bound >= BUDGET:
        raise ArithmeticError("R1_PRE_DOT_RANGE_NOT_ESTABLISHED")
    return exact_integer_matmul(x,y,bound)


def exact_integer_matmul(x,y,absolute_partial_sum_bound,operand_abs_bounds=None):
    """Plaintext R1 acceleration only; float64 stores exact bounded integers.

    Callers prove the absolute sum of all products is below this bound. Below
    2^52, each integer product and every partial sum is exactly representable
    in IEEE binary64, so dgemm's summation order/FMA cannot change the integer.
    Larger admitted dots keep int64 arithmetic. No rescale occurs here.
    """
    x,y=arr(x),arr(y)
    if absolute_partial_sum_bound>=BUDGET:
        raise ArithmeticError('R1_MATMUL_BOUND_EXCEEDS_BUDGET')
    operand_abs_bounds=operand_abs_bounds or (maximum_abs(x),maximum_abs(y))
    if (absolute_partial_sum_bound<(1<<52)
        and max(operand_abs_bounds)<(1<<52)):
        result=np.matmul(x.astype(np.float64),y.astype(np.float64))
        integers=result.astype(np.int64)
        if result.dtype!=np.float64 or not np.array_equal(result,integers.astype(np.float64)):
            raise ArithmeticError('R1_BINARY64_INTEGER_RESULT_NOT_EXACT')
        return integers
    return x@y


def floor_shift(x, shift: int):
    if shift < 0:
        if maximum_abs(x) * (1 << -shift) >= BUDGET:
            raise ArithmeticError("R1_PRE_SHIFT_RANGE_NOT_ESTABLISHED")
        return arr(x) * (1 << -shift)
    return arr(x) // (1 << shift)


@dataclass
class IntegerNumerics:
    constants: dict
    failed: bool = False
    first_sum_failure: dict | None = None

    @classmethod
    def load(cls, path=Path("artifacts/constants/constants_v1.json")):
        return cls(json.loads(Path(path).read_text()))

    def checked(self, x, bound=MAIN, dummy=0, lower=None):
        x = arr(x)
        bad = (x < (-bound if lower is None else lower)) | (x > bound)
        self.failed |= bool(np.any(bad))
        return np.where(bad, dummy, x)

    def shift_by_exponents(self, x, exponents):
        x, exponents = np.broadcast_arrays(arr(x), arr(exponents))
        out = np.empty_like(x)
        for exponent in np.unique(exponents):
            select = exponents == exponent
            out[select] = floor_shift(x[select], -int(exponent))
        return out

    def inverse_sqrt(self, a, bounds):
        a = arr(a)
        positive = a > 0
        self.failed |= not bool(np.all(positive))
        safe = np.where(positive, a, F24)
        bit = arr(np.frompyfunc(lambda n: int(n).bit_length()-1, 1, 1)(safe))
        k = (bit-24)//2
        allowed = (k >= bounds[0]) & (k <= bounds[1])
        self.failed |= not bool(np.all(allowed))
        k, safe = np.where(allowed, k, 0), np.where(allowed, safe, F24)
        m = self.shift_by_exponents(safe, -2*k)
        y = np.full_like(m, F24//2)
        for _ in range(6):
            y = self.checked(y, 2*F24, F24//2, lower=0)
            u = floor_shift(multiply(y, y), 24)
            v = floor_shift(multiply(m, u), 24)
            w = floor_shift(multiply(y, 3*F24-v), 24)
            y = w//2
        return self.checked(y, 2*F24, F24//2, lower=0), k

    def reciprocal(self, a):
        a = arr(a)
        allowed = (a >= F24) & (a <= 1280*F24)
        self.failed |= not bool(np.all(allowed))
        safe = np.where(allowed, a, F24)
        bit = arr(np.frompyfunc(lambda n: int(n).bit_length()-1, 1, 1)(safe))
        k = bit-24
        m = self.shift_by_exponents(safe, -k)
        r = np.full_like(m, F24//2)
        for _ in range(6):
            r = self.checked(r, 2*F24, F24//2, lower=0)
            v = floor_shift(multiply(m, r), 24)
            r = floor_shift(multiply(r, 2*F24-v), 24)
        r = self.checked(r, 2*F24, F24//2, lower=0)
        return self.shift_by_exponents(r, -k)

    def exp_negative(self, x):
        x = arr(x)
        self.failed |= bool(np.any(x < 0))
        eligible = (x >= 0) & (x <= 32*4096)
        safe = np.where(eligible, x, 0)
        tables = arr(self.constants["exp_f24"])
        e0 = tables[0, safe % 256]
        e1 = tables[1, (safe//256) % 256]
        e2 = tables[2, safe//65536]
        e = floor_shift(multiply(floor_shift(multiply(e0,e1),24),e2),24)
        return np.where(eligible, e, 0)

    def silu(self, x):
        x = self.checked(x)
        e = self.exp_negative(np.abs(x))
        r = self.reciprocal(F24+e)
        j = floor_shift(multiply(e,r),24)
        sigmoid = np.where(x >= 0,r,j)
        return self.checked(floor_shift(multiply(x,sigmoid),24))

    def square_sum_guard(self, x, coordinate_bound, sum_limit):
        """R1 version of the fixed-scan pre-addition secret budget guard."""
        original=arr(x)
        row_bad=np.any((original < -coordinate_bound)|(original > coordinate_bound),axis=-1)
        safe=self.checked(original,coordinate_bound)
        squares=multiply(safe,safe)
        total=np.zeros(safe.shape[:-1],dtype=np.int64)
        for j in range(safe.shape[-1]):
            room=sum_limit-total
            term=squares[...,j]
            over=term>room
            if bool(np.any(over)) and self.first_sum_failure is None:
                self.first_sum_failure={'operator':'pre_accumulation_square_sum_guard',
                    'shape':list(safe.shape),'sum_limit':int(sum_limit),'first_feature':j,
                    'maximum_term':int(np.max(term)),'maximum_partial_sum':int(np.max(total))}
            row_bad |= over
            total += np.where(over,0,term)
        self.failed |= bool(np.any(row_bad))
        return np.where(row_bad[...,None],0,safe),np.where(row_bad,0,total),row_bad

    def rms_norm(self, x, gamma, output_bound=MAIN):
        x, gamma = arr(x), arr(gamma)
        d = x.shape[-1]
        if d not in (128,1024,2048) or gamma.shape != (d,) or maximum_abs(gamma)>WEIGHT:
            raise ValueError("R1_INVALID_RMS_WEIGHT_OR_SHAPE")
        import math
        limit=d*((1<<42)-16)-1
        coordinate_bound=min(MAIN,math.isqrt(limit)) if d==128 else math.isqrt(limit)
        x,s,row_bad=self.square_sum_guard(x,coordinate_bound,limit)
        a = np.where(row_bad,F24,s//d+16)
        y,k = self.inverse_sqrt(a,(-10,8))
        b = self.shift_by_exponents(x,12-k[...,None])
        b = self.checked(b,96*F24)
        h24 = floor_shift(multiply(b,y[...,None]),24)
        h12 = floor_shift(h24,12)
        return self.checked(floor_shift(multiply(h12,gamma),12),output_bound)

    def l2(self, hidden, record_valid=True, last_position_valid=True):
        hidden = arr(hidden)
        if hidden.shape != (1024,):
            raise ValueError("R1_INVALID_L2_SHAPE")
        hidden,s,row_bad=self.square_sum_guard(hidden,MAIN,(1<<52)-1)
        s=int(s)
        valid = bool(record_valid and last_position_valid and s > 0 and not row_bad)
        y,k = self.inverse_sqrt(s if valid else F24,(-12,13))
        safe_hidden = np.where(valid,hidden,0)
        b = self.shift_by_exponents(safe_hidden,12-k)
        b = self.checked(b,2*F24)
        h24 = floor_shift(multiply(b,y),24)
        z = self.checked(floor_shift(h24,4),2 << 20)
        valid = valid and not self.failed
        return np.where(valid,z,0), int(valid)

    def softmax(self, scores, key_allowed, query_valid):
        scores, allowed = arr(scores), np.asarray(key_allowed,dtype=bool).copy()
        query_valid = np.asarray(query_valid,dtype=bool)
        if scores.shape != allowed.shape or scores.shape[-1]>1280:
            raise ValueError("R1_INVALID_SOFTMAX_SHAPE")
        invalid = ~query_valid
        allowed = np.where(invalid[...,None],False,allowed)
        allowed[...,0] |= invalid
        work = np.where(invalid[...,None],0,scores)
        # Complete plaintext R1 may use array selection; this is not R2 lookup.
        self.failed |= bool(np.any(~np.any(allowed,axis=-1)))
        missing = ~np.any(allowed,axis=-1)
        allowed[...,0] |= missing
        work = np.where(missing[...,None],0,work)
        maximum = np.max(np.where(allowed,work,-(1<<57)),axis=-1,keepdims=True)
        e = np.where(allowed,self.exp_negative(np.where(allowed,maximum-work,0)),0)
        denominator = np.sum(e,axis=-1,keepdims=True,dtype=np.int64)
        return self.checked(floor_shift(multiply(e,self.reciprocal(denominator)),24),F24,lower=0)

    def rope(self, x, positions, input_bound=MAIN, output_bound=MAIN):
        x = self.checked(x,input_bound)
        positions = arr(positions)
        if np.any(positions<0) or np.any(positions>=1280) or x.shape[-1]!=128:
            raise ValueError("R1_INVALID_ROPE_SHAPE")
        c = arr(self.constants["rope_cos_f20"])[positions]
        s = arr(self.constants["rope_sin_f20"])[positions]
        a,b=x[...,:64],x[...,64:]
        first=floor_shift(multiply(a,c)-multiply(b,s),20)
        second=floor_shift(multiply(b,c)+multiply(a,s),20)
        return self.checked(np.concatenate((first,second),axis=-1),output_bound)

    def probability_value(self,p,value):
        p,value=arr(p),self.checked(value)
        if p.shape[-1]!=value.shape[-2] or p.shape[-1]>1280:
            raise ValueError("R1_INVALID_PV_SHAPE")
        # The exact existing Softmax algorithm, including its safe substitutes,
        # establishes nonnegative p and sum(p)<4F. This is a pre-multiply check.
        if np.any(p<0) or np.any(p>F24) or np.any(np.sum(p,axis=-1,dtype=np.int64)>=4*F24):
            raise ArithmeticError("R1_SOFTMAX_MASS_BOUND_NOT_ESTABLISHED")
        if 4*F24*maximum_abs(value)>=BUDGET:
            raise ArithmeticError("R1_PV_PRE_DOT_RANGE_NOT_ESTABLISHED")
        return exact_integer_matmul(p,value,4*F24*maximum_abs(value))

    def linear(self,x,weight,raw=False,input_bound=MAIN,output_bound=MAIN,row_l1_bound=None,weight_domain_verified=False):
        x = self.checked(x,input_bound)
        if not weight_domain_verified and maximum_abs(weight)>WEIGHT:
            raise ValueError("R1_WEIGHT_DOMAIN_REJECTED")
        if row_l1_bound is None:
            row_l1_bound=max(int(np.abs(arr(weight[i:i+128])).sum(axis=-1).max())
                             for i in range(0,len(weight),128))
        if input_bound*int(row_l1_bound)>=BUDGET or x.shape[-1]!=weight.shape[-1]:
            raise ArithmeticError("R1_WEIGHT_L1_PRE_DOT_RANGE_NOT_ESTABLISHED")
        # Ordinary matrices fit in bounded layer-local memory. Only the large
        # tied vocabulary head needs output-row blocks (no split dots/floors).
        result=np.empty(x.shape[:-1]+(weight.shape[0],),dtype=np.int64)
        actual_x_max=maximum_abs(x);actual_bound=actual_x_max*int(row_l1_bound)
        block=len(weight) if len(weight)<=8192 else 4096
        for first in range(0,len(weight),block):
            result[...,first:first+block]=exact_integer_matmul(x,arr(weight[first:first+block]).T,actual_bound,(actual_x_max,WEIGHT))
        return result if raw else self.checked(floor_shift(result,12),output_bound)


def logical_positions(validity):
    validity=arr(validity)
    return np.where(validity,np.cumsum(validity)-1,0)


def last_valid(hidden,validity):
    validity=np.asarray(validity,dtype=bool)
    # Explicit last physical position works even when fixed private slots have holes.
    selectors=validity & (np.cumsum(validity[::-1])[::-1]==1)
    return np.sum(arr(hidden)*selectors[:,None],axis=0),bool(np.any(validity))


def greedy(logits):
    logits=arr(logits).reshape(-1)
    if logits.shape != (151936,):
        raise ValueError("R1_INVALID_LOGIT_SHAPE")
    allowed=np.concatenate((np.arange(151644),np.array([151645])))
    return int(allowed[np.argmax(logits[allowed])])
