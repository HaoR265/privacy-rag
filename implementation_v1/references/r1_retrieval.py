"""R1 plaintext exact Top-2/I7 oracle, confined to independent authorized references."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalReference:
    scores_f40: tuple[int,...]
    top_indices: tuple[int,int]
    top_validity: tuple[int,int]
    lower_f40: tuple[int,...]
    upper_f40: tuple[int,...]
    keep: tuple[int,...]


def retrieve(query,documents,query_valid,document_validity):
    if len(query)!=1024 or len(documents)!=len(document_validity) or any(len(z)!=1024 for z in documents):
        raise ValueError("R1_RETRIEVAL_SHAPE")
    if any(abs(int(x))>2**21 for x in query) or any(abs(int(x))>2**21 for z in documents for x in z):
        raise ValueError("R1_RETRIEVAL_PRE_RANGE")
    valid=[int(bool(query_valid and v)) for v in document_validity]
    scores=[sum(int(a)*int(b) for a,b in zip(query,z)) for z in documents]
    ranked=sorted(range(len(documents)),key=lambda i:(-valid[i],-scores[i],i))
    winners=[(i,valid[i]) for i in ranked[:2]]
    winners += [(0,0)]*(2-len(winners))
    bound=1024*(2*2**21*4095+4095**2)
    coarse=[sum((int(a)//4096)*(int(b)//4096) for a,b in zip(query,z))*4096**2 for z in documents]
    lower=[t-bound for t in coarse]; upper=[t+bound for t in coarse]
    lower_ranked=sorted((i for i in range(len(documents)) if valid[i]),key=lambda i:(-lower[i],i))
    threshold=lower[lower_ranked[1]] if len(lower_ranked)>=2 else None
    keep=[int(bool(valid[i] and (threshold is None or upper[i]>=threshold))) for i in range(len(documents))]
    return RetrievalReference(tuple(scores),tuple(v[0] for v in winners),tuple(v[1] for v in winners),
                              tuple(lower),tuple(upper),tuple(keep))
