"""Current fixed schedule BFV subset in bytes; not runtime or acceptance."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INPUT=2*16384*6*8+96
RETURN=16384*6*8+96
DESCRIPTOR=2*(80+96)
GC_FIXED=240+128*(16384//8+96)+3*96
GC_PER_COORD=2*2*16+61*16+59
GC_CHUNK=65536+96
def chunks(t):return (122*t*32+65535)//65536
def tile_bytes(t):return GC_FIXED+GC_PER_COORD*t+GC_CHUNK*chunks(t)
def empty():return dict(inputs=0,ciphertexts=0,targets=0,matrix_calls=0,linear_calls=0,gc_tiles=0,gc_chunks=0)
def add(a,b,times=1):
    for key in a:a[key]+=b[key]*times
def matrix(batch,inner,output):
    cap=512//batch;full,tail=divmod(output,cap);ct=full+bool(tail)
    return dict(inputs=(inner+15)//16,ciphertexts=ct,targets=batch*output,
                matrix_calls=1,linear_calls=0,gc_tiles=ct,
                gc_chunks=full*chunks(batch*cap)+(chunks(batch*tail) if tail else 0))
def token(hidden):
    full,tail=divmod(hidden,512)
    return dict(inputs=19,ciphertexts=hidden,targets=hidden,matrix_calls=0,linear_calls=1,
                gc_tiles=full+bool(tail),gc_chunks=full*chunks(512)+(chunks(tail) if tail else 0))
def sequence(length,hidden,ffn):
    result=empty();add(result,token(hidden),length)
    add(result,matrix(1,hidden,hidden),length*(2*28+1))
    add(result,matrix(1,128,128),length*28*(16+8))
    maps=[(hidden,2048),(hidden,1024),(hidden,1024),(2048,hidden),(hidden,ffn),(hidden,ffn),(ffn,hidden)]
    for begin in range(0,length,16):
        for inner,output in maps:add(result,matrix(min(16,length-begin),inner,output),28)
    assert result['targets']==length*(2*hidden+28*(4*hidden+7168+2*ffn))
    return result
def bytes_for(c):
    return c['inputs']*INPUT+c['ciphertexts']*RETURN+c['targets']*GC_PER_COORD+c['gc_tiles']*GC_FIXED+c['gc_chunks']*GC_CHUNK+(c['matrix_calls']+c['linear_calls'])*DESCRIPTOR
def main():
    counts={'prepare_embedding_257':sequence(257,1024,3072),'query_embedding_160':sequence(160,1024,3072),
            'generation_prefill_1024':sequence(1024,2048,6144),'generation_decode_255':empty(),'generation_logits_256':empty()}
    add(counts['generation_decode_255'],sequence(1,2048,6144),255)
    add(counts['generation_logits_256'],matrix(1,2048,151936),256)
    counts['one_decode_token_including_logits']=sequence(1,2048,6144)
    add(counts['one_decode_token_including_logits'],matrix(1,2048,151936))
    counts['generation_total']=empty()
    for name in ['generation_prefill_1024','generation_decode_255','generation_logits_256']:add(counts['generation_total'],counts[name])
    counts['nonempty_query_total']=dict(counts['generation_total']);add(counts['nonempty_query_total'],counts['query_embedding_160'])
    old=json.loads((ROOT/'records/crypto_static_cost_v1_0_7.json').read_text())
    assert counts['nonempty_query_total']['targets']==old['counts']['nonempty_query_total']['targets']
    sources=['tools/build_tile_cost.py','native/src/bfv.cpp','native/src/bfv_matrix.inc','native/src/bfv_tile_decoder.inc',
             'native/include/prag/gc_batch_channel.hpp','third_party/EzPC/SCI/src/OT/iknp.h','compute/model.hpp','compute/numerics.hpp']
    report={'kind':'static_public_schedule_BFV_communication_subset_not_benchmark','native_version':'1.0.8','counts':counts,
            'BFV_bytes_including_GC_input_OT_excluding_setup_and_other_operators':{key:bytes_for(c) for key,c in counts.items()},
            'GC_AND_gates':{key:122*c['targets'] for key,c in counts.items()},'GC_OT_choice_bits':{key:2*c['targets'] for key,c in counts.items()},
            'per_unit_bytes':dict(input_ciphertext=INPUT,returned_c1=RETURN,descriptor=DESCRIPTOR,tile_fixed=GC_FIXED,
                                 tile_per_coordinate=GC_PER_COORD,GC_padded_chunk=GC_CHUNK),
            'GC_table_chunk_expression':'ceil(122*T*32/65536), 1<=T<=512','GC_division_rounds':0,
            'previous_v107_query_bytes_same_subset':old['BFV_bytes_including_GC_input_OT_excluding_setup_and_other_operators']['nonempty_query_total'],
            'excludes':['base OT and public-key setup','all non-BFV arithmetic, boolean gates, range checks, secret lookups',
                        'TLS/TCP/IP overhead, management, heartbeat','wall time or hardware throughput estimates'],
            'token_lookup':'19 fixed input ciphertexts, one c1 per coordinate, a single shared GC per at most 512 returned targets',
            'gamma':'unchanged separate diagonal matrix per RMS call; no cross-token gamma batching',
            'full_model_R2':False,'formal_acceptance':False,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (ROOT/'records/crypto_static_cost_v1_0_8.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'query_bytes':bytes_for(counts['nonempty_query_total']),'one_decode_bytes':bytes_for(counts['one_decode_token_including_logits']),
                      'query_counts':counts['nonempty_query_total']}))
if __name__=='__main__':main()
