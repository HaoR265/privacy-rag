"""Read-only coordinated domain candidate. Does NOT modify active profiles/code."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from model_tools.fetch_models import sha256


def derive():
    budget=1<<56
    models=[]
    for kind in ('embedding','generation'):
        root=Path('artifacts/models')/kind/'quantized_f12'
        manifest=json.loads((root/'manifest.json').read_text())
        entries=[]
        for tensor in manifest['tensors']:
            if len(tensor['shape'])!=2:continue
            path=root/tensor['path']
            if sha256(path)!=tensor['sha256']:raise ValueError('CANDIDATE_WEIGHT_HASH_MISMATCH')
            w=np.load(path,mmap_mode='r',allow_pickle=False)
            maximum_l1=0;maximum_abs=0
            for first in range(0,len(w),128):
                absolute=np.abs(w[first:first+128].astype(np.int64))
                maximum_l1=max(maximum_l1,int(absolute.sum(axis=1).max()))
                maximum_abs=max(maximum_abs,int(absolute.max()))
            if maximum_l1!=tensor['maximum_row_l1_integer']:raise ValueError('CANDIDATE_L1_MISMATCH')
            token=tensor['name']=='model.embed_tokens.weight'
            down=tensor['name'].endswith('.mlp.down_proj.weight')
            entry={'tensor':tensor['name'],'sha256':tensor['sha256'],'shape':tensor['shape'],
                   'maximum_row_l1_f12_integer':maximum_l1,'maximum_absolute_weight_f12':maximum_abs,
                   'use':'down_proj' if down else 'tied LM head and token table' if token and kind=='generation' else 'token table only' if token else 'ordinary linear'}
            for cap in (1024,2048,4096):
                input_cap=(cap*cap if down else cap)<<12
                entry[f'cap_{cap}_raw_abs_f24_bound']=input_cap*maximum_l1
                entry[f'cap_{cap}_below_2pow56']=input_cap*maximum_l1<budget
            entries.append(entry)
        if len(entries)!=197:raise ValueError('CANDIDATE_MATRIX_COUNT')
        down_entries=[e for e in entries if e['use']=='down_proj']
        if len(down_entries)!=28:raise ValueError('CANDIDATE_DOWN_COUNT')
        max_down_l1=max(e['maximum_row_l1_f12_integer'] for e in down_entries)
        ordinary=[e for e in entries if e['use'] not in ('down_proj','token table only')]
        models.append({'model':kind,'model_id':manifest['model'],'revision':manifest['revision'],
                       'quantized_manifest_sha256':sha256(root/'manifest.json'),
                       'matrix_count':len(entries),'linear_matrix_count':len(ordinary)+28,
                       'matrices':entries,'maximum_down_row_l1_f12_integer':max_down_l1,
                       'maximum_ordinary_row_l1_f12_integer':max(e['maximum_row_l1_f12_integer'] for e in ordinary),
                       'maximum_integer_main_real_cap_fitting_down_static_bound':math.isqrt((budget-1)//((1<<12)*max_down_l1))})
    alternatives=[]
    for cap in (1024,2048,4096):
        ordinary_bound=max(e[f'cap_{cap}_raw_abs_f24_bound'] for m in models for e in m['matrices'] if e['use'] not in ('down_proj','token table only'))
        down_bound=max(e[f'cap_{cap}_raw_abs_f24_bound'] for m in models for e in m['matrices'] if e['use']=='down_proj')
        alternatives.append({'main_real_abs_cap':cap,'main_abs_f12':cap<<12,
            'joint_real_abs_cap':cap*cap,'joint_abs_f12':cap*cap*(1<<12),
            'joint_product_raw_abs_f24':(cap<<12)**2,
            'maximum_ordinary_linear_raw_abs_f24':ordinary_bound,
            'maximum_down_proj_raw_abs_f24':down_bound,
            'PV_raw_abs_f36_exclusive_bound':4*cap*(1<<36),
            'SiLU_raw_x_sigmoid_abs_f36':2*cap*(1<<36),
            'all_these_bounds_below_2pow56':max((cap<<12)**2,ordinary_bound,down_bound,4*cap*(1<<36),2*cap*(1<<36))<budget})
    rms={}
    for d in (128,1024,2048):
        limit=d*((1<<42)-16)-1
        rms[str(d)]={'dimension':d,'sum_square_f24_limit_inclusive':limit,
                     'coordinate_abs_f12_derived_cap':math.isqrt(limit),
                     'coordinate_real_cap':math.isqrt(limit)/4096,
                     'maximum_mean_square_epsilon_f24':limit//d+16,
                     'maximum_k':((limit//d+16).bit_length()-1-24)//2,
                     'sum_and_each_square_below_2pow56':limit<budget}
    l2_limit=(1<<52)-1
    observations=[]
    for kind,path in [('embedding','records/models_swiglu_revision_r0_activation_statistics.json'),
                      ('generation','records/models_r0_generation_prefill_observation.json')]:
        document=json.loads(Path(path).read_text());statistics=document['statistics']
        def summarize(predicate):
            selected=[(key,value) for key,value in statistics.items() if predicate(key)]
            coordinate=max(selected,key=lambda pair:pair[1]['max_abs_all_positions'])
            result={'maximum_coordinate_key':coordinate[0],
                    'maximum_coordinate_real_all_positions':coordinate[1]['max_abs_all_positions']}
            square_key='all_floor_f12_square_sum_max' if kind=='generation' else 'floor_f12_row_square_sum_f24_max'
            exponent_key='all_inferred_RMS_k_max' if kind=='generation' else 'inferred_RMS_exponent_k_max'
            with_squares=[pair for pair in selected if square_key in pair[1]]
            if with_squares:
                squares=max(with_squares,key=lambda pair:pair[1][square_key])
                result.update({'maximum_row_square_sum_key':squares[0],
                               'maximum_row_square_sum_f24':squares[1][square_key],
                               'maximum_inferred_RMS_k':max(pair[1][exponent_key] for pair in with_squares),
                               'square_sum_scope':'all physical positions' if kind=='generation' else 'valid physical positions only'})
            return result
        observations.append({'model':kind,'source_path':path,'sha256':sha256(Path(path)),
            'ordinary':summarize(lambda key:key.endswith('.output') and not any(t in key for t in ('q_norm','k_norm','down_proj'))),
            'down_output':summarize(lambda key:key.endswith('.down_proj.output')),
            'residual':summarize(lambda key:'residual' in key),
            'RMS_inputs':summarize(lambda key:key.endswith('.input') and any(t in key for t in ('layernorm','_norm.','final_norm'))),
            'joint':summarize(lambda key:key.endswith('.input_joint')),
            'note':'R0 observed float32 activations floored to f12 for statistics only; these are not propagated R1 states or proofs of arbitrary inputs'})
    report={'status':'PROPOSED_FOR_DISCUSSION_NOT_APPLIED',
            'scope':'Coordinated ordinary-main/SwiGLU/down/residual/RMS/L2 domain and exact pre-accumulation guard candidate; arithmetic scales and floor order unchanged',
            'basis':'actual locked matrix hashes and all row L1 values; domain-conditional analysis, not sample maxima as universal bounds',
            'recommended_main_real_abs_cap':2048,
            'recommendation_reason':'Largest tested power-of-two main cap whose coupled SwiGLU/down static bound fits 2^56 for all actual down matrices; 4096 fails that sufficient bound. Runtime guards remain necessary; this does not prove all model inputs or decode activations fit.',
            'engineering_budget_exclusive':budget,'semantic_ring_bits_unchanged':59,
            'weight_real_domain_unchanged':256,'models':models,'alternatives':alternatives,
            'R0_observations':observations,
            'independent_analysis':{'path':'records/numeric_residual_domain_directions.json','sha256':sha256(Path('records/numeric_residual_domain_directions.json'))},
            'RMS_domains':rms,'RMS_exponent_unchanged':[-10,8],
            'residual_and_down_output_coordinate_rule':'For each model hidden dimension d, use floor(sqrt(d*(2^42-16)-1)) f12 as guarded down-output/residual coordinate cap; each residual RMS input must additionally satisfy the full row-square budget. For Q/K RMS input use min(main cap, the d=128 coordinate cap) plus the same row budget.',
            'RMS_sum_guard_rule':'First secretly range-check every coordinate, square only its safe replacement. For each term, compare term > limit - current_sum, accumulate secret failure, select zero for an over-budget term, and only then add; neither subtraction nor addition exceeds the public limit. Do not add then check. On any row error select the entire vector zero and a=F24 (k=0) before NR; otherwise use exactly floor(S/d)+16. All scans and choices are fixed-shape secure operations.',
            'RMS_mean_proof':'floor(S/d)+16 <= 2^42-1 iff S <= d*(2^42-16)-1. Positive epsilon16 gives minimum k=-10; maximum k=8. The d<=2048 scaled-vector96 and the existing six-round NR bounds therefore remain applicable.',
            'L2_domain':{'sum_square_f24_limit_inclusive':l2_limit,'coordinate_abs_f12_derived_cap':math.isqrt(l2_limit),
                         'effective_input_coordinate_rule':'min(ordinary main cap, derived cap)',
                         'exponent_range_unchanged':[-12,13],
                         'proof':'For 1<=S<=2^52-1, floor((bit_length(S)-1-24)/2) lies in [-12,13]. Use the same pre-accumulation guard and prescribed invalid/zero dummy, preserving retrieval f20 output bounds.'},
            'PV_probability_mass_proof':'All exponential outputs E_j are nonnegative and <=F. The safe maximum/dummy-key rule guarantees one E_j=F, so F<=D=sum(E_j)<=1280F. k=floor(log2 D)-24 gives D<2^(k+1)F. After NR state checks, 0<=r<=2F and R=floor(r/2^k), so sum floor(E_j R/F)<=D R/F<4F. Any subsequent probability range replacement only reduces this mass. Thus abs(any raw PV partial sum)<4F*V_cap_f12, not 1280F*V_cap_f12.',
            'PV_safe_substitution_branches':'Invalid query or no allowed key uses exactly one dummy key whose exponent is F, giving D=F. NR state substitution remains in [0,2F]. Even an independently considered D<F reciprocal fallback gives mass <=D*(2F)/F<2F; D>1280F cannot occur under E_j<=F and N<=1280. Per-probability out-of-range substitution is zero and can only reduce nonnegative mass. After floor24 the attention-output ordinary-domain check is still required; <4M raw-derived output is not a guarantee of <=M.',
            'SiLU_safe_substitution_bound':'For denominator F+exp(|x|) in [F,2F], checked NR state only implies reciprocal<=2F, and exp<=F implies negative-side sigmoid<=2F. Therefore raw x*sigmoid is bounded by 2*M*2^36, without assuming convergence or sigmoid<=F. The existing output-domain check at M is retained BEFORE the SwiGLU joint; checked up also has bound M, so joint<=M^2 remains valid.',
            'other_raw_bounds':{'RMS_scaled_times_NR_f48':192*(1<<48),'NR_y_times_correction_coarse_f48':26*(1<<48),'RMS_h12_times_gamma_f24':(192<<12)*(256<<12),
                               'residual_addition_abs_f12':2*rms['2048']['coordinate_abs_f12_derived_cap'],
                               'worst_RMS_unselected_shift_abs_integer':rms['2048']['coordinate_abs_f12_derived_cap']*(1<<22),
                               'worst_L2_unselected_shift_abs_integer':(2048<<12)*(1<<24)},
            'implementation_requirements_not_yet_applied':['Replace old dimension*maxX*maxW coarse linear precondition with SHA-bound actual matrix maximum row L1; never split/floor partial dot products.',
                'Keep guards before every multiplication whose domain was enlarged, and apply guarded full-row reset on any RMS/L2 budget failure before exponent/shift/NR.',
                'Use the existing local Q/K caps and score bounds; change neither to the ordinary-main cap.',
                'Retain ordinary output guards after PV, SiLU, all ordinary linear/RMS outputs; the arithmetic proof does not imply these guards cannot fail.',
                'Retain fixed-shape failure propagation and suppression, and bind the approved implementation to a new profile/configuration before enabling.'],
            'unchanged_local_QK':'Retain previously authorized per-head gamma/RoPE/QK exact bounds; ordinary main cap is NOT a replacement for those. Their proof uses the safe RMS row conditions, not a 256 per-coordinate assumption.',
            'unchanged_misc':['Actual weights and quantization','BFV scheme and ring59','Exp LUTs and tail32','NR six rounds and exact floor locations','Fixed 1024/256/1280 budgets','R0/R1/R2 isolation'],
            'limitations':['This is not an applied profile or blanket authorization','Generation prefill and selected public observations do not cover 255 decode calls','Guards define accepted numerical domain; no proof all possible token inputs remain inside it','Secret pre-accumulation guard implementation and R1/R2 agreement must be reviewed before activation','BFV arithmetic/static interface supports stay the same; no overall cryptographic certification is inferred']}
    Path('records/models_coordinated_domain_candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'alternatives':alternatives,'RMS_domains':rms,
                      'matrix_count':sum(m['matrix_count'] for m in models)}),flush=True)


if __name__=='__main__':derive()
