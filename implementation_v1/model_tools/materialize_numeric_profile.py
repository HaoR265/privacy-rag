"""Materialize the authorized coordinated domain with real matrix provenance."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def generate():
    source_path=Path('records/models_coordinated_domain_candidate.json')
    source=json.loads(source_path.read_text())
    qk_source=json.loads(Path('records/models_numeric_domain_candidate.json').read_text())
    constants_path=Path('artifacts/constants/constants_v1.json')
    constants=json.loads(constants_path.read_text())
    version='v1-f12-coordinated-20261013'
    main=2048<<12;weight_cap=256<<12;joint=1<<34;budget=1<<56
    rms={str(d):{'dimension':d,'sum_square_f24_limit_inclusive':d*((1<<42)-16)-1,
                'coordinate_abs_f12_derived_cap':math.isqrt(d*((1<<42)-16)-1)} for d in (128,1024,2048)}
    rope_bound=max(abs(c)+abs(s) for cr,sr in zip(constants['rope_cos_f20'],constants['rope_sin_f20']) for c,s in zip(cr,sr))
    if rope_bound!=1482912:raise ValueError('COORDINATED_ROPE_CONSTANT_MISMATCH')
    models=[]
    for candidate in source['models']:
        kind=candidate['model'];d=1024 if kind=='embedding' else 2048
        directory=Path('artifacts/models')/kind/'quantized_f12'
        manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
        if sha(manifest_path)!=candidate['quantized_manifest_sha256']:raise ValueError('COORDINATED_MANIFEST_HASH')
        tensors={t['name']:t for t in manifest['tensors']}
        matrices=[]
        for item in candidate['matrices']:
            t=tensors[item['tensor']];path=directory/t['path']
            if sha(path)!=item['sha256'] or item['sha256']!=t['sha256']:raise ValueError('COORDINATED_MATRIX_HASH')
            w=np.load(path,mmap_mode='r',allow_pickle=False)
            l1=0;maximum=0
            for first in range(0,len(w),128):
                absolute=np.abs(w[first:first+128].astype(np.int64))
                l1=max(l1,int(absolute.sum(axis=1).max()));maximum=max(maximum,int(absolute.max()))
            if (l1!=item['maximum_row_l1_f12_integer'] or l1!=t['maximum_row_l1_integer']
                or maximum!=item['maximum_absolute_weight_f12'] or maximum>weight_cap):raise ValueError('COORDINATED_MATRIX_RANGE')
            cap=joint if item['use']=='down_proj' else main
            if cap*l1>=budget:raise ValueError('COORDINATED_RAW_BUDGET')
            matrices.append({key:item[key] for key in ['tensor','sha256','shape','maximum_row_l1_f12_integer','maximum_absolute_weight_f12','use']}|
                            {'input_abs_f12':cap,'full_dot_and_any_partial_sum_abs_f24_bound':cap*l1})
        if len(matrices)!=197:raise ValueError('COORDINATED_MATRIX_COUNT')
        qk=next(m for m in qk_source['models'] if m['model']==kind)
        gamma_sources=[];maxima={'q':0,'k':0}
        for t in manifest['tensors']:
            side='q' if t['name'].endswith('.self_attn.q_norm.weight') else 'k' if t['name'].endswith('.self_attn.k_norm.weight') else None
            if side is None:continue
            path=directory/t['path']
            if sha(path)!=t['sha256']:raise ValueError('COORDINATED_QK_GAMMA_HASH')
            maximum=int(np.abs(np.load(path,allow_pickle=False).astype(np.int64)).max())
            maxima[side]=max(maxima[side],maximum)
            gamma_sources.append({'name':t['name'],'side':side,'sha256':t['sha256'],'abs_max_f12':maximum,
                                  'source_file':t['source_file'],'source_tensor':t['source_tensor']})
        if maxima!=qk['gamma_abs_max_f12'] or len(gamma_sources)!=56:raise ValueError('COORDINATED_QK_GAMMA_BOUND')
        down=[m|{'layer':int(m['tensor'].split('.')[2])} for m in matrices if m['use']=='down_proj']
        linear=[m for m in matrices if m['use']!='token table only']
        models.append({**qk,'model_id':manifest['model'],'revision':manifest['revision'],
            'quantized_manifest_sha256':sha(manifest_path),'checkpoint_manifest_sha256':sha(directory.parent/'checkpoint_manifest.json'),
            'gamma_abs_max_f12':maxima,'gamma_sources':gamma_sources,'hidden_dimension':d,
            'residual_abs_f12':rms[str(d)]['coordinate_abs_f12_derived_cap'],
            'down_proj_output_abs_f12':rms[str(d)]['coordinate_abs_f12_derived_cap'],
            'matrix_bounds':matrices,'linear_bounds':linear,'down_proj_bounds':down,
            'ordinary_linear_maximum_row_l1_f12_integer':max(m['maximum_row_l1_f12_integer'] for m in linear if m['use']!='down_proj'),
            'down_proj_maximum_row_l1_f12_integer':max(m['maximum_row_l1_f12_integer'] for m in down),
            'down_proj_maximum_raw_dot_abs_f24_bound':max(m['full_dot_and_any_partial_sum_abs_f24_bound'] for m in down)})
    coordinated={**source,'status':'AUTHORIZED_ENGINEERING_REVISION_APPLIED','numeric_profile_version':version,
        'scope':'Coordinated ordinary-main/SwiGLU/down/residual/RMS/L2 domains and exact pre-accumulation guard; arithmetic scales and floor order unchanged',
        'authorization':'User authorized reasonable numerical/parameter adjustments within existing safety boundaries on 2026-10-13; this preserves the ring, strict 2^56 budget, locked models/scales/floors and fixed protocol budgets',
        'source_analysis_sha256':sha(source_path),'current_native_bfv_source_sha256':sha('native/src/bfv.cpp'),
        'current_scope':'Actual matrix provenance and analytic conditional bounds; successful R1/R2 model invocations have separate run records',
        'limitations':['Conditional admitted domains do not guarantee every input or decode activation fits',
                       'R0 observations are not propagated R1 states or complete R2 execution',
                       'Secret guard implementations and actual reference/protocol execution require separate evidence']}
    coordinated['implementation_requirements']=coordinated.pop('implementation_requirements_not_yet_applied')
    coordinated['implementation_requirements'][-1]='Retain fixed-shape failure propagation and suppression, and bind the implementation to the current numeric profile/configuration.'
    coordinated['selected_main_real_abs_cap']=coordinated.pop('recommended_main_real_abs_cap')
    coordinated['selection_reason']=coordinated.pop('recommendation_reason')
    coordinated_path=Path('artifacts/constants/coordinated_domain_evidence_v1.json')
    coordinated_path.write_text(json.dumps(coordinated,indent=2)+'\n')
    swiglu_path=Path('artifacts/constants/swiglu_domain_evidence_v1.json')
    swiglu={'status':'AUTHORIZED_COORDINATED_REVISION_APPLIED','numeric_profile_version':version,
        'joint_and_down_proj_input_abs_f12':joint,'raw_elementwise_product_abs_f24_bound':main*main,
        'models':[{'model':m['model'],'layers':m['down_proj_bounds'],
                   'maximum_row_l1_f12_integer':m['down_proj_maximum_row_l1_f12_integer'],
                   'maximum_raw_dot_abs_f24_bound':m['down_proj_maximum_raw_dot_abs_f24_bound']} for m in models]}
    swiglu_path.write_text(json.dumps(swiglu,indent=2)+'\n')
    profile={'format_version':'numeric-profile-v1-coordinated','profile_id':version,'profile_version':version,
        'authorization':coordinated['authorization'],'scope':'Coordinated ordinary, SwiGLU, down/residual, RMS/L2 row guards; previously approved exact Q/K domains unchanged',
        'constants_json_sha256':sha(constants_path),'main_abs_f12':main,'weight_abs_f12':weight_cap,
        'fractional_bits':{'main':12,'rope':20,'NR':24},'engineering_budget_exclusive':budget,'semantic_ring_bits':59,
        'normalized_B_f24_bound':379625064,'NR_Y_f24_bound':16777221,'normalized_H_f12_bound':92682,
        'rope_abs_cos_plus_abs_sin_f20_max':rope_bound,'attention_scale_f20':constants['attention_scale_f20'],'models':models,
        'rms_domains':rms,'rms_exponent_range':[-10,8],
        'l2_domain':{'sum_square_f24_limit_inclusive':(1<<52)-1,'coordinate_abs_f12':main,'exponent_range':[-12,13]},
        'row_guard':'Coordinate check before square; term>limit-acc selects zero before addition and accumulates row failure; any row failure resets whole vector and sum to zero. RMS then selects A=F/k=0; L2 revokes eligibility and selects zero vector/A=F. All loops remain fixed shape.',
        'probability_mass_exclusive_f24':4*(1<<24),'PV_raw_exclusive_bound':4*(1<<24)*main,
        'swiglu':{'joint_and_down_proj_input_abs_f12':joint,'joint_real_abs_bound':2048**2,
                   'raw_elementwise_product_abs_f24_bound':main*main,'evidence_path':str(swiglu_path),'evidence_sha256':sha(swiglu_path),
                   'floor_order':'complete elementwise product then floor12; complete down_proj dot then floor12; output/residual checks use each model residual_abs_f12'},
        'coordinated_evidence_path':str(coordinated_path),'coordinated_evidence_sha256':sha(coordinated_path),
        'tighter_NR_bound':'For m in [1,4), y in [0,2], two floors give v=m*y*y-delta with 0<=delta<5/2^24. The maximum of y*(3-m*y*y)/2 is 1/sqrt(m)<=1, so Y_next<=2^24+5. The upper bound is used as an absolute bound only after the prescribed nonnegative state check: a negative or greater-than-2F state is replaced by F/2. The initial state and every safe substitution therefore satisfy 0<=Y<=F+5.',
        'limitations':['Conditional bounds do not prove every token sequence remains inside the admitted domains','R1 and R2 model executions require separate evidence','This artifact does not activate production']}
    output=Path('artifacts/constants/numeric_profile_v1_qk.json')
    output.write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps({'profile':str(output),'sha256':sha(output),'matrix_count':394,'linear_count':393,'gamma_source_count':112}),flush=True)


if __name__=='__main__':generate()
