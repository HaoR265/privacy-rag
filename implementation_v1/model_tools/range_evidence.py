"""Materialize domain-conditional integer bounds; not a runtime success claim."""
import json
import hashlib
from pathlib import Path


def generate():
    path=Path('artifacts/constants/numeric_profile_v1_qk.json')
    profile=json.loads(path.read_text())
    if profile['profile_version']!='v1-f12-coordinated-20261013':raise ValueError('RANGE_PROFILE_VERSION')
    budget=1<<56;main=profile['main_abs_f12'];F=1<<24
    bounds={'ordinary_main_f12':main,'weight_f12':profile['weight_abs_f12'],
        'RMS_prechecked_square_sum_2048':profile['rms_domains']['2048']['sum_square_f24_limit_inclusive'],
        'L2_prechecked_square_sum':profile['l2_domain']['sum_square_f24_limit_inclusive'],
        'PV_any_partial_sum_exclusive':4*F*main,'SiLU_raw_x_sigmoid':2*main*F,
        'retrieval_dot_1024':1024*4*(1<<40),'RMS_scaled_vector_times_NR':192*(1<<48),
        'RMS_h12_times_gamma':(192<<12)*profile['weight_abs_f12'],
        'NR_m_times_y_square_after_rescale':16*(1<<48),'NR_y_times_correction':26*(1<<48),
        'reciprocal_r_times_correction':8*(1<<48),'exp_two_factor':1<<48,
        'softmax_probability_product':2*(1<<48),'softmax_denominator':1280*F,
        'I7_error_bound':1024*(2*(1<<21)*4095+4095**2)}
    qk_bounds={};swiglu_bounds={};linear_bounds={}
    for model in profile['models']:
        name=model['model'];residual=model['residual_abs_f12']
        local={'QK_raw_f24':model['QK_raw_abs_f24_bound'],
               'QK_after_floor12_f12':model['QK_after_floor12_abs_f12_bound'],
               'attention_scale_raw_f32':model['attention_scale_raw_abs_f32_bound'],
               'softmax_score_difference_f12':2*((model['attention_scale_raw_abs_f32_bound']+(1<<20)-1)//(1<<20))}
        for side in ('q','k'):
            local[side+'_gamma_product_f24']=profile['normalized_H_f12_bound']*model['gamma_abs_max_f12'][side]
            local[side+'_rope_pair_f32']=model['qk_norm_abs_candidate_f12'][side]*profile['rope_abs_cos_plus_abs_sin_f20_max']
        qk_bounds[name]=local;bounds.update({name+'_'+key:value for key,value in local.items()})
        swiglu_bounds[name]={'joint_abs_f12':profile['swiglu']['joint_and_down_proj_input_abs_f12'],
                            'raw_elementwise_product_abs_f24':profile['swiglu']['raw_elementwise_product_abs_f24_bound'],
                            'down_proj_full_and_partial_sum_abs_f24':model['down_proj_maximum_raw_dot_abs_f24_bound'],
                            'down_proj_actual_maximum_row_l1_f12':model['down_proj_maximum_row_l1_f12_integer'],
                            'down_proj_output_abs_f12':model['down_proj_output_abs_f12'],'layers':model['down_proj_bounds']}
        linear_bounds[name]=model['linear_bounds']
        bounds[name+'_ordinary_linear_full_and_partial_sum']=max(t['full_dot_and_any_partial_sum_abs_f24_bound'] for t in model['linear_bounds'] if t['use']!='down_proj')
        bounds[name+'_down_proj_full_and_partial_sum']=model['down_proj_maximum_raw_dot_abs_f24_bound']
        bounds[name+'_swiglu_product']=profile['swiglu']['raw_elementwise_product_abs_f24_bound']
        bounds[name+'_residual_add']=2*residual
        bounds[name+'_RMS_worst_unselected_shift']=residual*(1<<22)
    bounds['L2_worst_unselected_shift']=main*(1<<24)
    if not all(v<budget for v in bounds.values()):raise ArithmeticError('COORDINATED_BOUND_EXCEEDS_BUDGET')
    report={'format_version':'numeric-range-evidence-v1-coordinated','numeric_profile_version':profile['profile_version'],
        'numeric_profile_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rule_ids':['NUM-003','NUM-004'],
        'integer_upper_bounds':bounds,'engineering_budget_exclusive':budget,'signed_boundary_exclusive':1<<58,
        'all_listed_bounds_below_budget':True,'qk_operator_bounds':qk_bounds,'swiglu_operator_bounds':swiglu_bounds,
        'actual_linear_matrix_bounds':linear_bounds,'rms_domains':profile['rms_domains'],'l2_domain':profile['l2_domain'],
        'assumptions':['All actual matrix hashes and maximum row L1 values are bound; complete and any partial dot is bounded by input cap times actual row L1, never a prematurely truncated partial sum',
            'Weights and gamma remain abs<=256; ordinary checked activations use 2048; Q/K specialized caps are unchanged',
            'Checked SiLU and up outputs imply joint real cap4194304; each model has its own down/residual coordinate cap',
            'Before square, coordinate failures are zeroed; before add, over-budget terms are zeroed. Any row failure resets the entire vector and sum before exponent/NR use',
            'RMS uses A=F for a failed row, otherwise floor(S/d)+16. L2 invalid/zero/failed input uses zero vector/A=F',
            'Frozen exp values in [0,F], a maximal or dummy key gives E=F, checked reciprocal states in [0,2F] imply nonnegative probability mass<4F; individual probability replacement only lowers mass',
            'Secret failures propagate and suppress release; a substituted result is never a normal successful answer'],
        'QK_tighter_NR_derivation':profile['tighter_NR_bound'],
        'generic_NR_correction_bound':'Checked 0<=Y<=2F and F<=M<4F give 0<=V<16F, abs(3F-V)<=13F and abs(Y*(3F-V))<=26F^2.',
        'RMS_scaled_component_derivation':'For accepted rows A=floor(S/d)+16>S/d; A<4F*4^k gives scaled abs<2sqrt(d)*F with negative-floor unit allowance. d<=2048 remains inside96F. Failed rows select x0,A=F before scaling.',
        'L2_scaled_component_derivation':'For accepted nonzero S<2^52, k remains[-12,13] and |X|<=sqrt(S)<2^(k+1), giving scaled bound2F. Invalid or failed row uses zero vector/A=F.',
        'PV_safe_substitution_derivation':'Valid/dummy key logic gives F<=D<=1280F. D<2F*2^k and final r<=2F imply sum floor(E_i*floor(r/2^k)/F)<4F. An independently considered D<F fallback gives mass<2F; D>1280F is impossible under exp/shape bounds. Hence every raw PV partial sum<4F*Vcap.',
        'SiLU_safe_bound':'Checked reciprocal only implies sigmoid<=2F; raw x*sigmoid<=2*M*2^36. SiLU output is checked at M before joint.',
        'scope':'Analytic operator-domain conditional no-wrap bounds; does not prove all token inputs remain inside the admitted domains',
        'known_limitations':['Not a BFV noise or session failure-probability proof','R1 and R2 actual model execution must be reported separately','Finite precision constant agreement is not an interval arithmetic certificate']}
    data=json.dumps(report,indent=2)+'\n'
    Path('artifacts/constants/range_evidence.json').write_text(data)
    Path('artifacts/constants/range_evidence_v1_coordinated.json').write_text(data)
    print(json.dumps({'numeric_profile_version':profile['profile_version'],'all_listed_bounds_below_budget':True}))


if __name__=='__main__':generate()
