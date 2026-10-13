#!/usr/bin/env python3
"""Bind current BFV proof, native build and actual primitive evidence, not acceptance."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir',default='native/build')
parser.add_argument('--output-dir',default='artifacts')
parser.add_argument('--context',default='artifacts/bfv_context.json')
parser.add_argument('--extra-evidence',action='append',default=[])
args=parser.parse_args()
def rooted(p):
    p=(ROOT/p).resolve();p.relative_to(ROOT);return p
def sha(p):return hashlib.sha256(rooted(p).read_bytes()).hexdigest()
def read(p):return json.loads(rooted(p).read_text())
files=set()
def bind(p):files.add(p);return read(p)
def verify(mapping):
    for p,expected in mapping.items():
        if sha(p)!=expected:raise RuntimeError('Evidence identity changed: '+p)
        files.add(p)
def ledgers(items,terminal='closed'):
    for item in items:
        p=item.get('path',item.get('ledger_path'));expected=item.get('sha256',item.get('ledger_sha256'))
        if sha(p)!=expected:raise RuntimeError('Ledger identity changed: '+p)
        if rooted(p).read_text().splitlines()[-1]!=f'terminal={terminal} all_reserved=burned':
            raise RuntimeError('Ledger terminal mismatch: '+p)
        files.add(p)
context=bind(args.context);q=int(context['levels'][1]['Q']);p=int(context['levels'][0]['primes'][-1]);t=1<<59;n=16384
key_error=21*(2*n+1);first_error=Fraction(key_error,p)+Fraction(n+1,2)
assert first_error<8193
max_l1=8192*(1<<58);bound=8194*(512*max_l1+1)
assert 4*t*bound<q
current_library=sha(args.build_dir+'/libprag_crypto.a')
assert current_library==sha('experiments/pack109_production/build/native/libprag_crypto.a')==sha('build/compute/native/libprag_crypto.a')
runs=[f'records/pack109_production_{s}.json' for s in ('3x17x7_m0','3x17x7_m1','16x17x33_m0','1x2048x4096_m2','1x2048x12288_m3','16x2048x2048_m0','16x2048x6144_m0','16x6144x2048_m0')]
for path in runs:
    e=bind(path)
    assert e['native_version']=='1.0.9' and e['result']=='matched_CUDA_R1' and e['native_TLS13_mTLS']
    assert e['production_integrated'] and not e['candidate_only'] and not e['full_model_R2']
    assert e['source_sha256']['experiments/pack109_production/build/native/libprag_crypto.a']==current_library
    verify(e['source_sha256']);ledgers(e['ledgers'])
token_runs=['records/pack109_token_production_H1024_ids0_8191_8192_155647.json','records/pack109_token_production_H2048_ids151935_155647.json']
for path in token_runs:
    e=bind(path);verify(e['source_sha256']);ledgers(e['ledgers'])
    assert e['R1_reference']['model_checkpoint_used'] and not e['R1_reference']['CPU_reference_arithmetic']
    assert e['per_session_key_setup']==201861440 and e['actual_scan_entries_per_call']==155648
    assert all(c['result']=='matched_CUDA_R1' and c['input_ciphertexts']==19 and c['returned_c1']==c['hidden']//512 for c in e['cases'])
    assert all(c['initial_key_bytes']==c['switch_key_bytes']==0 for c in e['cases'][1:])
diag=bind('records/pack109_diagonal_production.json');verify(diag['source_sha256']);ledgers(diag['ledgers'])
assert len(diag['cases'])==3
long=bind('records/pack109_long_mtls.json');verify(long['source_and_binary_sha256']);ledgers(long['ledgers'])
assert all(c['result']=='matched' for c in long['cases'])
phase=bind('records/pack109_token_phase_mtls.json');verify(phase['source_and_binary_sha256']);ledgers(phase['ledgers'])
assert phase['result']=='matched_exact_token_unscale_boundaries' and phase['constructed_cases']==3500
assert phase['mTLS'] and not phase['ciphertext_lookup_executed']
failure=bind('records/pack109_production_failure_runs.json');verify(failure['source_and_binary_sha256'])
assert failure['result']=='expected_failure_and_cleanup' and len(failure['cases'])==8
for case in failure['cases']:ledgers(case['parties'],'closed' if case['mode'] in ['public_shape_limit','token_shape_limit'] else 'aborted')
rms=bind('records/pack109_production_rms_run.json');verify(rms['source_and_binary_sha256'])
for case in rms['cases']:ledgers(case['ledgers'],'closed' if case['full_RMS_kernel_executed'] else 'aborted')
for path in ['records/pack109_integration_review.json','records/pack109_layout_check.json','records/crypto_static_cost_v1_0_9.json','records/compute_pack109_build.json']:
    e=bind(path);verify(e.get('source_sha256',e.get('source_and_binary_sha256',{})))
assert read('records/crypto_static_cost_v1_0_9.json')['all_available_measured_bytes_match']
assert read('records/compute_pack109_build.json')['worker_crypto_library_matches_mtls_library']
assert read('records/pack109_layout_check.json')['four_t_E_less_q']
source_paths=['third_party/SEAL/native/src/seal/util/rlwe.cpp','third_party/SEAL/native/src/seal/util/scalingvariant.cpp',
 'third_party/SEAL/native/src/seal/util/rns.cpp','third_party/SEAL/native/src/seal/encryptor.cpp',
 'third_party/SEAL/native/src/seal/util/defines.h','third_party/SEAL/build/native/src/seal/util/config.h',
 'native/src/bfv.cpp','native/src/bfv_matrix.inc','native/src/bfv_tile_decoder.inc','native/include/prag/bfv.hpp',
 'native/src/gc_batch_channel.cpp','native/include/prag/gc_batch_channel.hpp','third_party/EzPC/SCI/src/OT/iknp.h']
files.update(source_paths)
files.update(['native/CMakeLists.txt','tools/build_crypto.sh','tools/build_crypto_evidence.py','records/crypto-bfv-correctness.md',
 'records/crypto_pack109_implementation.md','records/crypto-dependencies.json','records/pack109_production_rms_run.json'])
# Earlier attempts retain their original identities and are limitations/history;
# never reinterpret a previous run as a run of current source files.
for base in ('native/include/prag','native/src'):
    files.update(str(p.relative_to(ROOT)) for p in sorted(rooted(base).glob('*')) if p.is_file())
files.update([args.build_dir+'/libprag_crypto.a',args.build_dir+'/prag_bfv_context'])
files.update(args.extra_evidence)
report={'artifact':'ART-002','kind':'conditional_deterministic_BFV_decoding_bound','source_commit':'119dc32e135cb89c1062076a69310d4413ebc824',
 'assumptions':['Locked SEAL 4.1.2 CBD, ternary secret/u; asymmetric key-level encryption then exactly one drop',
 'Ordinary matrices: at most 512 same-level products, at most 8192 centered weight coefficients each; no ciphertext multiplication',
 'Token: fixed one-hot, bounded actual weights, nine independent acyclic key transitions with separate E_token bound',
 'Full polynomial R subtraction and fresh Enc(0); only c1 returned; valid honest BFV phases',
 'Conforming implementation and honest execution; excludes hardware faults and implementation bugs'],
 'noise_sample_abs_max':21,'key_zero_phase_abs_bound':key_error,'dropped_prime':str(p),'first_zero_phase_abs_strict_bound':8193,
 'plaintext_l1_max':str(max_l1),'maximum_input_blocks':512,'noise_bound_expression':'8194*(512*(8192*2^58)+1)',
 'returned_phase_error_abs_strict_bound':str(bound),'decode_required_inequality':'4*t*B < Q','left_hand_side':str(4*t*bound),'Q':str(q),
 'inequality_holds':True,'per_return_noise_failure_probability':0,'finite_session_noise_failure_probability':0,
 'scope':'Exact valid-phase tile conversion; not whole-protocol security certification, model-domain proof or software guarantee',
 'native_interface_version':'prag-native-v1.0.9','native_primitive_enabled':True,'production_activation':'native_primitives_integrated_full_model_still_gated',
 'implementation_review':'Local algebra/source review and actual mTLS diagnostics, not an independent audit',
 'proof_document_sha256':sha('records/crypto-bfv-correctness.md'),
 'exact_decode_circuit':{'profile':'bounded_phase_exact_tile_carry_v2','tile_coordinates_max':512,'GC_instances_per_tile':1,
 'local_component_bits':360,'local_scaled_product_bits':419,'local_quarter_product_bits':362,'local_division':'GMP exact quotient/remainder',
 'division_rounds':0,'AND_gates_per_coordinate':122,'carry_AND_gates':4,'mask_subtraction_AND_gates':118,
 'C_input_bits_per_coordinate':2,'S_input_bits_per_coordinate':61,'output_bits_per_coordinate':59,
 'GC_OT_extension_batches_per_tile':1,'output_shares':'C=A+k-rho; S=B+rho; final S adds WX_S+R',
 'precondition':'4*t*B512<Q and locked valid BFV phase; no general arbitrary-phase decode',
 'equivalence_document_sha256':sha('records/crypto_pack109_implementation.md')},
 'matrix_coefficient_layout':{'version':'adaptive_coefficient_matrix_with_tile_decode_v3','batch_rows_max':16,'input_columns_max':8192,
 'output_columns_max':155648,'reduction_tile':'public_power_of_two_16_to_floor8192divB','stride':'2*d-1','tile_output_coordinates_max':512,
 'output_tile_columns_expression':'min(floor(512/B),floor(8192/(B*d)))','maximum_message_degree':16383,
 'maximum_weight_slots_per_polynomial':8192,'maximum_input_blocks_per_call':512,
 'noise_bound_expression':'B_H=8194*(H*(8192*2^58)+1), 1<=H<=512',
 'input_ciphertext_coefficient_storage_max_bytes_per_party':805306368,
 'target_coefficient':'(o*B+b)*(2*d-1)+d-1','GC_instances':'ceil(O/floor(512/B))','input_ciphertext_reuse':True,
 'per_ciphertext_full_mask_and_fresh_Enc0':True,'per_target_fresh_GC_object':False,'per_tile_fresh_GC_object':True,
 'public_cost_selection':'min ceil(K/d)*1572960+R(d)*786528; ties smallest d; R counts subdivisions within GC tiles'},
 'token_coefficient_layout':{'version':'acyclic_512_target_token_v3','fixed_scan':155648,'input_ciphertexts':19,
 'tile_coordinates':512,'returns_expression':'H/512','target_coefficient':'32*bitreverse9(i)',
 'key_transitions':9,'independent_auxiliary_keys':9,'switch_key_ciphertexts':108,'acyclic_only':True,
 'all_auxiliary_key_setup_bytes':200026336,'same_session_only_key_reuse':True,
 'weight_integer_abs_max':1<<20,'internal_scale':512,'internal_scaled_abs_max':1<<29,
 'error_bound':read('records/pack109_layout_check.json')['token_return_error_bound'],
 'decode_required_inequality':'4*t*E_token<q','GC_ANDs_per_coordinate':358,
 'C_input_bits_per_coordinate':61,'S_input_bits_per_coordinate':120,'GC_OT_batches_per_tile':1,
 'C_output':'arithmetic_shift9(A+B+R+k mod 2^59)-rho','S_output':'rho',
 'no_ring_inverse_of_512':True,'not_general_matrix_unscale':True},
 'private_gamma':{'method':'asymmetric_ring_SplitIKNP','OT_choices_per_coordinate':59,'raw_scale':'f_X+f_W','implicit_floor':False},
 'long_input_layout':{'maximum_input_columns':155648,'maximum_input_blocks':19,'block_dimension':8192,'target_coefficient':8191,
 'GC_instances':'ceil(O/512)','one_returned_c1_per_output':True,'input_ciphertext_reuse':True,'short_input_routes_to_matrix':True},
 'context_width_metadata':'Context export phase/numerator/remainder widths describe generic mathematical decode bounds, not current GC wires',
 'source_sha256':{s:sha(s) for s in source_paths}}
output=rooted(args.output_dir);output.mkdir(exist_ok=True,parents=True)
bound_path=output/'bfv_correctness_bound.json';manifest_path=output/'bfv_native_manifest.json'
bound_raw=(json.dumps(report,indent=2)+'\n').encode();bound_name=str(bound_path.relative_to(ROOT));files.add(bound_name)
files.update(['records/pack109_layout_check.json','records/crypto_static_cost_v1_0_9.json','records/crypto_pack109_implementation.md',
 'tools/check_pack109_layout.py','tools/build_pack109_cost.py','compute/model.hpp','compute/numerics.hpp',
 'records/input109_v108_baseline.json','records/v108_source_snapshot.tar.gz',
 'records/crypto_tile108_implementation.md','records/pack109_rms_localization.json','tools/run_pack109_matrix.py'])
manifest={'artifact':'ART-002-native-materialization','native_interface_version':'prag-native-v1.0.9','native_primitive_enabled':True,
 'full_system_production_enabled':False,'production_mtls_evidence':{'records':runs,'cases':8,'current_library_matches_run':True,
 'source_and_ledger_hashes_verified':True,'current_library_sha256':current_library,'full_model_R2':False},
 'token_real_checkpoint_evidence':token_runs,'gamma_evidence':'records/pack109_diagonal_production.json',
 'long_input_evidence':'records/pack109_long_mtls.json','failure_evidence':'records/pack109_production_failure_runs.json',
 'token_decode_valid_phase_evidence':'records/pack109_token_phase_mtls.json',
 'complete_RMS_attempt':{'record':'records/pack109_production_rms_run.json','native_version':'1.0.9',
 'full_RMS_completed':all(c['full_RMS_kernel_executed'] for c in rms['cases'])},
 'historical_evidence':'v108 records remain unchanged and are not treated as current source runs',
 'hash_algorithm':'SHA-256',
 'files':{s:hashlib.sha256(bound_raw).hexdigest() if s==bound_name else sha(s) for s in sorted(files)}}
# Validate all dependencies before replacing current artifacts.
bound_temp=bound_path.with_suffix('.json.tmp');manifest_temp=manifest_path.with_suffix('.json.tmp')
bound_temp.write_bytes(bound_raw);manifest_temp.write_text(json.dumps(manifest,indent=2)+'\n')
bound_temp.replace(bound_path);manifest_temp.replace(manifest_path)
print(json.dumps({'native_version':'1.0.9','noise_bound_bits':bound.bit_length(),'four_t_B_less_q':True,
 'manifest_files':len(files),'full_system_production_enabled':False}))
