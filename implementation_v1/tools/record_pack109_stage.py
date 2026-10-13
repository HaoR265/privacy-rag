"""Record a verified implementation stage; never enables or certifies full R2."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
    manifest=read('artifacts/bfv_native_manifest.json')
    for path,expected in manifest['files'].items():
        if sha(path)!=expected:raise RuntimeError('Current artifact changed: '+path)
    config=(ROOT/'artifacts/configuration.sha256').read_text().strip()
    if sha('artifacts/configuration.json')!=config:raise RuntimeError('configuration changed')
    bindings=read('records/configuration_bindings_debug.json');boundary=read('records/compute_process_debug.json')
    assert bindings['configuration_sha256']==boundary['configuration_sha256']==config
    assert bindings['server_all_files_valid'] and bindings['client_without_server_weights_valid']
    spec=read('records/spec_revision_manifest.json');package=ROOT.parent/'项目文档/第一版机读规范'
    for entry in spec['machine_files']:
        if hashlib.sha256((package/entry['name']).read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Specification changed: '+entry['name'])
    assert not spec['machine_package_consistency']['errors']
    cost=read('records/crypto_static_cost_v1_0_9.json')
    build=read('records/compute_pack109_build.json')
    assert build['worker_crypto_library_matches_mtls_library'] and not build['full_R2']
    old=read('records/crypto_tile108_stage_completion.json');old_count=0
    for path,expected in old['evidence_record_sha256'].items():
        if path.startswith(('records/tile108_','records/compute_tile108_')):
            assert sha(path)==expected;old_count+=1
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:exe=str((p/'exe').resolve(strict=True))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if exe.startswith(str(ROOT)+'/') and Path(exe).name.startswith(('prag_','token_phase_probe','phase_probe')):
            active.append({'pid':int(p.name),'executable':exe})
    if active:raise RuntimeError('Protocol process still active')
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip():raise RuntimeError('GPU compute remains; inspect ownership before finalizing')
    owned_prefixes=['prag-pack109-','prag-token109-production-','prag-token-phase-','prag-diag109-','prag-token109-rms-']
    leftovers=[str(p) for p in Path('/tmp').iterdir() if any(p.name.startswith(x) for x in owned_prefixes)]
    if leftovers:raise RuntimeError('Diagnostic temporary directories remain: '+repr(leftovers))
    paths=list(manifest['production_mtls_evidence']['records'])+manifest['token_real_checkpoint_evidence']
    paths += ['records/pack109_diagonal_production.json','records/pack109_token_phase_mtls.json','records/pack109_production_failure_runs.json','records/pack109_production_rms_run.json','records/pack109_rms_localization.json','records/pack109_integration_review.json','records/pack109_long_mtls.json','records/pack109_repro_entry_3x17x7_m0.json','records/compute_pack109_build.json','records/crypto_static_cost_v1_0_9.json','records/pack109_layout_check.json','records/spec_revision_manifest.json','records/configuration_bindings_debug.json','records/compute_process_debug.json','records/crypto_pack109_implementation.md','README.md','records/implementation_progress.md','records/engineering_decisions.md','tools/record_pack109_stage.py']
    report={'kind':'reusable_implementation_stage_completion_not_full_R2','native_version':'1.0.9',
        'recorded_at':datetime.now(timezone.utc).isoformat(),'configuration_sha256':config,
        'implementation':['adaptive matrix packing','call-local joint QKV/gate-up','OT private diagonal gamma','fixed 155648 one-hot Token lookup with acyclic 512-target returns','wide GC input-OT framing adapter'],
        'native_library_sha256':sha('native/build/libprag_crypto.a'),'worker_sha256':sha('build/compute/prag_worker'),
        'native_manifest_file_hashes_verified':len(manifest['files']),
        'production_library_matrix_mTLS_runs':8,'fresh_reproducible_entry_run':1,'registered_real_Token_weight_runs':6,
        'private_gamma_lengths':[128,1024,2048],'public_failure_cases':8,'Token_valid_phase_cases':3500,'Token_valid_phase_padded_coordinates':3584,
        'Token_H1024_application_bytes':{'first':277488067,'later_same_session':75453344,'input_ciphertexts':19,'returned_c1':2},
        'Token_H2048_application_bytes':{'first':296004771,'later_same_session':93970048,'input_ciphertexts':19,'returned_c1':4},
        'per_session_key_and_base_OT_setup_bytes':202034723,
        'full_query_input_ciphertexts':cost['counts']['nonempty_query_total']['inputs'],
        'static_full_query_changed_operator_subset_bytes':cost['online_application_bytes']['nonempty_query_total'],
        'static_cost_boundary':cost['excludes'],'measured_matrix_and_Token_bytes_match':cost['all_available_measured_bytes_match'],
        'current_RMS_attempt':{'dimension':128,'diagnostic_deadline_seconds':300,'elapsed_seconds':300.001,'application_bytes':3329176758,'complete':False,'gamma_reached':False,'both_ledgers_aborted_and_burned':True,'production_total_session_deadline_changed':False},
        'configuration_server_files_verified':bindings['checked_files_server'],'configuration_client_files_verified':bindings['checked_files_client'],
        'machine_specification_files':len(spec['machine_files']),'rules_checked':spec['machine_package_consistency']['rules'],'specification_errors':[],
        'machine_current_originals_updated':True,'prior_v108_raw_evidence_hashes_unchanged':old_count,
        'unchanged':['model revisions','BFV parameters and modulus chain','ring59 and numeric scales','floor order and six NR iterations','participants and permitted leakage','fixed token budgets'],
        'full_GPU_R1':'prior complete RTX2060 run retained; not rerun','full_R2':False,'full_RMS_or_model_layer_completed':False,
        'private_GPU_backend_ready':False,'full_system_production_enabled':False,'CPU_full_model_run':False,'formal_acceptance_defined':False,
        'proof_scope':'local derivation and source review plus finite diagnostics, not independent security audit',
        'owned_protocol_processes_remaining':active,'NVIDIA_compute_processes':[],
        'temporary_PKI_and_reference_cleanup':True,'MCP_browser_or_VM_resources_created':False,
        'next_work':['batch exact shared guards, bit conversion, floor and nonlinear kernels without changing arithmetic semantics','implement genuine private RTX2060 backend; references are not a protocol backend','complete RMS, model layer, Decode and fixed private RAG only after their prerequisites hold'],
        'evidence_record_sha256':{p:sha(p) for p in paths}}
    target=ROOT/'records/crypto_pack109_stage_completion.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['native_version','configuration_sha256','native_manifest_file_hashes_verified','machine_specification_files','rules_checked','prior_v108_raw_evidence_hashes_unchanged','full_R2','owned_protocol_processes_remaining']},ensure_ascii=False))
if __name__=='__main__':main()
