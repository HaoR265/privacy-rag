"""Bind materialized artifacts without claiming configuration activation."""
from pathlib import Path
import json
import hashlib
from model_tools.fetch_models import sha256


def build():
    paths=["artifacts/template/template_v1.json","artifacts/template/manifest.json",
           "artifacts/constants/constants_v1.json","artifacts/constants/constants_v1.bin",
           "artifacts/constants/native_manifest.json","artifacts/constants/range_evidence.json",
           "artifacts/constants/numeric_profile_v1_qk.json", "artifacts/constants/swiglu_domain_evidence_v1.json",
           "artifacts/constants/coordinated_domain_evidence_v1.json"]
    for kind in ("embedding","generation"):
        paths += [f"artifacts/models/{kind}/checkpoint_manifest.json",f"artifacts/models/{kind}/quantized_f12/manifest.json"]
    files=[{"path":p,"bytes":Path(p).stat().st_size,"sha256":sha256(Path(p))} for p in paths]
    profile=json.loads(Path("artifacts/constants/numeric_profile_v1_qk.json").read_text())
    numeric={"profile_version":profile["profile_version"],"ring_bits":59,"weights_fractional_bits":12,"main_fractional_bits":12,
             "NR_exp_probability_fractional_bits":24,"rope_fractional_bits":20,
             "retrieval_fractional_bits":20,"epsilon_f24":16,"main_absolute_domain":2048,"weight_absolute_domain":256,
             "public_generation_slots":1024,"output_slots":256,"decode_calls":255,
             "approved_local_swiglu_domain":profile["swiglu"],
             "rms_domains":profile["rms_domains"],"l2_domain":profile["l2_domain"],
             "residual_domains":[{"model":m["model"],"absolute_f12":m['residual_abs_f12']} for m in profile['models']],
             "approved_local_qk_domains":[{"model":m["model"],"qk_norm_abs_f12":m["qk_norm_abs_candidate_f12"],
                 "rope_abs_f12":m["rope_abs_candidate_f12"]} for m in profile["models"]]}
    binding={"files":files,"numeric_configuration":numeric}
    version=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    run_paths=[Path(f"records/models_coordinated_{kind}_run.json") for kind in ("question","document1","document2")]
    runs=[json.loads(p.read_text()) for p in run_paths if p.exists()]
    valid_runs=[r for r in runs if r.get("numeric_profile")==profile["profile_version"]]
    embedding_ok=len(valid_runs)==3 and all(r["eligible"]==1 and not r["numeric_failure"] for r in valid_runs)
    gpu_run_paths=[Path(f"records/models_coordinated_cuda_{kind}_run.json") for kind in ("question","document1","document2")]
    gpu_runs=[json.loads(p.read_text()) for p in gpu_run_paths if p.exists()]
    gpu_embedding_ok=(len(gpu_runs)==3 and all(r.get("numeric_profile")==profile["profile_version"] and
                      r.get("device_name")=="NVIDIA GeForce RTX 2060" and r.get("eligible")==1 and
                      not r.get("numeric_failure",True) and r.get("model_CPU_fallback") is False for r in gpu_runs))
    failures=[r for r in valid_runs if r.get("numeric_failure")]
    full_path=Path("records/models_coordinated_full_r1_cuda_rag_run.json")
    full=json.loads(full_path.read_text()) if full_path.exists() else {}
    full_current=(full.get("numeric_profile_version")==profile["profile_version"] and
                  full.get("numeric_profile_sha256")==sha256(Path("artifacts/constants/numeric_profile_v1_qk.json")))
    full_complete=bool(full_current and full.get("fixed_generation_executed") and
                       full.get("start_to_finish_identity_unchanged") is True and
                       full.get("device_name")=="NVIDIA GeForce RTX 2060" and
                       full.get("model_CPU_fallback") is False and
                       full.get("fixed_generation_prefill_calls")==1 and
                       full.get("fixed_generation_decode_calls")==255 and
                       len(full.get("tokens",[]))==len(full.get("validity",[]))==256)
    full_normal=bool(full_complete and not full.get("numeric_failure",True) and
                     not full.get("insufficient_evidence",True) and full.get("first_failure") is None)
    if full_current and full.get("numeric_failure"):failures.append(full)
    operator_path=Path("records/models_cuda_operator_development_run.json")
    operator_run=json.loads(operator_path.read_text()) if operator_path.exists() else {}
    operator_summary=({"record":str(operator_path),"record_sha256":sha256(operator_path),
                       "count":len(operator_run.get("checks",[])),
                       "all_bitwise_equal":operator_run.get("all_bitwise_equal",False),
                       "start_to_finish_identity_unchanged":operator_run.get("start_to_finish_identity_unchanged",False),
                       "device_name":operator_run.get("device_name"),
                       "scope":"Small public independent operator checks only, not model/R2 execution or formal acceptance"}
                      if operator_run else None)
    activation=("blocked_current_numeric_domain_conflict" if failures else
                "R1_full_reference_checked_R2_not_verified" if full_normal else
                "R1_GPU_embeddings_checked_R2_not_verified" if gpu_embedding_ok else
                "R1_embedding_checked_R2_not_verified" if embedding_ok else
                "current_profile_R1_validation_pending_R2_not_verified")
    record={"format_version":"model-numeric-manifest-v1-coordinated","model_artifact_set_sha256":version,**binding,
            "global_configuration_binding":"artifacts/configuration.json owns the global configuration digest; this manifest intentionally does not embed or hash it, avoiding a cyclic dependency",
            "materialization_status":"materialized","activation_status":activation,
            "current_blocking_rule":"NUM-003" if failures else None,
            "current_first_failure":failures[0].get("first_failure") if failures else None,
            "historical_swiglu_conflict":"records/models_qk_revision_mlp_domain_conflict.json",
            "historical_original_qk_conflict":"records/models_numeric_domain_conflict.json",
            "approved_qk_change_applied":True,
            "approved_swiglu_change_applied":True,"swiglu_evidence":"artifacts/constants/swiglu_domain_evidence_v1.json",
            "authorized_coordinated_revision_applied":True,"coordinated_evidence":"artifacts/constants/coordinated_domain_evidence_v1.json",
            "full_R0_reference_execution_verified":True,"full_R0_record":"records/models_r0_full_rag_run.json",
            "full_R1_current_profile_embedding_runs":[str(p) for p in run_paths if p.exists()],
            "current_R1_embedding_normal_output_verified":embedding_ok,
            "current_R1_GPU_embedding_normal_output_verified":gpu_embedding_ok,
            "current_R1_GPU_embedding_runs":[str(p) for p in gpu_run_paths if p.exists()],
            "GPU_operator_development_run":operator_summary,
            "GPU_first_full_attempt_memory_failure_record":"records/models_full_r1_cuda_attempt1_oom.json",
            "GPU_memory_fix":"Replaced a 57-times expanded log2 threshold tensor with six integer shift stages; mathematical log2/floor semantics unchanged",
            "full_R1_current_profile_RAG_execution_verified":full_complete,
            "current_R1_full_RAG_normal_output_verified":full_normal,
            "full_R1_current_profile_RAG_record":str(full_path) if full_current else None,
            "full_R1_current_profile_RAG_record_sha256":sha256(full_path) if full_current else None,
            "GPU_full_completion_review":"records/models_cuda_full_completion_review.json" if full_complete else None,
            "GPU_full_public_answer_record":"records/models_cuda_full_public_answer.json" if full_complete else None,
            "runtime_result_scope":"Public independent reference execution only; no semantic quality assessment, formal acceptance or full R2 execution",
            "local_model_resource_policy":{"CPU_large_model_inference_permitted":False,
                 "R0_requires_actual_CUDA_device":True,"R0_CPU_fallback_permitted":False,
                 "R1_GPU_integer_backend_implemented":True,"R1_full_model_entry_enabled":True,
                 "required_GPU":"exactly one NVIDIA RTX 2060; actual CUDA allocation must succeed",
                 "GPU_operator_execution_record":"records/models_cuda_operator_development_run.json" if Path("records/models_cuda_operator_development_run.json").exists() else None,
                 "R1_GPU_full_model_execution_verified":full_normal,
                 "interrupted_run_record":"records/models_coordinated_full_r1_rag_attempt2_interrupted.json"},
            "full_R2_model_execution_verified":False}
    Path("artifacts/models/model_numeric_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"model_artifact_set_sha256":version,"activation_status":record["activation_status"],
                      "numeric_profile_version":profile["profile_version"]}))


if __name__=="__main__":build()
