"""Copy public materialized metadata into the standalone specification package.

No model weights, private inputs, certificates, keys or databases are copied.
This is an implementation record, not activation or acceptance.
"""
from pathlib import Path
import hashlib
import json
import platform
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT.parent / "项目文档/第一版机读规范"
OUT = SPEC / "实施工件"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def component(relative):
    raw = (ROOT / relative).read_bytes()
    content = json.loads(raw) if relative.endswith(".json") else raw.decode()
    return {"name": Path(relative).name, "sha256": digest(raw), "content": content}


def main():
    OUT.mkdir(exist_ok=True)
    groups = {
        "ART-001": ("template.json", "materialized", ["artifacts/template/template_v1.json", "artifacts/template/native_manifest.json"]),
        "ART-003": ("interfaces.json", "partially_materialized", ["artifacts/interface_delivery.json", "records/compute_process.md"]),
        "ART-004": ("dependencies.json", "materialized_current_host", ["requirements.lock", "pyproject.toml", "records/crypto-dependencies.json",
            "records/cuda_environment.json", "records/cuda_uvm_recovery_run.json", "records/private_model_device_policy.json",
            "records/compute_batch111_build.json"]),
        "ART-006": ("models.json", "materialized_coordinated_domains_full_R2_not_run", [
            "artifacts/models/model_numeric_manifest.json",
            "artifacts/models/embedding/checkpoint_manifest.json", "artifacts/models/embedding/quantized_f12/manifest.json",
            "artifacts/models/generation/checkpoint_manifest.json", "artifacts/models/generation/quantized_f12/manifest.json",
            "artifacts/constants/native_manifest.json", "artifacts/constants/numeric_profile_v1_qk.json",
            "artifacts/constants/range_evidence_v1_coordinated.json", "artifacts/constants/swiglu_domain_evidence_v1.json",
            "artifacts/constants/coordinated_domain_evidence_v1.json",
            "records/models_coordinated_question_run.json",
            "records/models_coordinated_document1_run.json", "records/models_coordinated_document2_run.json",
            "records/models_r0_full_rag_run.json",
            "records/models_coordinated_full_r1_rag_attempt1_interrupted.json",
            "records/models_coordinated_full_r1_rag_attempt2_interrupted.json",
            "records/models_cuda_backend_static_evidence.json",
            "records/models_cuda_operator_development_run.json",
            "records/models_coordinated_cuda_question_run.json",
            "records/models_coordinated_cuda_document1_run.json", "records/models_coordinated_cuda_document2_run.json",
            "records/models_cuda_embedding_postrun_identity_review.json",
            "records/r1_cuda_independent_review.json", "records/models_cuda_lifecycle_review.json",
            "records/models_full_r1_cuda_attempt1_oom.json",
            "records/models_cuda_full_completion_review.json", "records/models_cuda_full_public_answer.json",
            "records/models_r0_generation_prefill_observation.json"]),
    }
    groups["ART-002"] = ("bfv.json", "materialized_conditional_correctness", [
        "artifacts/bfv_context.json", "artifacts/bfv_correctness_bound.json", "artifacts/bfv_native_manifest.json",
        "records/crypto-bfv-correctness.md", "records/crypto_pack109_implementation.md",
        "records/pack109_layout_check.json", "records/crypto_static_cost_v1_0_9.json",
        "records/input109_v108_baseline.json", "records/compute_pack109_build.json",
        "records/pack109_diagonal_production.json", "records/pack109_long_mtls.json",
        "records/pack109_production_failure_runs.json", "records/pack109_production_rms_run.json",
        "records/pack109_token_production_H1024_ids0_8191_8192_155647.json",
        "records/pack109_token_production_H2048_ids151935_155647.json"])
    groups["ART-002"][2].extend(f"records/pack109_production_{shape}.json" for shape in
        ("3x17x7_m0","3x17x7_m1","16x17x33_m0","1x2048x4096_m2","1x2048x12288_m3",
         "16x2048x2048_m0","16x2048x6144_m0","16x6144x2048_m0"))
    groups["ART-002"][2].extend(["records/pack109_token_phase_mtls.json", "records/pack109_rms_localization.json", "records/pack109_integration_review.json"])
    groups["ART-002"][2].extend(["records/batch111_implementation.md", "records/batch111_stage.json",
        "records/compute_batch111_build.json", "records/batch110_baseline_profile.json",
        "records/matrix_triple_bfv_candidate.md", "records/batch111_cost_scope.json"])
    # This manifest validates old execution sources in their own archives and
    # distinguishes active registrations from completed model milestones.
    if (ROOT / 'artifacts/hybrid112_evidence.json').is_file():
        groups['ART-002'][2].extend(['artifacts/hybrid112_evidence.json', 'records/hybrid112_implementation.md'])
    if (ROOT / 'records/hybrid112_M2_closure.json').is_file():
        groups['ART-002'][2].extend(['records/hybrid112_M2_closure.json', 'records/hybrid112_M2_closure.md'])
    if (ROOT / 'records/hybrid112_materialization.json').is_file():
        groups['ART-002'][2].append('records/hybrid112_materialization.json')
        groups['ART-004'][2].extend(['records/hybrid112_production_build.json',
                                    'records/hybrid112_actual_context.json'])
    if (ROOT / 'records/hybrid112_fresh_build_entry.json').is_file():
        groups['ART-004'][2].append('records/hybrid112_fresh_build_entry.json')
    if (ROOT / 'records/hybrid112_registered_development_build.json').is_file():
        groups['ART-004'][2].append('records/hybrid112_registered_development_build.json')
    # A partial progress file is never registered as a completed full run.
    full_cuda_run = "records/models_coordinated_full_r1_cuda_rag_run.json"
    if (ROOT / full_cuda_run).is_file():
        groups["ART-006"][2].append(full_cuda_run)
    registry_path = SPEC / "决策与工件.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    for entry in registry["artifacts"]:
        filename, status, paths = groups[entry["id"]]
        blob = {"artifact": entry["id"], "status": status, "production_enabled": False,
                "python_version": platform.python_version(),
                "contents": [component(p) for p in paths],
                "boundary": "Only public definitions and evidence are bundled; runtime weights and libraries remain separate. No full R2 success or formal acceptance claim."}
        raw = (json.dumps(blob, ensure_ascii=False, indent=2) + "\n").encode()
        (OUT / filename).write_bytes(raw)
        entry.update(status=status, path="实施工件/" + filename, sha256=digest(raw))
    registry_path.write_text(yaml.safe_dump(registry, allow_unicode=True, sort_keys=False, width=110))
    profile = ROOT / "artifacts/constants/numeric_profile_v1_qk.json"
    (OUT / profile.name).write_bytes(profile.read_bytes())
    evidence = ROOT / "artifacts/constants/swiglu_domain_evidence_v1.json"
    (OUT / evidence.name).write_bytes(evidence.read_bytes())
    coordinated = ROOT / "artifacts/constants/coordinated_domain_evidence_v1.json"
    (OUT / coordinated.name).write_bytes(coordinated.read_bytes())
    # These fields describe the actually adopted profile, not a proposal.
    params_path = SPEC / "参数与接口.yaml"
    params = yaml.safe_load(params_path.read_text())
    numeric = params["numeric_pipeline"]
    numeric["profile_version"] = json.loads(profile.read_bytes())["profile_id"]
    numeric["range_policy"]["qk_local_domains"]["sha256"] = digest(profile.read_bytes())
    numeric["range_policy"]["swiglu_local_domain"]["profile_sha256"] = digest(profile.read_bytes())
    numeric["range_policy"]["swiglu_local_domain"]["evidence_sha256"] = digest(evidence.read_bytes())
    numeric["range_policy"]["coordinated_domains"]["profile_sha256"] = digest(profile.read_bytes())
    numeric["range_policy"]["coordinated_domains"]["evidence_sha256"] = digest(coordinated.read_bytes())
    numeric["range_policy"]["coordinated_domains"]["evidence_artifact"] = "实施工件/" + coordinated.name
    # Preserve the surrounding readable YAML instead of reformatting all
    # model/interface declarations for these implementation metadata changes.
    text = params_path.read_text()
    start = text.index("numeric_pipeline:\n")
    end = text.index("\nnonlinear_algorithms:\n", start)
    block = yaml.safe_dump({"numeric_pipeline": numeric}, allow_unicode=True, sort_keys=False, width=110)
    params_path.write_text(text[:start] + block + text[end:])
    print(json.dumps({"copied_artifact_registrations": len(groups), "production_enabled": False}))


if __name__ == "__main__":
    main()
