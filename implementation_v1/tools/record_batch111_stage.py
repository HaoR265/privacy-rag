"""Verify final public bindings and save a resumable kernel-stage record."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    manifest = read('artifacts/bfv_native_manifest.json')
    for path, expected in manifest['files'].items():
        assert sha(ROOT / path) == expected, path
    configuration = (ROOT / 'artifacts/configuration.sha256').read_text().strip()
    assert sha(ROOT / 'artifacts/configuration.json') == configuration
    bindings = read('records/configuration_bindings_debug.json')
    process = read('records/compute_process_debug.json')
    assert bindings['configuration_sha256'] == process['configuration_sha256'] == configuration
    assert bindings['server_all_files_valid'] and bindings['client_without_server_weights_valid']
    spec = read('records/spec_revision_manifest.json')
    assert not spec['machine_package_consistency']['errors']
    package = ROOT.parent / '项目文档/第一版机读规范'
    for item in spec['machine_files']:
        assert sha(package / item['name']) == item['sha256'], item['name']
    stage = read('records/batch111_stage.json')
    assert stage['matrix_preprocessing_integrated'] and not stage['full_model_R2']
    assert len(stage['rms_runs']) == 9 and all(c['complete'] for c in stage['attention_runs'])
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            executable = (proc / 'exe').resolve(strict=True)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(executable).startswith(str(ROOT) + '/') and executable.name.startswith('prag_'):
            active.append({'pid': int(proc.name), 'executable': str(executable)})
    assert not active, active
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True)
    records = ['records/batch111_stage.json', 'records/compute_batch111_build.json', 'records/batch111_implementation.md',
               'records/spec_revision_manifest.json', 'records/configuration_bindings_debug.json',
               'records/compute_process_debug.json', 'records/batch111_cost_scope.json',
               'README.md', 'records/implementation_progress.md', 'records/engineering_decisions.md']
    result = {'kind': 'reusable_kernel_stage_completion_not_full_system', 'native_version': '1.0.11',
              'recorded_at': datetime.now(timezone.utc).isoformat(), 'configuration_sha256': configuration,
              'manifest_files_verified': len(manifest['files']),
              'server_files_verified': bindings['checked_files_server'], 'client_files_verified': bindings['checked_files_client'],
              'machine_specification_files': len(spec['machine_files']), 'rules_checked': spec['machine_package_consistency']['rules'],
              'machine_originals_updated': True, 'specification_errors': [],
              'complete_RMS_runs': 9, 'complete_attention_head_runs': 4, 'primitive_cases': 29, 'failure_modes': 13,
              'full_layer_R2': False, 'full_Decode_R2': False, 'full_model_R2': False, 'private_GPU_backend': False,
              'CPU_full_model_run': False, 'formal_acceptance_defined': False,
              'owned_protocol_processes_remaining': active, 'observed_GPU_compute_processes': gpu.stdout.splitlines(),
              'temporary_PKI_and_reference_cleanup': 'TemporaryDirectory scopes exited after each diagnostic',
              'MCP_browser_or_VM_resources_created': False,
              'next_work': ['reduce measured Softmax exp/LUT communication with an equivalent shared circuit',
                            'batch remaining SiLU/RoPE/L2 and residual/range paths',
                            'implement actual private RTX2060 backend before any full model execution',
                            'then verify full layer, Decode and fixed-budget system integration'],
              'evidence_sha256': {p: sha(ROOT / p) for p in records}}
    target = ROOT / 'records/crypto_batch111_stage_completion.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('native_version', 'configuration_sha256', 'manifest_files_verified',
        'server_files_verified', 'client_files_verified', 'machine_specification_files', 'rules_checked', 'full_model_R2')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
