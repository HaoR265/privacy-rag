"""Capture small public artifact/source identities before reference execution."""
from pathlib import Path
import hashlib,json


def snapshot(entrypoint):
    paths=[Path(entrypoint),Path(__file__).resolve().relative_to(Path.cwd()),Path('private_rag/model_input.py')]
    paths+=sorted(Path('references').glob('*.py'))
    paths += [Path(p) for p in ['artifacts/constants/numeric_profile_v1_qk.json','artifacts/constants/constants_v1.json',
              'artifacts/constants/swiglu_domain_evidence_v1.json','artifacts/constants/coordinated_domain_evidence_v1.json',
              'artifacts/template/template_v1.json','artifacts/template/manifest.json']]
    for kind in ('embedding','generation'):
        root=Path('artifacts/models')/kind
        paths += [root/'quantized_f12/manifest.json',root/'checkpoint_manifest.json']
        # Hash tokenizer/config inputs; checkpoint tensors are represented by
        # their immutable manifest, and each quantized tensor is verified on load.
        paths+=sorted((root/'checkpoint').glob('*.json'))
        paths+=sorted((root/'checkpoint').glob('*.jinja'))
        paths+=sorted((root/'checkpoint').glob('*.txt'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def identify(entrypoint):
    files=snapshot(entrypoint)
    return {'captured_at':'before_model_execution','files_sha256':files,
            'snapshot_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def unchanged(identity,entrypoint):return identity['files_sha256']==snapshot(entrypoint)
