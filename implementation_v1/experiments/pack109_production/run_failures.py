"""Build a separate diagnostic against the already formed matrix library."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid
from private_rag.pki import create_local_pki

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT/'records/pack109_production_failure_runs.json'
    if output.exists():
        raise RuntimeError('refusing to overwrite failure evidence')
    library = ROOT/'experiments/pack109_production/build/native/libprag_crypto.a'
    source = ROOT/'experiments/pack109_production/failure_probe.cpp'
    binary = ROOT/'experiments/pack109_production/build/prag_matrix_failure_probe'
    compile_command = ['c++', '-O3', '-std=c++17', '-maes', '-msse4.1', '-mavx', '-mavx2',
                       '-DEMP_USE_RANDOM_DEVICE=1', '-I'+str(ROOT/'native/include'),
                       '-I'+str(ROOT/'third_party/EzPC/SCI/src'), str(source), str(library),
                       str(ROOT/'third_party/SEAL/build/lib/libseal-4.1.a'),
                       '-lssl', '-lcrypto', '-lgmpxx', '-lgmp', '-o', str(binary)]
    subprocess.run(compile_command, check=True, capture_output=True)
    identities = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [source, library, binary, Path(__file__)]}
    cases = []
    with tempfile.TemporaryDirectory(prefix='prag-matrix-failure-') as temp:
        pki = Path(temp)/'pki';create_local_pki(pki)
        for mode in range(8):
            prefix = ROOT/'records'/f'pack109_production_failure_{mode}_{uuid.uuid4().hex}'
            run = subprocess.run(['nice', '-n', '10', str(binary), str(prefix), str(pki), str(mode)],
                                 capture_output=True, text=True, timeout=90)
            if run.returncode:
                raise RuntimeError(run.stderr)
            parties = []
            for party in ['C', 'S']:
                data = json.loads(Path(str(prefix)+f'.{party}.json').read_text())
                ledger = Path(str(prefix)+f'.{party}.ledger')
                terminal = ledger.read_text().splitlines()[-1]
                expected = 'closed' if mode in [2,7] else 'aborted'
                assert terminal == f'terminal={expected} all_reserved=burned'
                data.update(ledger_path=str(ledger.relative_to(ROOT)),
                            ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
                            terminal=terminal)
                parties.append(data)
            cases.append({'mode': ['bad_callback', 'descriptor_mismatch', 'public_shape_limit','joint_descriptor','token_descriptor','token_weight_bound','diagonal_descriptor','token_shape_limit'][mode],
                          'parties': parties, 'exit_code': run.returncode, 'stderr': run.stderr})
    report = {'native_version': '1.0.9', 'result': 'expected_failure_and_cleanup',
              'source_and_binary_sha256': identities, 'compile_command': compile_command,
              'cases': cases, 'temporary_PKI_removed': not Path(temp).exists(),
              'full_model_R2': False}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'result': report['result'], 'cases': len(cases)}))


if __name__ == '__main__':
    main()
