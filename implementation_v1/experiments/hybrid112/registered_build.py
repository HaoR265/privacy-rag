"""Bind future diagnostic executions to a completed fresh build and its sources.

This helper neither starts a process nor touches an existing execution. A
source change requires a fresh build, even when an older binary is still valid
for its own archived run.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def registered_build(directory, executable):
    build = (ROOT / directory).resolve()
    if not build.is_relative_to((ROOT / 'build').resolve()):
        raise RuntimeError('BUILD_OUTSIDE_PROJECT')
    record_path = build / 'build_record.json'
    record = json.loads(record_path.read_text())
    if (record.get('result') != 'compiled_context_matched_gate_closed' or
            record.get('development') is not True or record.get('model_executed') is not False or
            record.get('production_enabled') is not False):
        raise RuntimeError('REGISTERED_DEVELOPMENT_BUILD_REQUIRED')
    binary = build / executable
    key = str(binary.relative_to(ROOT))
    identities = {**record['source_and_dependency_sha256'], **record['output_sha256']}
    if key not in record['output_sha256']:
        raise RuntimeError('EXECUTABLE_NOT_REGISTERED')
    for name, expected in identities.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise RuntimeError('BUILD_SOURCE_OR_OUTPUT_CHANGED: ' + name)
    identities[str(record_path.relative_to(ROOT))] = sha(record_path)
    return binary, identities


def freeze(prefix, identities):
    archive = Path(str(prefix) + '.source.tar.gz')
    # Never replace the source identity of an earlier execution.
    with archive.open('xb') as output, tarfile.open(fileobj=output, mode='w:gz') as tar:
        for name, expected in sorted(identities.items()):
            path = ROOT / name
            if sha(path) != expected:
                raise RuntimeError('SOURCE_CHANGED_DURING_REGISTRATION: ' + name)
            tar.add(path, arcname=name, recursive=False)
    return archive


def add_sources(identities, sources):
    result = dict(identities)
    for name in sources:
        actual = sha(ROOT / name)
        if name in result and result[name] != actual:
            raise RuntimeError('COMPILED_SOURCE_CHANGED: ' + name)
        result[name] = actual
    return result


def copy_binary(binary, destination, expected):
    shutil.copy2(binary, destination)
    if sha(destination) != expected:
        raise RuntimeError('REGISTERED_BINARY_CHANGED_DURING_COPY')
    return destination
