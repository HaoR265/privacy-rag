"""Bind optional native GPU operators before creating any protocol session."""
from contextlib import contextmanager
import os
import re
import subprocess


def canonical_pci(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    if not match or int(match[1], 16) > 0xffff:
        raise ValueError('INVALID_GPU_PCI')
    return f'{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}'


def resolve_binding(build, uuid, pci, inventory, environ):
    prefixes = [prefix for option, prefix in
                (('silu_gpu', 'PRAG_SILU_GPU'), ('enc0_gpu', 'PRAG_ENC0_GPU'),
                 ('client_phase_gpu', 'PRAG_ENC0_GPU'))
                if build.get(option) is True]
    prefixes = list(dict.fromkeys(prefixes))
    if not prefixes:
        return {}
    if not uuid or not pci:
        raise ValueError('EXPLICIT_GPU_UUID_AND_PCI_REQUIRED')
    pci = canonical_pci(pci)
    devices = []
    for line in inventory.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) != 3:
            raise ValueError('INVALID_GPU_INVENTORY')
        devices.append((fields[0], canonical_pci(fields[1]), fields[2]))
    matches = [d for d in devices if d[0] == uuid and d[1] == pci and 'RTX 2060' in d[2]]
    if len(matches) != 1:
        raise ValueError('RTX2060_BINDING_NOT_FOUND_OR_AMBIGUOUS')
    selected = {}
    for prefix in prefixes:
        for suffix, value in (('_UUID', uuid), ('_PCI', pci)):
            key = prefix + suffix
            existing = environ.get(key)
            if existing is not None:
                normalized = canonical_pci(existing) if suffix == '_PCI' else existing
                if normalized != value:
                    raise ValueError('GPU_ENVIRONMENT_CONFLICT_' + key)
            selected[key] = value
    return selected


@contextmanager
def operator_device_binding(build, uuid, pci):
    required = any(build.get(k) is True for k in ('silu_gpu', 'enc0_gpu', 'client_phase_gpu'))
    inventory = ''
    if required:
        if not uuid or not pci:
            raise ValueError('EXPLICIT_GPU_UUID_AND_PCI_REQUIRED')
        inventory = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,pci.bus_id,name', '--format=csv,noheader'],
            check=True, capture_output=True, text=True, timeout=10).stdout
    selected = resolve_binding(build, uuid, pci, inventory, os.environ)
    old = {key: os.environ.get(key) for key in selected}
    try:
        os.environ.update(selected)
        yield {'operator_environment': selected, 'inventory': inventory.strip(),
               'native_device_validation_required': required,
               'CPU_fallback_permitted': False}
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
