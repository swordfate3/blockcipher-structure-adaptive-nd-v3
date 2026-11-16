"""Injected platform readings only; no real RSS, CUDA or nvidia-smi execution."""
import ctypes
import sys
from typing import Any

import pytest
from cipher_lab_plugins.differential_relations import resources
from cipher_lab_plugins.differential_relations.resources import (
    GPUProbe,
    ResourceError,
    rss_bytes,
    validate_device_binding,
)

from cipher_lab.execution import Stopped

# The driver only selects a card for the GPU- prefix plus lower-case hex; the
# canonical UUID and CUDA_VISIBLE_DEVICES keep the upper-case form.
UUID = 'GPU-12345678-9ABC-DEF0-1234-56789ABCDEF0'
UUID_SELECTOR = 'GPU-12345678-9abc-def0-1234-56789abcdef0'
PCI4 = '0000:1a:2b.3'
PCI8 = '00000000:1A:2B.3'
QUERY = ['nvidia-smi', '--query-gpu=uuid,pci.bus_id,memory.used',
         '--format=csv,noheader,nounits', '-i', UUID_SELECTOR]


def test_linux_rss_delegates_to_production_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr('cipher_lab.execution.production_rss_bytes', lambda: 4321)
    assert rss_bytes() == 4321


@pytest.mark.parametrize('value', [-1, True, 1.5, '12'])
def test_linux_rss_rejects_non_byte_results(monkeypatch: pytest.MonkeyPatch,
                                            value: object) -> None:
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr('cipher_lab.execution.production_rss_bytes', lambda: value)
    with pytest.raises(Stopped) as caught:
        rss_bytes()
    assert caught.value.reason == 'RSS_UNAVAILABLE'


def test_linux_rss_preserves_unavailable_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'linux')

    def broken() -> int:
        raise RuntimeError('private monitor detail')
    monkeypatch.setattr('cipher_lab.execution.production_rss_bytes', broken)
    with pytest.raises(Stopped) as caught:
        rss_bytes()
    assert caught.value.reason == 'RSS_UNAVAILABLE'
    assert 'private' not in str(caught.value)


def test_unsupported_platform_is_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with pytest.raises(Stopped) as caught:
        rss_bytes()
    assert caught.value.reason == 'RSS_UNSUPPORTED'


def test_windows_rss_uses_working_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(resources, '_windows_api', lambda: (object(), object()))
    monkeypatch.setattr(resources, '_windows_working_set', lambda kernel, psapi: 87654)
    assert rss_bytes() == 87654


@pytest.mark.parametrize('outcome', ['error', 'bad_type'])
def test_windows_rss_failures_are_sanitized(monkeypatch: pytest.MonkeyPatch,
                                            outcome: str) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(resources, '_windows_api', lambda: (object(), object()))

    def broken(kernel: object, psapi: object) -> object:
        if outcome == 'error':
            raise RuntimeError('private Windows detail')
        return True
    monkeypatch.setattr(resources, '_windows_working_set', broken)
    with pytest.raises(Stopped) as caught:
        rss_bytes()
    assert caught.value.reason == 'RSS_UNAVAILABLE'
    assert 'private' not in str(caught.value)


def test_windows_working_set_structure_and_success() -> None:
    class Kernel:
        def __init__(self) -> None:
            self.GetCurrentProcess = lambda: 7

    def memory_info(handle: object, pointer: Any, size: int) -> int:
        assert handle == 7
        assert size == ctypes.sizeof(resources._ProcessMemoryCounters)
        counters = ctypes.cast(
            pointer, ctypes.POINTER(resources._ProcessMemoryCounters)).contents
        assert counters.cb == size
        counters.WorkingSetSize = 54321
        return 1

    class Psapi:
        def __init__(self) -> None:
            self.GetProcessMemoryInfo = memory_info

    assert resources._windows_working_set(Kernel(), Psapi()) == 54321


@pytest.mark.parametrize('outcome', ['no_handle', 'failed_call'])
def test_windows_working_set_failures(outcome: str) -> None:
    class Kernel:
        def __init__(self) -> None:
            self.GetCurrentProcess = lambda: 0 if outcome == 'no_handle' else 7

    class Psapi:
        def __init__(self) -> None:
            self.GetProcessMemoryInfo = lambda handle, pointer, size: 0

    with pytest.raises(OSError):
        resources._windows_working_set(Kernel(), Psapi())


@pytest.mark.parametrize(('uuid', 'pci', 'runner'), [
    ('GPU-123', PCI4, None),
    (UUID, '1a:2b.3', None),
    (UUID, '12340000:af:00.0', None),
    (UUID, PCI4, object()),
])
def test_gpu_probe_rejects_invalid_parameters(uuid: str, pci: str,
                                              runner: object) -> None:
    with pytest.raises(ValueError):
        GPUProbe(uuid, pci, runner=runner)  # type: ignore[arg-type]


def test_gpu_probe_query_identity_and_limits() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str]) -> str:
        calls.append(list(args))
        return f'GPU-{UUID[4:].lower()}, {PCI8}, 1024\n'

    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError('default external runner executed')
    original = resources._nvidia_smi_runner
    resources._nvidia_smi_runner = forbidden
    try:
        probe = GPUProbe(f'GPU-{UUID[4:].lower()}', PCI4, runner=runner)
        assert probe.check(2**30) == {'uuid': UUID, 'pci': '00000000:1a:2b.3',
                                      'used_bytes': 1024 * 2**20}
    finally:
        resources._nvidia_smi_runner = original
    assert calls == [QUERY]


def test_gpu_probe_pci_zero_extension_matches_both_identities() -> None:
    probe = GPUProbe(UUID, PCI8, runner=lambda args: f'{UUID}, {PCI4}, 0\n')
    assert probe.check(0)['pci'] == '00000000:1a:2b.3'


def test_gpu_probe_selects_driver_card_by_lowercase_hex_selector() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str]) -> str:
        calls.append(list(args))
        if args[-1] != UUID_SELECTOR:
            # Observed field behaviour: the upper-case hex selector makes
            # nvidia-smi report "No devices found" (exit 6) for a visible card.
            raise RuntimeError('No devices found')
        return f'GPU-{UUID[4:].lower()}, {PCI4}, 129\n'

    probe = GPUProbe(UUID, PCI8, runner=runner)
    assert probe.check(2**30) == {'uuid': UUID, 'pci': '00000000:1a:2b.3',
                                  'used_bytes': 129 * 2**20}
    assert calls == [QUERY]


def test_gpu_probe_mixed_case_row_must_still_match_exact_identity() -> None:
    other_card = UUID_SELECTOR[:-1] + '1'
    probe = GPUProbe(UUID, PCI8, runner=lambda args: f'{other_card}, {PCI8}, 1\n')
    with pytest.raises(ResourceError) as caught:
        probe.check(4096)
    assert caught.value.reason == 'GPU_UNAVAILABLE'


@pytest.mark.parametrize('output', [
    f'{UUID}, {PCI8}, N/A\n',
    f'{UUID}, {PCI8}\n',
    f'{UUID}, {PCI8}, 1\n{UUID}, {PCI8}, 1\n',
    f'GPU-00000000-0000-0000-0000-000000000000, {PCI8}, 1\n',
    f'{UUID}, 00000000:1a:2b.4, 1\n',
    f'{UUID}, 12340000:af:00.0, 1\n',
    f'{UUID}, {PCI8}, -1\n',
    f'{UUID}, {PCI8}, 1.5\n',
    f'{UUID}, {PCI8}, inf\n',
    '',
])
def test_gpu_probe_bad_rows_stop(output: str) -> None:
    probe = GPUProbe(UUID, PCI8, runner=lambda args: output)
    with pytest.raises(ResourceError) as caught:
        probe.check(4096)
    assert caught.value.reason == 'GPU_UNAVAILABLE'


def test_gpu_probe_runner_failure_is_sanitized() -> None:
    def broken(args: list[str]) -> str:
        raise RuntimeError('private nvidia-smi detail')
    probe = GPUProbe(UUID, PCI8, runner=broken)
    with pytest.raises(ResourceError) as caught:
        probe.check(4096)
    assert caught.value.reason == 'GPU_UNAVAILABLE'
    assert 'private' not in str(caught.value)


@pytest.mark.parametrize(('used_mib', 'limit'), [(4097, 4096), (20481, 2**40)])
def test_gpu_probe_limit_is_enforced(used_mib: int, limit: int) -> None:
    probe = GPUProbe(UUID, PCI8, runner=lambda args: f'{UUID}, {PCI8}, {used_mib}\n')
    with pytest.raises(ResourceError) as caught:
        probe.check(limit)
    assert caught.value.reason == 'GPU_LIMIT'


@pytest.mark.parametrize(('used_mib', 'limit', 'stops'), [
    (20480, 20 * 2**30, False),
    (20481, 2**40, True),
])
def test_gpu_probe_driver_mib_is_converted_to_bytes(
        used_mib: int, limit: int, stops: bool) -> None:
    probe = GPUProbe(UUID, PCI8, runner=lambda args: f'{UUID}, {PCI8}, {used_mib}\n')
    if stops:
        with pytest.raises(ResourceError) as caught:
            probe.check(limit)
        assert caught.value.reason == 'GPU_LIMIT'
    else:
        assert probe.check(limit)['used_bytes'] == used_mib * 2**20


@pytest.mark.parametrize('limit', [-1, True, 1.5])
def test_gpu_probe_rejects_invalid_limits(limit: object) -> None:
    probe = GPUProbe(UUID, PCI8, runner=lambda args: f'{UUID}, {PCI8}, 0\n')
    with pytest.raises(ValueError):
        probe.check(limit)  # type: ignore[arg-type]


def test_device_binding_requires_exact_uuid_and_logical_zero() -> None:
    environ = {'CUDA_VISIBLE_DEVICES': UUID}
    validate_device_binding('cuda:0', UUID, PCI4, environ)


@pytest.mark.parametrize(('device', 'environ'), [
    ('cuda:1', {'CUDA_VISIBLE_DEVICES': UUID}),
    ('cpu', {'CUDA_VISIBLE_DEVICES': UUID}),
    ('cuda:0', {}),
    ('cuda:0', {'CUDA_VISIBLE_DEVICES': '0'}),
    ('cuda:0', {'CUDA_VISIBLE_DEVICES': UUID + ','}),
    ('cuda:0', {'CUDA_VISIBLE_DEVICES': 'GPU-00000000-0000-0000-0000-000000000000'}),
    ('cuda:0', {'CUDA_VISIBLE_DEVICES': f'GPU-{UUID[4:].lower()}'}),
])
def test_device_binding_mismatch_stops(device: str,
                                       environ: dict[str, str]) -> None:
    with pytest.raises(ResourceError) as caught:
        validate_device_binding(device, UUID, PCI4, environ)
    assert caught.value.reason == 'DEVICE_BINDING'


def test_device_binding_bad_parameters() -> None:
    with pytest.raises(ValueError):
        validate_device_binding(1, UUID, PCI4, {})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_device_binding('cuda:0', 'GPU-123', PCI4, {})
    with pytest.raises(ValueError):
        validate_device_binding('cuda:0', UUID, '1a:2b.3', {})
    with pytest.raises(ValueError):
        validate_device_binding('cuda:0', UUID, PCI4, [])  # type: ignore[arg-type]
