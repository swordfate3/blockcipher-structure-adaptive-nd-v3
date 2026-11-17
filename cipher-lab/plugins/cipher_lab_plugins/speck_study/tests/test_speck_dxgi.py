"""Task5A fake COM/driver only: no DLL or CUDA initialization."""
from __future__ import annotations

from typing import Any

import pytest

from cipher_lab_plugins.speck_study.platform_resources.dxgi import DxgiUsage
from cipher_lab_plugins.speck_study.platform_resources.gpu import GpuError, actual_attestor
from cipher_lab_plugins.speck_study.tests.test_speck_resources import (  # noqa: F401
    Driver,
    resource_guard,
)

LUID = '0100000000000000'


class COM:
    def __init__(self) -> None:
        self.released: list[int] = []
        self.failed = False
        self.release_failed = False

    def adapters(self) -> list[int]:
        return [1, 2]

    def description(self, adapter: int) -> tuple[str, str]:
        return (LUID if adapter == 1 else '0200000000000000', 'same model')

    def node_count(self, identity: str) -> int:
        return 1

    def query(self, adapter: int, node: int) -> int | None:
        if self.failed:
            raise OSError('query failed')
        return 0 if node == 0 else None  # explicit successful zero, not missing

    def release(self, adapter: int) -> None:
        self.released.append(adapter)
        if self.release_failed:
            raise OSError('release failed')

    def close(self) -> None:
        pass


def test_bootstrap_all_adapters_explicit_zero_no_cuda() -> None:
    api = COM()
    rows = DxgiUsage(api_factory=lambda: api).sample()
    assert len(rows) == 2
    assert rows[0] == dict(luid=LUID, description='same model',
                           nodes=[dict(node=0, current_usage=0)])
    assert api.released == [1, 2]


@pytest.mark.parametrize('fault', ['query', 'release', 'missing', 'negative', 'overflow',
                                  'bool', 'duplicate', 'luid', 'nan'])
def test_dxgi_failure_never_missing_zero(fault: str) -> None:
    api = COM()
    if fault == 'query':
        api.failed = True
    elif fault == 'release':
        api.release_failed = True
    elif fault == 'duplicate':
        api.description = lambda adapter: (LUID, 'same')  # type: ignore[method-assign]
    elif fault == 'luid':
        api.description = lambda adapter: ('bogus', 'same')  # type: ignore[method-assign]
    else:
        value: Any = dict(missing=None, negative=-1, overflow=2**64, bool=True,
                          nan=float('nan'))[fault]
        api.query = lambda adapter, node: value  # type: ignore[method-assign]
    with pytest.raises(GpuError):
        DxgiUsage(api_factory=lambda: api).sample()
    assert sorted(api.released) == [1, 2]


class LuidDriver(Driver):
    def cuDeviceGetLuid(self, luid: Any, mask: Any, device: Any) -> int:
        luid[0] = b'\x01'
        mask._obj.value = 1
        return 0


def test_current_context_luid_no_init_or_gpu0_context_creation(monkeypatch: Any) -> None:
    import os
    from types import SimpleNamespace

    from cipher_lab_plugins.speck_study.platform_resources import gpu
    monkeypatch.setattr(gpu, 'os', SimpleNamespace(name='nt', getpid=os.getpid))
    evidence = actual_attestor(driver=LuidDriver())
    assert evidence['luid'] == LUID and evidence['device_node_mask'] == 1


def test_current_context_luid_failure_is_not_optional(monkeypatch: Any) -> None:
    import os
    from types import SimpleNamespace

    from cipher_lab_plugins.speck_study.platform_resources import gpu
    monkeypatch.setattr(gpu, 'os', SimpleNamespace(name='nt', getpid=os.getpid))
    class Bad(LuidDriver):
        def cuDeviceGetLuid(self, *args: Any) -> int:
            return 801
    with pytest.raises(GpuError):
        actual_attestor(driver=Bad())


def test_kmt_node_count_sdk_enum_not_render_or_miracast() -> None:
    import ctypes as C

    from cipher_lab_plugins.speck_study.platform_resources.dxgi import KmtAdapter, _Address
    class KMT:
        def __init__(self) -> None:
            self.closed = False
        def D3DKMTOpenAdapterFromLuid(self, ptr: Any) -> int:
            ptr._obj.handle = 42
            return 0
        def D3DKMTQueryAdapterInfo(self, ptr: Any) -> int:
            query = ptr._obj
            if query.kind == 6:
                output = C.cast(query.data, C.POINTER(_Address)).contents
                output.bus, output.device, output.function = 2, 0, 0
            else:
                assert query.kind == 30  # actual SDK explicit enum, docs list omits obsolete hole
                C.cast(query.data, C.POINTER(C.c_uint32)).contents.value = 1
            return 0
        def D3DKMTCloseAdapter(self, ptr: Any) -> int:
            assert ptr._obj.value == 42
            self.closed = True
            return 0
    api = KMT()
    assert KmtAdapter(LUID, api).information() == ((2, 0, 0), 1)
    assert api.closed


def test_parent_mapping_independent_pci_not_same_description() -> None:
    from cipher_lab_plugins.speck_study.platform_resources.dxgi import NativeBindingResolver
    from cipher_lab_plugins.speck_study.platform_resources.gpu import GpuBinding
    class Mapping(COM):
        def by_luid(self, identity: str) -> tuple[str, str]:
            return identity, 'same model'
    api = Mapping()
    def snapshot() -> list[dict[str, Any]]:
        return [dict(index=0, gpu_uuid='GPU-zero', pci_bus_id='0000:01:00.0'),
                dict(index=1, gpu_uuid='GPU-one', pci_bus_id='0000:02:00.0')]
    resolver = NativeBindingResolver(api_factory=lambda: api, snapshot=snapshot,
                                    location=lambda value: ((2 if value == LUID else 1, 0, 0), 1))
    assert resolver(GpuBinding(1, 'GPU-one', '0000:02:00.0')) == dict(
        luid=LUID, device_node_mask=1)
    with pytest.raises(GpuError):
        resolver(GpuBinding(1, 'GPU-zero', '0000:02:00.0'))


def test_linked_nodes_all_local_usage_not_just_node_zero() -> None:
    api = COM()
    api.node_count = lambda identity: 2  # type: ignore[method-assign]
    api.query = lambda adapter, node: 7 + node  # type: ignore[method-assign]
    rows = DxgiUsage(api_factory=lambda: api).sample()
    assert rows[0]['nodes'] == [dict(node=0, current_usage=7), dict(node=1, current_usage=8)]


def test_native_com_abi_slots_and_local_currentusage() -> None:
    import ctypes as C

    from cipher_lab_plugins.speck_study.platform_resources.dxgi import NativeDxgi, _Desc, _Memory
    api = object.__new__(NativeDxgi)
    api.factory = C.c_void_p(1)
    calls: list[int] = []
    def call(pointer: Any, slot: int, result: Any, args: Any, *values: Any) -> int:
        calls.append(slot)
        if slot == 12:
            if values[0] == 1:
                return -2005270526  # 0x887a0002 DXGI_ERROR_NOT_FOUND
            values[1]._obj.value = 2
        elif slot == 0:
            values[1]._obj.value = 3
        elif slot == 10:
            desc = values[0]._obj
            assert isinstance(desc, _Desc)
            desc.luid.low = 1
            desc.description[0] = ord('X')
        elif slot == 14:
            assert values[:2] == (0, 0)  # node0 LOCAL, not NON_LOCAL
            info = values[2]._obj
            assert isinstance(info, _Memory)
            info.budget, info.current, info.current_reservation = 999, 17, 888
        elif slot == 26:
            values[2]._obj.value = 3
        elif slot == 2:
            return 2  # Release reference count is not an HRESULT
        else:
            raise AssertionError(slot)
        return 0
    api._call = call  # type: ignore[method-assign]
    adapters = api.adapters()
    assert len(adapters) == 1
    assert api.description(adapters[0]) == (LUID, 'X')
    assert api.query(adapters[0], 0) == 17
    assert api.by_luid(LUID) == (LUID, 'X')
    api.release(adapters[0])
    api.close()
    assert 26 in calls and calls.count(2) == 4


@pytest.mark.parametrize('fault', ['query', 'close', 'open', 'count'])
def test_kmt_failures_release_and_no_synthetic_count(fault: str) -> None:
    import ctypes as C

    from cipher_lab_plugins.speck_study.platform_resources.dxgi import KmtAdapter
    class KMT:
        closed = False
        def D3DKMTOpenAdapterFromLuid(self, ptr: Any) -> int:
            ptr._obj.handle = 5
            return -1 if fault == 'open' else 0
        def D3DKMTQueryAdapterInfo(self, ptr: Any) -> int:
            if ptr._obj.kind == 30:
                C.cast(ptr._obj.data, C.POINTER(C.c_uint32)).contents.value = (
                    0 if fault == 'count' else 1)
            return -1 if fault == 'query' else 0
        def D3DKMTCloseAdapter(self, ptr: Any) -> int:
            self.closed = True
            return -1 if fault == 'close' else 0
    api = KMT()
    with pytest.raises(GpuError):
        KmtAdapter(LUID, api).information()
    assert api.closed is (fault != 'open')


@pytest.mark.parametrize('fault', ['domain', 'ambiguous', 'linked', 'lookup'])
def test_native_binding_mapping_unprovable_is_blocked(fault: str) -> None:
    from cipher_lab_plugins.speck_study.platform_resources.dxgi import NativeBindingResolver
    from cipher_lab_plugins.speck_study.platform_resources.gpu import GpuBinding
    class Mapping(COM):
        def by_luid(self, identity: str) -> tuple[str, str]:
            return ('0300000000000000' if fault == 'lookup' else identity), 'same model'
    def snapshot() -> list[dict[str, Any]]:
        return [dict(index=0, gpu_uuid='GPU-zero', pci_bus_id='0001:01:00.0' if fault == 'domain'
                     else '0000:01:00.0'),
                dict(index=1, gpu_uuid='GPU-one', pci_bus_id='0000:02:00.0')]
    resolver = NativeBindingResolver(api_factory=Mapping, snapshot=snapshot,
                                    location=lambda value: ((2 if value == LUID
                                                             or fault == 'ambiguous' else 1, 0, 0),
                                                            2 if fault == 'linked' else 1))
    with pytest.raises(GpuError):
        resolver(GpuBinding(1, 'GPU-one', '0000:02:00.0'))


def test_enum_failure_release_attempts_every_acquired_interface() -> None:
    import ctypes as C

    from cipher_lab_plugins.speck_study.platform_resources.dxgi import NativeDxgi
    api = object.__new__(NativeDxgi)
    api.factory = C.c_void_p(1)
    released: list[int] = []
    def call(pointer: Any, slot: int, result: Any, args: Any, *values: Any) -> int:
        if slot == 12:
            values[1]._obj.value = 10 + values[0]
            return 0
        if slot == 0:
            values[1]._obj.value = pointer.value + 10
            return 0
        if slot == 2:
            released.append(pointer.value)
            if pointer.value in (11, 20):
                raise OSError('fake COM dispatch failed')
            return 0
        raise AssertionError(slot)
    api._call = call  # type: ignore[method-assign]
    with pytest.raises((OSError, GpuError)):
        api.adapters()
    assert released == [10, 11, 20, 21]


@pytest.mark.parametrize('platform', ['posix', 'nt'])
def test_fix1_shared_attestor_luid_required_only_windows(platform: str,
                                                       monkeypatch: Any) -> None:
    import os
    from types import SimpleNamespace

    from cipher_lab_plugins.speck_study.platform_resources import gpu
    class NoLuid:
        cuCtxGetCurrent = Driver.cuCtxGetCurrent
        cuCtxGetDevice = Driver.cuCtxGetDevice
        cuDeviceGetUuid = Driver.cuDeviceGetUuid
        cuDeviceGetPCIBusId = Driver.cuDeviceGetPCIBusId
        current, device = 1, 0
    monkeypatch.setattr(gpu, 'os', SimpleNamespace(name=platform, getpid=os.getpid))
    if platform == 'posix':
        evidence = actual_attestor(driver=NoLuid())
        assert evidence['logical_device'] == 0
        assert evidence['pci_bus_id'] == '0000:02:00.0'
        assert evidence['gpu_uuid'] == 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f'
        assert 'luid' not in evidence and 'device_node_mask' not in evidence
    else:
        with pytest.raises(GpuError):
            actual_attestor(driver=NoLuid())


@pytest.mark.parametrize('platform', ['posix', 'nt'])
def test_fix1_native_driver_symbol_declarations_platform_scoped(platform: str,
                                                              monkeypatch: Any) -> None:
    import ctypes
    import os
    from types import SimpleNamespace

    from cipher_lab_plugins.speck_study.platform_resources import gpu
    driver = SimpleNamespace()
    base = Driver()
    for name in ('cuCtxGetCurrent', 'cuCtxGetDevice', 'cuDeviceGetUuid', 'cuDeviceGetPCIBusId'):
        method = getattr(base, name)
        setattr(driver, name, lambda *args, fn=method: fn(*args))
    monkeypatch.setattr(gpu, 'os', SimpleNamespace(name=platform, getpid=os.getpid))
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a: driver)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a: driver, raising=False)
    if platform == 'posix':
        assert actual_attestor()['logical_device'] == 0
        assert driver.cuCtxGetCurrent.restype is ctypes.c_int
    else:
        with pytest.raises(GpuError):
            actual_attestor()
