# ruff: noqa: UP007, UP045
"""Current-PID DXGI LOCAL usage, not allocator or independent process accounting.

Import is inert. Each sample owns/releases its factory and adapter3 interfaces.
LUID is the eight Windows LUID bytes in memory order, lower-case hex (CUDA uses
that same byte layout). A description is bootstrap metadata, NEVER GPU1 proof.
"""
from __future__ import annotations

import ctypes as C
import os
import re
import subprocess
import uuid
from typing import Any, Callable

from cipher_lab.platform_resources.gpu import GpuBinding, GpuError, pci

MAX_ADAPTERS = 64
MAX_NODES = 32


def luid(value: Any) -> str:
    if type(value) is not str or not re.fullmatch('[0-9a-f]{16}', value):
        raise GpuError()
    if value == '0' * 16:
        raise GpuError()
    return value


def validate_rows(rows: Any) -> list[dict[str, Any]]:
    if type(rows) is not list or not 1 <= len(rows) <= MAX_ADAPTERS:
        raise GpuError()
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != {'luid', 'description', 'nodes'}:
            raise GpuError()
        identity = luid(row['luid'])
        if identity in seen:
            raise GpuError()
        seen.add(identity)
        if (type(row['description']) is not str or len(row['description']) > 128
                or type(row['nodes']) is not list or not 1 <= len(row['nodes']) <= MAX_NODES):
            raise GpuError()
        for index, node in enumerate(row['nodes']):
            if (type(node) is not dict or set(node) != {'node', 'current_usage'}
                    or type(node['node']) is not int or node['node'] != index
                    or type(node['current_usage']) is not int
                    or not 0 <= node['current_usage'] < 2**64):
                raise GpuError()
    return rows


class _Luid(C.Structure):
    _fields_ = [('low', C.c_uint32), ('high', C.c_int32)]


class _Desc(C.Structure):
    _fields_ = [('description', C.c_uint16 * 128), ('vendor', C.c_uint32),
                ('device', C.c_uint32), ('subsystem', C.c_uint32), ('revision', C.c_uint32),
                ('dedicated_video', C.c_size_t), ('dedicated_system', C.c_size_t),
                ('shared_system', C.c_size_t), ('luid', _Luid), ('flags', C.c_uint32)]


class _Memory(C.Structure):
    _fields_ = [('budget', C.c_uint64), ('current', C.c_uint64),
                ('available_reservation', C.c_uint64), ('current_reservation', C.c_uint64)]


class _Guid(C.Structure):
    _fields_ = [('data', C.c_ubyte * 16)]

    @classmethod
    def parse(cls, value: str) -> _Guid:
        result = cls()
        result.data[:] = uuid.UUID(value).bytes_le
        return result


class NativeDxgi:
    """Small ctypes COM ABI adapter; injected fakes never call an actual DLL.

    Physical-node count comes from D3DKMT, never inferred from query failure.
    """
    def __init__(self) -> None:
        if os.name != 'nt':
            raise GpuError()
        self.dll = getattr(C, 'WinDLL')('dxgi.dll')
        create = self.dll.CreateDXGIFactory1
        create.argtypes = [C.POINTER(_Guid), C.POINTER(C.c_void_p)]
        create.restype = C.c_int32
        self.factory = C.c_void_p()
        iid = _Guid.parse('1bc6ea02-ef36-464f-bf0c-21ca39e5168a')  # IDXGIFactory4
        self._ok(create(C.byref(iid), C.byref(self.factory)))
        if not self.factory.value:
            raise GpuError()

    @staticmethod
    def _ok(status: int) -> None:
        if status != 0:
            raise GpuError()

    @staticmethod
    def _call(pointer: Any, slot: int, result: Any, args: list[Any], *values: Any) -> Any:
        table = C.cast(pointer, C.POINTER(C.POINTER(C.c_void_p))).contents
        function = getattr(C, 'WINFUNCTYPE')(result, C.c_void_p, *args)(table[slot])
        return function(pointer, *values)

    def release(self, pointer: Any) -> None:
        # Release returns a reference count, NOT an HRESULT. Nonzero is legal.
        self._call(pointer, 2, C.c_uint32, [])

    def close(self) -> None:
        if self.factory.value:
            self.release(self.factory)
            self.factory = C.c_void_p()

    def adapters(self) -> list[Any]:
        adapters: list[Any] = []
        try:
            for index in range(MAX_ADAPTERS + 1):
                original = C.c_void_p()
                status = self._call(self.factory, 12, C.c_int32,
                                    [C.c_uint32, C.POINTER(C.c_void_p)], index, C.byref(original))
                if status & 0xffffffff == 0x887a0002:  # DXGI_ERROR_NOT_FOUND
                    return adapters
                self._ok(status)
                try:
                    if index == MAX_ADAPTERS or not original.value:
                        raise GpuError()
                    iid = _Guid.parse('645967a4-1392-4310-a798-8053ce3e93fd')  # Adapter3
                    adapter = C.c_void_p()
                    self._ok(self._call(original, 0, C.c_int32,
                                        [C.POINTER(_Guid), C.POINTER(C.c_void_p)],
                                        C.byref(iid), C.byref(adapter)))
                    if not adapter.value:
                        raise GpuError()
                    adapters.append(adapter)
                finally:
                    self.release(original)
        except BaseException:
            # Cleanup legs are independent; a failed Release dispatch cannot
            # skip the remaining acquired interfaces. Original error stays fatal.
            for adapter in adapters:
                try:
                    self.release(adapter)
                except BaseException:
                    pass
            raise
        raise GpuError()

    def description(self, adapter: Any) -> tuple[str, str]:
        desc = _Desc()
        self._ok(self._call(adapter, 10, C.c_int32, [C.POINTER(_Desc)], C.byref(desc)))
        raw = bytes(desc.description)
        text = raw.decode('utf-16-le').split('\0', 1)[0]
        return bytes(desc.luid).hex(), text

    def node_count(self, identity: str) -> int:
        return KmtAdapter(identity).information()[1]

    def query(self, adapter: Any, node: int) -> int:
        info = _Memory()
        status = self._call(adapter, 14, C.c_int32,
                            [C.c_uint32, C.c_int32, C.POINTER(_Memory)],
                            node, 0, C.byref(info))  # DXGI_MEMORY_SEGMENT_GROUP_LOCAL
        self._ok(status)
        return int(info.current)

    def by_luid(self, identity: str) -> tuple[str, str]:
        value = _Luid.from_buffer_copy(bytes.fromhex(luid(identity)))
        iid = _Guid.parse('645967a4-1392-4310-a798-8053ce3e93fd')
        adapter = C.c_void_p()
        self._ok(self._call(self.factory, 26, C.c_int32,
                            [_Luid, C.POINTER(_Guid), C.POINTER(C.c_void_p)],
                            value, C.byref(iid), C.byref(adapter)))
        if not adapter.value:
            raise GpuError()
        try:
            return self.description(adapter)
        finally:
            self.release(adapter)


class DxgiUsage:
    def __init__(self, *, api_factory: Callable[[], Any] = NativeDxgi) -> None:
        self.api_factory = api_factory

    def sample(self) -> list[dict[str, Any]]:
        api = self.api_factory()
        adapters: list[Any] = []
        failed = False
        rows: list[dict[str, Any]] = []
        try:
            adapters = api.adapters()
            if not 1 <= len(adapters) <= MAX_ADAPTERS:
                raise GpuError()
            for adapter in adapters:
                identity, description = api.description(adapter)
                nodes: list[dict[str, int]] = []
                count = api.node_count(identity)
                if type(count) is not int or not 1 <= count <= MAX_NODES:
                    raise GpuError()
                for node in range(count):
                    nodes.append(dict(node=node, current_usage=api.query(adapter, node)))
                rows.append(dict(luid=identity, description=description, nodes=nodes))
            validate_rows(rows)
        except Exception:
            failed = True
        finally:
            for adapter in adapters:
                try:
                    api.release(adapter)
                except Exception:
                    failed = True
            try:
                api.close()
            except Exception:
                failed = True
        if failed:
            raise GpuError()
        return rows


class _Open(C.Structure):
    _fields_ = [('luid', _Luid), ('handle', C.c_uint32)]


class _Query(C.Structure):
    _fields_ = [('handle', C.c_uint32), ('kind', C.c_int32),
                ('data', C.c_void_p), ('size', C.c_uint32)]


class _Address(C.Structure):
    _fields_ = [('bus', C.c_uint32), ('device', C.c_uint32), ('function', C.c_uint32)]


class KmtAdapter:
    """Read-only LUID->PCI bus/device/function and physical node count.

    No CUDA/D3D device/context creation. Nonzero PCI domains cannot be proved by
    D3DKMT_ADAPTERADDRESS and are deliberately blocked in NativeBindingResolver.
    """
    def __init__(self, identity: str, library: Any = None) -> None:
        self.identity = luid(identity)
        if library is None:
            if os.name != 'nt':
                raise GpuError()
            library = getattr(C, 'WinDLL')('gdi32.dll')
            for name, argument in [('D3DKMTOpenAdapterFromLuid', _Open),
                                   ('D3DKMTQueryAdapterInfo', _Query),
                                   ('D3DKMTCloseAdapter', C.c_uint32)]:
                function = getattr(library, name)
                function.argtypes, function.restype = [C.POINTER(argument)], C.c_int32
        self.lib = library

    def information(self) -> tuple[tuple[int, int, int], int]:
        opened = _Open(_Luid.from_buffer_copy(bytes.fromhex(self.identity)), 0)
        NativeDxgi._ok(self.lib.D3DKMTOpenAdapterFromLuid(C.byref(opened)))
        if not opened.handle:
            raise GpuError()
        try:
            address, count = _Address(), C.c_uint32()
            # SDK uses explicit values (obsolete enum slots are not in the
            # Learn syntax listing): ADAPTERADDRESS=6, PHYSICALADAPTERCOUNT=30.
            for kind, output in [(6, address), (30, count)]:
                query = _Query(opened.handle, kind, C.addressof(output), C.sizeof(output))
                NativeDxgi._ok(self.lib.D3DKMTQueryAdapterInfo(C.byref(query)))
            if not 1 <= count.value <= MAX_NODES:
                raise GpuError()
            return (address.bus, address.device, address.function), count.value
        finally:
            handle = C.c_uint32(opened.handle)
            NativeDxgi._ok(self.lib.D3DKMTCloseAdapter(C.byref(handle)))


class NativeBindingResolver:
    """Parent independent physical snapshot + DXGI/KMT LUID mapping; no cuInit.

    Does not use per-PID NVML memory. Single-node GPU1 only for scientific bind;
    linked/multi-node or nonzero-domain mapping remains GPU_UNVERIFIED.
    """
    def __init__(self, *, api_factory: Callable[[], Any] = NativeDxgi,
                 location: Callable[[str], Any] = lambda value: KmtAdapter(value).information(),
                 snapshot: Any = None) -> None:
        self.api_factory, self.location, self.snapshot = api_factory, location, snapshot

    def __call__(self, binding: GpuBinding) -> dict[str, Any]:
        if self.snapshot is None:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                     '--format=csv,noheader,nounits'], capture_output=True,
                                    text=True, check=True, timeout=5)
            snapshot: list[dict[str, Any]] = []
            for line in result.stdout.splitlines():
                fields = [f.strip() for f in line.split(',')]
                if len(fields) != 3:
                    raise GpuError()
                snapshot.append(dict(index=int(fields[0]), gpu_uuid=fields[1],
                                     pci_bus_id=fields[2]))
        else:
            snapshot = self.snapshot()
        if GpuBinding.from_snapshot(1, snapshot) != binding:
            raise GpuError()
        # Without the segment field from Windows, require domain zero for ALL
        # NVIDIA devices and unique bus/device/function to avoid an alias.
        buses = [pci(row['pci_bus_id']) for row in snapshot]
        if any(not value.startswith('0000:') for value in buses) or len(set(buses)) != len(buses):
            raise GpuError()
        bus = binding.pci_bus_id
        expected = (int(bus[5:7], 16), int(bus[8:10], 16), int(bus[11], 16))
        api = self.api_factory()
        adapters: list[Any] = []
        matches: list[str] = []
        try:
            adapters = api.adapters()
            for adapter in adapters:
                identity, _ = api.description(adapter)
                address, count = self.location(identity)
                if address == expected:
                    if count != 1 or api.by_luid(identity)[0] != identity:
                        raise GpuError()
                    matches.append(luid(identity))
            if len(matches) != 1:
                raise GpuError()
            return dict(luid=matches[0], device_node_mask=1)
        finally:
            failed = False
            for adapter in adapters:
                try:
                    api.release(adapter)
                except Exception:
                    failed = True
            try:
                api.close()
            except Exception:
                failed = True
            if failed:
                raise GpuError()
