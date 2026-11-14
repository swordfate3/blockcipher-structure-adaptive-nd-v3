# ruff: noqa: UP007, UP045
"""Lazy standard-library CUDA/NVML observation. Import does NOT initialize CUDA.

WDDM unavailable process memory is NOT an allocator estimate: fail closed. No
unreliable Windows performance-counter/adapter heuristic is used as a fallback.
This module and the trusted provider wrapper are Python 3.9 source-safe.
"""
from __future__ import annotations

import ctypes as C
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Optional


class GpuError(RuntimeError):
    def __init__(self) -> None:
        super().__init__('GPU_UNVERIFIED')


def pci(value: str) -> str:
    match = re.fullmatch(
        r'(?:0000)?([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    if not match:
        raise GpuError()
    return ':'.join((match[1].lower(), match[2].lower(), match[3].lower())) + '.' + match[4]


@dataclass(frozen=True)
class GpuBinding:
    index: int
    gpu_uuid: str
    pci_bus_id: str

    @classmethod
    def from_snapshot(cls, index: int = 1,
                      snapshot: Optional[list[dict[str, Any]]] = None) -> GpuBinding:
        if type(index) is not int or index != 1:
            raise GpuError()
        if snapshot is None:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                timeout=5, check=True)
            snapshot = []
            for line in result.stdout.splitlines():
                fields = [f.strip() for f in line.split(',')]
                if len(fields) != 5:
                    raise GpuError()
                snapshot.append(dict(index=int(fields[0]), gpu_uuid=fields[1],
                                     pci_bus_id=fields[2], free_mib=int(fields[3]),
                                     utilization=int(fields[4])))
            selected = [v for v in snapshot if v['index'] == index]
            if (len(selected) != 1 or selected[0]['free_mib'] < 24 * 1024
                    or selected[0]['utilization'] > 30):
                raise GpuError()
        selected = [v for v in snapshot if v['index'] == index]
        if (len(selected) != 1 or len({v['gpu_uuid'] for v in snapshot}) != len(snapshot)
                or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', selected[0]['gpu_uuid'])):
            raise GpuError()
        return cls(index, selected[0]['gpu_uuid'], pci(selected[0]['pci_bus_id']))


def verify_runtime(binding: GpuBinding, identity: dict[str, Any]) -> None:
    if (identity.get('gpu_uuid') != binding.gpu_uuid
            or pci(identity.get('pci_bus_id', '')) != binding.pci_bus_id
            or type(identity.get('logical_device')) is not int
            or identity['logical_device'] != 0
            or type(identity.get('pid')) is not int
            or identity['pid'] <= 0
            or identity.get('evidence') != 'cuda-current-context'):
        raise GpuError()


def _ok(result: int) -> None:
    if result != 0:
        raise GpuError()


def actual_attestor(runtime: Any = None, *, driver: Any = None) -> dict[str, Any]:
    """Inspect the CURRENT context, never cuInit/retain/create a new context.

    TF may leave no current context on the calling thread; that is UNKNOWN and
    intentionally stops. Not equivalent to looking up ordinal0 and echoing env.
    """
    del runtime
    if driver is None:
        driver = (getattr(C, 'WinDLL')('nvcuda.dll') if os.name == 'nt'
                  else C.CDLL('libcuda.so.1'))
        signatures = {
            'cuCtxGetCurrent': [C.POINTER(C.c_void_p)],
            'cuCtxGetDevice': [C.POINTER(C.c_int)],
            'cuDeviceGetUuid': [C.c_void_p, C.c_int],
            'cuDeviceGetPCIBusId': [C.c_void_p, C.c_int, C.c_int],
        }
        if os.name == 'nt':
            signatures['cuDeviceGetLuid'] = [C.c_void_p, C.POINTER(C.c_uint), C.c_int]
        for name, arguments in signatures.items():
            try:
                function = getattr(driver, name)
            except AttributeError:
                raise GpuError() from None
            function.argtypes, function.restype = arguments, C.c_int
    context = C.c_void_p()
    _ok(driver.cuCtxGetCurrent(C.byref(context)))
    if not context.value:
        raise GpuError()
    device = C.c_int()
    _ok(driver.cuCtxGetDevice(C.byref(device)))
    # Runtime ordinal is independently obtained from the current context.
    if device.value != 0:
        raise GpuError()
    uuid = (C.c_ubyte * 16)()
    _ok(driver.cuDeviceGetUuid(C.byref(uuid), device))
    raw = bytes(uuid).hex()
    gpu_uuid = 'GPU-' + '-'.join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))
    bus = C.create_string_buffer(32)
    _ok(driver.cuDeviceGetPCIBusId(bus, len(bus), device))
    evidence = dict(gpu_uuid=gpu_uuid, pci_bus_id=pci(bus.value.decode('ascii')),
                    pid=os.getpid(), logical_device=0, evidence='cuda-current-context')
    # LUID is required by Windows DXGI only, not the shared POSIX/NVML path.
    if os.name == 'nt':
        local_id = C.create_string_buffer(8)
        node_mask = C.c_uint()
        try:
            _ok(driver.cuDeviceGetLuid(local_id, C.byref(node_mask), device))
        except AttributeError:
            raise GpuError() from None
        if not node_mask.value or bytes(local_id) == b'\0' * 8:
            raise GpuError()
        evidence.update(luid=bytes(local_id).hex(), device_node_mask=node_mask.value)
    return evidence


class Nvml:
    """Independent parent proof and actual per-PID memory, compute+graphics union.

    No process entry is not interpreted as zero. Missing/N/A memory is fatal;
    WDDM installations without reliable counters remain blocked by design.
    """
    def __init__(self, library: Any = None) -> None:
        self.lib = library if library is not None else C.CDLL(
            'nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
        _ok(self.lib.nvmlInit_v2())

    def _handle(self, binding: GpuBinding) -> C.c_void_p:
        handle = C.c_void_p()
        _ok(self.lib.nvmlDeviceGetHandleByUUID(binding.gpu_uuid.encode('ascii'), C.byref(handle)))
        return handle

    def processes(self, binding: GpuBinding) -> dict[int, Optional[int]]:
        class Info(C.Structure):
            _fields_ = [('pid', C.c_uint), ('used', C.c_ulonglong),
                        ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]
        handle = self._handle(binding)
        result: dict[int, Optional[int]] = {}
        for name in ('nvmlDeviceGetComputeRunningProcesses_v2',
                     'nvmlDeviceGetGraphicsRunningProcesses_v2'):
            count = C.c_uint(4096)
            rows = (Info * count.value)()
            _ok(getattr(self.lib, name)(handle, C.byref(count), rows))
            if count.value > 4096:
                raise GpuError()
            for row in rows[:count.value]:
                value = None if row.used == 2**64 - 1 else int(row.used)
                if row.pid in result and result[row.pid] != value:
                    raise GpuError()
                result[row.pid] = value
        return result

    def verify(self, evidence: dict[str, Any], pid: int, uuid: str, bus: str) -> bool:
        binding = GpuBinding(1, uuid, pci(bus))
        verify_runtime(binding, evidence)
        # Independent physical identity, not ready JSON/environment echoes.
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,pci.bus_id',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, check=True, timeout=5)
        matches = [f.strip().split(',') for f in result.stdout.splitlines()]
        if not any(len(f) == 2 and f[0].strip() == uuid and pci(f[1].strip()) == bus
                   for f in matches):
            raise GpuError()
        rows = self.processes(binding)
        return evidence['pid'] == pid and pid in rows and rows[pid] is not None


def configure_torch(torch: Any, binding: GpuBinding, *,
                    attestor: Any = actual_attestor) -> dict[str, Any]:
    """Caller freezes UUID/env before importing Torch; configure before work."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != binding.gpu_uuid:
        raise GpuError()
    if torch.cuda.device_count() != 1:
        raise GpuError()
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((8 * 2**30) / total, device=0)
    torch.cuda.init()
    evidence: dict[str, Any] = attestor(torch)
    verify_runtime(binding, evidence)
    return evidence
