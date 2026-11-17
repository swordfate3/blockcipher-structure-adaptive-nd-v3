"""CUDA current-context identity; narrow provenance extraction (see Task2 report)."""
from __future__ import annotations

import ctypes as C
import os
from typing import Any

from .resources import ResourceStop, pci


def _ok(result: int) -> None:
    if result != 0:
        raise ResourceStop()


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
                raise ResourceStop() from None
            function.argtypes, function.restype = arguments, C.c_int
    context = C.c_void_p()
    _ok(driver.cuCtxGetCurrent(C.byref(context)))
    if not context.value:
        raise ResourceStop()
    device = C.c_int()
    _ok(driver.cuCtxGetDevice(C.byref(device)))
    # Runtime ordinal is independently obtained from the current context.
    if device.value != 0:
        raise ResourceStop()
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
            raise ResourceStop() from None
        if not node_mask.value or bytes(local_id) == b'\0' * 8:
            raise ResourceStop()
        evidence.update(luid=bytes(local_id).hex(), device_node_mask=node_mask.value)
    return evidence

