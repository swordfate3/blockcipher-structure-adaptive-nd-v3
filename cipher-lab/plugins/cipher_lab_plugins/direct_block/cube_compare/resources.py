"""Local resource samples, never a scheduler or a scientific decision."""
from __future__ import annotations

import ctypes as C
import os
import sys
from collections.abc import Callable

from cipher_lab.execution import Stopped

from .config import GPU_DEVICE, GPU_PHYSICAL_INDEX, GPU_UUID


# PROCESS_MEMORY_COUNTERS: Windows DWORD is fixed 32-bit, SIZE_T is native.
class _Counters(C.Structure):
    _fields_ = [('cb', C.c_uint32), ('PageFaultCount', C.c_uint32),
                *[(name, C.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]


_windows_binding: tuple[Callable[[], int], Callable[[int, object, int], int]] | None = None


def _read_windows_working_set() -> int:
    # Meter is single-threaded. Cache only the FFI binding, never the sample.
    global _windows_binding
    if _windows_binding is None:
        win_dll = getattr(C, 'WinDLL')
        get_process = win_dll('kernel32', use_last_error=True).GetCurrentProcess
        get_info = win_dll('psapi', use_last_error=True).GetProcessMemoryInfo
        get_process.argtypes = []
        get_process.restype = C.c_void_p
        get_info.argtypes = [C.c_void_p, C.POINTER(_Counters), C.c_uint32]
        get_info.restype = C.c_int
        _windows_binding = get_process, get_info
    current_process, memory_info = _windows_binding
    counters = _Counters()
    counters.cb = C.sizeof(counters)
    if not memory_info(current_process(), C.byref(counters), counters.cb):
        raise OSError('working set unavailable')
    return int(counters.WorkingSetSize)


def working_set_bytes(*, platform: str | None = None) -> int:
    target = sys.platform if platform is None else platform
    if target != 'win32':
        raise Stopped('RSS_UNSUPPORTED')
    try:
        value = _read_windows_working_set()
        if type(value) is not int or value <= 0:
            raise ValueError
        return value
    except Exception:
        raise Stopped('RSS_UNAVAILABLE') from None


def device_identity() -> dict[str, object]:
    """Require both the UUID selector and the UUID actually reported by CUDA."""
    import torch

    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise ValueError('GPU_IDENTITY')
    try:
        props = torch.cuda.get_device_properties(GPU_DEVICE)
        uuid = str(props.uuid).lower().removeprefix('gpu-')
        if (uuid != GPU_UUID.lower().removeprefix('gpu-')
                or not isinstance(props.name, str) or not props.name
                or type(props.total_memory) is not int or props.total_memory <= 0):
            raise ValueError
    except Exception:
        raise ValueError('GPU_IDENTITY') from None
    return {'logical_device': GPU_DEVICE, 'physical_gpu_index': GPU_PHYSICAL_INDEX,
            'uuid': GPU_UUID, 'name': props.name, 'total_bytes': props.total_memory}
