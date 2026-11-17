"""Platform resource readings and identity binding only, never training.

RSS is a current-process reading: Linux delegates to the existing production
reader, Windows reads the current process WorkingSetSize via psapi. Unsupported
platforms and failed reads fail closed; a missing value is never reported as 0.

GPU memory is the driver-reported ``memory.used`` for one explicitly named GPU;
it is not a Torch allocator estimate. Device binding requires logical ``cuda:0``
and an exact UUID in ``CUDA_VISIBLE_DEVICES`` so another card is never selected
silently. Import probes no host and starts no thread or monitoring platform.
"""
from __future__ import annotations

import ctypes as C
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from ctypes import wintypes
from typing import Any

from cipher_lab.execution import Stopped

_GPU_REASONS = frozenset({'GPU_UNSUPPORTED', 'GPU_UNAVAILABLE', 'GPU_LIMIT',
                          'DEVICE_BINDING'})
_GPU_UUID = re.compile(
    r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
_PCI_BUS = re.compile(
    r'([0-9a-fA-F]{4}|0000[0-9a-fA-F]{4}):'
    r'([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])')
_USED = re.compile(r'[0-9]+')
_MAX_GPU_BYTES = 20 * 2**30
_RUNNER_TIMEOUT = 5
_GPU_QUERY = '--query-gpu=uuid,pci.bus_id,memory.used'
_GPU_FORMAT = '--format=csv,noheader,nounits'


class ResourceError(RuntimeError):
    """Fixed plugin-local GPU/binding reasons; not part of the core Stopped set."""

    def __init__(self, reason: str) -> None:
        if reason not in _GPU_REASONS:
            raise ValueError('unknown resource reason')
        self.reason = reason
        super().__init__(reason)


class _ProcessMemoryCounters(C.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', C.c_size_t),
        ('WorkingSetSize', C.c_size_t),
        ('QuotaPeakPagedPoolUsage', C.c_size_t),
        ('QuotaPagedPoolUsage', C.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', C.c_size_t),
        ('QuotaNonPagedPoolUsage', C.c_size_t),
        ('PagefileUsage', C.c_size_t),
        ('PeakPagefileUsage', C.c_size_t),
    ]


def _checked_bytes(value: object) -> int:
    if type(value) is not int or value < 0:
        raise Stopped('RSS_UNAVAILABLE')
    return value


def _windows_api() -> tuple[Any, Any]:
    """Load kernel32/psapi lazily; tests always replace this seam."""
    loader = getattr(C, 'WinDLL')
    return loader('kernel32', use_last_error=True), loader('psapi', use_last_error=True)


def _windows_working_set(kernel32: Any, psapi: Any) -> int:
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel32.GetCurrentProcess()
    if not handle:
        raise OSError('no current process handle')
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, C.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _ProcessMemoryCounters()
    counters.cb = C.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(handle, C.byref(counters), counters.cb):
        raise OSError('process memory info unavailable')
    value = int(counters.WorkingSetSize)
    if value < 0:
        raise ValueError('negative working set')
    return value


def rss_bytes() -> int:
    """Return current resident bytes or stop; unsupported is never zero."""
    if sys.platform == 'win32':
        try:
            kernel32, psapi = _windows_api()
            value = _windows_working_set(kernel32, psapi)
        except Stopped:
            raise
        except Exception:
            raise Stopped('RSS_UNAVAILABLE') from None
        return _checked_bytes(value)
    if sys.platform != 'linux':
        raise Stopped('RSS_UNSUPPORTED')
    try:
        from cipher_lab.execution import production_rss_bytes
        value = production_rss_bytes()
    except Stopped:
        raise
    except Exception:
        raise Stopped('RSS_UNAVAILABLE') from None
    return _checked_bytes(value)


def _canonical_uuid(value: object) -> str:
    if type(value) is not str or _GPU_UUID.fullmatch(value.strip()) is None:
        raise ValueError('invalid GPU UUID')
    return value.strip().upper()


def _canonical_pci(value: object) -> str:
    if type(value) is not str:
        raise ValueError('invalid PCI bus id')
    match = _PCI_BUS.fullmatch(value.strip())
    if match is None:
        raise ValueError('invalid PCI bus id')
    domain, bus, device, function = match.groups()
    return f'{int(domain, 16):08x}:{bus.lower()}:{device.lower()}.{function}'


def _driver_selector(canonical_uuid: str) -> str:
    """Render the ``-i`` selector: ``GPU-`` prefix plus lower-case hex only.

    The driver rejects the canonical upper-case hex with "No devices found",
    while CUDA_VISIBLE_DEVICES keeps accepting and requiring upper case.
    """
    return f'GPU-{canonical_uuid[4:].lower()}'


def _used_bytes(value: str) -> int:
    # nvidia-smi memory.used with nounits is MiB; the public result is bytes.
    if _USED.fullmatch(value) is None:
        raise ValueError('invalid used memory')
    used_mib = int(value)
    if used_mib > 2**64 - 1:
        raise ValueError('used memory out of range')
    return used_mib * 2**20


def _nvidia_smi_runner(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=_RUNNER_TIMEOUT,
                            check=True, shell=False)
    return result.stdout


class GPUProbe:
    """Query one expected physical GPU by UUID and verify UUID+PCI together.

    The canonical identity and ``CUDA_VISIBLE_DEVICES`` stay upper-case; only
    the driver selector is rendered as ``GPU-`` plus lower-case hex.
    """

    def __init__(self, expected_uuid: str, expected_pci: str, *,
                 runner: Callable[[list[str]], str] | None = None) -> None:
        self._uuid = _canonical_uuid(expected_uuid)
        self._pci = _canonical_pci(expected_pci)
        if runner is not None and not callable(runner):
            raise ValueError('runner must be callable')
        self._runner = _nvidia_smi_runner if runner is None else runner

    def check(self, limit_bytes: int) -> dict[str, object]:
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise ValueError('limit must be a nonnegative integer')
        args = ['nvidia-smi', _GPU_QUERY, _GPU_FORMAT, '-i',
                _driver_selector(self._uuid)]
        try:
            output = self._runner(args)
            if type(output) is not str:
                raise ValueError('runner did not return text')
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError('expected exactly one GPU row')
            fields = [field.strip() for field in lines[0].split(',')]
            if len(fields) != 3:
                raise ValueError('expected three GPU fields')
            uuid = _canonical_uuid(fields[0])
            pci = _canonical_pci(fields[1])
            used = _used_bytes(fields[2])
            if uuid != self._uuid or pci != self._pci:
                raise ValueError('GPU identity mismatch')
        except Exception:
            raise ResourceError('GPU_UNAVAILABLE') from None
        if used > min(limit_bytes, _MAX_GPU_BYTES):
            raise ResourceError('GPU_LIMIT')
        return {'uuid': uuid, 'pci': pci, 'used_bytes': used}


def validate_device_binding(device: str, expected_uuid: str, expected_pci: str,
                            environ: Mapping[str, str]) -> None:
    """Require explicit logical cuda:0 and the canonical uppercase UUID.

    CUDA_VISIBLE_DEVICES must contain exactly the canonical upper-case UUID, not
    an index, list or lower-case variant. CLI/start setup must normalize the
    public expected UUID the same way; this function does not select a card.
    """
    if type(device) is not str:
        raise ValueError('device must be text')
    uuid = _canonical_uuid(expected_uuid)
    _canonical_pci(expected_pci)
    if not isinstance(environ, Mapping):
        raise ValueError('environ must be a mapping')
    if device != 'cuda:0' or environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ResourceError('DEVICE_BINDING')
