"""Shared cooperative resource ledger; CUDA limits belong to the allocator."""
from __future__ import annotations

import ctypes as C
import os
import sys
from datetime import timedelta
from typing import Any, cast

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Kind, Meter, Stopped

from .config import Recipe, estimate


class ResourceStop(RuntimeError):
    def __init__(self, detail_code: str) -> None:
        self.reason = ('SOURCE_CHANGED' if detail_code == 'SOURCE_CHANGED' else
                       'GPU_LIMIT' if detail_code == 'GPU_MEMORY_LIMIT' else 'GPU_UNAVAILABLE')
        self.detail_code = detail_code
        super().__init__(self.reason)


def production_rss_bytes() -> int:
    """Current process RSS: Linux procfs or Windows GetProcessMemoryInfo; fail closed."""
    try:
        if sys.platform == 'linux':
            with open('/proc/self/statm', encoding='ascii') as stream:
                pages = int(stream.read(4096).split()[1])
            value = pages * os.sysconf('SC_PAGE_SIZE')
        elif sys.platform == 'win32':
            class Counters(C.Structure):
                _fields_ = [('cb', C.c_ulong), ('PageFaultCount', C.c_ulong),
                            *[(name, C.c_size_t) for name in (
                                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]
            kernel = getattr(C, 'WinDLL')('kernel32', use_last_error=True)
            psapi = getattr(C, 'WinDLL')('psapi', use_last_error=True)
            kernel.GetCurrentProcess.restype = C.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [C.c_void_p, C.c_void_p, C.c_ulong]
            psapi.GetProcessMemoryInfo.restype = C.c_int
            counters = Counters()
            counters.cb = C.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), C.byref(counters),
                                             C.sizeof(counters)):
                raise ValueError
            value = int(counters.WorkingSetSize)
        else:
            raise Stopped('RSS_UNSUPPORTED')
        if value <= 0:
            raise ValueError
        return value
    except Stopped:
        raise
    except Exception:
        raise Stopped('RSS_UNAVAILABLE') from None


def policy(recipe: Recipe, clocks: Clocks | None = None) -> Policy:
    count, limits = estimate(recipe), recipe.resources
    deadline = ((clocks or Clocks()).utc() + timedelta(seconds=limits.wall_seconds)).isoformat()
    return Policy(2, float(limits.wall_seconds), float(limits.cpu_seconds), deadline,
                  count['real_blocks'], 0, limits.rss_bytes, limits.artifact_bytes,
                  count['optimizer_updates'], count['model_groups'])


class Resources:
    def __init__(self, meter: Meter) -> None:
        self.meter = meter

    def check(self) -> None:
        self.meter.check()

    def reserve(self, kind: str, amount: int) -> None:
        self.meter.reserve(cast(Kind, kind), amount)

    def complete(self, kind: str, amount: int) -> None:
        self.meter.complete(cast(Kind, kind), amount)

    def snapshot(self) -> dict[str, Any]:
        return self.meter.snapshot()


def manual_resources(recipe: Recipe) -> Resources:
    """Artificial loop tests only; not reachable from the product entry point."""
    return Resources(Meter(policy(recipe), Clocks(), lambda: 0))
