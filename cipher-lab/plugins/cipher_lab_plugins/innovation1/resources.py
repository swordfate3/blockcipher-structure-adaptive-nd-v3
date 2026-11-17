"""Cooperative FP32 single-GPU limits, cached physical telemetry, no fallback.

GPU accounting conservatively charges the WHOLE selected device, not allocator
estimates. Samples every 1s; a gap >5s fails closed instead of renewing stale proof.
Not an OS hard limit or a transient peak guarantee. Windows N/A GPU memory stops.
"""
from __future__ import annotations

import ctypes as C
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Kind, Meter, Stopped

from .config import Recipe, estimate


class ResourceStop(RuntimeError):
    def __init__(self, reason: str = 'GPU_UNVERIFIED') -> None:
        if reason not in ('GPU_UNVERIFIED', 'GPU_LIMIT', 'SOURCE_CHANGED'):
            raise ValueError('invalid public resource code')
        self.reason = reason
        super().__init__(reason)


def pci(value: str) -> str:
    match = re.fullmatch(
        r'(?:0000)?([\da-fA-F]{4}):([\da-fA-F]{2}):([\da-fA-F]{2})\.([0-7])', value)
    if not match:
        raise ResourceStop()
    return ':'.join((match[1].lower(), match[2].lower(), match[3].lower())) + '.' + match[4]


class GpuGate:
    def __init__(self, uuid: str, bus: str, limit: int, sample: Callable[[], dict[str, Any]],
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', uuid):
            raise ResourceStop()
        self.uuid, self.bus, self.limit = uuid, pci(bus), limit
        self.sample, self.clock = sample, clock
        self.last: float | None = None
        self.latest: dict[str, Any] = {}
        self.failure: ResourceStop | None = None

    def check(self) -> None:
        if self.failure:
            raise self.failure
        try:
            now = self.clock()
            if not math.isfinite(now) or (self.last is not None and (
                    now < self.last or now - self.last > 5.0)):
                raise ResourceStop()
            if self.last is not None and now - self.last < 1.0:
                return
            row = self.sample()
            if (row.get('uuid') != self.uuid or pci(row.get('pci', '')) != self.bus
                    or type(row.get('bytes')) is not int or row['bytes'] < 0):
                raise ResourceStop()
            after = self.clock()
            if not math.isfinite(after) or not 0 <= after - now <= 1.0:
                raise ResourceStop()
            if row['bytes'] > self.limit:
                raise ResourceStop('GPU_LIMIT')
            self.last, self.latest = now, dict(row, sampled_at=now)
        except Exception as error:
            self.failure = error if isinstance(error, ResourceStop) else ResourceStop()
            raise self.failure from None


def gpu_sample(uuid: str) -> dict[str, Any]:
    try:
        result = subprocess.run(['nvidia-smi', f'--id={uuid}',
            '--query-gpu=uuid,pci.bus_id,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=1)
        rows = [row.split(',') for row in result.stdout.strip().splitlines()]
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError
        identity, bus, used = (x.strip() for x in rows[0])
        return dict(uuid=identity, pci=pci(bus), bytes=int(used) * 2**20)
    except Exception:
        raise ResourceStop() from None


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


def policy(recipe: Recipe) -> Policy:
    count, limits = estimate(recipe), recipe.resources
    deadline = (datetime.now(UTC) + timedelta(seconds=limits.wall_seconds)).isoformat()
    return Policy(2, float(limits.wall_seconds), float(limits.cpu_seconds), deadline,
                  count['real_blocks'], 0, limits.rss_bytes, limits.artifact_bytes,
                  count['optimizer_updates'], count['model_groups'])


class Resources:
    def __init__(self, meter: Meter, gpu: GpuGate | None = None) -> None:
        self.meter, self.gpu = meter, gpu

    def check(self) -> None:
        self.meter.check()
        if self.gpu is not None:
            self.gpu.check()

    def reserve(self, kind: str, amount: int) -> None:
        self.check()
        self.meter.reserve(cast(Kind, kind), amount)

    def complete(self, kind: str, amount: int) -> None:
        self.meter.complete(cast(Kind, kind), amount)

    def snapshot(self) -> dict[str, Any]:
        return dict(self.meter.snapshot(), gpu=None if self.gpu is None else self.gpu.latest,
                    sampled_limits=True)


def manual_resources(recipe: Recipe) -> Resources:
    return Resources(Meter(policy(recipe), Clocks(), lambda: 0))
