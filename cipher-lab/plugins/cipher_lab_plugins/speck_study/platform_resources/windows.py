"""Windows Job ownership established before any child user code runs.

CREATE_SUSPENDED + assign + creation-time handle + NtResumeProcess avoids the
Popen/ready registration race. No PID-only kill, no scanning/killing other models.
Native calls are lazy; tests inject an API and never probe this workstation.
"""
from __future__ import annotations

import ctypes as C
import math
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any


class ProcessError(RuntimeError):
    def __init__(self) -> None:
        super().__init__('PROCESS_UNVERIFIED')


class WindowsAPI:
    def __init__(self) -> None:
        if os.name != 'nt':
            raise ProcessError()
        from ctypes import wintypes as W
        self.W = W
        self.k = getattr(C, 'WinDLL')('kernel32', use_last_error=True)
        self.ps = getattr(C, 'WinDLL')('psapi', use_last_error=True)
        self.nt = getattr(C, 'WinDLL')('ntdll')
        # ctypes defaults truncate HANDLEs on 64-bit Windows. Declare every API.
        signatures = {
            'CreateJobObjectW': ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            'SetInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
            'IsProcessInJob': ([W.HANDLE, W.HANDLE, C.c_void_p], W.BOOL),
            'QueryInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD,
                                          C.c_void_p], W.BOOL),
            'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            'WaitForSingleObject': ([W.HANDLE, W.DWORD], W.DWORD),
            'GetProcessTimes': ([W.HANDLE] + [C.c_void_p] * 4, W.BOOL),
            'CloseHandle': ([W.HANDLE], W.BOOL),
            'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result
        self.ps.GetProcessMemoryInfo.argtypes = [W.HANDLE, C.c_void_p, W.DWORD]
        self.ps.GetProcessMemoryInfo.restype = W.BOOL
        self.nt.NtResumeProcess.argtypes = [W.HANDLE]
        self.nt.NtResumeProcess.restype = C.c_long

    @staticmethod
    def required(result: Any) -> Any:
        if not result:
            raise ProcessError()
        return result

    def create_job(self) -> Any:
        W = self.W

        class Basic(C.Structure):
            _fields_ = [('process_time', C.c_longlong), ('job_time', C.c_longlong),
                        ('flags', W.DWORD), ('min_ws', C.c_size_t), ('max_ws', C.c_size_t),
                        ('active', W.DWORD), ('affinity', C.c_size_t),
                        ('priority', W.DWORD), ('scheduling', W.DWORD)]

        class Extended(C.Structure):
            _fields_ = [('basic', Basic), ('io', C.c_ulonglong * 6),
                        ('process_memory', C.c_size_t), ('job_memory', C.c_size_t),
                        ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]

        handle = self.required(self.k.CreateJobObjectW(None, None))
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway
        try:
            self.required(self.k.SetInformationJobObject(
                handle, 9, C.byref(limits), C.sizeof(limits)))
        except BaseException:
            self.k.CloseHandle(handle)
            raise
        return handle

    def assign(self, job: Any, handle: Any) -> None:
        self.required(self.k.AssignProcessToJobObject(job, handle))

    def belongs(self, job: Any, handle: Any) -> bool:
        result = self.W.BOOL()
        self.required(self.k.IsProcessInJob(handle, job, C.byref(result)))
        return bool(result.value)

    def resume(self, handle: Any) -> None:
        if self.nt.NtResumeProcess(handle) != 0:
            raise ProcessError()

    def pids(self, job: Any) -> list[int]:
        class Pids(C.Structure):
            _fields_ = [('assigned', C.c_uint32), ('listed', C.c_uint32),
                        ('values', C.c_size_t * 4096)]
        value = Pids()
        self.required(self.k.QueryInformationJobObject(
            job, 3, C.byref(value), C.sizeof(value), None))
        if value.assigned != value.listed or value.listed > 4096:
            raise ProcessError()
        return list(value.values[:value.listed])

    def open(self, pid: int) -> Any:
        return self.required(self.k.OpenProcess(0x1000 | 0x10 | 0x800 | 0x100000, False, pid))

    def times(self, handle: Any) -> tuple[int, float]:
        values = [C.c_ulonglong() for _ in range(4)]
        self.required(self.k.GetProcessTimes(handle, *(C.byref(v) for v in values)))
        return values[0].value, (values[2].value + values[3].value) / 10_000_000

    def exited(self, handle: Any) -> bool:
        result = self.k.WaitForSingleObject(handle, 0)
        if result not in (0, 258):
            raise ProcessError()
        return bool(result == 0)

    def rss(self, handle: Any) -> int:
        class Memory(C.Structure):
            _fields_ = [('cb', C.c_uint32), ('faults', C.c_uint32),
                        ('peak_ws', C.c_size_t), ('ws', C.c_size_t),
                        ('rest', C.c_size_t * 6)]
        value = Memory()
        value.cb = C.sizeof(value)
        self.required(self.ps.GetProcessMemoryInfo(handle, C.byref(value), value.cb))
        return int(value.ws)

    def cpu(self, job: Any) -> float:
        class Accounting(C.Structure):
            _fields_ = [('times', C.c_longlong * 4), ('faults', C.c_uint32),
                        ('total', C.c_uint32), ('active', C.c_uint32), ('terminated', C.c_uint32)]
        value = Accounting()
        self.required(self.k.QueryInformationJobObject(
            job, 1, C.byref(value), C.sizeof(value), None))
        return float(value.times[0] + value.times[1]) / 10_000_000

    def close_handle(self, handle: Any) -> None:
        self.required(self.k.CloseHandle(handle))

    def kill_job(self, job: Any) -> None:
        self.required(self.k.TerminateJobObject(job, 1))


@dataclass
class OrphanCleanup:
    """Retain Popen's original process HANDLE even if assign never succeeded."""
    process: Any
    reaped: bool = False
    streams_closed: set[str] = field(default_factory=set)
    errors: list[BaseException] = field(default_factory=list)


class WindowsJob:
    def __init__(self, api: Any = None) -> None:
        self.api = api if api is not None else WindowsAPI()
        self.job = self.api.create_job()
        self.handles: dict[int, tuple[int, Any]] = {}
        self.lock = threading.RLock()
        self.closed = False
        self.orphans: list[OrphanCleanup] = []

    @property
    def cleanup_unknown(self) -> bool:
        return bool(self.orphans)

    def retry_cleanup(self) -> None:
        """Independent/retryable kill, reap and each stream close. Never PID kill."""
        with self.lock:
            for record in list(self.orphans):
                record.errors.clear()
                if not record.reaped:
                    try:
                        record.process.kill()
                    except BaseException as error:
                        record.errors.append(error)
                    try:
                        record.process.wait(timeout=2)
                        record.reaped = True
                    except BaseException as error:
                        record.errors.append(error)
                for name in ('stdin', 'stdout', 'stderr'):
                    if name not in record.streams_closed:
                        try:
                            stream = getattr(record.process, name)
                            if stream is not None:
                                stream.close()
                            record.streams_closed.add(name)
                        except BaseException as error:
                            record.errors.append(error)
                if record.reaped and len(record.streams_closed) == 3:
                    self.orphans.remove(record)
            if self.orphans:
                raise ProcessError()

    def creation_time(self, pid: int) -> int:
        with self.lock:
            created, handle = self.handles[pid]
            if self.api.times(handle)[0] != created:
                raise ProcessError()
            return int(created)

    def adopt_suspended(self, pid: int, *, before_resume: Any) -> None:
        """Child-inherited outer Job; independently open/verify before resume.

        Caller holds the lifecycle lock also used by parent sampling. The child
        owns Popen/streams, while this parent retains its own process handle.
        """
        with self.lock:
            if self.closed or self.cleanup_unknown:
                raise ProcessError()
            self._register(pid)
            created, handle = self.handles[pid]
            before_resume(pid, created)
            self.api.resume(handle)

    def spawn(self, args: list[str], *, before_resume: Any = None,
              **kwargs: Any) -> subprocess.Popen[bytes]:
        """Task2's optional Popen-shaped seam, never used by legacy clients."""
        with self.lock:
            if self.closed or self.cleanup_unknown:
                raise ProcessError()
            process = subprocess.Popen(args, creationflags=0x4, **kwargs)
            record = OrphanCleanup(process)
            self.orphans.append(record)  # owner exists before assign/open/resume can fail
            handle = process._handle  # type: ignore[attr-defined]  # Windows Popen owned handle
            try:
                self.api.assign(self.job, handle)
                self._register(process.pid)
                if before_resume is not None:
                    before_resume(process.pid, self.handles[process.pid][0])
                self.api.resume(handle)
                self.orphans.remove(record)  # ownership transferred only on success
                return process
            except BaseException:
                try:
                    self.retry_cleanup()
                except BaseException:
                    pass  # retained handle-owned records expose UNKNOWN for retry
                raise ProcessError() from None

    def _register(self, pid: int) -> None:
        handle = self.api.open(pid)
        try:
            if not self.api.belongs(self.job, handle):
                raise ProcessError()
            created, _ = self.api.times(handle)
            if pid in self.handles:
                if self.handles[pid][0] != created:
                    raise ProcessError()
                self.api.close_handle(handle)
            else:
                self.handles[pid] = created, handle
        except BaseException:
            self.api.close_handle(handle)
            raise

    def sample(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.closed or self.cleanup_unknown:
                raise ProcessError()
            active = self.api.pids(self.job)
            rows: list[dict[str, Any]] = []
            for pid in active:
                self._register(pid)
                created, handle = self.handles[pid]
                current, cpu = self.api.times(handle)
                if current != created or not math.isfinite(cpu) or cpu < 0:
                    raise ProcessError()
                rows.append(dict(pid=pid, created=created, cpu=cpu, rss=self.api.rss(handle)))
            return rows

    def cpu_seconds(self) -> float:
        """Job accounting retains exited descendants, unlike live PID deltas."""
        with self.lock:
            return float(self.api.cpu(self.job))

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.api.kill_job(self.job)
            # Terminate is asynchronous: caller polls empty() before COMPLETE.

    def empty(self) -> bool:
        with self.lock:
            return not self.cleanup_unknown and self.api.pids(self.job) == []

    def close_job_handle(self) -> None:
        """Kill-on-close fallback WITHOUT dropping independent process identities."""
        with self.lock:
            if not self.closed:
                self.api.close_handle(self.job)
                self.closed = True

    def retain_members(self) -> None:
        """Capture even a just-spawned owned descendant before asynchronous TERM."""
        with self.lock:
            if not self.closed:
                for pid in self.api.pids(self.job):
                    if pid not in self.handles:
                        self._register(pid)
                    self.creation_time(pid)

    def dispose(self, *, release_processes: bool = True,
                require_exit: bool = False) -> None:
        with self.lock:
            errors: list[BaseException] = []
            try:
                self.close_job_handle()
            except BaseException as error:
                errors.append(error)
            try:
                self.retry_cleanup()  # includes children that never joined the Job
            except BaseException as error:
                errors.append(error)
            for pid, (created, handle) in list(self.handles.items()) if release_processes else []:
                try:
                    if require_exit and (self.api.times(handle)[0] != created
                                         or not self.api.exited(handle)):
                        raise ProcessError()  # retain unresolved owned handle; never guess reap
                    self.api.close_handle(handle)
                    del self.handles[pid]
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise ProcessError()
