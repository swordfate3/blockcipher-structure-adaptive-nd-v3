# ruff: noqa: UP007, UP045
"""Bounded DXGI self-report transport and independent parent identity checks.

Trusted process self reports, not NVML-equivalent/fully independent accounting.
Parent supplies all paths/expected process identities from its Job/owned handles.
No labels, model locks, key material, training or scientific admission here.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from cipher_lab.platform_resources.dxgi import (
    DxgiUsage,
    NativeBindingResolver,
    luid,
    validate_rows,
)
from cipher_lab.platform_resources.gpu import GpuBinding, GpuError, verify_runtime

MAX_FRAME = 64 * 1024
EVIDENCE = 'dxgi-self-per-process+parent-luid'


def _integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**64:
        raise GpuError()
    return value


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise GpuError()
    return float(value)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise GpuError()
        result[key] = value
    return result


def _invalid(value: str) -> Any:
    raise GpuError()


def _parse(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid)


def _path(root: Path, path: Path) -> Path:
    """Parent-chosen absolute path; validate EVERY ancestor, including root.

    Reject links/reparse points and nonregular destinations. A trusted root ACL
    remains required: lstat/open is not an adversarial namespace sandbox.
    """
    if not root.is_absolute() or not path.is_absolute() or '..' in path.parts:
        raise GpuError()
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise GpuError() from None
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(i) for i in range(1, 10)),
                *('LPT' + str(i) for i in range(1, 10))}
    for part in relative.parts:
        if (any(c in part for c in ':\\\\\x00') or part.endswith(('.', ' '))
                or part.split('.', 1)[0].upper() in reserved):
            raise GpuError()
    if path == root or path.name in ('', '.'):
        raise GpuError()
    for parent in [*reversed(path.parent.parents), path.parent]:
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise GpuError()
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400 or info.st_nlink != 1):
            raise GpuError()
    return path


def _validate(value: Any) -> dict[str, Any]:
    fields = {'version', 'evidence', 'phase', 'seq', 'tick', 'monotonic', 'pid', 'created', 'rows'}
    if type(value) is not dict or set(value) != fields:
        raise GpuError()
    if (type(value['version']) is not int or value['version'] != 1
            or value['evidence'] != EVIDENCE or value['phase'] != 'bootstrap-description'):
        raise GpuError()
    for key in ('seq', 'tick', 'pid', 'created'):
        _integer(value[key], 1)
    if value['seq'] != value['tick']:
        raise GpuError()
    _number(value['monotonic'])
    validate_rows(value['rows'])
    return value


class _FileInformation(C.Structure):
    # BY_HANDLE_FILE_INFORMATION: DWORD attrs, three FILETIMEs, six DWORDs.
    _fields_ = [('attributes', C.c_uint32), ('times', C.c_uint32 * 6),
                ('volume', C.c_uint32), ('size_high', C.c_uint32),
                ('size_low', C.c_uint32), ('links', C.c_uint32),
                ('index_high', C.c_uint32), ('index_low', C.c_uint32)]


def _read_frame_bytes(root: Path, path: Path, *, kernel: Any = None) -> bytes:
    """Bounded read; Windows handle permits concurrent atomic replace/unlink.

    Builtin open('rb') does not grant FILE_SHARE_DELETE on Windows. Retaining
    that handle would turn a legal publisher replace/stop into sharing failure.
    Validate the opened handle too; OPEN_REPARSE_POINT never follows a final
    reparse target. Ancestor/ACL assumptions remain those of _path.
    """
    target = _path(root, path)
    if kernel is None and os.name != 'nt':
        with target.open('rb') as stream:
            return stream.read(MAX_FRAME + 1)
    if kernel is None:
        kernel = getattr(C, 'WinDLL')('kernel32.dll', use_last_error=True)
        signatures = {
            'CreateFileW': ([C.c_wchar_p, C.c_uint32, C.c_uint32, C.c_void_p,
                             C.c_uint32, C.c_uint32, C.c_void_p], C.c_void_p),
            'GetFileType': ([C.c_void_p], C.c_uint32),
            'GetFileInformationByHandle': ([C.c_void_p, C.POINTER(_FileInformation)], C.c_int32),
            'ReadFile': ([C.c_void_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_uint32),
                          C.c_void_p], C.c_int32),
            'CloseHandle': ([C.c_void_p], C.c_int32),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = arguments, result
    # GENERIC_READ; SHARE_READ | SHARE_WRITE | SHARE_DELETE; OPEN_EXISTING;
    # FILE_FLAG_OPEN_REPARSE_POINT. No overlapped I/O or device/pipe path.
    handle = kernel.CreateFileW(str(target), 0x80000000, 7, None, 3, 0x00200000, None)
    if handle in (None, 0, -1, C.c_void_p(-1).value):
        raise GpuError()
    try:
        info = _FileInformation()
        if (kernel.GetFileType(handle) != 1
                or not kernel.GetFileInformationByHandle(handle, C.byref(info))
                or info.attributes & (0x400 | 0x10) or info.links != 1):
            raise GpuError()
        size = (info.size_high << 32) | info.size_low
        if size > MAX_FRAME:
            raise GpuError()
        raw = bytearray()
        while len(raw) <= MAX_FRAME:
            remaining = MAX_FRAME + 1 - len(raw)
            buffer, count = C.create_string_buffer(remaining), C.c_uint32()
            if (not kernel.ReadFile(handle, buffer, remaining, C.byref(count), None)
                    or count.value > remaining):
                raise GpuError()
            if count.value == 0:
                break
            raw.extend(buffer.raw[:count.value])
        if len(raw) > MAX_FRAME or len(raw) != size:
            raise GpuError()
        return bytes(raw)
    finally:
        if not kernel.CloseHandle(handle):
            raise GpuError()


def read_sample(root: Path, path: Path) -> dict[str, Any]:
    try:
        raw = _read_frame_bytes(root, path)
        if len(raw) > MAX_FRAME:
            raise GpuError()
        envelope = _parse(raw)
        if type(envelope) is not dict or set(envelope) != {'size', 'sha256', 'payload'}:
            raise GpuError()
        if type(envelope['payload']) is not str:
            raise GpuError()
        payload = envelope['payload'].encode('ascii')
        if (_integer(envelope['size'], 1) != len(payload)
                or envelope['sha256'] != hashlib.sha256(payload).hexdigest()):
            raise GpuError()
        return _validate(_parse(payload))
    except Exception:
        raise GpuError() from None


class _FileTime(C.Structure):
    _fields_ = [('low', C.c_uint32), ('high', C.c_uint32)]


def current_process_identity(kernel: Any = None) -> tuple[int, int]:
    """Own PID + FILETIME creation identity; no RSS probe or pseudo-handle close."""
    if kernel is None:
        if os.name != 'nt':
            raise GpuError()
        kernel = getattr(C, 'WinDLL')('kernel32.dll', use_last_error=True)
        signatures = {
            'GetCurrentProcess': ([], C.c_void_p),
            'GetCurrentProcessId': ([], C.c_uint32),
            'GetProcessTimes': ([C.c_void_p] + [C.POINTER(_FileTime)] * 4, C.c_int32),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = arguments, result
    created, exited, system, user = (_FileTime() for _ in range(4))
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), C.byref(created),
                                  C.byref(exited), C.byref(system), C.byref(user)):
        raise GpuError()
    return (_integer(kernel.GetCurrentProcessId(), 1),
            _integer((created.high << 32) | created.low, 1))


def windows_monotonic(kernel: Any = None) -> float:
    """Raw QPC epoch is shared across Windows PIDs and Python 3.9/3.10.

    Task5B must use this SAME domain for parent origin/deadlines and samples.
    Python 3.9 monotonic's per-process epoch must not be compared to 3.10.
    """
    if kernel is None:
        if os.name != 'nt':
            raise GpuError()
        kernel = getattr(C, 'WinDLL')('kernel32.dll', use_last_error=True)
        for name in ('QueryPerformanceFrequency', 'QueryPerformanceCounter'):
            function = getattr(kernel, name)
            function.argtypes, function.restype = [C.POINTER(C.c_int64)], C.c_int32
    frequency, counter = C.c_int64(), C.c_int64()
    if (not kernel.QueryPerformanceFrequency(C.byref(frequency))
            or not kernel.QueryPerformanceCounter(C.byref(counter))
            or frequency.value <= 0 or counter.value < 0):
        raise GpuError()
    return _number(counter.value / frequency.value)


class TelemetryPublisher:
    """One owned path; independently polls even while teacher.predict blocks.

    Native calls cannot be forcibly interrupted by Python; stop(timeout) then
    fails sticky instead of claiming join/live/clean exit. Parent TTL also stops.
    Publisher must remain running through closing until owner is reaped (5B).
    """
    def __init__(self, root: Path, path: Path, *, pid: int, created: int,
                 sample: Callable[[], list[dict[str, Any]]],
                 clock: Callable[[], float] = windows_monotonic, interval: float = .5) -> None:
        self.root, self.path = root, _path(root, path)
        self.pending = _path(root, path.with_name(path.name + '.pending'))
        self.pid, self.created = _integer(pid, 1), _integer(created, 1)
        if (type(interval) not in (int, float) or not math.isfinite(interval)
                or not 0 < interval <= .5):
            raise GpuError()
        self.sample, self.clock, self.interval = sample, clock, interval
        self.seq = 0
        self.stop_event = threading.Event()
        self.failure = False
        self.thread: Optional[threading.Thread] = None
        self.last_time: Optional[float] = None
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.started = False

    def _fail(self) -> None:
        self.failure = True
        raise GpuError()

    def publish_once(self) -> None:
        """Synchronous seam for deterministic synthetic tests; thread is production path."""
        with self.write_lock:
            try:
                if self.failure or self.stop_event.is_set():
                    raise GpuError()
                now = _number(self.clock())  # start-of-sample timestamp exposes slow probes
                rows = self.sample()
                if self.last_time is not None and now < self.last_time:
                    raise GpuError()
                value = _validate(dict(version=1, evidence=EVIDENCE, phase='bootstrap-description',
                                       seq=self.seq + 1, tick=self.seq + 1, monotonic=now,
                                       pid=self.pid, created=self.created, rows=rows))
                payload = _json(value)
                raw = _json(dict(size=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                                 payload=payload.decode('ascii')))
                if len(raw) > MAX_FRAME:
                    raise GpuError()
                # Stop and replace share a lock: a hung probe returning after stop
                # may NOT resurrect a live-looking frame.
                with self.lock:
                    if self.stop_event.is_set() or self.failure:
                        return
                    _path(self.root, self.path)
                    _path(self.root, self.pending)
                    with self.pending.open('xb') as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if self.stop_event.is_set() or self.failure:
                        self.pending.unlink()
                        return
                    os.replace(self.pending, self.path)
                    self.seq += 1
                    self.last_time = now
            except Exception:
                self._fail()

    def start(self) -> None:
        if self.started or self.failure or self.stop_event.is_set():
            self._fail()
        self.started = True
        def loop() -> None:
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    self.publish_once()
                except BaseException:
                    self.failure = True
                    return
                if self.stop_event.wait(max(0., self.interval - (time.monotonic() - started))):
                    return
        self.thread = threading.Thread(target=loop, name='dxgi-telemetry', daemon=True)
        self.thread.start()

    def check(self) -> None:
        try:
            # Read immutable publication state BEFORE the clock. The publisher
            # can advance last_time while clock() is running; rereading it after
            # taking t1 would compare unordered t2>t1 and falsely report future.
            # Never acquire the publication/fsync lock on this owner check path.
            stamp, thread, started = self.last_time, self.thread, self.started
            now = _number(self.clock())
            if (self.failure or self.stop_event.is_set() or not started
                    or thread is None or not thread.is_alive()
                    or stamp is None or not 0 <= now - stamp <= 1.5):
                raise GpuError()
        except Exception:
            self._fail()

    def stop(self, *, timeout: float = 1.) -> None:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 2:
            self._fail()
        end = time.monotonic() + timeout
        # Event.set is independent of a blocked file fsync/replace. Acquiring
        # the publication lock itself is bounded, not an unbounded stop wait.
        self.stop_event.set()
        if not self.lock.acquire(timeout=max(0., end - time.monotonic())):
            self._fail()
        try:
            target = _path(self.root, self.path)
            if target.exists():
                target.unlink()  # cannot leave an apparently live final frame
        finally:
            self.lock.release()
        if self.thread is not None:
            self.thread.join(max(0., end - time.monotonic()))
        if self.failure or (self.thread is not None and self.thread.is_alive()):
            self._fail()


class BootstrapPending(GpuError):
    """Measured subset plus explicit unknown PIDs; NEVER scientific admission."""
    def __init__(self, usage: dict[int, int], pending: set[int]) -> None:
        super().__init__()
        self.usage, self.pending = usage, pending


class TelemetryGpu:
    """StudyResources processes/verify adapter; exact Job members plus parent PID.

    `members()` is independently sampled from Job ownership, never from frames;
    `creation_time(pid)` uses a retained/verified owned process handle. Include
    the parent through this same callback separately (parent not inside own Job).
    Slots are registered by parent before suspended children resume. Missing or
    extra live slots fail; remove reaped slots via unregister, never guess exit.
    `resolve(binding)` must independently check physical index1/UUID/PCI -> LUID.
    """
    requires_exact_members = True

    def __init__(self, root: Path, slots: dict[int, tuple[int, Path]], *,
                 members: Callable[[], list[dict[str, Any]]],
                 creation_time: Callable[[int], int], science_deadline: float,
                 inventory: Optional[dict[str, int]] = None,
                 scientific_pids: Optional[set[int]] = None,
                 resolve: Callable[[GpuBinding], dict[str, Any]] = NativeBindingResolver(),
                 clock: Callable[[], float] = windows_monotonic, ttl: float = 1.5) -> None:
        self.root, self.members, self.creation_time = root, members, creation_time
        self.resolve, self.clock = resolve, clock
        self.scientific_pids = set() if scientific_pids is None else set(scientific_pids)
        for pid in self.scientific_pids:
            _integer(pid, 1)
        if inventory is None:
            inventory = {row['luid']: len(row['nodes']) for row in DxgiUsage().sample()}
        if type(inventory) is not dict or not 1 <= len(inventory) <= 64:
            raise GpuError()
        self.inventory = dict(inventory)
        for identity, count in self.inventory.items():
            luid(identity)
            if type(count) is not int or not 1 <= count <= 32:
                raise GpuError()
        self.deadline = _number(science_deadline)
        if type(ttl) not in (int, float) or not math.isfinite(ttl) or not .5 <= ttl <= 1.5:
            raise GpuError()
        self.ttl = ttl
        self.slots: dict[int, tuple[int, Path]] = {}
        self.seen: dict[int, tuple[int, float, str]] = {}
        self.retired: set[int] = set()
        self.retired_identities: dict[int, int] = {}
        self.first_frame_deadlines: dict[int, float] = {}
        self.bound: Optional[dict[str, Any]] = None
        self.attested: set[int] = set()
        self.failure = False
        self.last_time: Optional[float] = None
        self.lock = threading.RLock()
        for pid, (created, path) in slots.items():
            self.register(pid, created, path)

    @property
    def ready(self) -> bool:
        # Roles come from parent spawn contracts, NEVER from frames. CPU-only
        # parent/bootstrap has explicit samples but need not create a context.
        return (not self.failure and self.bound is not None and bool(self.scientific_pids)
                and self.scientific_pids <= self.attested)

    def register(self, pid: int, created: int, path: Path, *,
                 first_frame_deadline: Optional[float] = None) -> None:
        with self.lock:
            _integer(pid, 1)
            _integer(created, 1)
            path = _path(self.root, path)
            if (self.failure or pid in self.slots or pid in self.retired
                    or path in [p for _, p in self.slots.values()]):
                raise GpuError()
            if first_frame_deadline is not None:
                deadline = _number(first_frame_deadline)
                if not 0 < deadline - self.clock() <= 5.:
                    raise GpuError()
                self.first_frame_deadlines[pid] = deadline
            self.slots[pid] = created, path

    def unregister(self, pid: int) -> None:
        """Only after parent handle proves reaped and Job membership excludes PID."""
        with self.lock:
            if self.failure or pid not in self.slots or pid in self._members():
                raise GpuError()
            self.retired_identities[pid] = self.slots[pid][0]
            del self.slots[pid]
            self.attested.discard(pid)
            self.scientific_pids.discard(pid)
            self.retired.add(pid)

    def _members(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for row in self.members():
            pid, created = _integer(row['pid'], 1), _integer(row['created'], 1)
            if pid in result or _integer(self.creation_time(pid), 1) != created:
                raise GpuError()
            result[pid] = created
        return result

    def _collect(self) -> dict[int, dict[str, Any]]:
        if self.failure:
            raise GpuError()
        now = _number(self.clock())
        if self.last_time is not None and now < self.last_time:
            raise GpuError()
        before = self._members()
        if before != {pid: created for pid, (created, _) in self.slots.items()}:
            raise GpuError()
        frames = {}
        next_seen = dict(self.seen)
        pending: set[int] = set()
        for pid, (created, path) in list(self.slots.items()):
            if (pid in self.first_frame_deadlines and not path.exists()
                    and now < self.first_frame_deadlines[pid]):
                pending.add(pid)
                continue
            frame = read_sample(self.root, path)
            # First acceptance is bounded by PARENT receipt completion time,
            # not a producer timestamp which could arrive after the launch cap.
            first_deadline = self.first_frame_deadlines.get(pid)
            # A publisher can advance after our initial tick but before read.
            # Compare against one post-read parent clock, including first receipt.
            read_time = _number(self.clock())
            if first_deadline is not None and read_time >= first_deadline:
                raise GpuError()
            self.first_frame_deadlines.pop(pid, None)
            if (read_time < now or frame['pid'] != pid or frame['created'] != created
                    or not 0 <= read_time - frame['monotonic'] <= self.ttl):
                raise GpuError()
            if {row['luid']: len(row['nodes']) for row in frame['rows']} != self.inventory:
                raise GpuError()
            seq, stamp = frame['seq'], frame['monotonic']
            digest = hashlib.sha256(_json(frame)).hexdigest()
            if pid in self.seen:
                previous, old_time, old_digest = self.seen[pid]
                if seq < previous or stamp < old_time or (seq == previous and digest != old_digest):
                    raise GpuError()
            next_seen[pid] = seq, stamp, digest
            if self.bound is not None:
                selected = [row for row in frame['rows'] if row['luid'] == self.bound['luid']]
                if len(selected) != 1:
                    raise GpuError()
                mask = self.bound['device_node_mask']
                if mask >= 1 << len(selected[0]['nodes']):
                    raise GpuError()
                for row in frame['rows']:
                    for node in row['nodes']:
                        if (row['luid'] != self.bound['luid'] or not mask & (1 << node['node'])):
                            if node['current_usage'] != 0:
                                raise GpuError()
            frames[pid] = frame
        members_after = self._members()
        # Only a parent-confirmed retirement may change this collection. No
        # new PID/reuse, guessed exit, or disappearing slot is accepted.
        if (any(before.get(pid) != created for pid, created in members_after.items())
                or any(self.retired_identities.get(pid) != created
                       for pid, created in before.items() if pid not in members_after)):
            raise GpuError()
        frames = {pid: frame for pid, frame in frames.items() if pid in members_after}
        pending &= members_after.keys()
        self.seen.update({pid: value for pid, value in next_seen.items()
                          if pid in members_after})
        after = _number(self.clock())
        if after < now or any(not 0 <= after - f['monotonic'] <= self.ttl
                              for f in frames.values()):
            raise GpuError()
        self.last_time = after
        if pending:
            raise BootstrapPending({pid: sum(n['current_usage'] for r in f['rows']
                                              for n in r['nodes'])
                                    for pid, f in frames.items()}, pending)
        return frames

    def processes(self, binding: GpuBinding) -> dict[int, int]:
        with self.lock:
            try:
                if self.bound is not None and self.bound['binding'] != binding:
                    raise GpuError()
                frames = self._collect()
                return {pid: sum(node['current_usage'] for row in frame['rows']
                                 for node in row['nodes']) for pid, frame in frames.items()}
            except BootstrapPending:
                raise
            except Exception:
                self.failure = True
                raise GpuError() from None

    def verify(self, evidence: dict[str, Any], pid: int, uuid: str, bus: str) -> bool:
        # Native identity lookup may block; NEVER hold the sampling lock across
        # it. Parent monitoring must keep checking samples/TTL during ready.
        try:
            if self.failure or _number(self.clock()) >= self.deadline:
                raise GpuError()
            binding = GpuBinding(1, uuid, bus)
            verify_runtime(binding, evidence)
            expected = self.resolve(binding)  # independent, NOT evidence/frame-derived
        except Exception:
            self.failure = True
            raise GpuError() from None
        with self.lock:
            try:
                if (self.failure or _number(self.clock()) >= self.deadline
                        or evidence['pid'] != pid or pid not in self.slots):
                    raise GpuError()
                selected = luid(expected['luid'])
                mask = _integer(expected['device_node_mask'], 1)
                if mask >= 2**32 or evidence.get('luid') != selected:
                    raise GpuError()
                if (type(evidence.get('device_node_mask')) is not int
                        or evidence['device_node_mask'] != mask):
                    raise GpuError()
                value = dict(binding=binding, luid=selected, device_node_mask=mask)
                if self.bound is not None and value != self.bound:
                    raise GpuError()
                self.bound = value
                self._collect()
                # A blocking resolver must not move first init outside the budget.
                if _number(self.clock()) >= self.deadline:
                    raise GpuError()
                self.attested.add(pid)
                return True
            except Exception:
                self.failure = True
                raise GpuError() from None
