"""Study-only sampled limits and parent resource supervision, never RunStore.

0.5s sampling is not an OS memory sandbox/peak guarantee; OS probes and process
termination may fail or block. UNKNOWN is never reported as COMPLETE.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cipher_lab.execution import Meter
from cipher_lab_plugins.speck_study.platform_resources.gpu import GpuBinding

_CODES = frozenset({'GPU_UNVERIFIED', 'GPU_LIMIT', 'RSS_UNAVAILABLE', 'RSS_LIMIT',
                    'CPU_UNAVAILABLE', 'CLOCK_UNAVAILABLE', 'PROCESS_UNVERIFIED',
                    'OUTPUT_UNAVAILABLE', 'OUTPUT_LIMIT', 'SCIENCE_DEADLINE',
                    'CLOSE_DEADLINE', 'CLEANUP_UNKNOWN', 'FINALIZE_FAILED'})


class ResourceStop(RuntimeError):
    def __init__(self, reason: str) -> None:
        if reason not in _CODES:
            raise ValueError('unknown resource reason')
        self.reason = reason
        super().__init__(reason)


def directory_bytes(root: Path) -> int:
    """Count all outputs/cache/temp; reject links/reparse points, never follow."""
    info = root.lstat() if root.exists() else None
    if (info is None or not stat.S_ISDIR(info.st_mode)
            or root.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise ResourceStop('OUTPUT_UNAVAILABLE')
    def walk_error(error: OSError) -> None:
        raise ResourceStop('OUTPUT_UNAVAILABLE')
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in dirs + files:
            path = Path(base) / name
            info = path.lstat()
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, 'st_file_attributes', 0) & 0x400):
                raise ResourceStop('OUTPUT_UNAVAILABLE')
            if not stat.S_ISDIR(info.st_mode):
                if not stat.S_ISREG(info.st_mode):
                    raise ResourceStop('OUTPUT_UNAVAILABLE')
                total += info.st_size
    return total


class StudyResources:
    def __init__(self, binding: GpuBinding, root: Path, wall_origin: float,
                 science_deadline: float, close_deadline: float, process_tree: Any, *,
                 gpu: Any, clock: Callable[[], float] = time.monotonic,
                 output_bytes: Callable[[], int] | None = None) -> None:
        if (not all(math.isfinite(v) for v in (wall_origin, science_deadline, close_deadline))
                or science_deadline != wall_origin + 28200
                or close_deadline != wall_origin + 28800):
            raise ValueError('one origin and fixed 28200/28800 contract required')
        self.binding, self.root, self.tree, self.gpu = binding, root, process_tree, gpu
        self.origin, self.science_deadline = wall_origin, science_deadline
        self.close_deadline, self.clock = close_deadline, clock
        self.output_bytes = output_bytes or (lambda: directory_bytes(root))
        self.last_wall = wall_origin
        self.cpu_seconds = 0.
        self.cpu_by_identity: dict[tuple[int, int], float] = {}
        self.failure: ResourceStop | None = None
        self.closing = False
        self.lock = threading.RLock()
        self.latest: dict[str, Any] = {}

    def verify_teacher(self, evidence: dict[str, Any], pid: int,
                       uuid: str, bus: str) -> bool:
        """Task2 verifier: Job ownership+creation time bracket independent GPU proof."""
        # Do not hold sampling lock across a verifier/driver/subprocess call.
        # Constructor/ready/proof blocking cannot suppress independent monitoring.
        try:
            if uuid != self.binding.gpu_uuid or bus != self.binding.pci_bus_id:
                return False
            before = {r['pid']: r['created'] for r in self.tree.sample()}
            if pid not in before:
                return False
            if self.gpu.verify(evidence, pid, uuid, bus) is not True:
                return False
            after = {r['pid']: r['created'] for r in self.tree.sample()}
            return pid in after and before[pid] == after[pid]
        except Exception:
            return False

    def _stop(self, code: str) -> None:
        if self.failure is None:
            self.failure = ResourceStop(code)
        raise self.failure

    def begin_closing(self) -> None:
        """Child single writer calls before acknowledging quiescence to parent.

        Irreversible: blocks all further science reservations/completions. Saving
        and parent monitoring use check(science=False), retaining every hard cap.
        """
        with self.lock:
            self.closing = True

    def check(self, *, science: bool = True) -> None:
        with self.lock:
            if self.failure:
                raise self.failure
            try:
                now = self.clock()
                if (type(now) not in (int, float) or not math.isfinite(now)
                        or now < self.last_wall):
                    raise ValueError
            except Exception:
                self._stop('CLOCK_UNAVAILABLE')
            self.last_wall = now
            if now >= self.close_deadline:
                self._stop('CLOSE_DEADLINE')
            try:
                rows = self.tree.sample()
            except Exception:
                self._stop('PROCESS_UNVERIFIED')
            rss = 0
            for row in rows:
                if type(row.get('rss')) is not int or row['rss'] < 0:
                    self._stop('RSS_UNAVAILABLE')
                rss += row['rss']
                cpu = row.get('cpu')
                identity = row['pid'], row['created']
                if (type(cpu) not in (int, float) or not math.isfinite(cpu)
                        or cpu < self.cpu_by_identity.get(identity, 0.)):
                    self._stop('CPU_UNAVAILABLE')
                self.cpu_by_identity[identity] = cpu
            if hasattr(self.tree, 'cpu_seconds'):
                try:
                    cpu = self.tree.cpu_seconds()
                    if not math.isfinite(cpu) or cpu < self.cpu_seconds:
                        raise ValueError
                    self.cpu_seconds = cpu
                except Exception:
                    self._stop('CPU_UNAVAILABLE')
            else:
                self.cpu_seconds = sum(self.cpu_by_identity.values())
            if rss > 32 * 2**30:
                self._stop('RSS_LIMIT')
            from cipher_lab_plugins.speck_study.gpu_telemetry import BootstrapPending
            pending: set[int] = set()
            try:
                try:
                    usage = self.gpu.processes(self.binding)
                except BootstrapPending as bootstrap:
                    if science:
                        raise ValueError('bootstrap cannot admit science') from None
                    usage, pending = bootstrap.usage, bootstrap.pending
                if set(usage) & pending:
                    raise ValueError('overlapping measured/unknown')
                # Native lifecycle adapter reconciles only independently proven
                # exits between the initial resource rows and GPU collection.
                # It must not sample membership again or turn missing into zero.
                if hasattr(self.gpu, 'reconcile_rows'):
                    rows = self.gpu.reconcile_rows(rows, set(usage) | pending)
                total = 0
                for row in rows:
                    if row['pid'] in pending:
                        continue  # explicitly unknown, recorded below, never zero
                    value = usage.get(row['pid'])
                    if type(value) is not int or value < 0:
                        raise ValueError
                    total += value
            except Exception:
                self._stop('GPU_UNVERIFIED')
            if total > 20 * 2**30:
                self._stop('GPU_LIMIT')
            # DXGI bootstrap is measured accounting, not physical GPU1 proof.
            # Parent must include itself in both the owned-tree view and GPU
            # callback; exact PID sets prevent silently dropping its usage.
            if getattr(self.gpu, 'requires_exact_members', False) and set(usage) | pending != {
                    row['pid'] for row in rows}:
                self._stop('GPU_UNVERIFIED')
            if science and getattr(self.gpu, 'ready', True) is not True:
                self._stop('GPU_UNVERIFIED')
            try:
                output = self.output_bytes()
                if type(output) is not int or output < 0:
                    raise ValueError
            except Exception:
                self._stop('OUTPUT_UNAVAILABLE')
            ceiling = 4 * 2**30 if self.closing else 4 * 2**30 - 64 * 2**20
            if output > ceiling:
                self._stop('OUTPUT_LIMIT')
            self.latest = dict(wall_elapsed_seconds=now - self.origin,
                               cpu_seconds=self.cpu_seconds, rss_bytes=rss,
                               gpu_bytes=None if pending else total,
                               measured_gpu_bytes=total, bootstrap_pending_pids=sorted(pending),
                               output_bytes=output, sampled=True)
            if science and (self.closing or now >= self.science_deadline):
                # Science cutoff is an admission stop, not a fatal monitor fault.
                # Keep the 600s close/save window usable; all hard checks above run.
                raise ResourceStop('SCIENCE_DEADLINE')


class StudyMeter:
    """Narrow adapter; teacher uses the original forward_rows ledger, once.

    Other descriptors are counts of attempted/completed arithmetic rows only.
    Resource supervisor never calls or mutates this single-writer ledger.
    """
    def __init__(self, meter: Meter, resources: StudyResources) -> None:
        self.meter, self.resources = meter, resources
        self.counts: dict[str, list[int]] = {k: [0, 0] for k in ('scoring', 'parity')}

    def check(self) -> None:
        self.resources.check()
        self.meter.check()

    def reserve(self, kind: str, rows: int) -> None:
        if kind not in ('teacher', 'scoring', 'parity') or type(rows) is not int or rows < 0:
            raise ValueError('invalid study counter')
        self.check()
        if kind == 'teacher':
            self.meter.reserve('forward_rows', rows)
        else:
            self.counts[kind][0] += rows

    def complete(self, kind: str, rows: int) -> None:
        self.resources.check()  # stopped work cannot become valid completion
        if kind not in ('teacher', 'scoring', 'parity') or type(rows) is not int or rows < 0:
            raise ValueError('invalid study counter')
        if kind == 'teacher':
            self.meter.complete('forward_rows', rows)
        elif self.counts[kind][1] + rows <= self.counts[kind][0]:
            self.counts[kind][1] += rows
        else:
            raise ValueError('completion exceeds reservation')


def verify_sources(root: Path, plugin_root: Path, manifest: dict[str, str]) -> None:
    """Verify explicit source closure BEFORE import, reject managed bytecode.

    Hashes must come from an independently frozen Git manifest, not child claims.
    .pyc plus missing/changed source cannot bypass the freeze via import caches.
    """
    if not manifest:
        raise ValueError('empty source closure')
    from cipher_lab_plugins.speck_study.paths import source_path
    from cipher_lab_plugins.speck_study.runtime.source import package_source_files

    package_source_files(root, plugin_root)
    for name, expected in manifest.items():
        path = source_path(root, plugin_root, name)
        if Path(name).is_absolute() or '..' in Path(name).parts or path.suffix != '.py':
            raise ValueError('invalid source closure path')
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('source symlink')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('source identity mismatch')


def durable_receipt(path: Path, record: dict[str, Any], *,
                    sync_directory: Callable[[Path], None] | None = None,
                    platform: str | None = None) -> dict[str, Any]:
    """Parent resource receipt, NOT child owner proof or crash-recovery token.

    Windows namespace/powerloss durability is explicitly UNVERIFIED. Retains
    file fsync+replace and target readback contract; POSIX dirsync stays required.
    Explicit sync callbacks failing on ANY platform still fail finalization.
    """
    platform = os.name if platform is None else platform
    if platform not in ('nt', 'posix'):
        raise ResourceStop('FINALIZE_FAILED')
    saved = dict(record, namespace_powerloss='UNVERIFIED',
                 write_protocol='file-fsync+replace+target-readback-len-sha256')
    data = (json.dumps(saved, sort_keys=True, allow_nan=False) + '\n').encode()
    temporary = path.with_suffix('.pending')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if sync_directory is not None:
        sync_directory(path.parent)
    elif platform == 'posix':
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    actual = path.read_bytes()
    expected_sha = hashlib.sha256(data).hexdigest()
    if len(actual) != len(data) or hashlib.sha256(actual).hexdigest() != expected_sha:
        raise ResourceStop('FINALIZE_FAILED')
    return dict(target_size=len(actual), target_sha256=expected_sha,
                namespace_powerloss='UNVERIFIED', write_protocol=saved['write_protocol'])


class Supervisor:
    """Independent parent/thread loop starts BEFORE child constructor/spawn.

    Can run as a thread in a dedicated parent process; does not wait on IPC or
    teacher lock. Native sampler hangs are an unverified OS boundary, not hard
    real-time enforcement. Never writes child's terminal, even on kill.
    """
    def __init__(self, resources: StudyResources, *,
                 request_closing: Callable[[], None] | None = None,
                 closing_acknowledged: Callable[[], bool] | None = None) -> None:
        self.resources = resources
        self.request_closing = request_closing
        self.closing_acknowledged = closing_acknowledged
        self.closing_requested = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.complete = False
        self.receipt_evidence: dict[str, Any] = {}

    def cleanup_tree(self) -> list[BaseException]:
        """Never swallow TerminateJob failure without immediate kill-on-close."""
        errors: list[BaseException] = []
        try:
            self.resources.tree.close()
        except BaseException as error:
            errors.append(error)
            try:
                self.resources.tree.dispose()
            except BaseException as fallback:
                errors.append(fallback)
        return errors

    def tick(self) -> None:
        try:
            # Hard monitoring remains live throughout cooperation/save. Parent
            # never treats science cutoff alone as permission to kill at 28200.
            self.resources.check(science=False)
            if (self.resources.closing
                    or self.resources.last_wall >= self.resources.science_deadline):
                self.resources.begin_closing()
                if not self.closing_requested:
                    self.closing_requested = True
                    if self.request_closing is not None:
                        # Contract: bounded nonblocking IPC signal, no wait for ack.
                        self.request_closing()
        except BaseException as error:
            self.error = error
            self.cleanup_tree()
            raise

    def start(self) -> None:
        self.tick()  # fail before spawn on missing monitoring
        def loop() -> None:
            while not self.stop_event.wait(.5):
                try:
                    self.tick()
                except BaseException:
                    return
        self.thread = threading.Thread(target=loop, daemon=True, name='study-resource-supervisor')
        self.thread.start()

    def finish(self, receipt: Path, *, child_exit: int | None, child_terminal: bool,
               write: Callable[[Path, dict[str, Any]], Any] = durable_receipt) -> None:
        self.complete = False
        try:
            if self.error or self.resources.failure:
                raise ResourceStop('CLEANUP_UNKNOWN')
            self.tick()
            self.resources.begin_closing()
            if (self.closing_requested and (self.closing_acknowledged is None
                    or self.closing_acknowledged() is not True)):
                raise ResourceStop('CLEANUP_UNKNOWN')
            if (type(child_exit) is not int or child_exit != 0 or child_terminal is not True
                    or self.resources.tree.empty() is not True):
                raise ResourceStop('CLEANUP_UNKNOWN')
            evidence = write(receipt, dict(status='RESOURCE_CLOSED', **self.resources.latest))
            if isinstance(evidence, dict):
                self.receipt_evidence = evidence
            # Includes rename/fsync time. Persisted receipt alone never means complete.
            self.resources.check(science=False)
            self.stop_event.set()
            if self.thread:
                self.thread.join(1.)
                if self.thread.is_alive():
                    raise ResourceStop('CLEANUP_UNKNOWN')
            if self.error:
                raise ResourceStop('CLEANUP_UNKNOWN')
            self.resources.check(science=False)  # joins consume the same 28800s window
            # A background tick may have requested closing during receipt I/O,
            # after the first ACK gate. Joined sampler cannot add another request.
            if (self.closing_requested and (self.closing_acknowledged is None
                    or self.closing_acknowledged() is not True)):
                raise ResourceStop('CLEANUP_UNKNOWN')
            self.complete = True
        except BaseException:
            self.error = ResourceStop('FINALIZE_FAILED')
            self.cleanup_tree()
            raise
        finally:
            self.stop_event.set()
