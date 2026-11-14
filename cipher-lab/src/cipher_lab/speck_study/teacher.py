"""Python 3.10 client owning exactly one local, serial teacher subprocess.

No scientific ledger: caller reserves attempted rows BEFORE predict and completes
only after a validated reply. Partial/failed requests are never refunded/retried.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .teacher_network import ALLOCATOR_MIB, MODEL_SPEC
from .teacher_protocol import (
    READY,
    REPLY,
    REQUEST,
    DeadlineIO,
    Frame,
    ProtocolError,
    pack_points,
    read_frame,
    unpack_reply,
    write_frame,
)

STDERR_LIMIT = 1 << 20
GpuVerifier = Callable[[dict[str, Any], int, str, str], bool]


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError('duplicate ready field')
        result[key] = value
    return result


class TeacherClient:
    """Absolute monotonic deadline, no reconnect; close has separate bounded grace.

    gpu_pci_bus_id is frozen by the parent alongside UUID; verify_gpu MUST inspect
    runtime evidence independently (Task3). No default string-self-attestation.
    Provider PID is exposed for the parent's process-tree/Job supervisor.
    """

    def __init__(self, python_exe: str, provider_module: str, weights_path: str,
                 expected_sha: str, gpu_uuid: str, root: Path, deadline: float, *,
                 gpu_pci_bus_id: str, verify_gpu: GpuVerifier | None = None,
                 close_grace: float = 2.,
                 spawn: Callable[..., subprocess.Popen[bytes]] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: DeadlineIO | None = None
        self.writer: DeadlineIO | None = None
        self.stderr_thread: threading.Thread | None = None
        self.stderr_error: BaseException | None = None
        self.lock = threading.Lock()
        self.closed = False
        self.cleanup_complete = False
        self.seq = 0
        self.deadline = deadline
        self.close_grace = close_grace
        self.ready: dict[str, Any] = {}
        if (not math.isfinite(deadline) or deadline <= clock()
                or not math.isfinite(close_grace) or not 0 < close_grace <= 10):
            raise ValueError('invalid teacher deadline/grace')
        if not re.fullmatch(r'[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*', provider_module):
            raise ValueError('provider must be a source module, not a command')
        if not re.fullmatch('[0-9a-f]{64}', expected_sha):
            raise ValueError('invalid expected weight SHA256')
        if verify_gpu is None:
            raise ProtocolError('GPU_UNVERIFIED: parent verifier required')
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('teacher root must exist')
        env = os.environ.copy()
        if env.get('CUDA_VISIBLE_DEVICES') not in (None, gpu_uuid):
            raise ProtocolError('inherited GPU binding mismatch')
        env['CUDA_VISIBLE_DEVICES'] = gpu_uuid
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        for name in ('TMP', 'TEMP', 'TMPDIR', 'CUDA_CACHE_PATH', 'KERAS_HOME',
                     'XDG_CACHE_HOME'):
            path = root / ('teacher-' + name.lower())
            path.mkdir(exist_ok=False)
            env[name] = str(path)
        log = (root / 'teacher-stderr.log').open('xb')
        try:
            # Study passes WindowsJob.spawn: suspended/assigned before user code.
            # Legacy default deliberately remains Popen.
            self.process = (spawn or subprocess.Popen)(
                [python_exe, '-B', '-m', provider_module, '--weights', str(weights_path),
                 '--sha256', expected_sha, '--gpu-uuid', gpu_uuid,
                 '--gpu-pci', gpu_pci_bus_id, '--deadline', repr(deadline)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, env=env)
            assert self.process.stdin and self.process.stdout and self.process.stderr
            self.reader = DeadlineIO(self.process.stdout, clock=clock)
            self.writer = DeadlineIO(self.process.stdin, clock=clock)
            stderr = self.process.stderr

            def drain() -> None:
                remaining = STDERR_LIMIT
                try:
                    while True:
                        block = stderr.read(8192)
                        if not block:
                            break
                        if remaining:
                            chunk = block[:remaining]
                            log.write(chunk)
                            remaining -= len(chunk)
                except BaseException as exc:
                    self.stderr_error = exc
                finally:
                    log.close()

            self.stderr_thread = threading.Thread(target=drain, daemon=True,
                                                  name='teacher-stderr')
            self.stderr_thread.start()
            frame = read_frame(self.reader, deadline, expected_kind=READY, expected_seq=0)
            self.ready = self._validate_ready(frame.payload, expected_sha, gpu_uuid,
                                               gpu_pci_bus_id, verify_gpu)
        except BaseException:
            try:
                self.close()
            finally:
                if self.stderr_thread is None:
                    log.close()
            raise

    @property
    def pid(self) -> int:
        if self.process is None:
            raise RuntimeError('teacher not started')
        return self.process.pid

    def _validate_ready(self, payload: bytes, sha: str, uuid: str, pci: str,
                        verifier: GpuVerifier) -> dict[str, Any]:
        try:
            ready = json.loads(payload, object_pairs_hook=_unique,
                               parse_constant=lambda _: (_ for _ in ()).throw(
                                   ProtocolError('nonfinite ready JSON')))
        except (ValueError, UnicodeError) as exc:
            raise ProtocolError('invalid ready JSON') from exc
        fields = {'python', 'tensorflow', 'model', 'weights_sha256', 'gpu', 'allocator_mib'}
        if not isinstance(ready, dict) or set(ready) != fields:
            raise ProtocolError('invalid ready fields')
        if (not isinstance(ready['python'], str) or not ready['python'].startswith('3.9.')
                or not isinstance(ready['tensorflow'], str) or not ready['tensorflow']
                or json.dumps(ready['model'], sort_keys=True)
                != json.dumps(MODEL_SPEC, sort_keys=True)
                or ready['weights_sha256'] != sha
                or type(ready['allocator_mib']) is not int
                or ready['allocator_mib'] != ALLOCATOR_MIB
                or not isinstance(ready['gpu'], dict)):
            raise ProtocolError('teacher ready identity mismatch')
        gpu = ready['gpu']
        if (gpu.get('gpu_uuid') != uuid or gpu.get('pci_bus_id') != pci
                or type(gpu.get('pid')) is not int or gpu.get('pid') != self.pid
                or verifier(gpu, self.pid, uuid, pci) is not True):
            raise ProtocolError('GPU_UNVERIFIED: parent rejected runtime evidence')
        return ready

    def predict(self, points: Sequence[int]) -> tuple[tuple[float, ...], tuple[int, ...]]:
        if not self.lock.acquire(blocking=False):
            raise ProtocolError('one teacher request in flight')
        try:
            if self.closed or self.process is None:
                raise ProtocolError('teacher closed')
            try:
                if self.stderr_error is not None:
                    raise ProtocolError('teacher stderr drain failed')
                payload = pack_points(points)
                assert self.reader is not None and self.writer is not None
                self.seq += 1
                write_frame(self.writer, Frame(REQUEST, self.seq, len(points), payload),
                            self.deadline)
                frame = read_frame(self.reader, self.deadline, expected_kind=REPLY,
                                   expected_seq=self.seq)
                if frame.count != len(points):
                    raise ProtocolError('teacher reply row mismatch')
                return unpack_reply(frame.payload, frame.count)
            except BaseException:
                self.close()
                raise
        finally:
            self.lock.release()

    def close(self) -> None:
        """Always attempt kill/reap even after TERM/wait error; idempotent.

        Never block on pipe.close before the child is dead and I/O workers joined.
        A pathological unkillable OS process yields explicit UNKNOWN, not success.
        The external Task3 Job/process-tree supervisor remains the hard backstop.
        """
        process = self.process
        if self.cleanup_complete:
            return
        self.closed = True
        if process is None:
            self.cleanup_complete = True
            return
        end = time.monotonic() + self.close_grace
        errors: list[BaseException] = []
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=max(.01, self.close_grace / 4))
            except BaseException as exc:
                errors.append(exc)
        if process.poll() is None:
            try:
                process.kill()
            except BaseException as exc:
                errors.append(exc)
        try:
            process.wait(timeout=max(.01, end - time.monotonic()))
        except BaseException as exc:
            errors.append(exc)
        joined = all([transport.join(end) for transport in (self.reader, self.writer)
                      if transport is not None])
        if self.stderr_thread is not None:
            self.stderr_thread.join(max(0., end - time.monotonic()))
            joined = joined and not self.stderr_thread.is_alive()
        if joined and process.poll() is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.cleanup_complete = True
        else:
            raise RuntimeError('teacher cleanup UNKNOWN: process/pipe worker not reaped') from (
                errors[-1] if errors else None)

    def __enter__(self) -> TeacherClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
