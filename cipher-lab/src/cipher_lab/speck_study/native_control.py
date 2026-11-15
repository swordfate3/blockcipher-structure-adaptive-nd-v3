"""Bounded local parent/owner control, not scientific data or task execution.

One serial request/reply mailbox in the parent-owned supervision directory.
Files use Windows delete-sharing reads; no pickle, code, labels or secrets.
The parent's independent resource sampler never waits for this client.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cipher_lab.speck_study.gpu_telemetry import _read_frame_bytes, windows_monotonic
from cipher_lab.speck_study.resources import ResourceStop


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate control field')
        result[key] = value
    return result


def read_message(path: Path) -> dict[str, Any]:
    raw = _read_frame_bytes(path.parent, path)
    value = json.loads(raw, object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
    if type(value) is not dict:
        raise ValueError('control object required')
    return value


def write_message(path: Path, value: dict[str, Any]) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > 65536:
        raise ValueError('control message too large')
    pending = path.with_name(path.name+'.pending')
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def write_diagnostic(path: Path, value: dict[str, Any]) -> None:
    """Same atomic finite JSON mailbox plus exact target-byte readback.

    No power-loss/Windows namespace durability claim. Caller handles all failures
    without suppressing Job termination. This does not change RPC frame policy.
    """
    expected = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    write_message(path, value)
    actual = _read_frame_bytes(path.parent, path)
    if len(actual) != len(expected) or hashlib.sha256(actual).digest() != hashlib.sha256(
            expected).digest():
        raise ResourceStop('FINALIZE_FAILED')


_CONTROL_KINDS = frozenset({'check', 'closing', 'expect-provider', 'adopt-provider',
                            'verify', 'provider-reaped'})


class ControlClient:
    def __init__(self, root: Path, deadline: float, *,
                 clock: Callable[[], float] = windows_monotonic) -> None:
        self.root, self.deadline, self.clock = root, deadline, clock
        self.seq = 0
        self.lock = threading.Lock()
        self.failed = False

    def __call__(self, kind: str, **payload: Any) -> dict[str, Any]:
        if self.failed or not self.lock.acquire(blocking=False):
            raise ResourceStop('CLEANUP_UNKNOWN')
        try:
            self.seq += 1
            write_message(self.root/'request.json', dict(seq=self.seq, kind=kind, payload=payload))
            # Formal origin/deadline stays QPC; no local epoch conversion.
            end = min(self.deadline, self.clock()+5.)
            while self.clock() < end:
                response = self.root/'response.json'
                if response.exists():
                    value = read_message(response)
                    if value.get('seq') == self.seq:
                        if set(value) != {'seq','ok','value'} or value['ok'] is not True:
                            raise ResourceStop('CLEANUP_UNKNOWN')
                        if type(value['value']) is not dict:
                            raise ResourceStop('CLEANUP_UNKNOWN')
                        return value['value']
                    if type(value.get('seq')) is not int or value['seq'] > self.seq:
                        raise ResourceStop('CLEANUP_UNKNOWN')
                time.sleep(.01)
            raise ResourceStop('CLEANUP_UNKNOWN')
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()


class ControlServer:
    def __init__(self, root: Path, dispatch: Callable[..., dict[str, Any]], *,
                 on_stage: Callable[[str], None] | None = None,
                 on_error: Callable[[BaseException], None] | None = None) -> None:
        self.root, self.dispatch = root, dispatch
        self.on_stage, self.on_error = on_stage, on_error
        self.seq = 0
        self.digest = ''
        self.error: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def poll(self) -> None:
        path = self.root/'request.json'
        if not path.exists():
            return
        request = read_message(path)
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if request.get('seq') == self.seq and digest == self.digest:
            return
        if (set(request) != {'seq','kind','payload'} or type(request['seq']) is not int
                or request['seq'] != self.seq+1 or type(request['kind']) is not str
                or type(request['payload']) is not dict):
            raise ValueError('invalid control sequence')
        self.seq, self.digest = request['seq'], digest
        if self.on_stage is not None and request['kind'] in _CONTROL_KINDS:
            self.on_stage('control-'+request['kind'])
        result = self.dispatch(request['kind'], **request['payload'])
        write_message(self.root/'response.json', dict(seq=self.seq, ok=True, value=result))

    def start(self) -> None:
        def loop() -> None:
            while not self.stop_event.wait(.01):
                try:
                    self.poll()
                except BaseException as error:
                    self.error = error
                    if self.on_error is not None:
                        try:
                            self.on_error(error)
                        except BaseException:
                            pass  # preserve original error, parent still kills
                    return
        self.thread = threading.Thread(target=loop, daemon=True, name='study-control')
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(1.)
            if self.thread.is_alive():
                raise ResourceStop('CLEANUP_UNKNOWN')
        if self.error:
            raise ResourceStop('CLEANUP_UNKNOWN') from self.error


class ChildResources:
    """Child consumes admission only; no parent RunStore writes or science override."""
    def __init__(self, rpc: Callable[..., dict[str, Any]], channel: Any,
                 publisher_check: Callable[[], None]) -> None:
        self.rpc, self.channel, self.publisher_check = rpc, channel, publisher_check
        self.closing = False
        self.last_wall = 0.
        self.science_deadline = 0.
        self.latest: dict[str, Any] = {}
        self.r0: Callable[[Any], bool] | None = None

    def check(self, *, science: bool = True) -> None:
        self.publisher_check()
        if self.channel.requested():
            self.closing = True
        if science and self.closing:
            raise ResourceStop('SCIENCE_DEADLINE')
        value = self.rpc('check', science=science)
        wall, deadline = value['wall'], value['science_deadline']
        if not all(type(v) in (int,float) and math.isfinite(v) for v in (wall,deadline)):
            raise ResourceStop('CLOCK_UNAVAILABLE')
        if wall < self.last_wall:
            raise ResourceStop('CLOCK_UNAVAILABLE')
        self.last_wall, self.science_deadline, self.latest = wall, deadline, value['latest']
        if value.get('closing') is True:
            self.closing = True
        if science and (self.closing or wall >= deadline):
            raise ResourceStop('SCIENCE_DEADLINE')

    def begin_closing(self) -> None:
        self.closing = True
        self.rpc('closing')  # parent request precedes sole-owner save/ACK

    def formal_r0(self, data: Any) -> bool:
        self.check()
        return self.r0 is not None and self.r0(data) is True

    def acknowledge_closing(self) -> None:
        if self.channel.requested():
            self.channel.acknowledge(saved=True, quiesced=self.closing)
