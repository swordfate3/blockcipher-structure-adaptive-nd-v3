# ruff: noqa: UP007, UP045
# Keep runtime-evaluable annotations Python 3.9 safe (no PEP604).
"""Versioned, bounded data-only teacher wire format (Python 3.9 stdlib).

Deadlines use the explicitly supplied clock (legacy default time.monotonic;
native Windows raw QPC). join() uses only a LOCAL monotonic timeout. A timed-out blocking pipe is
poisoned: its process owner MUST kill/reap the child, then join the I/O thread.
There is no queue, retry, resynchronisation or reusable timed-out transport.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import math
import struct
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import IO, Optional, TypeVar, Union, cast

MAGIC = b'SPKTCHR\x00'
VERSION = 1
REQUEST, REPLY, READY = 1, 2, 3
MAX_ROWS = 50_000
MAX_READY = 16_384
HEADER = struct.Struct('>8sBBQII32s')
T = TypeVar('T')


class ProtocolError(ValueError):
    """Fatal malformed/truncated/closed protocol. Never retry this request."""


@dataclass(frozen=True)
class Frame:
    kind: int
    seq: int
    count: int
    payload: bytes


class DeadlineIO:
    """One active blocking operation; daemon thread never prevents shutdown.

    Windows anonymous pipes cannot be select()ed. Process termination closes the
    peer ends, unblocking the worker; join() lets the owner verify that fact.
    No caller-side close of a buffered stream while its reader owns its lock.
    """

    def __init__(self, stream: IO[bytes], *, clock: Callable[[], float] = time.monotonic):
        self.stream = stream
        self.clock = clock
        self.thread: Optional[threading.Thread] = None
        self.poisoned = False
        self.lock = threading.Lock()

    def run(self, operation: Callable[[], T], deadline: float) -> T:
        if not math.isfinite(deadline) or self.clock() >= deadline:
            raise TimeoutError('teacher IPC deadline')
        if not self.lock.acquire(blocking=False):
            raise ProtocolError('concurrent teacher IPC')
        try:
            if self.poisoned:
                raise ProtocolError('poisoned teacher IPC')
            result: list[T] = []
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    result.append(operation())
                except BaseException as exc:
                    errors.append(exc)

            self.thread = threading.Thread(target=worker, daemon=True, name='teacher-pipe')
            self.thread.start()
            self.thread.join(max(0., deadline - self.clock()))
            if self.thread.is_alive():
                self.poisoned = True
                raise TimeoutError('teacher IPC deadline')
            if errors:
                self.poisoned = True
                raise errors[0]
            return result[0]
        finally:
            self.lock.release()

    def join(self, deadline: float) -> bool:
        if self.thread is not None:
            self.thread.join(max(0., deadline - time.monotonic()))
            return not self.thread.is_alive()
        return True


def _exact(stream: IO[bytes], length: int) -> bytes:
    parts = bytearray()
    while len(parts) < length:
        block = stream.read(length - len(parts))
        if not block:
            raise ProtocolError('teacher EOF/partial frame')
        parts.extend(block)
    return bytes(parts)


def _shape(kind: int, seq: int, count: int, length: int, max_rows: int) -> None:
    if type(kind) is not int or type(count) is not int:
        raise ProtocolError('header fields must be exact integers')
    if type(seq) is not int or not 0 <= seq < 1 << 64:
        raise ProtocolError('sequence out of range')
    if kind == READY:
        if seq != 0 or count != 0 or not 1 <= length <= MAX_READY:
            raise ProtocolError('invalid ready size/sequence')
    elif kind in (REQUEST, REPLY):
        if seq == 0 or type(count) is not int or not 1 <= count <= max_rows:
            raise ProtocolError('invalid row count/sequence')
        if length != count * (8 if kind == REQUEST else 5):
            raise ProtocolError('invalid payload size')
    else:
        raise ProtocolError('unknown frame kind')


def read_frame(stream: Union[IO[bytes], DeadlineIO], deadline: float,
               max_rows: int = MAX_ROWS, *, expected_kind: Optional[int] = None,
               expected_seq: Optional[int] = None) -> Frame:
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise ProtocolError('invalid max rows')
    transport = stream if isinstance(stream, DeadlineIO) else DeadlineIO(stream)

    def read() -> Frame:
        magic, version, kind, seq, count, length, digest = HEADER.unpack(
            _exact(transport.stream, HEADER.size))
        if magic != MAGIC or version != VERSION:
            raise ProtocolError('wrong magic/version')
        _shape(kind, seq, count, length, max_rows)
        if expected_kind is not None and kind != expected_kind:
            raise ProtocolError('wrong frame kind')
        if expected_seq is not None and seq != expected_seq:
            raise ProtocolError('wrong sequence')
        payload = _exact(transport.stream, length)
        if not hmac.compare_digest(hashlib.sha256(payload).digest(), digest):
            raise ProtocolError('checksum mismatch')
        return Frame(kind, seq, count, payload)

    return transport.run(read, deadline)


def write_frame(stream: Union[IO[bytes], DeadlineIO], frame: Frame, deadline: float) -> None:
    if type(frame.payload) is not bytes:
        raise ProtocolError('payload must be bytes')
    _shape(frame.kind, frame.seq, frame.count, len(frame.payload), MAX_ROWS)
    raw = HEADER.pack(MAGIC, VERSION, frame.kind, frame.seq, frame.count,
                      len(frame.payload), hashlib.sha256(frame.payload).digest()) + frame.payload
    transport = stream if isinstance(stream, DeadlineIO) else DeadlineIO(stream)

    def write() -> None:
        offset = 0
        while offset < len(raw):
            n = transport.stream.write(raw[offset:])
            if n is None or n <= 0:
                raise ProtocolError('teacher pipe write failed')
            offset += n
        transport.stream.flush()

    transport.run(write, deadline)


def pack_points(points: Sequence[int]) -> bytes:
    if not 1 <= len(points) <= MAX_ROWS:
        raise ProtocolError('invalid points count')
    output = io.BytesIO()
    for point in points:
        if type(point) is not int or not 0 <= point < 1 << 64:
            raise ProtocolError('point must be literal uint64')
        output.write(struct.pack('>Q', point))
    return output.getvalue()


def unpack_points(payload: bytes) -> tuple[int, ...]:
    if not payload or len(payload) % 8 or len(payload) > MAX_ROWS * 8:
        raise ProtocolError('invalid packed points')
    return tuple(x[0] for x in struct.iter_unpack('>Q', payload))


def pack_reply(probs: Sequence[float]) -> bytes:
    if not 1 <= len(probs) <= MAX_ROWS:
        raise ProtocolError('invalid reply count')
    output = bytearray()
    for prob in probs:
        if not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ProtocolError('probability out of range')
        # Derive label AFTER float32 rounding; receiver applies the same rule.
        rounded = cast(float, struct.unpack('>f', struct.pack('>f', prob))[0])
        output.extend(struct.pack('>fB', rounded, int(rounded > .5)))
    return bytes(output)


def unpack_reply(payload: bytes, count: int) -> tuple[tuple[float, ...], tuple[int, ...]]:
    if type(count) is not int or not 1 <= count <= MAX_ROWS or len(payload) != count * 5:
        raise ProtocolError('invalid reply size')
    probs, labels = [], []
    for prob, label in struct.iter_unpack('>fB', payload):
        if not math.isfinite(prob) or not 0 <= prob <= 1 or label != int(prob > .5):
            raise ProtocolError('invalid probability/hardlabel')
        probs.append(prob)
        labels.append(label)
    return tuple(probs), tuple(labels)
