# ruff: noqa: UP007, UP045
# Keep runtime-evaluable annotations Python 3.9 safe (no PEP604).
"""Single local Python 3.9 teacher adapter, never a task service.

stdin carries public data only. No SSH/Git, file tasks, retries, ledger or B key.
The parent owns process-tree containment. Missing runtime GPU attestor fails
closed; Task3 must supply the hook from an audited same-Git module entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any, BinaryIO, Optional

from .teacher_network import (
    ALLOCATOR_MIB,
    MODEL_SPEC,
    GpuAttestor,
    configure_tensorflow,
    make_official_teacher,
    points_to_bits,
    prepare_environment,
)
from .teacher_protocol import (
    READY,
    REPLY,
    REQUEST,
    DeadlineIO,
    Frame,
    ProtocolError,
    pack_reply,
    read_frame,
    write_frame,
)


def serve(reader: BinaryIO, writer: BinaryIO, deadline: float,
          ready: dict[str, Any], predict: Callable[[bytes], Sequence[float]], *,
          clock: Callable[[], float] = time.monotonic) -> None:
    """Exactly one request at a time; any EOF or error exits, no resync/retry.

    Both pipes are unbuffered in main. On timeout main uses os._exit so a worker
    blocked in a Windows pipe cannot hang interpreter finalization.
    """
    source, sink = DeadlineIO(reader, clock=clock), DeadlineIO(writer, clock=clock)
    raw = json.dumps(ready, allow_nan=False, separators=(',', ':')).encode('utf-8')
    write_frame(sink, Frame(READY, 0, 0, raw), deadline)
    seq = 1
    while True:
        request = read_frame(source, deadline, expected_kind=REQUEST, expected_seq=seq)
        probs = predict(request.payload)
        if len(probs) != request.count:
            raise ProtocolError('network returned wrong row count')
        reply = pack_reply(probs)
        write_frame(sink, Frame(REPLY, seq, request.count, reply), deadline)
        seq += 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--weights', required=True)
    result.add_argument('--sha256', required=True)
    result.add_argument('--gpu-uuid', required=True)
    result.add_argument('--gpu-pci', required=True)
    result.add_argument('--deadline', required=True, type=float)
    return result


def run_provider(args: argparse.Namespace, reader: BinaryIO, writer: BinaryIO,
                 attestor: Optional[GpuAttestor] = None, tf: Any = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 publisher_check: Callable[[], None] = lambda: None) -> None:
    """Explicit injection seam for Task3 attestation / synthetic fake TF tests.

    Actual weights may only be opened in authorized science, never engineering.
    Hash keeps a file handle and rechecks path identity after load_weights. The
    parent must keep the pinned asset read-only: Keras requires a pathname, so
    this is change detection, not an OS-enforced immutable-file guarantee.
    """
    if not math.isfinite(args.deadline) or clock() >= args.deadline:
        raise TimeoutError('teacher provider deadline')
    prepare_environment(args.gpu_uuid)
    if attestor is None:
        raise RuntimeError('GPU_UNVERIFIED: provider runtime attestor required')
    with open(args.weights, 'rb') as weights:
        digest = hashlib.sha256()
        for chunk in iter(lambda: weights.read(1 << 20), b''):
            if clock() >= args.deadline:
                raise TimeoutError('teacher weight hashing deadline')
            digest.update(chunk)
        sha = digest.hexdigest()
        before = os.fstat(weights.fileno())
        if sha != args.sha256:
            raise ProtocolError('weight SHA256 mismatch')
        publisher_check()
        if tf is None:
            tf = importlib.import_module('tensorflow')
        gpu = configure_tensorflow(tf, args.gpu_uuid, args.gpu_pci, attestor)
        model = make_official_teacher(tf)
        model.load_weights(args.weights)
        after = os.stat(args.weights)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ProtocolError('weights changed during load')
    ready = {'python': platform.python_version(), 'tensorflow': tf.__version__,
             'model': MODEL_SPEC, 'weights_sha256': sha, 'gpu': gpu,
             'allocator_mib': ALLOCATOR_MIB}

    def predict(payload: bytes) -> Sequence[float]:
        # Eager inference only, never compile/fit/tf.function/optimizer.
        publisher_check()
        if clock() >= args.deadline:
            raise TimeoutError('teacher provider deadline')
        with tf.device('/GPU:0'):
            output = model(points_to_bits(payload), training=False).numpy()
        if output.shape != (len(payload) // 8, 1):
            raise ProtocolError('network output shape mismatch')
        return [float(value) for value in output[:, 0]]

    serve(reader, writer, args.deadline, ready, predict, clock=clock)


def main(attestor: Optional[GpuAttestor] = None) -> int:
    args = parser().parse_args()
    # Native libraries/Python diagnostics cannot contaminate the binary stream.
    # Preserve a private duplicated stdout fd, route ordinary stdout to stderr.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    writer = os.fdopen(protocol_fd, 'wb', buffering=0)
    reader = os.fdopen(os.dup(sys.stdin.fileno()), 'rb', buffering=0)
    if sys.platform == 'win32':
        import msvcrt
        msvcrt.setmode(reader.fileno(), os.O_BINARY)
        msvcrt.setmode(writer.fileno(), os.O_BINARY)
    publisher = None
    try:
        if os.environ.get('SPECK_STUDY_NATIVE') != '1':
            raise RuntimeError('native parent contract required')
        from pathlib import Path

        from cipher_lab.platform_resources.dxgi import DxgiUsage
        from cipher_lab.speck_study.gpu_telemetry import (
            TelemetryPublisher,
            current_process_identity,
            windows_monotonic,
        )
        origin = float(os.environ['SPECK_STUDY_ORIGIN'])
        deadline = float(os.environ['SPECK_STUDY_SCIENCE_DEADLINE'])
        if (not math.isfinite(origin) or deadline != origin+28200
                or args.deadline != deadline or windows_monotonic() < origin):
            raise RuntimeError('invalid formal origin')
        root = Path(os.environ['SPECK_STUDY_CONTROL_ROOT'])
        pid, created = current_process_identity()
        publisher = TelemetryPublisher(root,root/'provider-telemetry.json',pid=pid,
                                       created=created,sample=DxgiUsage().sample)
        publisher.publish_once()  # explicit measured bootstrap, before TF import
        publisher.start()
        run_provider(args, reader, writer, attestor, clock=windows_monotonic,
                     publisher_check=publisher.check)
    except BaseException:
        # Avoid leaking arguments, arbitrary library exception repr, or secrets.
        os.write(2, b'teacher provider stopped (protocol/environment/inference failure)\n')
        return 2
    return 0


if __name__ == '__main__':
    # os._exit intentionally skips buffered pipe/destructor joins on partial frame.
    os._exit(main())
