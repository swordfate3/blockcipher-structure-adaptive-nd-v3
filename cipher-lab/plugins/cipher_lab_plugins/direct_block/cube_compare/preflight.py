"""The single public-vector and artificial-label gate; no private provider."""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from hashlib import sha256
from typing import TYPE_CHECKING

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter

from . import config

if TYPE_CHECKING:
    from ..fit import Engine
    from .data import Words
    from .training import Arm

# SIMON64/128 authors' known-answer vector, in the core's word/packing order.
_PUBLIC_KEY = (0x03020100, 0x0B0A0908, 0x13121110, 0x1B1A1918)
_KAT_PLAINTEXT = 0x656B696C20646E75
_KAT_CIPHERTEXT = 0x44C8FC20B9DFA07A
_POINTS = (0, 1, 0xFFFFFFFF, 1 << 32, 1 << 63, 2**64 - 1,
           0x0123456789ABCDEF, 0xFEDCBA9876543210)


def public_cipher_checks(meter: Meter) -> None:
    import numpy as np

    from cipher_lab.ciphers import simon

    for rounds, points in ((44, (_KAT_PLAINTEXT,)), (config.ROUNDS, _POINTS)):
        size = len(points)
        meter.reserve('real_blocks', size)
        scalar = simon.encrypt_many(points, _PUBLIC_KEY, rounds)
        if (type(scalar) is not tuple or len(scalar) != size
                or any(type(x) is not int or not 0 <= x < 2**64 for x in scalar)
                or (rounds == 44 and scalar != (_KAT_CIPHERTEXT,))):
            raise ValueError('public scalar check')
        meter.complete('real_blocks', size)
        supplied: Words = np.array(points, dtype=np.uint64)
        meter.reserve('real_blocks', size)
        actual = simon.encrypt_batch(supplied, _PUBLIC_KEY, rounds)
        if (type(actual) is not np.ndarray or actual.dtype != np.uint64
                or actual.shape != (size,) or not actual.flags.c_contiguous
                or tuple(map(int, supplied)) != points or tuple(map(int, actual)) != scalar):
            raise ValueError('public batch check')
        meter.complete('real_blocks', size)


class DeviceError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def require_selector() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config.GPU_UUID:
        raise DeviceError('GPU_IDENTITY')


def gpu_gate() -> dict[str, object]:
    require_selector()
    import torch

    from .resources import device_identity

    count = torch.cuda.device_count()
    if type(count) is not int or count != 1:
        raise DeviceError('GPU_VISIBLE_COUNT')
    try:
        identity = device_identity()
    except ValueError:
        raise DeviceError('GPU_IDENTITY') from None
    free, total = torch.cuda.mem_get_info(config.GPU_DEVICE)
    if (type(free) is not int or type(total) is not int
            or not config.GPU_FREE_BYTES <= free <= total):
        raise DeviceError('GPU_FREE_MEMORY')
    # The runtime's free/total sample and the device's capacity are distinct.
    return {**identity, 'visible_count': count, 'entry_free_bytes': free,
            'entry_runtime_total_bytes': total}


def gpu_peaks() -> dict[str, int]:
    import torch

    allocated = torch.cuda.max_memory_allocated(config.GPU_DEVICE)
    reserved = torch.cuda.max_memory_reserved(config.GPU_DEVICE)
    if type(allocated) is not int or type(reserved) is not int or not 0 <= allocated <= reserved:
        raise ValueError('GPU_PEAKS')
    return {'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved}


def run(meter: Meter, store: RunStore, summary: dict[str, object], *,
        make_engine: Callable[[Arm], Engine]) -> None:
    """Exactly one gate. Immutable preflight files do not occupy science best slots."""
    import numpy as np

    from ..fit import assert_source, predict_batch

    start = meter.snapshot()
    assert_source(store)
    public_cipher_checks(meter)
    # Deterministic artificial labels cover all 128 output columns, independent
    # of the public cipher checks. No metric/accuracy from this synthetic gate.
    rows = np.arange(8, dtype=np.int32)[:, None]
    cols = np.arange(config.WIDTH, dtype=np.int32)[None, :]
    x = ((rows + cols // 3) % 2).astype(np.float32)
    y = ((rows // 2 + cols) % 2).astype(np.uint8)
    for arm in config.ARMS:
        assert_source(store)
        meter.check()
        # Distinguish an entered artificial GPU gate with unknown peaks from
        # a public-vector or admission failure that never entered it.
        summary.setdefault('cuda_peaks_after_preflight', None)
        engine = make_engine(arm)
        meter.reserve('forward_rows', len(x))
        backward = engine.train_forward(x)
        meter.complete('forward_rows', len(x))
        meter.check()
        loss = backward(y)
        del backward
        if loss is None or not math.isfinite(loss) or loss < 0:
            raise ValueError('preflight loss')
        meter.reserve('optimizer_updates', 1)
        engine.update()
        meter.complete('optimizer_updates', 1)
        meter.check()
        raw = engine.dump()
        if type(raw) is not bytes or not 0 < len(raw) <= config.CHECKPOINT_BYTES:
            raise ValueError('preflight checkpoint size')
        digest = sha256(raw).hexdigest()
        meter.check()
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})
        # If checkpoint I/O seals the store, retain only the committed GPU-phase
        # unknown marker, not the live summary or any uncommitted peak sample.
        previous_key = store._failure_evidence_key
        store._failure_evidence_key = 'cuda_peaks_after_preflight'
        try:
            ref = store.write_model(f'{arm}-preflight', raw)
            del raw
            raw = store.read_model(ref, digest, config.CHECKPOINT_BYTES)
        finally:
            store._failure_evidence_key = previous_key
        meter.check()
        engine.restore(raw)
        del raw
        predict_batch(engine, x, y, meter)
        del engine
    assert_source(store)
    meter.check()
    end = meter.snapshot()
    for kind, n in config.preflight_counts().items():
        for prefix in ('attempted', 'completed'):
            key = f'{prefix}_{kind}'
            before, after = start[key], end[key]
            if not isinstance(before, int) or after != before + n:
                raise ValueError('preflight counts')
    summary['preflight'] = {
        'status': 'PASSED', 'counts': config.preflight_counts(),
        'end_wall_elapsed_seconds': end['wall_elapsed_seconds'],
        'end_cpu_elapsed_seconds': end['cpu_elapsed_seconds'],
    }
    store.write_snapshot({'counts': end, 'summary': summary})
