from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter


@pytest.mark.parametrize('bad_restore', [False, True])
def test_four_strict_reloads_before_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         bad_restore: bool) -> None:
    from cipher_lab_plugins.direct_block.fit import Floats, Labels
    from cipher_lab_plugins.direct_block.full32_frontier import config, data, training

    for name in ('TRAIN', 'DEV', 'TEST', 'BATCH'):
        monkeypatch.setattr(config, name, 2)
    monkeypatch.setattr(config, 'EPOCHS', 1)
    policy = Policy(2, 60, 60, (datetime.now(UTC) + timedelta(seconds=60)).isoformat(),
                    rss_stop_bytes=config.RSS_BYTES, output_bytes=config.OUTPUT_BYTES,
                    **config.CAPS)
    meter = Meter(policy, Clocks(), lambda: 1)
    store = RunStore.claim(tmp_path, 'probe', output_bytes=config.OUTPUT_BYTES,
                           needs_models=True, model_slots=config.SLOTS,
                           source_supplier=lambda: {'fixture/model.py': 'a' * 64})
    events: list[str] = []

    class Fake:
        def __init__(self, slot: str, depth: int) -> None:
            self.slot, self.depth = slot, depth

        def train_forward(self, plain: Floats) -> Callable[[Labels], float]:
            return lambda labels: .7

        def update(self) -> None:
            pass

        def predict(self, plain: Floats) -> Floats:
            return np.full((len(plain), 32), 1 if self.depth == 3 else -1, np.float32)

        def dump(self) -> bytes:
            return self.slot.encode()

        def restore(self, raw: bytes) -> None:
            assert raw == self.dump()
            events.append('RESTORE')
            if bad_restore and events.count('RESTORE') == 4:
                raise ValueError('fixture restore failure')

    # Keep label generation on the real data assembly with a substitute oracle.
    draws = iter((41, 42))

    def panels(m: Meter) -> data.CyclePanels:
        def encrypt(points: data.Words, key: data.Key, depth: int) -> data.Words:
            events.append(f'LABEL-{depth}')
            return np.full(points.shape, 0xffffffff, dtype=np.uint32)

        return data.make_panels(m, lambda _: next(draws), encrypt,
                                sample=lambda n, _: np.arange(n, dtype=np.uint32))

    def gate(m: Meter) -> None:
        m.reserve('real_blocks', 2)
        m.complete('real_blocks', 2)

    def execute() -> None:
        from cipher_lab_plugins.direct_block.fit import Engine

        def make(depth: int, slot: str, arm: dict[str, object], seed: int) -> Engine:
            arm.update(parameters=config.PARAMETERS[depth], initial_sha256='a' * 64)
            return Fake(slot, depth)

        training.run(meter, store, {}, gate=gate, make_panels=panels, make=make,
                     read_snapshot=lambda: read_saved(tmp_path, 'probe'))

    if bad_restore:
        with pytest.raises(ValueError, match='fixture restore failure'):
            execute()
        assert events == ['LABEL-3'] * 4 + ['RESTORE'] * 4
    else:
        execute()
        assert events[:10] == ['LABEL-3'] * 4 + ['RESTORE'] * 4 + ['LABEL-3'] * 2
        record = read_saved(tmp_path, 'probe')
        assert record['status'] == 'SUCCEEDED'
        assert meter.snapshot()['completed_forward_rows'] == config.counts(4)['forward_rows']
