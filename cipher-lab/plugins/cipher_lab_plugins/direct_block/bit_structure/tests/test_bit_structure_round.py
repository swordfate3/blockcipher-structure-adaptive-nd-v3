from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.bit_structure import config, training
from cipher_lab_plugins.direct_block.fit import Engine, Floats, Labels
from numpy.typing import NDArray

from cipher_lab.artifacts import RunStore
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter


def test_two_rounds_share_meter_and_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ('TRAIN', 'DEV', 'TEST', 'BATCH', 'EPOCHS', 'WARMUP', 'PATIENCE'):
        monkeypatch.setattr(config, key, 2 if key != 'PATIENCE' else 1)
    monkeypatch.setattr(config, 'MIN_EPOCHS', 2)
    deadline = (datetime.now(UTC) + timedelta(seconds=120)).isoformat()
    policy = Policy(2, 120, 120, deadline, rss_stop_bytes=config.RSS_BYTES,
                    output_bytes=config.OUTPUT_BYTES, **config.CAPS)
    meter = Meter(policy, Clocks(), lambda: 1)
    prefixes = ('c4-', 'c5-')
    store = RunStore.claim(tmp_path, 'fixture', output_bytes=config.OUTPUT_BYTES,
                           needs_models=True,
                           model_slots=tuple(p + s for p in prefixes for s in config.SLOTS),
                           source_supplier=lambda: {'fixture/engine.py': 'a' * 64})
    summary: dict[str, object] = {}
    events: list[object] = []
    params = {'mlp': 3, 'correct': 2, 'shuffled': 2}

    class Panels:
        def points(self, repeat: int, split: str, /) -> NDArray[np.uint32]:
            return np.array([0, 1], dtype=np.uint32)

        def label(self, repeat: int, split: str, /) -> Labels:
            events.append((repeat, split))
            meter.reserve('real_blocks', 2)
            meter.complete('real_blocks', 2)
            return np.array([[0], [1]], dtype=np.uint8)

    class FakeEngine:
        def __init__(self, slot: str) -> None:
            self.slot = slot

        def train_forward(self, inputs: Floats) -> Callable[[Labels], float | None]:
            return lambda _: 1.0

        def update(self) -> None:
            pass

        def predict(self, inputs: Floats) -> Floats:
            return np.zeros((len(inputs), 1), dtype=np.float32)

        def dump(self) -> bytes:
            return self.slot.encode()

        def restore(self, raw: bytes) -> None:
            assert raw == self.slot.encode()
            events.append('restore')

    def make(slot: str, arm: dict[str, object], seed: int) -> Engine:
        arm.update(initial_sha256='a' * 64, parameters=params[slot.split('-', 2)[2]])
        return FakeEngine(slot)

    meter.reserve('real_blocks', 7)
    meter.complete('real_blocks', 7)
    results = []
    for prefix in prefixes:
        events.clear()
        result = training.run_round(meter, store, summary, panels=Panels(), make=make,
                                    prefix=prefix, parameters=params)
        assert events[-8:] == ['restore'] * 6 + [(0, 'test'), (1, 'test')]
        assert set(result) == {'arms', 'panel_hashes', 'test', 'confirmation_candidate',
                               'epochs_completed', 'dev_evaluations', 'train_presentations'}
        assert result['epochs_completed'] == result['dev_evaluations'] == 12
        assert result['train_presentations'] == 24
        assert result['confirmation_candidate'] is False
        arms = cast(dict[str, dict[str, object]], result['arms'])
        assert tuple(arms) == tuple(prefix + slot for slot in config.SLOTS)
        for slot, arm in arms.items():
            assert arm['best_reloaded'] is True
            selected = cast(dict[str, object], arm['selected'])
            assert selected['ref'] == f'model/{slot}.pt'
        hashes = cast(dict[str, dict[str, str]], result['panel_hashes'])
        tests = cast(dict[str, dict[str, object]], result['test'])
        assert set(tests) == set(hashes) == {'r1', 'r2'}
        assert all(set(panel) == {'train', 'dev', 'test'} for panel in hashes.values())
        assert all(set(panel) == {'arms', 'constant', 'comparisons'} for panel in tests.values())
        assert summary['phase'] == 'TEST'
        assert 'active_slot' not in summary and 'test' not in summary
        results.append(result)
        store.write_snapshot({'counts': meter.snapshot(), 'summary': summary})

    assert results[0]['arms'] is not results[1]['arms']
    counts = meter.snapshot()
    for status in ('attempted', 'completed'):
        assert counts[f'{status}_real_blocks'] == 31
        assert counts[f'{status}_optimizer_updates'] == 24
        assert counts[f'{status}_forward_rows'] == 120
        assert counts[f'{status}_null_words'] == 0
