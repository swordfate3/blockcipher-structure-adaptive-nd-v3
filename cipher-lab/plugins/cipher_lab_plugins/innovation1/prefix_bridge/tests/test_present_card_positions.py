from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_lab.execution import Meter


def test_mask_positions_and_no_target_bypass() -> None:
    from cipher_lab_plugins.innovation1.present_card_mask.masking import prepare as old
    from cipher_lab_plugins.innovation1.present_card_positions import config, masking

    inputs = np.ones((8, 1, 2, 64), dtype=np.float32)
    for got, want in zip(masking.prepare(inputs, 9, config.RANDOM), old(inputs, 9), strict=True):
        np.testing.assert_array_equal(got, want)
    sides = []
    for slot, start in ((config.FIRST, 0), (config.FOURTH, 3)):
        view, index, targets = masking.prepare(inputs, 9, slot)
        flags = view[:, :, :2].astype(bool)
        hidden = flags.any(axis=2)
        assert np.all(hidden.sum(axis=2) == 16)
        assert np.flatnonzero(hidden[0, 0]).tolist() == list(range(start, 64, 4))
        assert not view[:, :, 3:5][flags].any() and not view[:, :, 5][hidden].any()
        sides.append(flags[:, :, 1][:, :, start::4])
        np.testing.assert_array_equal(targets, inputs.reshape(-1)[index])
        changed = inputs.copy()
        changed[flags] = 0
        altered, _, replacement = masking.prepare(changed, 9, slot)
        np.testing.assert_array_equal(altered, view)
        assert not replacement.any()
    np.testing.assert_array_equal(sides[0], sides[1])
    with pytest.raises(ValueError):
        masking.prepare(inputs, 9, 'unknown')


def test_exact_shared_panel_budget() -> None:
    from cipher_lab_plugins.innovation1.present_card_positions import config

    assert config.counts((30, 30, 30)) == dict(real_blocks=1_327_684, null_words=0,
                                             optimizer_updates=9000, forward_rows=93_342_336)
    assert config.counts((18, 20), test=False)['optimizer_updates'] == 3800
    with pytest.raises(ValueError):
        config.counts((30, 30))


def test_next_arm_resets_mask_and_schedule(monkeypatch: pytest.MonkeyPatch) -> None:

    from cipher_lab_plugins.innovation1.prefix_bridge import runtime as bridge
    from cipher_lab_plugins.innovation1.present_card_fresh import config as original
    from cipher_lab_plugins.innovation1.present_card_positions import config, masking, runtime

    engine = runtime.Runtime(config.FOURTH)
    engine.origin = 4100
    engine._state = 'TRAINING'
    seen: list[tuple[int, str]] = []

    def prepare(x: np.ndarray, step: int, slot: str) -> None:
        seen.append((step, slot))

    def update(self: runtime.Runtime, x: np.ndarray, y: np.ndarray, meter: Meter) -> float:
        assert self._learning_rate(meter) == original.learning_rate(0)
        self._batch_aux = .7
        return .6

    class Counter:
        def snapshot(self) -> dict[str, int]:
            return dict(completed_optimizer_updates=4100, attempted_optimizer_updates=4101)

    monkeypatch.setattr(masking, 'prepare', prepare)
    monkeypatch.setattr(bridge.Runtime, 'batch_update', update)
    batch = np.zeros((original.BATCH_SIZE, 1, 2, 64), dtype=np.float32)
    labels = np.zeros((len(batch), 1), dtype=np.uint8)
    assert getattr(engine, 'batch_update')(batch, labels, Counter()) == .6
    assert seen == [(1, config.FOURTH)]
    assert engine.aux_rows == len(batch) and engine._masked is None


@pytest.mark.parametrize('fail_restore', [False, True])
def test_test_panel_requires_all_restores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                          bridge_meter: Meter, fail_restore: bool) -> None:
    from dataclasses import replace

    from cipher_lab_plugins.innovation1.present_card_fresh import config as pure
    from cipher_lab_plugins.innovation1.present_card_mask import config as prior
    from cipher_lab_plugins.innovation1.present_card_positions import config, training

    from cipher_lab.artifacts import RunStore, read_saved

    for owner in (config, pure):
        for name in ('TRAIN_ROWS', 'DEV_ROWS', 'TEST_ROWS'):
            monkeypatch.setattr(owner, name, 8)
    monkeypatch.setattr(pure, 'BATCH_SIZE', 8)
    monkeypatch.setattr(config, 'CAPS', config.counts((30, 30, 30)))
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, **config.CAPS))
    source = {'fixture/source.py': 'a' * 64}
    store = RunStore.claim(tmp_path, 'positions', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS, source_supplier=lambda: source)
    store.write_snapshot({'recipe': config.recipe()})
    restored: list[str] = []
    origins: list[int] = []
    invoked_test: list[int] = []

    def blocks(n: int) -> None:
        meter.reserve('real_blocks', n)
        meter.complete('real_blocks', n)

    class Panel:
        def __len__(self) -> int:
            return 8

        def batch(self, at: Any) -> tuple[Any, Any]:
            return np.zeros((len(at), 1), np.float32), (at % 2).astype('u1')[:, None]

    class Panels:
        hashes = {'train': 'b' * 64, 'dev': 'c' * 64}

        def view(self, split: str, slot: str) -> Panel:
            assert slot == config.SLOT and split in self.hashes
            return Panel()

        def test(self) -> None:
            assert tuple(restored) == config.SLOTS
            invoked_test.append(1)
            blocks(16)
            self.hashes = {**self.hashes, 'test': 'd' * 64}

    class Engine:
        parameters = prior.PARAMETERS
        initial_sha = 'e' * 64
        geometry = {'example': 'f' * 64}

        def __init__(self, slot: str) -> None:
            self.slot = slot

        def batch_update(self, x: Any, y: Any, active: Meter) -> float:
            active.reserve('forward_rows', 2 * len(x))
            active.complete('forward_rows', 2 * len(x))
            return .5

        def predict(self, x: Any) -> Any:
            return x

        def dump(self) -> bytes:
            return self.slot.encode()

        def restore(self, raw: bytes) -> None:
            assert raw == self.slot.encode()
            if fail_restore and self.slot == config.FOURTH:
                raise ValueError('FAILED_THIRD_RESTORE')
            restored.append(self.slot)

        def release(self) -> None:
            pass

        def activate(self) -> None:
            assert invoked_test == [1]

    def make(slot: str, arm: dict[str, object], fit: bool, origin: int) -> Any:
        if fit:
            origins.append(origin)
        return Engine(slot)

    def panels_for(active: Meter) -> Any:
        assert active is meter
        blocks(32)
        return Panels()

    def run() -> None:
        training.run(meter, store, {}, gate=lambda _: blocks(4), panels_for=panels_for,
                     make=make, peaks=dict)

    if fail_restore:
        with pytest.raises(ValueError, match='FAILED_THIRD_RESTORE'):
            run()
        assert not invoked_test
    else:
        run()
        result: Any = read_saved(tmp_path, 'positions')['summary']
        assert set(result['test']) == set(config.SLOTS)
        assert result['paired'][f'{config.FOURTH}_vs_{config.FIRST}']['both_correct'] == 4
    assert origins == [0, 18, 36]
