"""Guarded experiment assembly with artificial panels and engines."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_lab.execution import Meter


@pytest.mark.parametrize('fail_restore', [False, True])
def test_test_panel_requires_all_restores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                          bridge_meter: Meter, fail_restore: bool) -> None:
    from dataclasses import replace

    from cipher_lab_plugins.innovation1.present_card_fresh import config as pure
    from cipher_lab_plugins.innovation1.present_card_p_layer_fusion import config
    from cipher_lab_plugins.innovation1.present_card_positions import training

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
        backbone_initial_sha = 'a' * 64
        backbone_geometry = {'example': 'b' * 64}
        initial_sha = 'e' * 64
        geometry = {'example': 'f' * 64}

        def __init__(self, slot: str) -> None:
            self.slot = slot
            self.parameters = config.PARAMETERS[slot]
            self.fusion_initial_sha = '' if slot == config.BASE else 'c' * 64

        def batch_update(self, x: Any, y: Any, active: Meter) -> float:
            active.reserve('forward_rows', len(x))
            active.complete('forward_rows', len(x))
            return .5

        def predict(self, x: Any) -> Any:
            return x

        def dump(self) -> bytes:
            return self.slot.encode()

        def restore(self, raw: bytes) -> None:
            assert raw == self.slot.encode()
            if fail_restore and self.slot == config.P_LAYER:
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
                     make=make, peaks=dict, _recipe=config)

    if fail_restore:
        with pytest.raises(ValueError, match='FAILED_THIRD_RESTORE'):
            run()
        assert not invoked_test
    else:
        run()
        result: Any = read_saved(tmp_path, 'positions')['summary']
        assert set(result['test']) == set(config.SLOTS)
        assert result['paired'][f'{config.P_LAYER}_vs_{config.SHUFFLED}']['both_correct'] == 4
    assert origins == [0, 18, 36]
