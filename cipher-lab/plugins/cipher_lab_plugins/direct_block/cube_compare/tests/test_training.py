from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare import config, data

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter


class ManualEngine:
    def __init__(self, arm: str, events: list[str], fault: str = '') -> None:
        self.arm, self.events, self.fault = arm, events, fault
        self.step = 0
        self.loaded = False
        self.orders: list[bytes] = []

    def train_forward(self, inputs: data.Floats) -> Callable[[data.Labels], float]:
        self.orders.append(inputs.tobytes())

        def backward(labels: data.Labels) -> float:
            assert labels.shape == (8, 128)
            return 0.125

        return backward

    def update(self) -> None:
        self.step += 1
        self.events.append(f'update/{self.arm}')

    def predict(self, inputs: data.Floats) -> data.Floats:
        if self.loaded:
            self.events.append(f'test/{self.arm}')
            if self.fault == 'score':
                raise ValueError('private sentinel')
        return np.full_like(inputs, -self.step)

    def dump(self) -> bytes:
        return f'{self.arm}/{self.step}'.encode()

    def restore(self, raw: bytes) -> None:
        self.events.append(f'reload/{self.arm}')
        assert raw == f'{self.arm}/4'.encode()
        if self.fault == 'reload':
            raise ValueError('private sentinel')
        self.loaded = True


class ManualPanels(data.CubePanels):
    def __init__(self, meter: Meter, events: list[str]) -> None:
        super().__init__(np.zeros_like, meter, families=(2, 1, 2))
        self.events = events

    def label_test(self) -> data.Panel:
        assert self.events[-2:] == ['reload/mlp', 'reload/sharedm']
        self.events.append('label/test')
        return super().label_test()


@pytest.mark.parametrize('fault', ['', 'reload', 'score'])
def test_dual_assembly(tmp_path: Path, cube_meter: Meter, fault: str) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import training

    store = RunStore.claim(tmp_path, 'manual', output_bytes=config.OUTPUT_BYTES,
                           needs_models=True, model_slots=config.SLOTS,
                           source_supplier=lambda: {'test/source.py': 'f' * 64})
    events: list[str] = []
    engines = {arm: ManualEngine(arm, events, fault if arm == 'sharedm' else '')
               for arm in config.ARMS}
    summary: dict[str, object] = {}

    def engine(arm: Literal['mlp', 'sharedm']) -> ManualEngine:
        return engines[arm]

    def execute() -> dict[str, object]:
        return training.run(cube_meter, store, summary, make_engine=engine,
                            make_panels=lambda meter: ManualPanels(meter, events),
                            plan=training.Plan((2, 1, 2), 2, 8))

    if fault:
        with pytest.raises(ValueError, match='private sentinel'):
            execute()
        assert ('label/test' in events) == (fault == 'score')
        assert not (tmp_path / 'data/manual/dataset/test.npz').exists()
    else:
        result = execute()
        assert result['denominator_bits'] == 128
        assert engines['mlp'].orders == engines['sharedm'].orders
        assert events[:8] == ['update/mlp'] * 4 + ['update/sharedm'] * 4
        assert cube_meter.snapshot()['completed_forward_rows'] == 128
        for slot, arm in zip(config.SLOTS, config.ARMS, strict=True):
            assert (tmp_path / 'models/manual' / f'{slot}.pt').read_bytes() == (
                f'{arm}/4'.encode())
    assert 'test' not in summary
