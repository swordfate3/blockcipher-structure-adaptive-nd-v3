from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter

Floats = NDArray[np.float32]
Labels = NDArray[np.uint8]


class ManualArm:
    def __init__(self, failure: str = '') -> None:
        self.steps = 0
        self.batches: list[bytes] = []
        self.failure = failure
        self.restored = False

    def train_forward(self, inputs: Floats) -> Callable[[Labels], float]:
        self.batches.append(inputs.tobytes())

        def backward(labels: Labels) -> float:
            assert labels.shape == (8, 128)
            assert np.array_equal(labels, 1 - inputs)
            return float(self.steps + 1)

        return backward

    def update(self) -> None:
        self.steps += 1

    def predict(self, inputs: Floats) -> Floats:
        from cipher_lab_plugins.direct_block.cube_compare.config import ACTIVE_COLUMNS

        assert self.steps in (3, 6, 9, 12, 15)
        strength = {3: 1, 6: 2, 9: 3, 12: 3, 15: 1}[self.steps]
        correct_sign = 1 - 2 * inputs
        logits = -strength * correct_sign
        logits[:, ACTIVE_COLUMNS] *= -1
        return logits

    def dump(self) -> bytes:
        if self.failure == 'oversize':
            return b'x' * (16 * 2**20 + 1)
        return f'fake-{self.steps}'.encode()

    def restore(self, raw: bytes) -> None:
        assert raw == b'fake-9'
        if self.failure == 'restore':
            raise ValueError('invalid fake restore')
        self.restored = True


@pytest.mark.parametrize('failure', ['', 'hash', 'restore', 'oversize', 'disk_size'])
def test_cube_shared_fit_epoch_order_selection_and_double_persisted_gate(
    cube_meter: Meter, tmp_path: Path, failure: str,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.config import ACTIVE_COLUMNS, CHECKPOINT_BYTES
    from cipher_lab_plugins.direct_block.cube_compare.data import CubePanels
    from cipher_lab_plugins.direct_block.fit import (
        FitSettings,
        fit,
        reload_then,
        seeded_order,
    )

    store = RunStore.claim(tmp_path, 'cube', needs_models=True, output_bytes=2**26,
                           model_slots=('mlp-best', 'sharedm-best'),
                           source_supplier=lambda: {'fake/synthetic.py': '0' * 64})
    frames: list[str] = []
    a, b = ManualArm(), ManualArm(failure)
    labels_called: list[int] = []

    def oracle(points: NDArray[np.uint64]) -> NDArray[np.uint64]:
        if len(labels_called) == 2:
            assert a.restored and b.restored
        labels_called.append(len(points))
        return ~points

    panels = CubePanels(oracle, cube_meter, families=(3, 1, 1))
    train, dev = panels.build('train'), panels.build('dev')
    summary: dict[str, object] = {}
    first: dict[str, object] = {}
    second: dict[str, object] = {}
    summary['arms'] = {'mlp': first, 'sharedm': second}
    options = FitSettings(epochs=5, batch_size=8, selected_columns=ACTIVE_COLUMNS,
                          checkpoint_bytes=CHECKPOINT_BYTES)
    order = seeded_order(len(train), 20261124)
    left = fit(train, dev, a, order, 'mlp-best', first, summary, options,
               cube_meter, store, phase=frames.append)
    if failure == 'oversize':
        with pytest.raises(ValueError, match='checkpoint size'):
            fit(train, dev, b, order, 'sharedm-best', second, summary, options, cube_meter, store)
        assert labels_called == [48, 16]
        return
    right = fit(train, dev, b, order, 'sharedm-best', second, summary, options, cube_meter, store)
    assert left is not None and right is not None
    assert a.batches == b.batches
    epochs = [a.batches[start:start + 3] for start in range(0, 15, 3)]
    assert len({tuple(epoch) for epoch in epochs}) == 5
    assert np.array_equal(order(1), np.random.default_rng(
        np.random.SeedSequence([20261124, 1])).permutation(24))
    points = first['dev']
    assert isinstance(points, list) and len(points) == 5
    assert [p['train_bce'] for p in points] == [2, 5, 8, 11, 14]
    assert points[2]['active_bce'] < points[0]['active_bce']
    assert points[2]['full_bce'] > points[0]['full_bce']
    selected = first['selected']
    assert isinstance(selected, dict) and selected['epoch'] == 3 and selected['step'] == 9
    assert left == ('model/mlp-best.pt', sha256(b'fake-9').hexdigest())
    assert frames.count('BEST_SAVE') == 3
    assert cube_meter.snapshot()['completed_forward_rows'] == 2 * (5 * 24 + 5 * 8)
    raw_path = tmp_path / 'models/cube/sharedm-best.pt'
    assert raw_path.read_bytes() == b'fake-9'
    if failure == 'hash':
        raw_path.write_bytes(b'fake-8')
    elif failure == 'disk_size':
        raw = b'0' * (CHECKPOINT_BYTES + 1)
        replacement = store.write_model('sharedm-best', raw, replace=True)
        right = replacement, sha256(raw).hexdigest()
    args = ((a, left, first), (b, right, second))
    if failure:
        reason = {'hash': 'HASH_MISMATCH', 'restore': 'invalid fake restore',
                  'disk_size': 'SIZE_LIMIT'}[failure]
        with pytest.raises((ArtifactError, ValueError), match=reason):
            reload_then(args, summary, options, cube_meter, store, panels.label_test)
        assert a.restored and not b.restored and labels_called == [48, 16]
    else:
        panel = reload_then(args, summary, options, cube_meter, store, panels.label_test)
        assert len(panel) == 8 and labels_called == [48, 16, 16]
        assert first['best_reloaded'] is second['best_reloaded'] is True
