from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter


def test_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.direct_block.bit_structure import config, report, training
    from cipher_lab_plugins.direct_block.cube_compare import checkpoint

    for key in ('TRAIN', 'DEV', 'TEST', 'BATCH'):
        monkeypatch.setattr(config, key, 2)
    events: list[object] = []
    points = np.array([0, 1], dtype=np.uint32)
    labels = np.array([[0], [1]], dtype=np.uint8)

    class Panels:
        def points(self, repeat: int, split: str) -> Any:
            return points

        def label(self, repeat: int, split: str) -> Any:
            events.append((repeat, split))
            meter.reserve('real_blocks', 2)
            meter.complete('real_blocks', 2)
            return labels

    class Engine:
        def __init__(self, slot: str) -> None:
            self.model = torch.nn.Linear(32, 1)
            self.slot = slot

        def train_forward(self, inputs: Any) -> Any:
            return lambda y: 1.0

        def update(self) -> None:
            pass

        def predict(self, inputs: Any) -> Any:
            return np.zeros((len(inputs), 1), dtype=np.float32)

        def dump(self) -> bytes:
            identity = config.SLOTS[0] if fail and self.slot == config.SLOTS[-1] else self.slot
            return checkpoint.dump(self.model, identity=identity)

        def restore(self, raw: bytes) -> None:
            events.append('restore')
            checkpoint.restore(self.model, raw, identity=self.slot)

    def gate(meter: Meter) -> None:
        meter.reserve('real_blocks', 2)
        meter.complete('real_blocks', 2)

    for fail in (True, False):
        events.clear()
        deadline = (datetime.now(UTC) + timedelta(seconds=120)).isoformat()
        policy = Policy(2, 120, 120, deadline, rss_stop_bytes=config.RSS_BYTES,
                        output_bytes=config.OUTPUT_BYTES, **config.CAPS)
        meter = Meter(policy, Clocks(), lambda: 1)
        run_id = f'fixture-{int(fail)}'
        store = RunStore.claim(tmp_path, run_id, output_bytes=config.OUTPUT_BYTES,
                               needs_models=True, model_slots=config.SLOTS,
                               source_supplier=lambda: {'fixture/engine.py': 'a' * 64})
        store.write_snapshot({'recipe': config.recipe()})
        summary: dict[str, object] = {}

        def execute() -> None:
            def make(slot: str, arm: dict[str, object], seed: int) -> Engine:
                arm['initial_sha256'] = 'a' * 64
                arm['parameters'] = config.PARAMS[slot.split('-', 1)[1]]
                return Engine(slot)

            training.run(meter, store, summary, gate=gate, make_panels=lambda _: Panels(),
                         make=make)

        if fail:
            with pytest.raises(ValueError, match='invalid checkpoint identity'):
                execute()
            assert events.count('restore') == 6
            assert summary['phase'] == 'BEST_RELOAD' and 'test' not in summary
            assert not any(isinstance(e, tuple) and e[1] == 'test' for e in events)
        else:
            execute()
            assert events[-8:] == ['restore'] * 6 + [(0, 'test'), (1, 'test')]
            saved = read_saved(tmp_path, run_id)
            files = report.report_files(saved)
            assert '正确接线减打乱接线' in files['report.md'].decode()
            counts = cast(dict[str, Any], saved['counts'])
            assert counts['completed_optimizer_updates'] == 6 * 18
            assert counts['completed_forward_rows'] == 6 * 18 * 4 + 6 * 2
            assert counts['completed_real_blocks'] == 14
