from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.execution import Meter


@pytest.mark.parametrize('fault', ['', 'ceiling', 'write', 'sha', 'restore', 'source', 'test',
                                  'final-source', 'final-write'])
def test_single_fit_disk_gate_and_success_only_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter, fault: str,
) -> None:
    from cipher_lab_plugins.innovation1.mask_aux import kat
    from cipher_lab_plugins.innovation1.present_card_fresh import config, training

    # One tiny TRAIN batch, and eval tails (pair counts 8 + 4 on both splits).
    for name, value in dict(TRAIN_ROWS=8, DEV_ROWS=12, TEST_ROWS=12, BATCH_SIZE=8).items():
        monkeypatch.setattr(config, name, value)
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, real_blocks=68))
    source = {'toy/a.py': 'a'*64}
    store = RunStore.claim(tmp_path, 'once', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS, source_supplier=lambda: source.copy())
    store.write_snapshot({'recipe': config.recipe()})
    calls: list[str] = []
    dev_batches: list[int] = []
    test_batches: list[int] = []
    selected_epochs: list[int] = []
    step = 0

    def gate(m: Meter) -> None:
        calls.append('gate')
        kat.check(m, lambda p, k, r: 0x5579C1387B228445 if r == 31 else 71,
                  lambda p, k, r: np.full((1, 1), 0x5579C1387B228445 if r == (31,) else 71,
                                          dtype=np.uint64))

    class Panel:
        def __init__(self, rows: int, value: float) -> None:
            self.rows, self.value = rows, value

        def __len__(self) -> int:
            return self.rows

        def batch(self, at: Any) -> tuple[Any, Any]:
            return np.full((len(at), 1), self.value, np.float32), (at % 2).astype('u1')[:, None]

    class Panels:
        hashes = {'train': 'b'*64, 'dev': 'c'*64}

        def view(self, split: str) -> Panel:
            return {'train': Panel(8, 1), 'dev': Panel(12, 2), 'test': Panel(12, 3)}[split]

        def test(self) -> None:
            assert calls[-3:] == ['release-train', 'reload', 'restore']
            calls.append('test')
            meter.reserve('real_blocks', 24)
            meter.complete('real_blocks', 24)
            self.hashes = {**self.hashes, 'test': 'd'*64}

    def panels(m: Meter) -> Panels:
        assert calls == ['gate'] and m.snapshot()['completed_real_blocks'] == 4
        calls.append('train-dev')
        m.reserve('real_blocks', 40)
        m.complete('real_blocks', 40)
        return Panels()

    class Engine:
        parameters = config.PARAMETERS
        initial_sha = 'e'*64
        geometry = {'artificial': 'f'*64}

        def __init__(self, slot: str, arm: dict[str, object], train: bool) -> None:
            assert slot == config.SLOT
            self.arm: Any = arm
            self.training = train
            calls.append('fit' if train else 'reload')

        def batch_update(self, x: Any, y: Any, m: Meter) -> float:
            nonlocal step
            assert self.training and len(x) == 8
            assert m.snapshot()['attempted_optimizer_updates'] == step + 1
            m.reserve('forward_rows', len(x))
            m.complete('forward_rows', len(x))
            step += 1
            return .5

        def predict(self, x: Any) -> Any:
            if self.training:
                assert x[0, 0] == 2
                dev_batches.append(len(x))
                value = 1 / step if fault == 'ceiling' else float(step == 1)
            else:
                assert x[0, 0] == 3
                test_batches.append(len(x))
                if fault == 'test' and len(test_batches) == 2:
                    raise ValueError('FAKE_TEST_FORWARD')
                value = 0.
            return np.full((len(x), 1), value, np.float32)

        def dump(self) -> bytes:
            point = self.arm['dev'][-1]
            selected_epochs.append(point['epoch'])
            return str(point['epoch']).encode()

        def restore(self, raw: bytes) -> None:
            assert raw == str(selected_epochs[-1]).encode()
            if fault == 'restore':
                raise ValueError('FAKE_STRICT_RESTORE')
            calls.append('restore')
            if fault == 'source':
                source['toy/a.py'] = '9'*64

        def release(self) -> None:
            calls.append('release-train' if self.training else 'release-score')
            if fault == 'sha' and self.training:
                # Same length; detection must come from SHA, not length.
                (tmp_path / 'models' / 'once' / (config.SLOT+'.pt')).write_bytes(b'9')

        def activate(self) -> None:
            assert calls[-1] == 'test'
            calls.append('activate')

        def train_forward(self, x: Any) -> Any:
            pytest.fail('not batch_update')

        def update(self) -> None:
            pytest.fail('second update')

    if fault in ('write', 'final-write'):
        atomic = RunStore._atomic

        def fail_write(self: RunStore, path: Path, content: bytes) -> None:
            if ((fault == 'write' and path.suffix == '.pt')
                    or (fault == 'final-write' and path.name == 'run.json'
                        and json.loads(content)['status'] == 'SUCCEEDED')):
                raise OSError('artificial write failure')
            atomic(self, path, content)

        monkeypatch.setattr(RunStore, '_atomic', fail_write)

    summary: dict[str, object] = {}

    def peaks() -> dict[str, int]:
        if fault == 'final-source':
            source['toy/a.py'] = '9'*64
        return {}

    def run() -> None:
        training.run(meter, store, summary, gate=gate, make=Engine, make_panels=panels,
                     peaks=peaks)

    failures = {'write': (ArtifactError, 'WRITE_FAILED'), 'sha': (ArtifactError, 'HASH_MISMATCH'),
                'restore': (ValueError, 'FAKE_STRICT_RESTORE'),
                'source': (ArtifactError, 'SOURCE_CHANGED'),
                'test': (ValueError, 'FAKE_TEST_FORWARD'),
                'final-source': (ArtifactError, 'SOURCE_CHANGED'),
                'final-write': (ArtifactError, 'WRITE_FAILED')}
    if fault in failures:
        error, reason = failures[fault]
        with pytest.raises(error, match=reason):
            run()
        if fault not in ('write', 'sha', 'final-write'):
            store.finalize({'status': 'FAILED', 'summary': summary, 'counts': meter.snapshot()})
        saved: Any = json.loads((tmp_path / 'data' / 'once' / 'run.json').read_text())
        assert saved['status'] == 'FAILED'
        assert saved['counts']['completed_optimizer_updates'] == step
        assert 'test' not in saved['summary']
        assert 'test' not in saved['summary'].get('panel_hashes', {})
        assert calls.count('test') == (fault in ('test', 'final-source', 'final-write'))
        return

    run()
    saved = read_saved(tmp_path, 'once')
    epochs = 30 if fault == 'ceiling' else 18
    assert saved['status'] == 'SUCCEEDED'
    assert saved['summary']['fit']['epochs_completed'] == step == epochs
    assert saved['summary']['fit']['stop_reason'] == (
        'EPOCH_CEILING' if fault == 'ceiling' else 'EARLY_STOPPING')
    assert saved['summary']['fit']['selected']['epoch'] == (30 if fault == 'ceiling' else 2)
    assert saved['summary']['fit']['best_reloaded'] is True
    assert dev_batches == [8, 4] * epochs and test_batches == [8, 4]
    assert calls.count('fit') == calls.count('test') == calls.count('restore') == 1
    assert list(saved['artifacts']) == [f'model/{config.SLOT}.pt']
    assert saved['counts']['completed_forward_rows'] == 20 * epochs + 12
    assert saved['summary']['test'] == {'unit': 'pair', 'pairs': 12, 'correct_pairs': 6,
                                       'bce': math.log(2), 'accuracy': .5,
                                       'by_label': {'0': {'pairs': 6, 'correct': 0},
                                                    '1': {'pairs': 6, 'correct': 6}},
                                       'confusion': {'tn': 0, 'fp': 6, 'fn': 0, 'tp': 6},
                                       'reference_accuracy': .5}
