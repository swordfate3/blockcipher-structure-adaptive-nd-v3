from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.innovation1.prefix_bridge import config, report, training

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter, Stopped


@pytest.mark.parametrize('fault', ['', 'KAT', 'LAST_RESTORE', 'LATE_SCORE', 'FIRST_SAVE',
                                   'SOURCE', 'DEADLINE'])
def test_full_fake_lifecycle_zero_retry_and_no_partial_terminal(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    monkeypatch.setattr(os, "fsync", lambda fd: None)  # temp fixture; keep actual atomic I/O
    loaded: list[str] = []
    made: list[str] = []
    test = False
    scored: list[str] = []
    stored: list[str] = []
    source = {'fake/a.py': 'a' * 64}
    store = RunStore.claim(tmp_path, 'manual', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS, source_supplier=lambda: dict(source))
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, real_blocks=302))
    for name, value in (('TRAIN_ROWS', 2), ('DEV_ROWS', 2), ('TEST_ROWS', 2), ('BATCH_SIZE', 2),
                        ('ROUNDSTATEOUTPUTS', 288)):
        monkeypatch.setattr(config, name, value)

    class Panel:
        def __init__(self, slot: str, split: str) -> None:
            self.slot, self.split = slot, split

        def __len__(self) -> int:
            return 2

        def batch(self, at: Any) -> tuple[Any, Any]:
            return (np.full((2, 4, 6, 64), int(self.split == 'test'), np.float32),
                    np.array([[0], [1]], np.uint8)[at])

    class Panels:
        hashes = {'train': 'b' * 64, 'dev': 'c' * 64}

        def train(self, slot: str) -> Panel:
            return Panel(slot, 'train')

        def dev(self, slot: str) -> Panel:
            return Panel(slot, 'dev')

        def view(self, split: str, slot: str) -> Panel:
            assert split == 'test' and test
            return Panel(slot, split)

        def test(self) -> None:
            nonlocal test
            assert not test and loaded == list(config.SLOTS)
            test = True
            self.hashes['test'] = 'd' * 64
            pay(96)

    def pay(n: int) -> None:
        meter.reserve('real_blocks', n)
        meter.complete('real_blocks', n)

    def gate(meter: Meter) -> None:
        pay(14)
        if fault == 'KAT':
            raise ValueError('secret-sentinel')

    def panels(meter: Meter) -> Any:
        pay(192)
        return Panels()

    class Engine:
        initial_sha = 'f' * 64
        parameters = 1

        def __init__(self, slot: str, training: bool) -> None:
            self.slot, self.training = slot, training
            self.epoch = 0
            made.append(slot)

        def activate(self) -> None:
            assert test and not self.training

        def release(self) -> None:
            pass

        def predict(self, inputs: Any) -> Any:
            if inputs[0, 0, 0, 0] == 1:
                scored.append(self.slot)
                if fault == 'LATE_SCORE' and self.slot == config.SLOTS[-1]:
                    raise ValueError('secret-sentinel')
                return np.array([[-1.], [1.]], np.float32)
            # S2 plateaus, everything else improves every seventh epoch.
            return np.full((2, 1), 0. if self.slot == 's2-only' else 1 / (1 + self.epoch // 7),
                           dtype=np.float32)

        def batch_update(self, inputs: Any, labels: Any, meter: Meter) -> float:
            assert self.training and not test
            self.epoch += 1
            meter.reserve('forward_rows', 2)
            meter.complete('forward_rows', 2)
            return .7

        def dump(self) -> bytes:
            return self.slot.encode()

        def restore(self, raw: bytes) -> None:
            assert not self.training and not test and raw == self.slot.encode()
            if fault == 'LAST_RESTORE' and self.slot == config.SLOTS[-1]:
                raise ValueError('secret-sentinel')
            loaded.append(self.slot)

    def make(slot: str, arm: dict[str, object], train: bool) -> Any:
        return Engine(slot, train)

    def encode(labels: Any, logits: Any) -> bytes:
        assert scored == list(config.SLOTS) and labels.shape == (2, 1) and len(logits) == 13
        return b'fake opaque NPZ'

    def fake_write(name: str, content: bytes) -> str:
        assert name == 'test-groups' and content == b'fake opaque NPZ'
        stored.append(name)
        return orig_dataset(name, content)

    orig_dataset = store.write_dataset
    monkeypatch.setattr(store, 'write_dataset', fake_write)
    orig_atomic = store._atomic

    def atomic(path: Path, content: bytes) -> None:
        if path.suffix == '.pt':
            if fault == 'FIRST_SAVE':
                raise OSError('secret-sentinel')
            if fault == 'SOURCE':
                source['fake/a.py'] = 'e' * 64
        orig_atomic(path, content)

    monkeypatch.setattr(store, '_atomic', atomic)
    if fault == 'DEADLINE':
        monkeypatch.setattr(meter, 'check', lambda: (_ for _ in ()).throw(Stopped('DEADLINE')))
    summary: dict[str, object] = {}
    store.write_snapshot({'recipe': config.recipe()})

    def run() -> None:
        training.run(meter, store, summary, gate=gate, make=make, make_panels=panels,
                     encode_test=encode, peaks=lambda: {'artificial': 1})

    if fault:
        with pytest.raises((ValueError, ArtifactError, Stopped)):
            run()
        saved = json.loads((tmp_path / 'data' / 'manual' / 'run.json').read_text())
        assert 'test' not in saved['summary'] and not stored
        assert test == (fault == 'LATE_SCORE')
        if fault in ('KAT', 'DEADLINE'):
            assert not made
        if fault == 'FIRST_SAVE':
            assert saved['counts']['completed_optimizer_updates'] == 1
            assert saved['counts']['completed_forward_rows'] == 4
        assert 'secret-sentinel' not in str(saved)
    else:
        run()
        saved = json.loads((tmp_path / 'data' / 'manual' / 'run.json').read_text())
        assert saved['status'] == 'SUCCEEDED'
        assert made == list(config.SLOTS) * 2 and scored == list(config.SLOTS)
        assert stored == ['test-groups']
        assert saved['summary']['fits']['s2-only']['epochs_completed'] == 18
        assert saved['summary']['fits']['r7-joint']['epochs_completed'] == 30
        assert len(saved['summary']['test']['rounds']) == 6
        assert {row['s2_slot'] for row in saved['summary']['test']['rounds']} == {'s2-only'}
        assert saved['counts']['completed_forward_rows'] == 4 * (18 + 12 * 30) + 2 * 13
        assert set(saved['artifacts']) == {*(f'model/{s}.pt' for s in config.SLOTS),
                                           'dataset/test-groups.npz'}
        assert saved['counts']['completed_optimizer_updates'] == 18 + 12 * 30
        page = report.report_files({**saved, 'run_ref': 'data/manual/run.json',
                                    'run_sha256': 'f' * 64})['report.md'].decode()
        assert '|s2-only|18|1|' in page and '|r7-joint|30|28|' in page
        assert page.count('2/2; 100.0000%;') == 18
        assert 'TEST（正确组/总组；ACC；BCE）' in page
