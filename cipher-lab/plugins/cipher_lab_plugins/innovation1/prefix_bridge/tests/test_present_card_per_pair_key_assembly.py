from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.innovation1.present_card_per_pair_key import config

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.execution import Kind, Meter


def test_caps() -> None:
    assert config.CAPS == dict(real_blocks=2_327_692, null_words=0,
                               optimizer_updates=6_000, forward_rows=32_228_224)
    assert config.SLOTS == ('fixed-key', 'per-pair-key')
    assert config.counts((), test=False)['real_blocks'] == 2_065_548
    with pytest.raises(ValueError):
        config.counts((30, 30), diagnostic=True)


@pytest.mark.parametrize('failure', [None, 'restore', 'identity', 'activation', 'hash'])
def test_matched_lifecycle(bridge_meter: Meter, monkeypatch: pytest.MonkeyPatch,
                           tmp_path: Path, failure: str | None) -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh import config as pure
    from cipher_lab_plugins.innovation1.present_card_per_pair_key import report
    from cipher_lab_plugins.innovation1.present_card_swap import training

    for owner in (pure, config):
        for key, value in [('TRAIN_ROWS', 8), ('DEV_ROWS', 6), ('TEST_ROWS', 10)]:
            monkeypatch.setattr(owner, key, value)
    monkeypatch.setattr(pure, 'BATCH_SIZE', 4)
    monkeypatch.setattr(config, 'CAPS', config.counts((30, 30)))
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, **config.CAPS))
    source = {'fixture/stub.py': 'a' * 64}
    store = RunStore.claim(tmp_path, 'matched', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS, source_supplier=lambda: source)
    store.write_snapshot({'recipe': config.recipe()})
    summary: dict[str, object] = {}
    hashes = {k: f'{i + 1:064x}' for i, k in enumerate(sorted(config.TRAIN_DEV_HASHES))}
    orders: dict[str, list[int]] = {slot: [] for slot in config.SLOTS}
    restored: list[str] = []
    activated: list[str] = []
    tested: list[bool] = []
    origins: list[int] = []

    def charge(counter: Kind, size: int) -> None:
        meter.reserve(counter, size)
        meter.complete(counter, size)

    class Panel:
        def __init__(self, marker: int, rows: int) -> None:
            self.marker, self.rows = marker, rows

        def __len__(self) -> int:
            return self.rows

        def batch(self, at: Any) -> tuple[Any, Any]:
            x = np.column_stack((at, np.full(len(at), self.marker))).astype('f4')
            return x, (at % 2).astype('u1')[:, None]

    class Panels:
        @property
        def hashes(self) -> dict[str, str]:
            return dict(hashes)

        def train(self, slot: str) -> Panel:
            return Panel(1 + config.SLOTS.index(slot), 8)

        def dev(self, slot: str) -> Panel:
            assert slot == config.SLOT
            return Panel(3, 6)

        def test(self) -> None:
            assert tuple(activated) == config.SLOTS and not tested
            tested.append(True)
            charge('real_blocks', 20)
            hashes['test'] = 'd' * 64
            if failure == 'hash':
                hashes['train-fixed-key'] = 'e' * 64

        def view(self, split: str, slot: str) -> Panel:
            assert tested and split == 'test' and slot == config.SLOT
            return Panel(4, 10)

    class Engine:
        parameters = pure.PARAMETERS
        classifier_initial_sha = 'a' * 64
        geometry = {'card': 'b' * 64}

        def __init__(self, slot: str, fitting: bool) -> None:
            self.slot, self.fitting = slot, fitting
            self.initial_sha = ('f' if failure == 'identity' and slot == config.PER_PAIR
                                else 'a') * 64
            self.updates = 0
            self.next_row = 0

        def batch_update(self, x: Any, y: Any, active: Meter) -> float:
            assert active is meter and self.fitting
            assert (x[:, 1] == 1 + config.SLOTS.index(self.slot)).all()
            assert np.array_equal(x[:, :1] % 2, y)
            orders[self.slot].extend(x[:, 0].astype(int))
            self.updates += 1
            charge('forward_rows', len(x))
            return .5

        def predict(self, x: Any) -> Any:
            assert (x[:, 1] == (3 if self.fitting else 4)).all()
            if not self.fitting:
                assert tested and self.slot in activated
            assert np.array_equal(x[:, 0], np.arange(self.next_row, self.next_row + len(x)))
            self.next_row += len(x)
            if self.fitting and self.next_row == 6:
                self.next_row = 0
            logits = 2 * (x[:, :1] % 2) - 1
            if not self.fitting and self.slot == config.FIXED:
                logits[::2] *= -1
            return logits

        def dump(self) -> bytes:
            return f'{self.slot}:{self.updates}'.encode()

        def restore(self, raw: bytes) -> None:
            assert raw == f'{self.slot}:2'.encode()  # Earliest tied DEV minimum.
            if failure == 'restore' and self.slot == config.PER_PAIR:
                raise ValueError('RESTORE_STUB')
            restored.append(self.slot)

        def activate(self) -> None:
            assert tuple(restored) == config.SLOTS
            if failure == 'activation' and self.slot == config.PER_PAIR:
                raise ValueError('ACTIVATION_STUB')
            activated.append(self.slot)

        def release(self) -> None:
            pass

    def make(slot: str, arm: dict[str, object], fit: bool, origin: int) -> Any:
        if fit:
            origins.append(origin)
        return Engine(slot, fit)

    def panels_for(active: Meter) -> Any:
        assert active is meter
        charge('real_blocks', 44)
        return Panels()

    def run() -> None:
        training.run(meter, store, summary, gate=lambda _: charge('real_blocks', 12),
                     panels_for=panels_for, make=make, peaks=dict, _recipe=config,
                     separate_train=True, train_dev_hashes=config.TRAIN_DEV_HASHES,
                     gate_name=config.KAT_NAME, gate_blocks=config.KAT_BLOCKS)

    if failure:
        with pytest.raises(ValueError, match={
            'restore': 'RESTORE_STUB', 'identity': 'MODEL_INITIAL_IDENTITY',
            'activation': 'ACTIVATION_STUB', 'hash': 'PANEL_IDENTITY',
        }[failure]):
            run()
        assert tested == ([True] if failure == 'hash' else [])
        assert 'test' not in summary and 'paired' not in summary
        return

    run()
    saved = read_saved(tmp_path, 'matched')
    result: Any = saved['summary']
    assert result['panel_hashes'] == hashes
    assert result['kat'] == dict(status='PASSED', name=config.KAT_NAME, real_blocks=12)
    assert 'dev_diagnostics' not in result
    assert result['paired'][config.CONTRAST]['percentage_points'] == 50
    assert origins == [0, 36] and orders[config.FIXED] == orders[config.PER_PAIR]
    assert meter.snapshot()['completed_forward_rows'] == 524
    assert meter.snapshot()['completed_optimizer_updates'] == 72
    assert meter.snapshot()['completed_real_blocks'] == 76
    assert b'+50.0000' in report.report_files(saved)['report.md']


def test_public_pair_gate_fixture(bridge_meter: Meter, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.present_card_per_pair_key import gate

    monkeypatch.setattr(bridge_meter, '_policy', replace(bridge_meter._policy, real_blocks=16))

    def scalar(point: int, key: tuple[int, int, int], rounds: int) -> int:
        assert rounds == 7
        return point + key[2]

    def vector(points: Any, low: Any, high: Any) -> Any:
        assert low.tolist() == [0, 0x0203040506070809]
        assert high.tolist() == [0, 1]
        return points + high[:, None]

    gate.check(bridge_meter, vector, scalar)
    assert bridge_meter.snapshot()['completed_real_blocks'] == 8
    with pytest.raises(ValueError, match='PUBLIC_PAIR_KEY_GATE'):
        gate.check(bridge_meter, lambda points, low, high: points.copy(), scalar)
    assert bridge_meter.snapshot()['attempted_real_blocks'] == 16
    assert bridge_meter.snapshot()['completed_real_blocks'] == 12


def test_static_recipe_and_source(capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.present_card_per_pair_key import cli, source
    from cipher_lab_plugins.innovation1.present_card_swap.source import source_identity

    path = Path('configs/innovation1-present-card-per-pair-key.json')
    assert json.loads(path.read_text(encoding='utf-8')) == config.recipe()
    assert cli.main(['check', '--config', str(path)]) == 0
    assert capsys.readouterr().err == ''
    parent = source_identity()
    extended = source.source_identity()
    assert parent.items() <= extended.items()
    prefix = 'cipher_lab_plugins/innovation1/present_card_per_pair_key/'
    assert extended.keys() - parent.keys() == {prefix + name + '.py' for name in (
        '__init__', 'cipher', 'cli', 'config', 'data', 'gate', 'report', 'runtime', 'source',
    )}
