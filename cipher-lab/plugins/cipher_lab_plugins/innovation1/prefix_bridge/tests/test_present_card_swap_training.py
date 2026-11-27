from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.execution import Meter


def test_fixed_work_caps() -> None:
    from cipher_lab_plugins.innovation1.present_card_swap import config

    assert config.SLOTS == ('original', 'swap-50') and config.SWAP_SEED == 20261128
    assert config.CAPS == dict(real_blocks=1_327_684, null_words=0,
                               optimizer_updates=6_000, forward_rows=32_359_296)
    assert config.recipe()['parameters'] == 1_087_509


def test_check_rejects_recipe_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.present_card_swap import cli, config

    path = tmp_path / 'recipe.json'
    changed = {**config.recipe(), 'swap_seed': config.SWAP_SEED + 1}
    path.write_text(json.dumps(changed), encoding='utf-8')
    assert cli.main(['check', '--config', str(path)]) == 2
    capture = capsys.readouterr()
    assert capture.out == ''
    assert json.loads(capture.err)['reason'] == 'INVALID_ARGUMENTS_OR_CONFIG'


def test_source_extends_fresh_card() -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh.source import source_identity as fresh
    from cipher_lab_plugins.innovation1.present_card_swap.source import source_identity

    baseline = fresh()
    extended = source_identity()
    assert {key: extended[key] for key in baseline} == baseline
    extra = {'innovation1/present_card_swap/' + name + '.py' for name in (
        '__init__', 'cli', 'config', 'diagnostics', 'report', 'runtime', 'source', 'training',
        'views',
    )} | {'innovation1/mask_aux/training.py'}
    assert extended.keys() - baseline.keys() == {'cipher_lab_plugins/' + path for path in extra}
    for path in extra:
        key = 'cipher_lab_plugins/' + path
        assert extended[key] == sha256(Path('plugins', key).read_bytes()).hexdigest()


@pytest.mark.parametrize('failure', [None, 'restore', 'diagnostic', 'test'])
def test_two_fit_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                           bridge_meter: Meter, failure: str | None) -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh import config as pure
    from cipher_lab_plugins.innovation1.present_card_swap import config, report, training

    for owner in (pure, config):
        for name, size in [('TRAIN_ROWS', 8), ('DEV_ROWS', 6), ('TEST_ROWS', 10)]:
            monkeypatch.setattr(owner, name, size)
    monkeypatch.setattr(pure, 'BATCH_SIZE', 4)
    monkeypatch.setattr(config, 'CAPS', config.counts((30, 30)))
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, **config.CAPS))
    store = RunStore.claim(tmp_path, 'swap', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS,
                           source_supplier=lambda: {'fixture/stub.py': 'a' * 64})
    store.write_snapshot({'recipe': config.recipe()})
    summary: dict[str, object] = {}
    orders: dict[str, list[tuple[int, ...]]] = {slot: [] for slot in config.SLOTS}
    restored: list[str] = []
    predictions: list[tuple[str, int, bool, int]] = []
    origins: list[int] = []
    test_calls: list[int] = []

    def blocks(n: int) -> None:
        meter.reserve('real_blocks', n)
        meter.complete('real_blocks', n)

    class Panel:
        def __init__(self, marker: int, rows: int) -> None:
            self.marker, self.rows = marker, rows

        def __len__(self) -> int:
            return self.rows

        def batch(self, at: Any) -> tuple[Any, Any]:
            x = np.zeros((len(at), 1, 2, 64), dtype=np.float32)
            x[:, 0, 0, :3] = np.column_stack((at, np.full(len(at), self.marker), at % 2))
            x[:, 0, 1, :3] = np.column_stack((at, np.full(len(at), -self.marker), 1 - at % 2))
            return x, (at % 2).astype('u1')[:, None]

    class Panels:
        hashes = {'train': 'b' * 64, 'dev': 'c' * 64}

        def view(self, split: str, slot: str) -> Panel:
            assert slot == config.SLOT and split in self.hashes
            return {'train': Panel(1, 8), 'dev': Panel(2, 6), 'test': Panel(3, 10)}[split]

        def train(self, slot: str) -> Panel:
            return self.view('train', slot)

        def dev(self, slot: str) -> Panel:
            return self.view('dev', slot)

        def test(self) -> None:
            assert tuple(restored) == config.SLOTS
            assert len(predictions) == 8 and {p[1] for p in predictions} == {2}
            assert not test_calls
            test_calls.append(1)
            blocks(20)
            self.hashes = {**self.hashes, 'test': 'd' * 64}

    class Engine:
        parameters = pure.PARAMETERS
        initial_sha = 'e' * 64
        geometry = {'card': 'f' * 64}

        def __init__(self, slot: str, fitting: bool) -> None:
            self.slot, self.fitting = slot, fitting
            self.updates = 0
            self.restored = False
            self.active = False

        def batch_update(self, x: Any, y: Any, active: Meter) -> float:
            assert self.fitting and (x[:, 0, 0, 1] == 1).all()
            orders[self.slot].append(tuple(x[:, 0, 0, 0].astype(int)))
            self.updates += 1
            active.reserve('forward_rows', len(x))
            active.complete('forward_rows', len(x))
            return .5

        def predict(self, x: Any) -> Any:
            marker = int(x[0, 0, 0, 1])
            if self.fitting:
                assert marker == 2
            else:
                assert self.active and self.restored and tuple(restored) == config.SLOTS
                assert abs(marker) == (3 if test_calls else 2)
                assert marker > 0 or not test_calls
                predictions.append((self.slot, abs(marker), marker < 0, len(x)))
                if ((failure == 'diagnostic' and marker == -2) or
                        (failure == 'test' and marker == 3 and self.slot == config.SLOTS[1])):
                    raise ValueError(f'fixture {failure} failure')
            sign = 2 * x[:, 0, 0, 2:] - 1
            if self.slot == config.SLOT and abs(marker) == 3:
                sign[::2] *= -1
            # The second fit improves at epoch 11, resetting its post-warmup patience.
            scale = 1 if self.fitting and self.slot != config.SLOT and self.updates <= 20 else 2
            return (scale * sign[:, :1]).copy()

        def dump(self) -> bytes:
            return f'{self.slot}:{self.updates}'.encode()

        def restore(self, raw: bytes) -> None:
            expected = 2 if self.slot == config.SLOT else 22
            assert raw == f'{self.slot}:{expected}'.encode()
            if failure == 'restore' and self.slot != config.SLOT:
                raise ValueError('fixture restore failure')
            restored.append(self.slot)
            self.restored = True

        def release(self) -> None:
            self.active = False

        def activate(self) -> None:
            assert self.restored and not self.active
            self.active = True

    def make(slot: str, arm: dict[str, object], fitting: bool, origin: int) -> Any:
        if fitting:
            origins.append(origin)
        return Engine(slot, fitting)

    def panels_for(active: Meter) -> Any:
        assert active is meter
        blocks(28)
        return Panels()

    def published() -> Any:
        return json.loads((tmp_path / 'data/swap/run.json').read_text(encoding='utf-8'))['summary']

    def run() -> None:
        training.run(meter, store, summary, gate=lambda _: blocks(4), panels_for=panels_for,
                     make=make, peaks=dict)

    if failure is not None:
        with pytest.raises(ValueError, match=f'fixture {failure} failure'):
            run()
        assert test_calls == ([1] if failure == 'test' else [])
        for state in (summary, published()):
            assert state['panel_hashes'] == {'train': 'b' * 64, 'dev': 'c' * 64}
            assert 'test' not in state and 'paired' not in state
        if failure == 'restore':
            assert not predictions
        store.finalize({'status': 'FAILED', 'counts': meter.snapshot(), 'summary': summary})
        assert read_saved(tmp_path, 'swap')['status'] == 'FAILED'
        assert 'test' not in published()['panel_hashes']
    else:
        run()
        saved = read_saved(tmp_path, 'swap')
        result: Any = saved['summary']
        assert result['panel_hashes']['test'] == 'd' * 64 and test_calls == [1]
        assert result['arms'][config.SLOT]['selected']['epoch'] == 1
        assert result['arms'][config.SLOTS[1]]['selected']['epoch'] == 11
        for slot in config.SLOTS:
            dev = result['dev_diagnostics'][slot]
            assert dev['flip_count'] == 6 and dev['flip_fraction'] == 1
            assert dev['mean_abs_logit_delta'] == 4
            assert dev['original']['accuracy'] == 1 and dev['swapped']['accuracy'] == 0
            assert dev['original']['bce'] == pytest.approx(np.logaddexp(0, -2))
            assert dev['swapped']['bce'] == pytest.approx(np.logaddexp(0, 2))
        assert result['paired']['swap-50_vs_original'] == dict(
            both_correct=5, both_wrong=0, joint_only_correct=5, control_only_correct=0,
            percentage_points=50,
        )
        assert result['test'][config.SLOT]['correct_pairs'] == 5
        assert result['test'][config.SLOTS[1]]['correct_pairs'] == 10
        assert meter.snapshot()['completed_optimizer_updates'] == 74
        assert meter.snapshot()['completed_forward_rows'] == 562
        assert meter.snapshot()['completed_real_blocks'] == 52
        files = report.report_files(saved)
        assert json.loads(files['source.json'])['run_sha256'] == saved['run_sha256']
        assert b'<meter' in files['index.html'] and b'+50.0000' in files['report.md']
        store.write_report_files(files)
    assert origins == [0, 36]
    assert orders[config.SLOT] == orders[config.SLOTS[1]][:36]
