from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.fit import Labels

from cipher_lab import cli as core
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.execution import Clocks, Meter

_CONFIG = Path('configs/direct-block-simon32-split8.json')
_PLUGIN = 'cipher_lab_plugins.direct_block.split8.cli'
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_split8_cli_check_is_strict_and_has_no_source_flag(tmp_path: Path) -> None:
    args = ['--plugin', _PLUGIN, '--config', str(_CONFIG)]
    assert core.main(['check', *args]) == 0
    assert core.main(['check', *args, '--source-root', str(tmp_path)]) == 2
    altered = tmp_path / 'config.json'
    actual = json.loads(_CONFIG.read_text())
    altered.write_text(json.dumps({**actual, 'plaintext_bits': 32.0}))
    assert core.main(['check', '--plugin', _PLUGIN, '--config', str(altered)]) == 2
    assert list(tmp_path.iterdir()) == [altered]


@pytest.mark.parametrize('fault', ['', 'kat', 'restore', 'wrong-sha', 'score', 'deadline',
                                  'baseline-rss', 'output', 'first-best-write', 'commit',
                                  'report', 'report-stop'])
def test_split8_cli_manual_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str], fault: str) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import engine, preflight, resources
    from cipher_lab_plugins.direct_block.split8 import cli, config, data, kat, metrics, model

    fixed = config.recipe()
    monkeypatch.setattr(config, "recipe", lambda: fixed)
    sizes = config.Sizes(4, 3, 5, 2)
    monkeypatch.setattr(config, 'SIZES', sizes)
    end = _NOW + timedelta(hours=3)
    utc = _NOW
    rss = 1
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: utc))
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: rss)
    order: list[str] = []
    targets = [18, 19, 50, 18] * 3
    fit_slots = config.SLOTS
    meters: list[Meter] = []
    monkeypatch.setattr(preflight, 'gpu_gate', lambda: {'manual': True})
    monkeypatch.setattr(preflight, 'gpu_peaks', lambda: {'manual': True})
    monkeypatch.setattr(engine, 'configure_runtime', lambda: {'manual': True})

    def scalar(p: int, key: data.Key, r: int) -> int:
        assert (p, r) == (0x65656877, 32)
        order.append('kat-scalar')
        return 0 if fault == 'kat' else 0xc69be9bb

    def batch(points: data.Words, key: data.Key, r: int) -> data.Words:
        assert points.tolist() == [0x65656877] and r == 32
        order.append('kat-batch')
        return np.array([0xc69be9bb], dtype=np.uint32)

    def public(meter: Meter) -> None:
        meters.append(meter)
        kat.check(meter, scalar, batch)

    def panels(meter: Meter) -> data.Panels:
        assert meter is meters[0]
        assert order == ['kat-scalar', 'kat-batch']

        def draw(bits: int) -> int:
            assert bits == 64
            order.append('draw')
            return 12345

        def label(points: data.Words, key: data.Key, r: int) -> data.Words:
            assert key == (12345, 0, 0, 0)
            if int(points[0]) >= 7:
                assert order.count('reload') == 12
                order.append('test-label')
                return points << np.uint32(24)
            else:
                order.append('fit-label')
            return np.zeros_like(points)

        return data.make_panels(meter, draw, label,
                                sample=lambda n, m: np.arange(n, dtype=np.intp).astype(np.uint32))

    class Manual:
        def __init__(self, r: int, j: int) -> None:
            self.id = (r - 1) * 4 + j
            self.steps = 0
            self.restored = False
            order.append(f'net-{r}-{j}')

        def train_forward(self, x: data.Floats) -> Callable[[Labels], float]:
            assert x.shape == (2, 32) and x.dtype == np.float32

            def backward(y: Labels) -> float:
                assert y.shape == (2, 2) and y.dtype == np.uint8
                return .5

            return backward

        def update(self) -> None:
            self.steps += 1

        def predict(self, x: data.Floats) -> data.Floats:
            assert x.shape[1] == 32
            if self.restored and fault == 'score' and self.id == 11:
                return np.ones((len(x), 8), dtype=np.float32)
            # DEV is all zeros; the zero-logit control plateaus at epoch 18.
            epoch = self.steps // 2
            strength = min(epoch, targets[self.id] - 8) * .01
            return np.full((len(x), 2), -strength if self.id % 4 != 3 else 0., np.float32)

        def dump(self) -> bytes:
            order.append('dump')
            return f'{self.id}/{self.steps}'.encode()

        def restore(self, raw: bytes) -> None:
            if self.id == 11 and fault == 'restore':
                raise ValueError('bad last checkpoint')
            assert raw.startswith(f'{self.id}/'.encode())
            self.restored = True
            order.append('reload')

    monkeypatch.setattr(kat, 'production', public)
    monkeypatch.setattr(data, 'production_panels', panels)
    monkeypatch.setattr(model, 'make_engine', Manual)
    real_dataset = RunStore.write_dataset
    real_read = RunStore.read_model
    real_atomic = RunStore._atomic
    report_calls: list[str] = []
    real_report = RunStore.write_report_files
    real_majority = metrics.train_majority

    def read(store: RunStore, ref: str, sha: str, cap: int) -> bytes:
        if fault == 'wrong-sha' and ref.endswith('r3-b3.pt'):
            sha = '0' * 64
        return real_read(store, ref, sha, cap)

    def majority(labels: data.Bytes, batch_size: int, meter: Meter) -> tuple[int, ...]:
        nonlocal rss, utc
        if fault == 'baseline-rss':
            rss = config.RSS_BYTES + 1
        if fault == 'deadline':
            utc = end
        return real_majority(labels, batch_size, meter)

    def dataset(store: RunStore, name: str, content: bytes) -> str:
        if fault == 'output':
            store._limit = 1
        order.append(f'dataset-{name}')
        return real_dataset(store, name, content)

    def atomic(store: RunStore, path: Path, content: bytes) -> None:
        if fault == 'first-best-write' and path.suffix == '.pt' and 'pt-write' not in order:
            order.append('pt-write')
            assert path.name == 'r1-b0.pt'
            raise OSError('first model write failed; other I/O remains writable')
        if fault == 'commit' and path.name == 'run.json' and json.loads(content).get(
            'status'
        ) == 'SUCCEEDED':
            raise OSError('terminal media fault')
        real_atomic(store, path, content)

    def report(store: RunStore, files: dict[str, bytes]) -> None:
        nonlocal utc
        report_calls.append('report')
        if fault == 'report':
            raise OSError('report failed')
        if fault == 'report-stop':
            utc = end
        real_report(store, files)

    monkeypatch.setattr(RunStore, 'read_model', read)
    monkeypatch.setattr(RunStore, 'write_dataset', dataset)
    monkeypatch.setattr(RunStore, '_atomic', atomic)
    monkeypatch.setattr(RunStore, 'write_report_files', report)
    monkeypatch.setattr(metrics, 'train_majority', majority)
    result = core.main(['run', '--plugin', _PLUGIN, '--config', str(_CONFIG), '--root',
                        str(tmp_path), '--run-id', 'split8-manual', '--deadline-utc',
                        end.isoformat()])
    raw = json.loads((tmp_path / 'data/split8-manual/run.json').read_bytes())
    counts = raw['counts']
    summary = raw['summary']
    if fault in ('', 'report', 'report-stop'):
        assert result == {'': 0, 'report': 4, 'report-stop': 3}[fault]
        assert raw['status'] == 'SUCCEEDED' and set(summary['test']) == {'r1', 'r2', 'r3'}
        assert order.count('fit-label') == 6 and order.count('test-label') == 3
        assert order[-3:] == ['dataset-train', 'dataset-dev', 'dataset-test']
        assert order.count('dump') > 12 and order.count('draw') == 1
        assert set(raw['artifacts']) == {*[f'model/{s}.pt' for s in config.SLOTS],
                                         *[f'dataset/{s}.npz' for s in config.SPLITS]}
        assert report_calls == ['report']
        epochs = tuple(a['epochs_completed'] for a in summary['fits'].values())
        assert epochs == tuple(targets)
        assert [summary['fits'][s]['selected']['epoch'] for s in fit_slots] == [10, 11, 42, 1] * 3
        for k, amount in config.counts(epochs).items():
            assert counts[f'attempted_{k}'] == counts[f'completed_{k}'] == amount
        assert all(r['bit_denominator'] == 40 for r in summary['test'].values())
        assert all(ref['best_reloaded'] for ref in summary['fits'].values())
        saved = read_saved(tmp_path, 'split8-manual')
        assert 'cipher_lab/ciphers/simon32.py' in cast(dict[str, str], saved['source'])
    else:
        assert result == (3 if fault in ('deadline', 'baseline-rss') else 4)
        assert raw['status'] != 'SUCCEEDED' and 'test' not in summary
        if fault in ('kat', 'restore', 'wrong-sha', 'deadline', 'baseline-rss'):
            assert 'test-label' not in order
        if fault in ('deadline', 'baseline-rss', 'output'):
            assert not report_calls
        if fault == 'kat':
            assert order == ['kat-scalar'] and not raw['artifacts']
            assert counts['attempted_real_blocks'] == 1 and counts['completed_real_blocks'] == 0
        if fault == 'first-best-write':
            assert order == ['kat-scalar', 'kat-batch', 'draw', 'fit-label', 'fit-label',
                             'net-1-0', 'dump', 'pt-write']
            assert raw['status'] == 'FAILED' and not raw['artifacts']
            native = json.loads(capsys.readouterr().err)
            assert native['status'] == 'FAILED' and native['reason'] == 'WRITE_FAILED'
            for kind, amount in {'real_blocks': 9, 'optimizer_updates': 2, 'forward_rows': 7,
                                 'null_words': 0}.items():
                assert native['counts'][f'attempted_{kind}'] == amount
                assert native['counts'][f'completed_{kind}'] == amount
            assert counts == native['counts']
