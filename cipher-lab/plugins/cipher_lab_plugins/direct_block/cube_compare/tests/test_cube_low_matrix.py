from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare import config, data
from cipher_lab_plugins.direct_block.cube_compare.tests.test_cube_reuse import cube_reuse50_bind
from cipher_lab_plugins.direct_block.cube_compare.tests.test_cube_reuse import (
    cube_reuse50_source as cube_reuse50_source,
)

from cipher_lab import artifacts
from cipher_lab.ciphers import simon
from cipher_lab.cli import main as dispatch
from cipher_lab.execution import Clocks, Meter, Stopped


@pytest.mark.parametrize('fault', ['', 'sha1', 'sha2', 'sha3', 'restore1', 'restore2', 'restore3',
                                  'testsha', 'score3', 'testdeadline', 'output', 'commit',
                                  'reportio', 'reportdeadline', 'oracle',
                                  'majoritydeadline', 'runtime', 'provider'])
def test_cube_low_cli_matrix_delayed_test_and_atomic_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    cube_reuse50_source: Callable[..., tuple[Path, str, dict[str, bytes]]], fault: str,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import (
        cli,
        engine,
        preflight,
        resources,
        training,
    )

    root, digest, datasets = cube_reuse50_source(two_test_families=True)
    cube_reuse50_bind(monkeypatch, digest, datasets)
    monkeypatch.setattr(config, 'FAMILIES', (1, 1, 2))
    # Mechanics use fake fixtures and one update/epoch; the formal JSON never shrinks.
    test_size = 16
    out = tmp_path / 'output'
    old_run = root / 'data' / config.SOURCE_RUN_ID
    if fault == 'testsha':
        (old_run / 'dataset/test.npz').write_bytes(b'!' + datasets['test'][1:])
    if fault == 'provider':
        (old_run / 'run.json').write_bytes(b'bad source record')
    events: list[str] = []
    orders: dict[int, list[bytes]] = {r: [] for r in (1, 2, 3)}
    oracle_calls: list[tuple[str, int, bytes]] = []
    keys: list[int] = []
    key = (1 << 127) | 13  # artificial words, never a real draw
    split = ''
    now = datetime(2026, 11, 24, tzinfo=UTC)
    deadline = now + timedelta(hours=3)
    opened = Path.open

    def open_source(path: Path, *a: Any, **kw: Any) -> Any:
        nonlocal split, now
        if path.is_relative_to(root):
            assert path.name in ('run.json', 'train.npz', 'dev.npz', 'test.npz')
            if path.suffix == '.npz':
                split = path.stem
                events.append(f'read/{split}')
                if split == 'test':
                    assert events[-4:-1] == ['loaded/1', 'loaded/2', 'loaded/3']
                    if fault == 'testdeadline':
                        now = deadline
        return opened(path, *a, **kw)

    def fake_draw(bits: int) -> int:
        keys.append(bits)
        assert keys == [128]
        return key

    def fake_oracle(points: data.Words, words: tuple[int, ...], rounds: int) -> data.Words:
        assert words == (13, 0, 0, 1 << 31) and rounds in (1, 2, 3)
        assert points.dtype == np.uint64 and points.ndim == 1 and points.flags.c_contiguous
        assert len(points) <= 65536
        oracle_calls.append((split, rounds, points.tobytes()))
        if fault == 'oracle' and rounds == 2:
            return points.astype(np.int64)
        return points ^ np.uint64(1 << rounds)

    monkeypatch.setattr(Path, 'open', open_source)
    monkeypatch.setattr(secrets, 'randbits', fake_draw)
    monkeypatch.setattr(simon, 'encrypt_batch', fake_oracle)
    monkeypatch.setattr(cli, 'Clocks', lambda: Clocks(lambda: 0., lambda: 0., lambda: now))
    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)
    monkeypatch.setattr(preflight, 'gpu_gate', lambda: {'uuid': config.GPU_UUID})
    monkeypatch.setattr(preflight, 'gpu_peaks', lambda: {'manual': True})
    def fake_runtime() -> dict[str, object]:
        if fault == 'runtime':
            raise ValueError('private fake runtime failure')
        return {'manual': True}

    monkeypatch.setattr(engine, 'configure_runtime', fake_runtime)
    majority_passes: list[int] = []
    majority_batches: list[tuple[int, int]] = []
    if fault == 'majoritydeadline':
        monkeypatch.setattr(config, 'BATCH', 4)  # two TRAIN batches per fake round
        from cipher_lab_plugins.direct_block.cube_compare.low_metrics import train_majority

        def watch_majority(batches: Iterable[data.Labels]) -> tuple[int, ...]:
            rounds = len(majority_passes) + 1
            majority_passes.append(rounds)

            def watch_batches() -> Iterator[data.Labels]:
                nonlocal now
                for i, batch in enumerate(batches, 1):
                    majority_batches.append((rounds, i))
                    yield batch
                    # Resume only after the pure reducer has consumed the batch.
                    if rounds == i == 1:
                        now = deadline

            return train_majority(watch_batches())

        monkeypatch.setattr(training, 'train_full_majority', watch_majority)
    for module, attr in ((preflight, 'run'), (data, 'production_panels'),
                         (engine, 'make_engine'), (engine, 'make_lstm_engine')):
        monkeypatch.setattr(module, attr, lambda *a, **k: pytest.fail('legacy path invoked'))

    class ManualEngine:
        def __init__(self, rounds: int) -> None:
            assert 'read/train' in events and 'read/dev' in events
            events.append(f'made/{rounds}')
            self.rounds, self.step, self.loaded = rounds, 0, False

        def train_forward(self, inputs: data.Floats) -> Callable[[data.Labels], float]:
            orders[self.rounds].append(inputs.tobytes())

            def backward(labels: data.Labels) -> float:
                np.testing.assert_array_equal(labels, self.target(inputs))
                return .125

            return backward

        def target(self, inputs: data.Floats) -> data.Labels:
            result = inputs.astype(np.uint8)
            result[:, [63 - self.rounds, 127 - self.rounds]] ^= 1
            return result

        def predict(self, inputs: data.Floats) -> data.Floats:
            if self.loaded:
                events.append(f'score/{self.rounds}')
                if fault == 'score3' and self.rounds == 3:
                    raise ValueError('private fake failure')
            # Last significant improvement at epochs 1, 11, 12 respectively;
            # strict earliest best replacement also includes sub-delta gains.
            change = {1: 1, 2: 11, 3: 12}[self.rounds]
            strength = 1 + (self.step >= change) * .2 + self.step * 1e-6
            return np.float32(strength) * (2 * self.target(inputs).astype(np.float32) - 1)

        def update(self) -> None:
            self.step += 1

        def dump(self) -> bytes:
            return f'{self.rounds}/{self.step:02d}'.encode()

        def restore(self, raw: bytes) -> None:
            assert raw == self.dump()
            if fault == f'restore{self.rounds}':
                raise ValueError('private fake restore')
            self.loaded = True
            events.append(f'loaded/{self.rounds}')

    monkeypatch.setattr(engine, 'make_low_engine', ManualEngine, raising=False)
    read_model = artifacts.RunStore.read_model

    def read_best(store: artifacts.RunStore, ref: str, sha: str, size: int) -> bytes:
        n = events.count('loaded/1') + events.count('loaded/2') + events.count('loaded/3') + 1
        assert ref == f'model/r{n}-best.pt' and size == 16 * 2**20
        if fault == f'sha{n}':
            (out / 'models/matrix' / f'r{n}-best.pt').write_bytes(b'fail')
        return read_model(store, ref, sha, size)

    monkeypatch.setattr(artifacts.RunStore, 'read_model', read_best)
    atomic = artifacts.RunStore._atomic

    def boundary(store: artifacts.RunStore, path: Path, raw: bytes) -> None:
        nonlocal now
        if fault == 'output' and path.name == 'r3-test.npz':
            monkeypatch.setattr(store, '_limit', store._total())
        if fault == 'commit' and path.name == 'run.json' and b'"test":{' in raw:
            raise OSError('private fake media failure')
        if path.parent == out / 'result/matrix':
            if fault == 'reportio':
                raise OSError('private fake report failure')
            if fault == 'reportdeadline':
                now = deadline
        atomic(store, path, raw)

    monkeypatch.setattr(artifacts.RunStore, '_atomic', boundary)
    cfg = tmp_path / 'matrix.json'
    cfg.write_text(json.dumps(config.low_recipe()), encoding='utf-8')
    argv = ['run', '--plugin', cli.__name__, '--config', str(cfg), '--root', str(out),
            '--source-root', str(root), '--run-id', 'matrix',
            '--deadline-utc', deadline.isoformat()]
    code = dispatch(argv)
    log = capsys.readouterr()
    assert 'private ' not in log.err and 'Traceback' not in log.err
    saved = json.loads((out / 'data/matrix/run.json').read_bytes())
    before_key = fault in ('runtime', 'provider')
    before_engine = before_key or fault in ('oracle', 'majoritydeadline')
    assert keys == ([] if before_key else [128])
    assert saved['recipe']['mode'] == 'plaintext-only-source'
    assert saved['summary'].get('mode') != 'saved-panels'
    attempts = sum(len(raw) // 8 for _, _, raw in oracle_calls)
    counts = saved['counts']
    assert counts['attempted_real_blocks'] == attempts
    assert counts['completed_real_blocks'] == attempts - (16 if fault == 'oracle' else 0)
    assert counts['attempted_null_words'] == counts['completed_null_words'] == 0
    assert all(events.count(f'made/{r}') == (0 if before_engine else 1) for r in (1, 2, 3))
    expect_test = not before_engine and not fault.startswith(('sha', 'restore'))
    assert ('read/test' in events) == expect_test
    if fault == 'majoritydeadline':
        assert majority_passes == [1] and majority_batches == [(1, 1)]
        assert saved['status'] == 'STOPPED'
        assert events == ['read/train', 'read/dev']
        assert counts['attempted_real_blocks'] == counts['completed_real_blocks'] == 96
        for unit in ('optimizer_updates', 'forward_rows'):
            assert counts[f'attempted_{unit}'] == counts[f'completed_{unit}'] == 0
    if before_key:
        assert not events and attempts == 0 and 'arms' not in saved['summary']
        for name in ('index.html', 'report.md'):
            text = (out / 'result/matrix' / name).read_text(encoding='utf-8')
            assert 'Planned method' in text and 'No final TEST score was committed.' in text
            assert all(claim not in text for claim in ('newly drawn private key',
                                                       'One synthetic private key.',
                                                       'Three independent LSTMs;'))
        assert 'No fit details survived in the record.' in text
    for s, raw in datasets.items():
        calls = [(r, points) for name, r, points in oracle_calls if name == s]
        if calls:
            with np.load(BytesIO(raw), allow_pickle=False) as old:
                assert all(points == old['plaintext'].tobytes() for _, points in calls)
                assert [r for r, _ in calls] == ([1, 2] if fault == 'oracle' else [1, 2, 3])
    if fault:
        reason = {'testsha': 'HASH_MISMATCH', 'score3': 'TEST_SCORE_FAILED',
                  'testdeadline': 'DEADLINE', 'output': 'OUTPUT_LIMIT', 'commit': 'WRITE_FAILED',
                  'reportio': 'REPORT_FAILED', 'reportdeadline': 'DEADLINE',
                  'oracle': 'ORACLE_RESPONSE', 'majoritydeadline': 'DEADLINE',
                  'runtime': 'RUNTIME_FAILED', 'provider': 'HASH_MISMATCH'}.get(fault)
        if reason is None:
            reason = 'HASH_MISMATCH' if fault.startswith('sha') else 'BEST_RELOAD_FAILED'
        assert code == (3 if fault.endswith('deadline') or fault == 'oracle' else 4)
        assert not log.out and json.loads(log.err)['reason'] == reason
        assert ('test' in saved['summary']) == fault.startswith('report')
        if fault == 'score3':
            assert counts['attempted_forward_rows'] - counts['completed_forward_rows'] == 8
        if not fault.startswith('report'):
            path = out / 'result/matrix/index.html'
            if path.exists():
                text = path.read_text(encoding='utf-8')
                assert 'No final TEST score' in text and 'Public preflight' not in text
            return
    else:
        assert code == 0
    assert saved['status'] == 'SUCCEEDED' and saved['summary']['preflight']['status'] != 'RUNNING'
    test = saved['summary']['test']['rounds']
    assert set(test) == {'r1', 'r2', 'r3'}
    epochs = (18, 19, 20)
    assert orders[1] == orders[2][:18] == orders[3][:18] and orders[2] == orders[3][:19]
    for kind, n in config.low_counts(epochs).items():
        assert counts[f'attempted_{kind}'] == counts[f'completed_{kind}'] == n
    for round_n, n_epochs in zip((1, 2, 3), epochs, strict=True):
        name = f'r{round_n}'
        arm = saved['summary']['arms'][name]
        assert arm['best_reloaded'] and arm['epochs_completed'] == n_epochs
        assert arm['selected']['epoch'] == n_epochs and arm['slot'] == f'{name}-best'
        assert arm['identity'] == f'{config.LOW_METHOD}/{name}'
        assert test[name]['scores']['lstm']['correct']['wholepair'] == test_size
        for s in config.SPLITS:
            raw = (out / f'data/matrix/dataset/{name}-{s}.npz').read_bytes()
            with np.load(BytesIO(raw), allow_pickle=False) as new, np.load(
                BytesIO(datasets[s]), allow_pickle=False,
            ) as old:
                np.testing.assert_array_equal(new['plaintext'], old['plaintext'])
                np.testing.assert_array_equal(new['family_ids'], old['family_ids'])
                np.testing.assert_array_equal(new['ciphertext'], old['plaintext'] ^ (1 << round_n))
                assert not np.array_equal(new['ciphertext'], old['ciphertext'])
    if not fault:
        for report in ('index.html', 'report.md'):
            text = (out / 'result/matrix' / report).read_text(encoding='utf-8')
            assert all(word in text for word in ('r1', 'r2', 'r3', 'wholepair', 'x64', digest))
            assert '49.9969%' not in text and 'interval' not in text
            if report == 'index.html':
                assert ('No additional model fits or paired inference. '
                        'Majority bits are estimated from TRAIN only.') in text
        assert dispatch(argv) == 4
        assert json.loads(capsys.readouterr().err)['reason'] == 'OCCUPIED'
        assert keys == [128]


def test_cube_low_chunks_reserve_each_round_then_complete_valid_response() -> None:
    from cipher_lab_plugins.direct_block.cube_compare.low_rounds import LowRounds

    draws: list[tuple[int, int, int, int]] = []
    points = np.arange(65538, dtype=np.uint64).reshape(-1, 2)
    ids = np.arange(len(points), dtype=np.uint64)
    reads = 0

    class PlainSource:
        def build_plaintext(self, split: str) -> tuple[data.Words, data.Words]:
            nonlocal reads
            assert split == 'train'
            reads += 1
            return points, ids

        def label_test_plaintext(self) -> tuple[data.Words, data.Words]:
            pytest.fail("TEST not in this probe")

    # Fake meter logs ordering so even a failed last chunk is not refunded.
    events: list[tuple[str, int]] = []

    class FakeMeter:
        def check(self) -> None:
            pass

        def reserve(self, kind: str, n: int) -> None:
            assert kind == 'real_blocks'
            events.append(('reserve', n))

        def complete(self, kind: str, n: int) -> None:
            assert kind == 'real_blocks'
            events.append(('complete', n))

    def encrypt(batch: data.Words, r: int) -> data.Words:
        assert batch.dtype == np.uint64 and batch.ndim == 1 and batch.flags.c_contiguous
        assert events[-1] == ('reserve', len(batch))
        draws.append((r, len(batch), int(batch[0]), int(batch[-1])))
        if r == 3 and len(batch) == 2:
            raise ValueError('private fake error')
        return batch.copy()

    provider = LowRounds(PlainSource(), cast(Meter, FakeMeter()), encrypt)
    with pytest.raises(Stopped, match='ORACLE_FAILED'):
        provider.build('train')
    with pytest.raises(ValueError):
        provider.build('train')
    assert reads == 1
    assert draws == [(r, n, first, last) for r in (1, 2, 3)
                     for n, first, last in ((65536, 0, 65535), (2, 65536, 65537))]
    assert events == [('reserve', n) if k == 0 else ('complete', n)
                      for _, n, _, _ in draws for k in (0, 1)][:-1]
