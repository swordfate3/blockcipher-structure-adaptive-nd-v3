from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import secrets
import socket
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def recipe() -> Any:
    from cipher_lab_plugins.innovation1.config import load_config
    return load_config(ROOT / 'configs/innovation1-fresh.json')


def small_recipe() -> Any:
    r = recipe()
    return dataclasses.replace(r, training=dataclasses.replace(
        r.training, epochs=2, batches_per_epoch=2, batch_groups=4,
        dev_groups=8, test_groups=12, eval_batch_groups=4))


def test_formal_recipe_cost_and_lr() -> None:
    from cipher_lab_plugins.innovation1.config import estimate, learning_rate
    r = recipe()
    assert r.method == 'innovation1-shared-vs-dedicated-v1'
    assert [t.cipher for t in r.tasks] == ['SIMECK', 'PRESENT', 'SIMON', 'SKINNY']
    assert [(t.rounds, t.difference) for t in r.tasks] == [
        (16, 0x40), (7, 0xD00000), (13, 0x40), (8, 0x2000)]
    assert (r.seeds.init, r.seeds.sample, r.seeds.shuffle, r.seeds.bootstrap) == (
        26111741, 26111742, 26111743, 26111744)
    e = estimate(r)
    expected = dict(cycles=24576, fits=5, optimizer_updates=196608,
                    shared_updates=98304, dedicated_updates_each=24576,
                    train_groups=50331648, dev_groups=262144, test_groups=1048576,
                    generated_groups=51642368, real_blocks=413138944,
                    train_model_groups=100663296, dev_model_groups=12582912,
                    test_model_groups=2097152, model_groups=115343360,
                    saved_panel_bytes=85196800)
    assert expected.items() <= e.items()
    assert learning_rate(r, 0) == 1e-3
    assert learning_rate(r, 24575) == 1e-4
    assert math.isclose(learning_rate(r, 10) + learning_rate(r, 24565), 0.0011)
    for cycle in (-1, 24576, True):
        with pytest.raises(ValueError):
            learning_rate(r, cycle)
    assert r.to_dict() == json.loads((ROOT / 'configs/innovation1-fresh.json').read_text())


@pytest.mark.parametrize('path,value', [
    (('extra',), 1), (('training', 'batch_groups'), 3),
    (('training', 'epochs'), True), (('optimizer', 'weight_decay'), 1e-5),
    (('optimizer', 'lr_min'), 0.01), (('statistics', 'confidence'), 0.95),
    (('statistics', 'replicates'), 0), (('model', 'top_k'), 1),
    (('tasks',), []), (('seeds', 'init'), -1),
    (('training', 'eval_batch_groups'), 3), (('optimizer', 'eps'), float('nan')),
])
def test_strict_config(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    from cipher_lab_plugins.innovation1.config import load_config
    raw = copy.deepcopy(recipe().to_dict())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    f = tmp_path / 'bad.json'
    f.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(f)


class Meter:
    def __init__(self) -> None:
        self.attempted = 0
        self.completed = 0

    def reserve(self, kind: str, amount: int) -> None:
        assert kind == 'real_blocks'
        self.attempted += amount

    def complete(self, kind: str, amount: int) -> None:
        assert kind == 'real_blocks'
        self.completed += amount

    def check(self) -> None:
        pass


class FakeOracle:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def __call__(self, points: tuple[int, ...], key: tuple[int, ...],
                 rounds: int) -> tuple[int, ...]:
        self.calls.append((points, key, rounds))
        return tuple(p ^ 0x55 for p in points)  # artificial bijection, not a cipher


class Private:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, size: int) -> bytes:
        self.calls += 1
        return self.calls.to_bytes(size, 'big')


def test_online_batch_and_late_panels() -> None:
    from cipher_lab_plugins.innovation1.data import Provider, derive_seed
    r = small_recipe()
    oracle, private, meter = FakeOracle(), Private(), Meter()
    p = Provider(r, {t.cipher: oracle for t in r.tasks}, private, meter)
    assert private.calls == 0
    assert not oracle.calls
    t = r.tasks[0]
    batch = p.train(t, 0)
    # Both arms consume this SAME returned object; provider has no arm API/cache.
    m_batch = d_batch = batch
    assert m_batch is d_batch
    assert batch.ciphertext.shape == (4, 4, 2, 8)
    assert batch.labels.shape == (4, 1)
    assert int(batch.labels.sum()) == 2
    assert (meter.attempted, meter.completed, private.calls) == (32, 32, 4)
    assert set(vars(batch)) == {'ciphertext', 'labels', 'task', 'split', 'index'}
    assert len(batch.digest) == 64
    x, y = batch.tensors()
    assert x.shape == (4, 4, 2, 64) and y.shape == (4, 1)
    assert torch.equal(x[0, 0, 0], torch.tensor([
        (int.from_bytes(batch.ciphertext[0, 0, 0].tobytes(), 'big') >> s) & 1
        for s in range(63, -1, -1)], dtype=torch.float32))
    assert not batch.ciphertext.flags.writeable and not batch.labels.flags.writeable
    for i, (points, key, rounds) in enumerate(oracle.calls):
        assert len(points) == 8 and rounds == 16
        assert key == (i + 1, 0, 0, 0)  # exactly one private draw/shared key per 8 blocks
        if batch.labels[i, 0]:
            assert all(points[j] ^ points[j + 1] == 0x40 for j in range(0, 8, 2))
        else:
            assert all(points[j] ^ points[j + 1] != 0x40 for j in range(0, 8, 2))
        assert all(int.from_bytes(batch.ciphertext[i, j // 2, j % 2].tobytes(),
                                  'big') == points[j] ^ 0x55 for j in range(8))
    with pytest.raises(ValueError):
        p.train(t, 0)  # no accidental replay
    assert p.dev(t) is p.dev(t)
    assert private.calls == 12  # train 4 + dev 8; still no TEST
    before = private.calls
    assert p.test(t) is p.test(t)
    assert private.calls == before + 12
    streams = {derive_seed(r, t, split, i, purpose) for split in ('train', 'dev', 'test')
               for i in (0, 1) for purpose in ('sample', 'shuffle', 'bootstrap')}
    assert len(streams) == 18
    assert derive_seed(r, t, 'train', 0, 'sample') != derive_seed(
        r, r.tasks[1], 'train', 0, 'sample')


def test_private_failures_and_invalid_oracle_are_sanitized() -> None:
    from cipher_lab_plugins.innovation1.data import DataError, Provider
    r = small_recipe()

    def bad_private(size: int) -> bytes:
        raise RuntimeError('SECRET_SENTINEL')

    meter = Meter()
    p = Provider(r, {t.cipher: FakeOracle() for t in r.tasks}, bad_private, meter)
    with pytest.raises(DataError, match='PRIVATE_SOURCE_FAILED') as error:
        p.train(r.tasks[0], 0)
    assert 'SECRET' not in str(error.value) and error.value.__suppress_context__
    assert (meter.attempted, meter.completed) == (8, 0)
    with pytest.raises(DataError, match='PROVIDER_STOPPED'):
        p.train(r.tasks[0], 0)

    def invalid(*args: object) -> tuple[int, ...]:
        return (1 << 64,) * 8

    p = Provider(r, {t.cipher: invalid for t in r.tasks}, Private(), Meter())
    with pytest.raises(DataError, match='ORACLE_RESPONSE'):
        p.train(r.tasks[0], 0)


def test_metadata_models_and_bn() -> None:
    from cipher_lab.models.dedicated import FeistelSEExpert, PresentExpert, SkinnyExpert
    for model, count, frozen_bn in (
        (FeistelSEExpert('simeck64'), 599809, False),
        (FeistelSEExpert('simon64'), 599809, False),
        (PresentExpert(), 651073, True), (SkinnyExpert(rounds=8), 289313, False),
    ):
        assert sum(p.numel() for p in model.parameters()) == count
        assert not hasattr(model, 'last_auxiliary_loss')
        for mode in (False, True):
            assert model.train(mode) is model
            for layer in model.modules():
                if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    assert layer.training is (False if frozen_bn else mode)
                    assert layer.eps == 0.001 and layer.momentum == 0.01
                    assert layer.weight.requires_grad and torch.equal(
                        layer.weight, torch.ones_like(layer.weight))
                    assert torch.count_nonzero(layer.bias) == 0
        # Poison MUST reject even model invocation, no true forward attempted.
        with pytest.raises(AssertionError, match='UNAUTHORIZED_SCIENCE'):
            model(torch.zeros((1, 4, 2, 64)))
    feistel = FeistelSEExpert('simon64')
    assert [cast(torch.nn.Conv1d, cast(torch.nn.Sequential, b)[0]).kernel_size
            for b in feistel.branches] == [(1,), (3,), (5,), (7,)]
    for layer in feistel.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv1d)):
            assert torch.count_nonzero(layer.bias) == 0 if layer.bias is not None else True
    present = PresentExpert()
    assert [b.kernel_size for b in present.branches] == [(1,), (2,), (4,)]
    assert all(cast(Any, b).conv1.dilation == (1,) for b in present.blocks)
    skinny = SkinnyExpert(rounds=8)
    assert len(skinny.encoder) == 20


def test_manual_public_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.representations import dedicated as rep
    bits = torch.zeros((1, 4, 2, 64))
    bits[:, :, 0, 0] = 1
    words = rep.relation_words(bits, 'simon64')
    assert words.shape == (1, 4, 8, 32)
    assert torch.equal(words[0, 0, 0], bits[0, 0, 0, :32])
    assert words[0, 0, 6, 0] == 1  # L xor f(R); R=0
    # PRESENT preserves raw 8x16 row-major then 4 XOR channels per pair.
    component = rep.component_input(bits)
    assert component.shape == (4, 12, 16)
    assert component[0, 0, 0] == component[0, 8, 0] == 1
    assert component.sum() == 8
    calls: list[int] = []

    def fake_backstep(x: torch.Tensor, rounds: int) -> torch.Tensor:
        calls.append(rounds)
        return x

    monkeypatch.setattr(rep, 'public_backstep', fake_backstep)
    layout = rep.author_pair_layout(bits, rounds=8)
    triplet = torch.cat((bits, (bits[:, :, :1] != bits[:, :, 1:]).float()), 2)
    expected = triplet.reshape(4, 3, 4, 16).permute(0, 2, 1, 3).contiguous()
    assert torch.equal(layout, expected)
    assert calls == [8]
    for bad in (torch.zeros((1, 512)), torch.full((1, 4, 2, 64), 0.5)):
        with pytest.raises(ValueError):
            rep.component_input(bad)


def test_manual_skinny_backstep() -> None:
    # Hand calculation at rounds=1 for zero ciphertext: public constants 1,0,2;
    # lower-half inverse Sbox maps 2->6 at cell8 and 0->3 elsewhere.
    from cipher_lab.representations.dedicated import author_pair_layout
    bits = torch.zeros((1, 4, 2, 64))
    layout = author_pair_layout(bits, rounds=1)
    cells = [1, 0, 0, 0, 0, 0, 0, 0, 6, 3, 3, 3, 3, 3, 3, 3]
    binary = [int(b) for c in cells for b in f'{c:04b}']
    expected = torch.tensor(binary + binary + [0] * 64).float().reshape(
        3, 4, 16).permute(1, 0, 2)
    assert torch.equal(layout[0], expected)


def test_paired_metrics_fixed_quota_and_bootstrap() -> None:
    from cipher_lab_plugins.innovation1.metrics import compare, difference_counts
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    m = np.array([-1., 1., 0., -1.])
    d = np.array([1., -1., 1., -1.])
    assert difference_counts(labels, m, d) == ((1, 0, 1), (0, 2, 0))
    result = compare(labels, m, d, replicates=20000, seed=26111744)
    assert result['shared']['accuracy'] == result['dedicated']['accuracy'] == 0.5
    assert result['accuracy_difference'] == 0
    assert math.isclose(result['shared']['bce'], float(np.mean(
        np.logaddexp(0, m) - labels * m)))
    # Exact finite distribution for fixture: neg-layer two draws ±1, pos-layer 0.
    # Delta support -0.5,0,+0.5 with probabilities 1/4,1/2,1/4.
    assert result['interval'] == [-0.5, 0.5]
    assert result['confidence'] == 0.9875 and result['replicates'] == 20000
    same = compare(labels, m, m, replicates=31, seed=2)
    assert same['interval'] == [0., 0.]
    for bad_labels, bad_m in (([0, 0, 0, 1], m), (labels, [0, 1, np.nan, 2])):
        with pytest.raises(ValueError):
            compare(bad_labels, bad_m, d)


def test_guard_is_effective() -> None:
    from cipher_lab.ciphers import simon
    from cipher_lab.execution import production_rss_bytes
    for action in (
        lambda: os.urandom(1), lambda: secrets.token_bytes(1),
        lambda: socket.socket(), lambda: simon.encrypt_many((0,), (0, 0, 0, 0), 1),
        lambda: torch.nn.Linear(1, 1)(torch.zeros((1, 1))),
        lambda: torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]),
        lambda: cast(Any, torch.ones(1, requires_grad=True).sum()).backward(),
        lambda: torch.cuda.is_available(), lambda: production_rss_bytes(),
    ):
        with pytest.raises(AssertionError, match='UNAUTHORIZED_SCIENCE'):
            action()


def test_bootstrap_exact_pushforward_and_linear_quantile(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import Counter
    from itertools import product

    from cipher_lab_plugins.innovation1.metrics import bootstrap_interval
    # Enumerate all paired-index draws for the negative stratum [-1,+1].
    # Indices (0,0),(0,1),(1,0),(1,1) push forward to multinomial counts.
    index_sums = Counter(sum((-1, 1)[i] for i in pair) for pair in product(range(2), repeat=2))
    count_weights = Counter({-2: math.comb(2, 0), 0: math.comb(2, 1), 2: math.comb(2, 2)})
    assert index_sums == count_weights
    draws = np.array([[2, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 2]])

    class Enumerated:
        def __init__(self) -> None:
            self.calls = 0

        def multinomial(self, n: int, p: np.ndarray[Any, Any],
                        size: int) -> np.ndarray[Any, Any]:
            assert n == 2 and size == 4
            self.calls += 1
            if self.calls == 1:
                np.testing.assert_array_equal(p, [0.5, 0., 0.5])
                return draws
            np.testing.assert_array_equal(p, [0., 1., 0.])
            return np.array([[0, 2, 0]] * size)

    fake = Enumerated()
    monkeypatch.setattr(np.random, 'Generator', lambda bitgen: fake)
    actual = bootstrap_interval(((1, 0, 1), (0, 2, 0)), replicates=4, seed=7, chunk_size=4)
    expected = np.quantile([-0.5, 0, 0, 0.5], [0.00625, 0.99375], method='linear')
    assert actual == expected.tolist()  # not nearest/rank quantiles
    assert fake.calls == 2


def test_explicit_dev_preparation_and_private_independence() -> None:
    from cipher_lab_plugins.innovation1.data import Provider
    r = small_recipe()
    oracle1, oracle2 = FakeOracle(), FakeOracle()
    private1, private2 = Private(), Private()
    private2.calls = 123  # Changing private draws cannot change public sampling.
    p1 = Provider(r, {t.cipher: oracle1 for t in r.tasks}, private1, Meter())
    p2 = Provider(r, {t.cipher: oracle2 for t in r.tasks}, private2, Meter())
    panels = p1.prepare_dev()
    assert list(panels) == [t.cipher for t in r.tasks]
    assert private1.calls == 32  # DEV only; no train/test data collection.
    for t in r.tasks:
        assert panels[t.cipher] is p1.dev(t)
    p1.train(r.tasks[0], 0)
    p2.train(r.tasks[0], 0)
    assert [call[0] for call in oracle1.calls[-4:]] == [call[0] for call in oracle2.calls]
    assert oracle1.calls[-4][1] != oracle2.calls[0][1]
    for t, expected_words in zip(r.tasks, (4, 3, 4, 2), strict=True):
        batch = p2.train(t, 1 if t == r.tasks[0] else 0)
        assert len(oracle2.calls[-1][1]) == expected_words
        if t.cipher == 'PRESENT':
            assert oracle2.calls[-1][1][2] <= 65535
        with pytest.raises(ValueError):
            batch.labels.setflags(write=True)


def test_same_seed_feistel_weights_and_native_initialization() -> None:
    from cipher_lab.models.dedicated import FeistelSEExpert, PresentExpert, SkinnyExpert
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(26111741)
        first = FeistelSEExpert('simeck64')
        torch.manual_seed(26111741)
        second = FeistelSEExpert('simon64')
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)
    for model in (PresentExpert(), SkinnyExpert()):
        conv = next(m for m in model.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)))
        # Original Torch default init, not Feistel's forced zero/Xavier policy.
        assert conv.bias is not None and torch.count_nonzero(conv.bias) > 0


def test_config_is_stdlib_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split('.')[0] not in {'numpy', 'torch', 'psutil', 'socket', 'subprocess'}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    from cipher_lab_plugins.innovation1.config import Recipe, estimate, load_config

    r = load_config(ROOT / 'configs/innovation1-fresh.json')
    assert isinstance(r, Recipe)
    assert estimate(r)['real_blocks'] == 413138944


@pytest.mark.parametrize('cipher,rotations', [('simeck64', (0, 5, 1)),
                                             ('simon64', (1, 8, 2))])
def test_feistel_hand_word_rotations(cipher: str, rotations: tuple[int, int, int]) -> None:
    from cipher_lab.representations.dedicated import relation_words
    # Scalar 32-bit hand oracle over artificial ciphertexts, not encryption.
    mask = 2**32 - 1

    def rol(x: int, shift: int) -> int:
        return ((x << shift) | (x >> (32 - shift))) & mask

    def f(x: int) -> int:
        a, b, c = rotations
        return (rol(x, a) & rol(x, b)) ^ rol(x, c)

    left, right, other_left, other_right = 0x12345678, 0x87654321, 0xFEDCBA98, 0x76543210
    raw = [left, right, other_left, other_right]
    bits = torch.tensor([int(b) for x in raw for b in f'{x:032b}']).reshape(1, 1, 2, 64)
    bits = bits.expand(1, 4, 2, 64)
    before, other_before = left ^ f(right), other_left ^ f(other_right)
    expected = [left ^ other_left, right ^ other_right, *raw,
                before ^ other_before, right ^ f(before) ^ other_right ^ f(other_before)]
    words = relation_words(bits, cipher)
    for i, value in enumerate(expected):
        assert words[0, 0, i].tolist() == [int(b) for b in f'{value:032b}']


def test_public_source_dependency_boundaries() -> None:
    import ast

    production = [*sorted((ROOT / 'src/cipher_lab/models/dedicated').glob('*.py')),
                  ROOT / 'src/cipher_lab/representations/dedicated.py',
                  *sorted((ROOT / 'plugins/cipher_lab_plugins/innovation1').glob('*.py'))]
    for path in production:
        tree = ast.parse(path.read_text())
        imported = [node.module or '' for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)]
        imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for alias in node.names]
        assert not any('backups' in name or 'blockcipher_nd_v3' in name for name in imported)
        assert not any(name.startswith('cipher_lab_plugins.') for name in imported)
        if 'dedicated' in str(path.relative_to(ROOT)):
            assert not any(name.startswith(('cipher_lab.data', 'cipher_lab.training',
                                             'cipher_lab_plugins')) for name in imported)


def test_duplicate_fields_and_bad_task_declarations(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.config import load_config
    path = tmp_path / 'config.json'
    path.write_text('{"method":"one","method":"two"}')
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        load_config(path)
    raw = recipe().to_dict()
    raw['tasks'][0]['rounds'] = 15
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        load_config(path)


@pytest.mark.parametrize('failure', ['private-length', 'oracle-exception', 'meter-reserve'])
def test_failures_stop_without_block_completion(failure: str) -> None:
    from cipher_lab_plugins.innovation1.data import DataError, Provider
    r = small_recipe()
    oracle, private, meter = FakeOracle(), Private(), Meter()

    def bad_oracle(*args: object) -> tuple[int, ...]:
        raise RuntimeError('PRIVATE_SENTINEL')

    class RejectMeter(Meter):
        def reserve(self, kind: str, amount: int) -> None:
            raise RuntimeError('PUBLIC_BUDGET_STOP')

    if failure == 'meter-reserve':
        meter = RejectMeter()
    p = Provider(r, {t.cipher: bad_oracle if failure == 'oracle-exception' else oracle
                     for t in r.tasks},
                 (lambda size: b'') if failure == 'private-length' else private, meter)
    with pytest.raises(RuntimeError) as error:
        p.train(r.tasks[0], 0)
    assert 'SENTINEL' not in str(error.value)
    assert meter.completed == 0
    if failure == 'meter-reserve':
        assert private.calls == 0 and not oracle.calls
    with pytest.raises(DataError, match='PROVIDER_STOPPED'):
        p.dev(r.tasks[0])
