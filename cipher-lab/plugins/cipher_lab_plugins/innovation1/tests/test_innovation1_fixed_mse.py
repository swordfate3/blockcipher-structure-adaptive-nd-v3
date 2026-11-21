"""Fixed-pool integration: guarded in-memory stand-ins, no science execution."""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from plugin_isolation import forbidden

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Clocks, Meter

CONFIG = Path('configs/innovation1-skinny-layout-fixed-mse.json')


def test_exact_config_and_static_budget(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.config import load_config
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, FIXED_MSE, official
    from cipher_lab_plugins.innovation1.single_train.config import estimate, learning_rate

    recipe = load_config(CONFIG)
    assert recipe == FIXED_MSE and recipe.run_id is None and recipe.fixed_pool
    old = load_config(CONFIG.with_name('innovation1-skinny-layout-early-stop.json'))
    assert old == EARLY_STOP and not old.fixed_pool
    assert (recipe.method, recipe.namespace) != (old.method, old.namespace)
    assert recipe.resources == old.resources
    assert recipe.to_dict()['model'] == old.to_dict()['model']
    assert recipe.to_dict()['seeds'] == old.to_dict()['seeds']
    assert recipe.to_dict()['optimizer']['loss'] == 'mean-MSE-on-sigmoid'
    assert recipe.to_dict()['selection']['metric'] == 'per-model-dev-mse'
    assert recipe.training.batch_groups == 5000 and recipe.training.epochs == 50
    assert recipe.training.eval_batch_groups == 512
    assert recipe.early_stopping == old.early_stopping
    totals = estimate(recipe)
    assert (totals['real_blocks'], totals['optimizer_updates'], totals['forward_rows']) == (
        20_524_288, 180_000, 1_000_524_288)
    assert totals['train_groups_per_epoch'] == 9_000_000
    assert totals['pool_groups'] == 10_000_000 and totals['dev_groups'] == 1_000_000
    assert totals['test_groups'] == 262_144
    assert estimate(old)['real_blocks'] == 105_512_960
    assert [learning_rate(recipe, c, cosine_epochs=recipe.cosine_epochs)
            for c in (0, 43199, 43200, 89999)] == [0.001, 0.0001, 0.0001, 0.0001]
    bound = replace(recipe, run_id='new-mse-run')
    assert official(bound, 'new-mse-run') == bound
    for field, value in (('run_id', 'other-run'), ('namespace', old.namespace),
                         ('method', old.method)):
        data = recipe.to_dict()
        data[field] = value
        invalid = tmp_path / 'invalid.json'
        invalid.write_text(json.dumps(data))
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            load_config(invalid)
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            overrides: dict[str, Any] = {field: value}
            official(replace(bound, **overrides), 'new-mse-run')


def test_assembly_selects_fixed_provider_and_mse_only_for_new_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import secrets

    from cipher_lab_plugins.innovation1.layout_probe import application, data, runtime
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, FIXED_MSE
    from cipher_lab_plugins.innovation1.resources import Resources
    from cipher_lab_plugins.innovation1.single_train.config import make_policy

    from cipher_lab.ciphers import skinny

    class BCEStub:
        def __init__(self, check: Callable[[], None]) -> None:
            check()

    class MSEStub(BCEStub):
        pass

    monkeypatch.setattr(runtime, 'Runtime', BCEStub)
    monkeypatch.setattr(runtime, 'MSERuntime', MSEStub)
    for recipe, provider_type, runtime_type in ((FIXED_MSE, data.FixedPoolProvider, MSEStub),
                                                (EARLY_STOP, data.Provider, BCEStub)):
        clocks = Clocks()
        resources = Resources(Meter(make_policy(recipe, clocks), clocks, lambda: 0))
        provider, model = application.assemble(recipe, resources)
        assert type(provider) is provider_type and type(model) is runtime_type
        assert provider.oracle is skinny.encrypt_many is forbidden
        assert provider.private is secrets.token_bytes is forbidden


def test_fixed_pool_split_and_sample_replay() -> None:
    from cipher_lab_plugins.innovation1.layout_probe.config import FIXED_MSE
    from cipher_lab_plugins.innovation1.layout_probe.data import FixedPoolProvider
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    recipe = replace(FIXED_MSE, training=replace(FIXED_MSE.training, epochs=2,
                     batches_per_epoch=81, batch_groups=64, dev_groups=576))

    class Artificial(FixedPoolProvider):
        calls = 0

        def _collect(self, split: str, index: int, size: int) -> PairBatch:
            assert split == 'train' and index == self.calls and size == 64
            self.calls += 1
            ids = np.arange(index*size, (index+1)*size, dtype='>u8')
            x = np.zeros((size, 1, 2, 8), np.uint8)
            x[:, 0, 0, :] = ids.view(np.uint8).reshape(-1, 8)
            y = np.repeat(np.array([[1], [0]], np.uint8), size // 2, axis=0)
            return PairBatch(x, y, split, index, recipe.namespace)

    def rows(x: Any) -> list[int]:
        return [int.from_bytes(row, 'big') for row in x[:, 0, 0]]

    def replay() -> tuple[list[int], list[int], list[int]]:
        provider = Artificial(recipe, forbidden, forbidden, SimpleNamespace(check=lambda: None))
        with pytest.raises(ValueError, match='POOL_NOT_READY'):
            provider.train(0)
        provider.prepare_pool()
        assert provider.calls == 90
        with pytest.raises(ValueError, match='POOL_NOT_READY'):
            provider.train(0)
        dev = provider.panel('dev')
        assert dev.slice(0, 512).groups == 512 and dev.slice(512, 576).groups == 64
        provider.activate()
        epochs: list[list[int]] = [[], []]
        for cycle in range(162):
            batch = provider.train(cycle)
            assert batch.groups == 64 and batch.labels.sum() == 32
            epochs[cycle // 81].extend(rows(batch.ciphertext))
        assert provider.calls == 90
        return epochs[0], epochs[1], rows(dev.ciphertext)

    first, second, dev = replay()
    assert (first, second, dev) == replay()
    assert first != second and len(set(first)) == len(first) == 5184
    assert set(first) == set(second) and len(set(dev)) == len(dev) == 576
    assert set(first).isdisjoint(dev) and set(first).union(dev) == set(range(5760))
    assert any(len({row // 64 for row in first[i:i + 64]}) > 1 for i in range(0, 5184, 64))


def test_sigmoid_mse_update_uses_shared_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.layout_probe.runtime import MSERuntime
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    from cipher_lab_plugins.innovation1.scoring import mse
    from cipher_lab_plugins.innovation1.single_train.config import ARCHITECTURE, TASK

    # P=(.5, .75, .25, .5), Y=(0, 1, 0, 1): (.25+.0625+.0625+.25)/4.
    expected = .15625
    z = torch.tensor([[0.], [math.log(3)], [-math.log(3)], [0.]])
    y = torch.tensor([[0.], [1.], [0.], [1.]])
    assert mse(y.numpy(), z.numpy()) == pytest.approx(expected)
    events: list[object] = []
    params: list[dict[str, float]] = [{'lr': 0.}]
    optimizer = SimpleNamespace(param_groups=params,
                                step=lambda: events.append('step'),
                                zero_grad=lambda **kw: events.append(('zero', kw)))
    model = SimpleNamespace(train=lambda: events.append('train'), parameters=lambda: ())
    runtime = object.__new__(MSERuntime)  # no construction, Adam, or device calls
    runtime.recipe, runtime.device = ARCHITECTURE, torch.device('cpu')
    monkeypatch.setattr(runtime, 'models', {'fake': model}, raising=False)
    runtime.optimizers = {'fake': optimizer}
    monkeypatch.setattr(runtime, '_forward', lambda *args: z)
    monkeypatch.setattr(torch.Tensor, 'backward', lambda _: events.append('backward'))
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_',
                        lambda _, limit, **kw: events.append(('clip', limit, kw)))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _: events.append('fake sync'))
    monkeypatch.setattr(torch.nn.functional, 'binary_cross_entropy_with_logits', forbidden)
    meter = SimpleNamespace(reserve=lambda *a: events.append(('reserve', *a)),
                            complete=lambda *a: events.append(('complete', *a)),
                            check=lambda: None)
    assert MSERuntime.update is TorchRuntime.update
    assert runtime.update('fake', TASK, z, y, .001, meter) == pytest.approx(expected)
    zero = ('zero', {'set_to_none': True})
    assert events == ['train', zero, ('reserve', 'forward_rows', 4),
                      ('complete', 'forward_rows', 4), 'backward',
                      ('clip', 1.0, {'error_if_nonfinite': True}), 'step', 'fake sync', zero]
    assert params == [{'lr': .001}]


@pytest.mark.parametrize('case', ['success', 'pool-write', 'reload-failure', 'dev-tail'])
def test_fixed_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.cli import _finish
    from cipher_lab_plugins.innovation1.layout_probe import application, weights
    from cipher_lab_plugins.innovation1.layout_probe.config import EARLY_STOP, FIXED_MSE, SLOTS
    from cipher_lab_plugins.innovation1.layout_probe.data import FixedPoolProvider
    from cipher_lab_plugins.innovation1.resources import Resources
    from cipher_lab_plugins.innovation1.single_train.config import estimate, make_policy
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    run_id = 'fixed-fixture'
    tail = case == 'dev-tail'
    batch_size, eval_size, test_size = (576, 512, 512) if tail else (4, 4, 4)
    pool_size = 10 * batch_size
    training = replace(FIXED_MSE.training, epochs=2 if tail else 40, batches_per_epoch=9,
                       batch_groups=batch_size, dev_groups=batch_size, test_groups=test_size,
                       eval_batch_groups=eval_size)
    recipe = replace(FIXED_MSE, run_id=run_id, training=training)
    if tail:
        recipe = replace(recipe, early_stopping=None, cosine_epochs=None)
    source = {'artificial/fake.py': 'a'*64}
    clocks = Clocks()
    resources = Resources(Meter(make_policy(recipe, clocks), clocks, lambda: 0))
    store = RunStore.claim(tmp_path, run_id, output_bytes=2**24, needs_models=True,
                           model_slots=SLOTS, source_supplier=lambda: source)
    record = tmp_path / 'data' / run_id / 'run.json'
    calls: list[str] = []
    dev_slices: list[int] = []
    draws = 0
    updates = 0
    carrier: tuple[torch.Tensor, torch.Tensor]

    def fake_bytes(size: int) -> bytes:
        nonlocal draws
        assert size == 8 and (updates == 0 or calls == ['release', *SLOTS])
        draws += 1
        return draws.to_bytes(size, 'big')

    def fake_oracle(blocks: tuple[int, ...], key: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        assert len(blocks) == 2 and key == (draws, 0) and rounds == 8
        return (0, draws)

    class Provider(FixedPoolProvider):
        def panel(self, split: str) -> PairBatch:
            if split == 'test':
                assert calls == ['release', *SLOTS] and draws == pool_size
                saved = json.loads(record.read_bytes())
                assert set(saved['summary']['persisted_reloads']) == set(SLOTS)
            return super().panel(split)

    template = {k: torch.tensor([1]) for k in weights.FIXED_BUFFERS}

    class Runtime:
        def metadata(self) -> dict[str, bool]:
            return {'artificial': True}

        def prepare(self, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal carrier
            if batch.split == 'dev':
                dev_slices.append(batch.groups)
            # Only tiny identity carriers, not prepared neural inputs.
            carrier = (torch.from_numpy(batch.ciphertext[:, 0, 1, -2:].copy()),
                       torch.from_numpy(batch.labels.astype(np.float32)))
            return carrier

        def update(self, slot: str, task: object, x: torch.Tensor, y: torch.Tensor,
                   lr: float, meter: Resources) -> float:
            nonlocal updates
            if not updates:
                saved = json.loads(record.read_bytes())
                assert saved['summary']['pool_status'] == 'READY'
                assert set(saved['artifacts']) == {'dataset/train-SKINNY.npz',
                                                   'dataset/dev-SKINNY.npz'}
                assert saved['counts']['completed_real_blocks'] == 2 * pool_size
            assert draws == pool_size and x is carrier[0] and y is carrier[1]
            assert slot == SLOTS[updates % 2]
            updates += 1
            meter.reserve('forward_rows', len(y))
            meter.complete('forward_rows', len(y))
            return .125

        def predict(self, slot: str, task: object, x: torch.Tensor) -> Any:
            # MSE prefers epoch 1 over epoch 2 for A (BCE instead prefers epoch 2).
            # A gains < min_delta at 25 and must still replace its saved MSE best.
            epoch = updates // 18
            margins = [-2.6, 5., 5., 5.] if slot == SLOTS[0] else [2.] * 4
            if slot == SLOTS[0] and epoch in (2, 26):
                margins = [.1 if epoch == 2 else .11] * 4
            elif slot == SLOTS[0] and epoch >= 25:
                margins[0] += .000001
            return (2 * carrier[1].numpy() - 1) * np.resize(margins, (len(x), 1))

        def checkpoint(self, slot: str, meta: dict[str, Any]) -> bytes:
            assert 'mse' in meta and 'score' not in meta
            return encode(meta, template)

        def release(self) -> None:
            calls.append('release')

        def load_best(self, slot: str, raw: bytes, rec: Any, src: dict[str, str], epoch: int,
                      score: float, check: Callable[[], None]) -> None:
            assert raw == (tmp_path / 'models' / run_id / f'{slot}.pt').read_bytes()
            if case == 'reload-failure' and slot == SLOTS[1]:
                rec = replace(EARLY_STOP, training=training, run_id=run_id)
            weights.decode(raw, rec, src, 'artificial', slot, epoch, score, template)
            calls.append(slot)

    real_atomic = store._atomic
    if case == 'pool-write':
        def fail_train(target: Path, content: bytes) -> None:
            if target.name == 'train-SKINNY.npz':
                raise OSError('fake storage error')
            real_atomic(target, content)
        monkeypatch.setattr(store, '_atomic', fail_train)
    application._execute(recipe, store, resources, 'artificial',
                         lambda rec, meter: (Provider(rec, fake_oracle, fake_bytes, meter),
                                             Runtime()))
    saved = json.loads(record.read_bytes())
    summary = saved['summary']
    if case == 'pool-write':
        assert saved['status'] == 'FAILED' and summary['reason'] == 'WRITE_FAILED'
        assert updates == 0 and draws == pool_size and not calls
        assert saved['counts']['completed_real_blocks'] == 2 * pool_size
        return
    if tail:
        assert saved['status'] == 'SUCCEEDED'
        assert dev_slices == [512, 64] * 2 and draws == pool_size + test_size
        assert saved['counts']['completed_forward_rows'] == 2 * (2 * (5184 + 576) + 512)
        return
    assert summary['training_termination'] == dict(reason='EARLY_STOPPING', completed_epochs=36)
    assert updates == 36*18 and summary['paired_completed'] == 36*9
    assert [summary['best'][s]['epoch'] for s in SLOTS] == [25, 1]
    assert set(summary['curves'][0]['train_mse'].values()) == {.125}
    assert 'train_bce' not in summary['curves'][0]
    first, second = (summary['curves'][i]['dev'][SLOTS[0]] for i in (0, 1))
    assert second['bce'] < first['bce'] and first['mse'] < second['mse']
    assert all('mse' in s and 'bce' not in s for s in summary['best'].values())
    if case == 'reload-failure':
        assert saved['status'] == 'FAILED' and draws == pool_size
        assert set(summary['persisted_reloads']) == {SLOTS[0]} and 'test_panel' not in summary
        return
    totals = estimate(replace(recipe, training=replace(training, epochs=36)))
    assert (totals['real_blocks'], totals['optimizer_updates'], totals['forward_rows']) == (
        88, 648, 2888)
    for kind in ('attempted', 'completed'):
        for unit in ('real_blocks', 'optimizer_updates', 'forward_rows', 'null_words'):
            assert saved['counts'][f'{kind}_{unit}'] == totals[unit]
    assert saved['status'] == 'SUCCEEDED' and draws == pool_size + test_size
    for split in ('train', 'dev', 'test'):
        with np.load(record.parent / summary[f'{split}_panel']['ref'], allow_pickle=False) as data:
            meta = json.loads(data['metadata'].tobytes())
            assert meta['run_id'] == run_id and meta['recipe'] == saved['recipe']
            assert meta['method'] == recipe.method and meta['split'] == split
            assert data['ciphertext'].dtype == data['labels'].dtype == np.uint8
            if split == 'test':
                assert meta['weights'] == summary['persisted_reloads']
            else:
                assert set(data.files) == {'ciphertext', 'labels', 'metadata'}
    assert _finish(store, tmp_path, run_id, resources=resources) == 0
    report = (tmp_path / 'result' / run_id / 'report.md').read_text(encoding='utf-8')
    assert all(text in report for text in ('|MSE|', '|ACC|', 'ΔBCE', 'EARLY_STOPPING'))
    assert '固定数据、批大小和损失函数同时改变' in report
    assert '不能把总体涨跌归因于其中任何一项' in report
    assert all(f"{score['mse']:.8f}" in report for score in summary['test_scores'].values())
