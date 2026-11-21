"""Tiny byte-pinned saved pools and a fake runtime behind the owner guards."""
from __future__ import annotations

import io
import json
import math
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from plugin_isolation import forbidden

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Clocks, Meter

CONFIG = Path('configs/innovation1-skinny-layout-fixed-bce.json')
OLD_CODE = {'historical/fake.py': 'a' * 64}
NEW_CODE = {'current/fake.py': 'b' * 64}
PARENT = 'innovation1-skinny-fixed-mse-20261121-01'
CHILD = 'fixture-fixed-bce'


def test_static_exact_config_and_policy(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.config import load_config
    from cipher_lab_plugins.innovation1.layout_probe.config import FIXED_BCE, FIXED_MSE, official
    from cipher_lab_plugins.innovation1.single_train.config import estimate, learning_rate

    recipe = load_config(CONFIG)
    assert recipe == FIXED_BCE and recipe.fixed_pool and recipe.reuse_pool
    assert recipe.namespace != FIXED_MSE.namespace
    data, prior = recipe.to_dict(), FIXED_MSE.to_dict()
    assert data['data']['shuffle_namespace'] == prior['namespace']
    for key in ('model', 'training', 'seeds', 'resources'):
        assert data[key] == prior[key]
    assert {k: v for k, v in data['optimizer'].items() if k != 'loss'} == {
        k: v for k, v in prior['optimizer'].items() if k != 'loss'}
    assert data['optimizer']['loss'] == 'mean-BCEWithLogits'
    assert data['selection'] == {**prior['selection'], 'metric': 'per-model-dev-bce'}
    expected = estimate(FIXED_MSE)
    expected['real_blocks'] = 524_288
    assert estimate(recipe) == expected
    assert (expected['optimizer_updates'], expected['forward_rows']) == (180_000, 1_000_524_288)
    assert [learning_rate(recipe, c, cosine_epochs=recipe.cosine_epochs)
            for c in (0, 43199, 43200, 89999)] == [0.001, 0.0001, 0.0001, 0.0001]
    assert official(replace(FIXED_BCE, run_id=CHILD), CHILD).run_id == CHILD
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        official(replace(FIXED_BCE, run_id=PARENT), PARENT)
    source = data['data']['source']
    assert (source['root'], source['run_id']) == (
        r'G:\lxy\shared-m-cipher-nd-runs\20261121-skinny-fixed-mse-01', PARENT)
    assert [source[k]['bytes'] for k in ('record', 'train', 'dev')] == [48314, 153012425, 17012423]
    assert [source[k]['sha256'] for k in ('record', 'train', 'dev')] == [
        '5c6c96d95ef39d941b841d051c3d3df22584bda1cc675ef74791c20e5e66c1d7',
        '78e6d1a219d3a20f5c3746ae5c45bfb14f33028f09a6cd13fa4f81287739637d',
        'ed8792908f1ddca0842e97a55153f02a1efe2467d2a588ae69feae199145441a',
    ]
    changed = tmp_path / 'changed.json'
    for item, key, value in ((source, 'root', str(tmp_path)), (source, 'run_id', CHILD),
                             (data['data'], 'shuffle_namespace', recipe.namespace),
                             (source['train'], 'sha256', '0' * 64),
                             (source['dev'], 'bytes', float(source['dev']['bytes']))):
        prior_value = item[key]
        item[key] = value
        changed.write_text(json.dumps(data))
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            load_config(changed)
        item[key] = prior_value


def _fixture(tmp_path: Path, case: str = '') -> tuple[Any, Any, dict[Path, bytes]]:
    from cipher_lab_plugins.innovation1.layout_probe.config import FIXED_BCE, FIXED_MSE, FilePin
    from cipher_lab_plugins.innovation1.layout_probe.data import FixedPoolProvider, encode_panel
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    old = replace(FIXED_MSE, run_id=PARENT, training=replace(FIXED_MSE.training,
                  batch_groups=4, batches_per_epoch=9, dev_groups=4))
    new = replace(FIXED_BCE, run_id=CHILD, training=replace(old.training, test_groups=4,
                  eval_batch_groups=4))

    class Prior(FixedPoolProvider):
        def _collect(self, split: str, index: int, size: int) -> PairBatch:
            assert (split, size) == ('train', 4)
            x = np.arange(64, dtype=np.uint8).reshape(4, 1, 2, 8) + index * 7
            y = np.array([[1], [1], [0], [0]], dtype=np.uint8)
            return PairBatch(x, y, split, index, old.namespace)

    prior = Prior(old, forbidden, forbidden, SimpleNamespace(check=lambda: None))
    train, dev = prior.prepare_pool(), prior.panel('dev')
    prior.activate()
    root = tmp_path / 'prior'
    store = RunStore.claim(root, PARENT, output_bytes=2**20, source_supplier=lambda: OLD_CODE)
    record_path = root / 'data' / PARENT / 'run.json'
    recipe = old.to_dict()
    evidence: dict[str, Any] = dict(method=old.method, backend_identity='ciphers', synthetic=False,
                                    pool_status='READY')
    pins: dict[str, FilePin] = {}
    files: dict[Path, bytes] = {}
    for split, panel in (('train', train), ('dev', dev)):
        npz = encode_panel(panel, old, OLD_CODE, 'artificial')
        with np.load(io.BytesIO(npz), allow_pickle=False) as loaded:
            meta = json.loads(loaded['metadata'].tobytes())
            arrays = {k: loaded[k] for k in ('ciphertext', 'labels')}
        meta['backend'] = 'ciphers'
        if split == 'dev':
            if case == 'metadata-namespace':
                meta['recipe']['namespace'] = new.namespace
            if case == 'metadata-source':
                meta['source'] = NEW_CODE
            if case == 'metadata-digest':
                meta['digest'] = '0' * 64
            if case == 'labels':
                arrays['labels'][:, 0] = [2, 0, 0, 0]  # sum passes; labels are invalid
            if case == 'dev-order':
                arrays['labels'] = arrays['labels'][::-1].copy()
            if case == 'shape':
                arrays['ciphertext'] = arrays['ciphertext'].reshape(4, 16)
            if case == 'dtype':
                arrays['labels'] = arrays['labels'].astype(np.int8)
            if case == 'object':
                arrays['labels'] = arrays['labels'].astype(object)
            if case == 'extra-member':
                arrays['extra'] = np.zeros(1, dtype=np.uint8)
        meta_bytes = json.dumps(meta, sort_keys=True, separators=(',', ':')).encode('ascii')
        buffer = io.BytesIO()
        np.savez(buffer, metadata=np.frombuffer(meta_bytes, dtype=np.uint8), **arrays)
        payload = buffer.getvalue()
        ref = store.write_dataset(f'{split}-SKINNY', payload)
        digest = sha256(payload).hexdigest()
        evidence[f'{split}_panel'] = dict(ref=ref, sha256=digest, digest=panel.digest)
        pins[split] = FilePin(len(payload), digest)
        path = record_path.parent / ref
        if (case, split) == ('panel-hash', 'dev'):
            path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        if (case, split) == ('panel-size', 'dev'):
            path.write_bytes(payload + b'x')
        if (case, split) == ('missing', 'dev'):
            path.unlink()
        if (case, split) == ('symlink', 'dev'):
            spare = tmp_path / 'byte-identical.npz'
            path.rename(spare)
            path.symlink_to(spare)
        if path.exists():
            files[path] = path.read_bytes()
    if case == 'not-ready':
        evidence['pool_status'] = 'PREPARING'
    if case == 'bad-ref':
        evidence['dev_panel']['ref'] = 'dataset/train-SKINNY.npz'
    if case == 'record-namespace':
        recipe['namespace'] = new.namespace
    record = json.loads(record_path.read_bytes())
    n = 78 if case == 'counts' else 80
    record.update(recipe=recipe, summary=evidence, counts=dict(attempted_real_blocks=n,
                  completed_real_blocks=n, attempted_null_words=0, completed_null_words=0))
    if case == 'manifest':
        record['artifacts']['dataset/dev-SKINNY.npz']['bytes'] -= 1
    raw = json.dumps(record).encode('ascii')
    disk = raw if case != 'record-hash' else raw.replace(b'RUNNING', b'STOPPED')
    record_path.write_bytes(disk)
    files[record_path] = disk
    assert new.pool_source is not None
    return (replace(new, pool_source=replace(new.pool_source, root=str(root),
                    record=FilePin(len(raw), sha256(raw).hexdigest()),
                    train=pins['train'], dev=pins['dev'])), prior, files)


@pytest.mark.parametrize('case', ['missing', 'record-hash', 'record-namespace', 'not-ready',
    'bad-ref', 'manifest', 'panel-hash', 'panel-size', 'symlink', 'metadata-namespace',
    'metadata-source', 'metadata-digest', 'labels', 'dev-order', 'shape', 'dtype', 'object',
    'counts', 'extra-member'])
def test_bad_archive_fails_before_updates_without_fallback(tmp_path: Path, case: str) -> None:
    _run(tmp_path, case)


@pytest.mark.parametrize('case', ['success', 'failed-second-reload'])
def test_stopped_ready_replay_and_test_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    # The fake runtime uses only this run's byte-backed best reads, not torch.load.
    monkeypatch.setattr(torch, 'load', forbidden)
    _run(tmp_path, case)


def _run(tmp_path: Path, case: str) -> None:
    from cipher_lab_plugins.innovation1.cli import _finish
    from cipher_lab_plugins.innovation1.config import canonical
    from cipher_lab_plugins.innovation1.layout_probe import application, weights
    from cipher_lab_plugins.innovation1.layout_probe.config import SLOTS
    from cipher_lab_plugins.innovation1.layout_probe.reuse import ReusedPoolProvider
    from cipher_lab_plugins.innovation1.resources import Resources
    from cipher_lab_plugins.innovation1.single_train.config import make_policy
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    recipe, prior, original = _fixture(tmp_path, case)
    valid = case in ('success', 'failed-second-reload')
    root = tmp_path / 'new'
    record = root / 'data' / CHILD / 'run.json'
    clocks = Clocks()
    resources = Resources(Meter(make_policy(recipe, clocks), clocks, lambda: 0))
    store = RunStore.claim(root, CHILD, output_bytes=2**24, needs_models=True,
                           model_slots=SLOTS, source_supplier=lambda: NEW_CODE)
    calls: list[str] = []
    seen: list[str] = []
    updates = 0
    labels: torch.Tensor

    class Provider(ReusedPoolProvider):
        def _generate(self, split: str, index: int, size: int) -> PairBatch:
            assert valid and (split, index, size) == ('test', 0, 4)
            assert calls == ['release', *SLOTS]
            saved = json.loads(record.read_bytes())
            assert set(saved['summary']['persisted_reloads']) == set(SLOTS)
            self.meter.reserve('real_blocks', 8)
            self.meter.complete('real_blocks', 8)
            calls.append('test')
            return PairBatch(np.zeros((4, 1, 2, 8), np.uint8), np.array([[0], [1]]*2, np.uint8),
                             split, index, recipe.namespace)

    class Runtime:
        def metadata(self) -> dict[str, bool]:
            return {'artificial': True}

        def prepare(self, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal labels
            assert not batch.ciphertext.flags.writeable and not batch.labels.flags.writeable
            if batch.split == 'train':
                previous = prior.train(len(seen))
                assert batch.digest == previous.digest
                np.testing.assert_array_equal(batch.ciphertext, previous.ciphertext)
                np.testing.assert_array_equal(batch.labels, previous.labels)
                seen.append(batch.digest)
            labels = torch.from_numpy(batch.labels.astype(np.float32))
            return torch.from_numpy(batch.ciphertext.copy()), labels

        def update(self, slot: str, task: object, x: torch.Tensor, y: torch.Tensor,
                   lr: float, meter: Resources) -> float:
            nonlocal updates
            assert valid and slot == SLOTS[updates % 2]
            assert meter.snapshot()['attempted_real_blocks'] == 0
            assert meter.snapshot()['completed_real_blocks'] == 0
            if updates == 0:
                saved = json.loads(record.read_bytes())
                assert saved['summary']['pool_status'] == 'READY' and not saved['artifacts']
                assert saved['summary']['pool_source']['status'] == 'RUNNING'
                for split in ('train', 'dev'):
                    panel = saved['summary'][f'{split}_panel']
                    assert panel['ref'] == f'dataset/{split}-SKINNY.npz'
                    assert panel['origin'] == 'reused'
            updates += 1
            meter.reserve('forward_rows', len(y))
            meter.complete('forward_rows', len(y))
            return .2

        def predict(self, slot: str, task: object, x: torch.Tensor) -> Any:
            # DEV epoch one wins on MSE but loses on BCE for row scan.
            margins = ([-2.6, 5., 5., 5.] if updates == 18 else [.1] * 4)
            if slot == SLOTS[1]:
                margins = [2.] * 4
            return (2 * labels.numpy() - 1) * np.asarray(margins).reshape(4, 1)

        def checkpoint(self, slot: str, meta: dict[str, Any]) -> bytes:
            assert meta == weights.identity(recipe, NEW_CODE, 'artificial', slot, meta['epoch'],
                                            meta['score'])
            assert meta['recipe']['data']['source'] == recipe.to_dict()['data']['source']
            return json.dumps(meta).encode('ascii')

        def release(self) -> None:
            calls.append('release')

        def load_best(self, slot: str, raw: bytes, rec: Any, source: dict[str, str], epoch: int,
                      score: float, check: Callable[[], None]) -> None:
            assert raw == (root / 'models' / CHILD / f'{slot}.pt').read_bytes()
            expected = weights.identity(rec, source, 'artificial', slot, epoch, score)
            assert canonical(json.loads(raw)) == canonical(expected)
            if slot == SLOTS[1] and case == 'failed-second-reload':
                raise ValueError('synthetic reload fault')
            calls.append(slot)

    application._execute(recipe, store, resources, 'artificial',
                         lambda rec, meter: (Provider(rec, forbidden, forbidden, meter), Runtime()))
    saved = json.loads(record.read_bytes())
    summary, counts = saved['summary'], saved['counts']
    assert all(path.read_bytes() == contents for path, contents in original.items())
    if not valid:
        assert saved['status'] == 'FAILED' and summary['reason'] == 'POOL_SOURCE_INVALID'
        assert not calls and not seen and not saved['artifacts']
        assert counts['attempted_optimizer_updates'] == counts['attempted_real_blocks'] == 0
        assert counts['attempted_forward_rows'] == 0
        return
    assert updates == 648 and len(seen) == 324 and seen[:9] != seen[9:18]
    # Independent of the shared provider implementation: changing both old/new
    # shuffle streams in lockstep must not make the comparison pass.
    assert {i: seen[i] for i in (0, 8, 9, 225, 323)} == {
        0: 'd378540dec646abecc4e7bfc6f3dd5be5ee16f4d00456d89c3689a690eacf6ca',
        8: '42a212941cc7b871f1fe346078d188dcf60de615435d328532f14aa890bf96f6',
        9: '13bb51b071b087f482e7c040db533e7967ae5e4bcf6c5927755e7da3dbb31150',
        225: 'fd4b4029133a2262f285c7d93ae53844b3639e8e5d3a5a0ee4f7e53632fec68a',
        323: '8b17435da986745aafa45b806d047ada92405b379cb5230691e4ef6398b2535b',
    }
    assert summary['training_termination'] == dict(reason='EARLY_STOPPING', completed_epochs=36)
    assert [summary['best'][slot]['epoch'] for slot in SLOTS] == [2, 1]
    assert all('bce' in s and 'mse' not in s for s in summary['best'].values())
    first, second = (summary['curves'][i]['dev'][SLOTS[0]] for i in (0, 1))
    assert second['bce'] < first['bce'] and first['mse'] < second['mse']
    assert summary['curves'][0]['train_bce'][SLOTS[0]] == pytest.approx(.2)
    if case == 'failed-second-reload':
        assert saved['status'] == 'FAILED' and calls == ['release', SLOTS[0]]
        assert counts['attempted_real_blocks'] == 0 and 'test_panel' not in summary
        return
    assert saved['status'] == 'SUCCEEDED' and calls == ['release', *SLOTS, 'test']
    for kind in ('attempted', 'completed'):
        assert counts[f'{kind}_real_blocks'] == 8 and counts[f'{kind}_optimizer_updates'] == 648
        assert counts[f'{kind}_forward_rows'] == 2888 and counts[f'{kind}_null_words'] == 0
    datasets = {k for k in saved['artifacts'] if k.startswith('dataset/')}
    assert datasets == {'dataset/test-SKINNY.npz'}
    with np.load(record.parent / 'dataset/test-SKINNY.npz', allow_pickle=False) as data:
        meta = json.loads(data['metadata'].tobytes())
        assert meta['recipe']['data']['source'] == recipe.to_dict()['data']['source']
        assert meta['weights'] == summary['persisted_reloads']
        assert meta['namespace'] == recipe.namespace
        assert meta['reused_pool'] == {k: summary[k] for k in ('pool_source', 'train_panel',
                                                               'dev_panel')}
    assert _finish(store, root, CHILD, resources=resources) == 0
    report = (root / 'result' / CHILD / 'report.md').read_text(encoding='utf-8')
    assert all(w in report for w in ('|MSE|', '|ACC|', '|BCE|', 'RUNNING', PARENT, 'TRAIN/DEV',
                                    'EARLY_STOPPING', 'DEV', '独立 TEST'))
    assert all(f"{score['mse']:.8f}" in report for score in summary['test_scores'].values())


def test_fixed_bce_uses_standard_logit_runtime_and_closed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1.layout_probe import application, runtime
    from cipher_lab_plugins.innovation1.layout_probe.config import FIXED_BCE
    from cipher_lab_plugins.innovation1.layout_probe.reuse import ReusedPoolProvider
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    from cipher_lab_plugins.innovation1.resources import Resources
    from cipher_lab_plugins.innovation1.single_train.config import make_policy
    from cipher_lab_plugins.innovation1.source import source_identity

    assert runtime.Runtime._loss is TorchRuntime._loss
    loss_only = object.__new__(runtime.Runtime)  # no models, Adam or device preparation
    z, y = torch.tensor([[0.], [math.log(3.)]]), torch.tensor([[0.], [1.]])
    monkeypatch.setattr(torch, 'sigmoid', forbidden)
    assert loss_only._loss(z, y).item() == pytest.approx((math.log(2.) + math.log(4. / 3.)) / 2.)

    class RuntimeStub:
        def __init__(self, check: Callable[[], None]) -> None:
            check()

    monkeypatch.setattr(runtime, 'Runtime', RuntimeStub)
    monkeypatch.setattr(runtime, 'MSERuntime', forbidden)
    clocks = Clocks()
    resources = Resources(Meter(make_policy(FIXED_BCE, clocks), clocks, lambda: 0))
    provider, model = application.assemble(FIXED_BCE, resources)
    assert type(provider) is ReusedPoolProvider and type(model) is RuntimeStub
    assert provider.oracle is forbidden and provider.private is forbidden
    with pytest.raises(ValueError, match='POOL_REUSE_NO_GENERATION'):
        provider._collect('train', 0, 4)
    assert resources.snapshot()['attempted_real_blocks'] == 0
    key = 'cipher_lab_plugins/innovation1/layout_probe/reuse.py'
    path = Path('plugins') / key
    assert source_identity()[key] == sha256(path.read_bytes()).hexdigest()


def test_cli_check_and_overlapping_root_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from cipher_lab_plugins.innovation1 import cli
    from cipher_lab_plugins.innovation1.layout_probe.config import FIXED_BCE

    assert FIXED_BCE.pool_source is not None
    monkeypatch.setattr(RunStore, 'claim', forbidden)
    assert cli.main(['check', '--config', str(CONFIG)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['scope'] == 'STATIC_ONLY_NO_DEVICE_ADMISSION'
    assert result['estimate']['real_blocks'] == 524_288
    assert result['recipe']['data']['source']['record']['bytes'] == 48314
    for root in (FIXED_BCE.pool_source.root.upper(),
                 FIXED_BCE.pool_source.root + r'\data', r'G:\lxy\shared-m-cipher-nd-runs'):
        assert cli.main(['run', '--config', str(CONFIG), '--root', root, '--id', CHILD,
                         '--gpu-uuid', 'GPU-01234567-0123-4567-89ab-0123456789ab']) == 2
        assert 'invalid settings' in capsys.readouterr().err
