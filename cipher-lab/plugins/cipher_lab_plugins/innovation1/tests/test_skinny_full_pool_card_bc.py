"""Full-pool paired card contract; all archives and execution are guarded fakes."""
from __future__ import annotations

import builtins
import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1 import application, cli, resources, source
from cipher_lab_plugins.innovation1.checkpoint import encode
from cipher_lab_plugins.innovation1.config import Task, load_config
from cipher_lab_plugins.innovation1.four_single import replay
from cipher_lab_plugins.innovation1.four_single.models import SinglePairM
from cipher_lab_plugins.innovation1.layout_probe import reuse
from cipher_lab_plugins.innovation1.models import TorchRuntime
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.serial_bc.models import NoCardM
from cipher_lab_plugins.innovation1.single_train.config import TASK, learning_rate
from cipher_lab_plugins.innovation1.single_train.data import PairBatch
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import SOURCE, _archive
from numpy.typing import NDArray
from plugin_isolation import forbidden
from torch import nn

from cipher_lab.artifacts import RunStore, _encode, read_saved
from cipher_lab.execution import Clocks, Meter
from cipher_lab.models.shared_m.router import PublicRouter

CONFIG = Path('configs/innovation1-skinny-full-pool-card-bc.json')
NOW = datetime(2026, 11, 25, 10, tzinfo=UTC)
DEADLINE = '2026-11-25T10:01:00Z'
TEST_SHA = '3a845204b3ccf3afb77b73625219fcd78b107155a9526406fe3693935f1f3c27'
TEST_DIGEST = 'abb5a3a9d6c0fa4303ac353e2ae9117ba8e7263507e60b8fdf322fd061119bb6'


def test_check_exact_maximum_and_no_numeric_imports(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.skinny_card.config import CARD, SLOTS, estimate

    real_import = builtins.__import__

    def stdlib(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split('.')[0] not in ('numpy', 'torch') and not name.endswith('scoring')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', stdlib)
    monkeypatch.setattr(application, 'make_resources', forbidden)
    monkeypatch.setattr(source, 'no_reparse', forbidden)
    monkeypatch.setattr(RunStore, 'claim', forbidden)
    assert cli.main(['check', '--config', str(CONFIG)]) == 0 and load_config(CONFIG) is CARD
    value = json.loads(capsys.readouterr().out)
    assert value['scope'] == 'STATIC_ONLY_NO_DEVICE_ADMISSION'
    assert value['recipe'] == CARD.to_dict()
    tr = CARD.training
    assert (tr.epochs, tr.batches_per_epoch, tr.batch_groups, tr.dev_groups, tr.test_groups) == (
        50, 1800, 5000, 1_000_000, 262_144)
    n = value['estimate']
    assert n == estimate(CARD)
    assert [n[k] for k in ('fits', 'retries', 'optimizer_updates', 'forward_rows', 'real_blocks',
                           'null_words', 'train_groups_per_epoch')] == [
        2, 0, 180000, 1000524288, 0, 0, 9000000]
    assert n['artifact_peak_bound_bytes'] < 2**30
    assert CARD.resources.wall_seconds == 50400
    assert CARD.resources.gpu_bytes == 20 * 2**30 and CARD.resources.rss_bytes == 16 * 2**30
    assert SLOTS == ('B-card-best', 'C-no-card-best')
    assert CARD.test_sha256 == TEST_SHA and CARD.test_digest == TEST_DIGEST
    assert [learning_rate(CARD, c, cosine_epochs=CARD.cosine_epochs)
            for c in (0, 43199, 43200, 89999)] == [.001, .0001, .0001, .0001]
    bad = tmp_path / 'bad.json'
    for k, wrong in [('run_id', 'other'), ('extra', None), ('routing', {}), ('data', {}),
                     ('training', {**value['recipe']['training'], 'epochs': 50.})]:
        bad.write_bytes(_encode({**value['recipe'], k: wrong}))
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            load_config(bad)


def _fixture(tmp_path: Path, case: str = 'early') -> tuple[Any, dict[Path, bytes],
                                                         dict[str, list[str]], Path]:
    from cipher_lab_plugins.innovation1.skinny_card.config import CARD

    archive, files, batches = _archive(tmp_path, case)
    base = Path(archive.archive.root) / 'data' / archive.archive.run_id
    test = base / 'dataset/test-SKINNY.npz'
    record_bytes = files[base / 'run.json']
    record = json.loads(b'{' + record_bytes[1:] if case == 'record-hash' else record_bytes)
    recipe = replace(CARD, training=archive.training, archive=archive.archive,
                     skinny_source=archive.skinny_source, run_id='guarded-card-pair',
                     test_digest=record['summary']['test']['SKINNY']['panel']['digest'],
                     test_sha256=sha256(files[test]).hexdigest())
    return recipe, files, batches, test


def test_reuse_order_and_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.skinny_card.config import SLOTS
    from cipher_lab_plugins.innovation1.skinny_card.data import CardPools

    recipe, files, batches, test = _fixture(tmp_path)
    pinned, zip_open = reuse._pinned, ZipFile.open
    seen: list[Path] = []

    def track(path: Path, *args: Any) -> Any:
        seen.append(path)
        return pinned(path, *args)

    def member(z: ZipFile, item: Any, mode: Literal['r', 'w'] = 'r', **kw: Any) -> Any:
        assert getattr(item, 'filename', item) != 'logits.npy'
        return zip_open(z, item, mode, **kw)

    monkeypatch.setattr(reuse, '_pinned', track)
    monkeypatch.setattr(replay, '_pinned', track)
    monkeypatch.setattr(ZipFile, 'open', member)
    pools = CardPools(recipe, lambda: None)
    with pytest.raises(ValueError, match='TEST_ORDER'):
        pools.panel('test')
    pools.prepare_pool()
    assert pools.panel('dev').digest == pools.pool_provenance()['dev_panel']['digest']
    pools.activate()
    assert [pools.train(c).digest for c in range(len(batches['SKINNY']))] == batches['SKINNY']
    with pytest.raises(ValueError, match='TEST_ORDER'):
        pools.load_test({SLOTS[0]: {}})
    reloads = {slot: dict(ref=f'model/{slot}.pt') for slot in SLOTS}
    test_panel = pools.load_test(reloads)
    assert test_panel.digest == recipe.test_digest
    with pytest.raises(ValueError, match='TEST_ORDER'):
        pools.load_test(reloads)
    assert [p.name for p in seen] == ['run.json', 'run.json', 'train-SKINNY.npz',
                                     'dev-SKINNY.npz', 'test-SKINNY.npz']
    assert seen[-1] == test and all(p.read_bytes() == data for p, data in files.items())
    for key in ('test_digest', 'test_sha256'):
        seen.clear()
        with pytest.raises(reuse.PoolSourceError):
            CardPools(replace(recipe, **{key: '0' * 64}), lambda: None).prepare_pool()
        assert seen == [test.parent.parent / 'run.json']


@pytest.mark.parametrize('api', ['installed', 'legacy'])
def test_test_archive_exact_numeric_members(api: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single.config import PANEL_OVERHEAD
    from cipher_lab_plugins.innovation1.skinny_card.config import CARD, SLOTS
    from cipher_lab_plugins.innovation1.skinny_card.data import encode_test

    def legacy_savez(file: io.BytesIO, *args: Any, **kwds: Any) -> None:
        # NumPy 2.0 treats every keyword after file as a named array, not a control.
        members = dict(kwds)
        for i, array in enumerate(args):
            name = f'arr_{i}'
            if name in members:
                raise ValueError(f'duplicate member: {name}')
            members[name] = array
        with ZipFile(file, 'w') as archive:
            for name, array in members.items():
                with archive.open(f'{name}.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=True)

    if api == 'legacy':
        monkeypatch.setattr(np, 'savez', legacy_savez)
    batch = PairBatch(np.arange(64, dtype=np.uint8).reshape(4, 1, 2, 8),
                      np.array([[0], [1], [1], [0]], dtype=np.uint8),
                      'test', 0, CARD.archive_reader_recipe().namespace)
    recipe = replace(CARD, training=replace(CARD.training, test_groups=batch.groups),
                     test_digest=batch.digest, test_sha256='c' * 64)
    logits = {SLOTS[0]: np.arange(4, dtype=np.float64).reshape(4, 1),
              SLOTS[1]: -np.arange(1, 5, dtype=np.float64).reshape(4, 1)}
    best = {slot: dict(ref=f'model/{slot}.pt', epoch=i + 1, sha256=str(i) * 64)
            for i, slot in enumerate(SLOTS)}
    reused = dict(ref='dataset/test-SKINNY.npz', sha256=recipe.test_sha256, digest=batch.digest)
    raw = encode_test(batch, recipe, SOURCE, 'artificial', logits=logits, weights=best,
                      reused_input=reused)
    assert len(raw) <= batch.groups * 33 + PANEL_OVERHEAD
    expected: dict[str, NDArray[np.uint8] | NDArray[np.float64]] = {
        'ciphertext': batch.ciphertext, 'labels': batch.labels, **logits}
    members = sorted(f'{name}.npy' for name in (*expected, 'metadata'))
    with ZipFile(io.BytesIO(raw)) as archive:
        assert sorted(archive.namelist()) == members
    with np.load(io.BytesIO(raw), allow_pickle=False) as panel:
        for name, original in expected.items():
            array = panel[name]
            assert array.dtype == original.dtype and not array.dtype.hasobject
            np.testing.assert_array_equal(array, original)
        metadata = panel['metadata']
        assert metadata.dtype == np.uint8 and not metadata.dtype.hasobject
        assert json.loads(metadata.tobytes()) == dict(
            schema=1, method=recipe.method, recipe=recipe.to_dict(), run_id=recipe.run_id,
            source=SOURCE, backend='artificial', cipher=TASK.cipher, split='test',
            rows=batch.groups, encoding='packed-msb0-single-pair-v1', digest=batch.digest,
            namespace=batch.namespace,
            reused_input=reused, best=best)


def test_different_router_input_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.skinny_card.config import CARD, SLOTS
    from cipher_lab_plugins.innovation1.skinny_card.runtime import CardRuntime

    def cpu(self: Any, recipe: Any, device: str, *, inference_only: bool) -> None:
        assert device == 'cuda:0' and inference_only
        self.recipe, self.device = recipe, torch.device('cpu')
        self.models, self.optimizers = {}, {}

    def adam(params: Any, **kw: Any) -> object:
        assert tuple(params) and kw['lr'] == .001
        return object()

    monkeypatch.setattr(TorchRuntime, '__init__', cpu)
    monkeypatch.setattr(torch.optim, 'Adam', adam)
    rng = torch.random.get_rng_state()
    fit = CardRuntime(CARD, lambda: None, forbidden)
    assert torch.equal(torch.random.get_rng_state(), rng)
    a, b = (fit.models[s] for s in SLOTS)
    assert type(a) is SinglePairM and type(b) is NoCardM
    torch.testing.assert_close(a.state_dict(), b.state_dict(), atol=0., rtol=0.)
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(a.state_dict().values(),
                                                          b.state_dict().values(), strict=True))
    assert fit.optimizers[SLOTS[0]] is not fit.optimizers[SLOTS[1]]
    inputs: list[torch.Tensor] = []
    preprocessing: list[tuple[Any, ...]] = []

    def gate(x: torch.Tensor) -> torch.Tensor:
        inputs.append(x)
        return x.new_tensor([.1, .3, .2])

    def fake_shared(shared: Any, slots: Any, public: torch.Tensor, *, rounds: int,
                    nibble: Any, route_for: Any) -> torch.Tensor:
        assert nibble.func is SinglePairM._column and nibble.args[0].shared is shared
        preprocessing.append((slots, public, rounds))
        route_for(public)  # fake scorer, no real layer is run
        return public.new_zeros((4, 1))

    monkeypatch.setattr('cipher_lab_plugins.innovation1.four_single.models.shared_logits',
                        fake_shared)
    from functools import partial

    class WiringOnly(nn.Module):
        def __init__(self, variant: type[SinglePairM]) -> None:
            super().__init__()
            self._route = partial(variant._route, self)  # type: ignore[arg-type]
            self._column = partial(SinglePairM._column, self)  # type: ignore[arg-type]
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(321)
                router = PublicRouter(1)
            monkeypatch.setattr(router.gate, 'forward', gate)
            self.shared = SimpleNamespace(router=router, parameters=lambda: iter([torch.zeros(1)]))

        def forward(self, x: torch.Tensor, public: torch.Tensor, *, rounds: int) -> torch.Tensor:
            return SinglePairM.forward(self, x, public, rounds=rounds)  # type: ignore[arg-type]

    assert type(a).forward is type(b).forward is SinglePairM.forward
    fit.models = dict(zip(SLOTS, (WiringOnly(SinglePairM), WiringOnly(NoCardM)), strict=True))
    x = torch.zeros((4, 1, 2, 64), dtype=torch.float32)
    for slot in SLOTS:
        fit._forward(slot, TASK, x)
    assert len(inputs) == 2 and len(preprocessing) == 2
    assert inputs[0] is preprocessing[0][1] is preprocessing[1][1]
    assert not torch.equal(inputs[0], inputs[1]) and torch.equal(inputs[1], torch.full((13,), .5))
    assert torch.equal(preprocessing[0][0], preprocessing[1][0])
    assert preprocessing[0][2] == preprocessing[1][2] == 8


@pytest.mark.parametrize('case', ['early', 'ceiling'])
def test_same_batches_and_common_stop(case: str, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    _fit(case, tmp_path, monkeypatch)


@pytest.mark.parametrize('case', ['strict-b', 'strict-c', 'disk', 'test-save', 'after-test'])
def test_strict_disk_best_before_test(case: str, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    _fit(case, tmp_path, monkeypatch)


@pytest.mark.parametrize('case', ['interrupt', 'deadline', 'ledger-fail'])
def test_deadline_and_failure_preserve_paid_evidence(case: str, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _fit(case, tmp_path, monkeypatch)


def _fit(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.skinny_card import application as card_app
    from cipher_lab_plugins.innovation1.skinny_card.config import SLOTS, make_policy
    from cipher_lab_plugins.innovation1.skinny_card.data import CardPools
    from cipher_lab_plugins.innovation1.skinny_card.runtime import CardRuntime

    recipe, _, batches, test_path = _fixture(tmp_path / 'inputs')
    current = NOW
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: current)
    res = Resources(Meter(make_policy(recipe, clocks, DEADLINE), clocks, lambda: 0))
    root = tmp_path / 'owner'
    run = root / 'data' / recipe.run_id / 'run.json'
    store = RunStore.claim(root, recipe.run_id, output_bytes=2**30, needs_models=True,
                           model_slots=SLOTS, source_supplier=lambda: SOURCE)
    paired: list[tuple[str, PairBatch, float]] = []
    events: list[str] = []
    finalizations: list[str] = []
    reads: list[str] = []
    models: list[nn.Module] = []
    adams: list[object] = []
    observed_routes: list[tuple[str, int]] = []
    per_epoch = recipe.training.batches_per_epoch

    def cpu(self: Any, streams: Any, device: str, *, inference_only: bool) -> None:
        assert device == 'cuda:0' and inference_only
        self.recipe, self.device = streams, torch.device('cpu')
        self.models, self.optimizers = {}, {}

    def adam(params: Any, **kw: Any) -> object:
        assert tuple(params) and kw['lr'] == .001
        adams.append(object())
        return adams[-1]

    def build(self: CardRuntime, slot: str) -> SinglePairM:
        # Small seeded CPU parameter template for repeated strict disk decoding.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(4321)
            model = nn.Linear(1, 1)
        models.append(model)
        return model  # type: ignore[return-value]

    def update(self: CardRuntime, slot: str, task: Task, x: PairBatch, y: Any,
               lr: float, meter: Resources) -> float:
        nonlocal current
        assert task == TASK and not events and self.optimizers[slot] is adams[SLOTS.index(slot)]
        assert slot == SLOTS[len(paired) % 2] and x.index == len(paired) // 2 and y is x.labels
        if slot == SLOTS[1]:
            assert x is paired[-1][1] and lr == paired[-1][2]
            if x.index == 2 and case == 'interrupt':
                raise KeyboardInterrupt
        paired.append((slot, x, lr))
        meter.reserve('forward_rows', x.groups)
        meter.complete('forward_rows', x.groups)
        with torch.no_grad():
            next(self.models[slot].parameters()).fill_(x.index // per_epoch + 1)
        if x.index == 2 and slot == SLOTS[0] and case == 'deadline':
            current += timedelta(seconds=60)
        return .6

    def predict(self: CardRuntime, slot: str, task: Task, x: PairBatch) -> Any:
        assert task == TASK
        epoch = int(next(self.models[slot].parameters()).item())
        if x.split == 'test':
            assert events[:3] == [*SLOTS, 'test-open'] and not self.optimizers
            assert epoch == (26 if slot == SLOTS[0] else 5) or case == 'ceiling'
            events.append(f'test:{slot}')
            return np.full(x.labels.shape, -.1 if slot == SLOTS[0] else .2)
        assert x.split == 'dev' and epoch == len(paired) // (2 * per_epoch)
        # Two different optima with exact ties; in B the later real minimum is
        # below the early anchor by less than min_delta and still must win.
        value = (1. if epoch == 1 else 1.00001 if epoch in (26, 27) else .5) if slot == SLOTS[0] \
            else (1. if epoch in (5, 6) else .5)
        if case == 'ceiling' and slot == SLOTS[1]:
            value = epoch / 50
        return (2. * x.labels - 1.) * value

    strict = CardRuntime.load_best
    path_open, zip_open = Path.open, ZipFile.open

    def load(self: CardRuntime, slot: str, raw: bytes, *args: Any) -> None:
        reads.append(slot)
        registered = json.loads(run.read_bytes())['artifacts'][f'model/{slot}.pt']
        assert registered['sha256'] == sha256(raw).hexdigest() and not self.optimizers
        if case == ('strict-b' if slot == SLOTS[0] else 'strict-c'):
            checkpoint = torch.load(io.BytesIO(raw), weights_only=True)
            raw = encode({**checkpoint['metadata'], 'slot': SLOTS[1 - SLOTS.index(slot)]},
                         checkpoint['state'])
        strict(self, slot, raw, *args)
        events.append(slot)

    def open_path(path: Path, *args: Any, **kw: Any) -> Any:
        if path.is_relative_to(tmp_path / 'inputs'):
            assert path.suffix != '.pt'
        if path == test_path:
            assert events == list(SLOTS)
            events.append('test-open')
        return path_open(path, *args, **kw)

    def member(z: ZipFile, item: Any, mode: Literal['r', 'w'] = 'r', **kw: Any) -> Any:
        assert mode != 'r' or getattr(item, 'filename', item) != 'logits.npy'
        return zip_open(z, item, mode, **kw)

    read_model, finalize, atomic = store.read_model, store.finalize, store._atomic

    def disk(ref: str, digest: str, limit: int) -> bytes:
        if case == 'disk':
            (root / 'models' / recipe.run_id / f'{SLOTS[0]}.pt').write_bytes(b'damaged')
        return read_model(ref, digest, limit)

    def terminal(value: dict[str, Any]) -> None:
        finalizations.append(value['status'])
        assert 'test_scores' not in json.loads(run.read_bytes())['summary']
        finalize(value)

    def write(path: Path, content: bytes) -> None:
        if ((case == 'ledger-fail' and path.name.startswith('batches-'))
                or (case == 'test-save' and path.name == 'test-SKINNY.npz')):
            raise OSError
        atomic(path, content)

    def observe(self: CardRuntime, slot: str) -> dict[str, float]:
        epoch = int(next(self.models[slot].parameters()).item())
        observed_routes.append((slot, epoch))
        return {'nibble': .25, 'word': .75}

    monkeypatch.setattr(TorchRuntime, '__init__', cpu)
    monkeypatch.setattr(torch.optim, 'Adam', adam)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(CardRuntime, '_build', build)
    monkeypatch.setattr(CardRuntime, 'prepare', lambda self, b: (b, b.labels))
    monkeypatch.setattr(CardRuntime, 'routes', observe)
    monkeypatch.setattr(CardRuntime, 'update', update)
    monkeypatch.setattr(CardRuntime, 'predict', predict)
    monkeypatch.setattr(CardRuntime, 'load_best', load)
    monkeypatch.setattr('cipher_lab_plugins.innovation1.skinny_card.runtime.geometry',
                        lambda m: {'fake': '1' * 64})
    monkeypatch.setattr('cipher_lab_plugins.innovation1.four_single.models.geometry',
                        lambda m: {'fake': '1' * 64})
    monkeypatch.setattr('cipher_lab_plugins.innovation1.four_single.models.FIXED_BUFFERS',
                        frozenset())
    monkeypatch.setattr('cipher_lab_plugins.innovation1.single_train.training.tensor_digest',
                        lambda x, y: x.digest)
    if case == 'after-test':
        def changed() -> dict[str, str]:
            return {'altered.py': 'c' * 64} if len(events) == 5 else SOURCE
        monkeypatch.setattr(store, 'current_source_identity', changed)
    monkeypatch.setattr(store, 'read_model', disk)
    monkeypatch.setattr(store, 'finalize', terminal)
    monkeypatch.setattr(store, '_atomic', write)
    with monkeypatch.context() as guard:
        guard.setattr(Path, 'open', open_path)
        guard.setattr(ZipFile, 'open', member)
        card_app.execute(recipe, store, res, launch={}, backend='artificial',
            assemble=lambda p, check: (pools := CardPools(p, check),
                                       CardRuntime(p, check, pools.pool_provenance, 'artificial')))
    saved = json.loads(run.read_bytes())
    assert store._sealed and store._frozen_sha == sha256(run.read_bytes()).hexdigest()
    ok = case in ('early', 'ceiling')
    assert saved['status'] == ('SUCCEEDED' if ok else 'STOPPED' if case in (
        'deadline', 'interrupt', 'after-test') else 'FAILED')
    auto_sealed = case in ('disk', 'ledger-fail', 'test-save')
    assert finalizations == ([] if auto_sealed else [saved['status']])
    assert len(adams) == 2 and len(models) <= 4
    counts, summary = saved['counts'], saved['summary']
    assert counts['completed_optimizer_updates'] == len(paired)
    assert counts['completed_real_blocks'] == counts['completed_null_words'] == 0
    if not ok:
        assert not {'test_panel', 'test_scores', 'paired_test', 'test_input'} & summary.keys()
        assert ('test-open' in events) == (case in ('test-save', 'after-test'))
        if case in ('deadline', 'interrupt'):
            assert summary['paired_completed'] == 2 and summary['pending'][-1][
                'updates_completed'] == [SLOTS[0]]
            assert counts['attempted_optimizer_updates'] == (6 if case == 'interrupt' else 5)
        if case == 'ledger-fail':
            assert summary['progress']['paired_completed'] == per_epoch
            assert len(summary['progress']['pending']) == per_epoch
        if case in ('strict-b', 'test-save', 'after-test'):
            from cipher_lab_plugins.innovation1.skinny_card.report import render

            # Only the post-registration stop has evidence of a TEST artifact.
            assert ('dataset/test-SKINNY.npz' in saved['artifacts']) == (case == 'after-test')
            report_saved = read_saved(root, recipe.run_id)
            with monkeypatch.context() as guard:
                guard.setattr(np, 'load', forbidden)
                guard.setattr(torch, 'load', forbidden)
                files = render(report_saved)
            report = files['report.md'].decode()
            flow = next(p for p in report.split('\n\n') if '同一原 TEST 密文及标签重新评分' in p)
            assert '未发布完整 TEST 比较。' in report
            assert '## TEST（不参与选模）' not in report and 'C−B 准确率差' not in report
            assert '<svg' not in files['index.html'].decode()
            assert flow.startswith('预定流程：')
            assert ('已登记 TEST 产物' in report) == (case == 'after-test')
            assert '未读取 TEST' not in report and '未评分 TEST' not in report
            assert json.loads(files['source.json']) == dict(
                schema=1, run_id=recipe.run_id, run_ref=f'data/{recipe.run_id}/run.json',
                run_sha256=sha256(run.read_bytes()).hexdigest())
        return
    epochs = 36 if case == 'early' else 50
    assert summary['training_termination'] == dict(completed_epochs=epochs,
        reason='EARLY_STOPPING' if case == 'early' else 'EPOCH_CEILING')
    assert summary['last_dev'] == dict(epoch=epochs, dev=summary['curves'][-1]['dev'])
    assert len(summary['curves']) == epochs and len(observed_routes) == 2 * (epochs + 1)
    assert summary['paired_completed'] == epochs * per_epoch
    assert counts['completed_forward_rows'] == counts['attempted_forward_rows'] == (
        2 * (epochs * (36 + 4) + 4))
    reference = batches['SKINNY']
    assert all(p.digest == reference[p.index] for _, p, _ in paired[:2 * len(reference)])
    for slot, selected in zip(SLOTS, (26, 5 if case == 'early' else 50), strict=True):
        assert summary['best'][slot]['epoch'] == selected
        assert summary['best'][slot] == summary['persisted_reloads'][slot]
        chosen = summary['curves'][selected-1]['dev'][slot]
        assert summary['best'][slot]['bce'] == chosen['bce']
        assert summary['best'][slot]['dev'] == chosen
        assert summary['reload_routes'][slot] == {'nibble': .25, 'word': .75}
        assert summary['updates'][slot] == dict(attempted=epochs * per_epoch,
                                                completed=epochs * per_epoch)
    assert reads == list(SLOTS) and summary['test_input']['digest'] == recipe.test_digest
    with np.load(run.parent / summary['test_panel']['ref'], allow_pickle=False) as panel:
        assert set(panel.files) == {'metadata', 'ciphertext', 'labels', *SLOTS}
        assert np.all(panel[SLOTS[0]] == -.1) and np.all(panel[SLOTS[1]] == .2)
    assert read_saved(root, recipe.run_id)['status'] == 'SUCCEEDED'
    assert cli._finish(store, root, recipe.run_id, resources=res) == 0
    report = (root / 'result' / recipe.run_id / 'report.md').read_text(encoding='utf-8')
    assert 'C−B' in report and '不是独立确认' in report


def test_admission_rechecks_deadline_after_full_source_scan(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.skinny_card.config import (
        DEADLINE_ENV,
        GPU_UUID,
        ROOT,
        RUN_ID,
        SLOTS,
        validate_root,
    )

    validate_root(PureWindowsPath(ROOT))
    args = ['run', '--config', str(CONFIG), '--root', ROOT, '--id', RUN_ID, '--gpu-uuid', GPU_UUID]
    monkeypatch.setattr(source, 'no_reparse', forbidden)
    real_claim = RunStore.claim
    monkeypatch.setattr(RunStore, 'claim', forbidden)
    for index, bad in ((4, str(tmp_path)), (6, 'other-id'), (8, GPU_UUID[:-1] + '9')):
        wrong = args.copy()
        wrong[index] = bad
        assert cli.main(wrong) == 2
    current = NOW
    monkeypatch.setattr(application, 'Clocks', lambda: Clocks(
        wall=lambda: 0., cpu=lambda: 0., utc=lambda: current))
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 0)
    monkeypatch.setattr(cli, 'validate_card_root', lambda _: None)
    for inherited in (None, '', 'invalid', '2026-11-25T10:01:00', NOW.isoformat(),
                      (NOW + timedelta(seconds=50401)).isoformat()):
        if inherited is None:
            monkeypatch.delenv(DEADLINE_ENV, raising=False)
        else:
            monkeypatch.setenv(DEADLINE_ENV, inherited)
        assert cli.main(args) == 3
        assert json.loads(capsys.readouterr().out)['reason'] == 'DEADLINE'
    monkeypatch.setenv(DEADLINE_ENV, DEADLINE)
    monkeypatch.setattr(source, 'no_reparse', lambda _: tmp_path / 'owner')

    def claim(*a: Any, **kw: Any) -> Any:
        assert kw['model_slots'] == SLOTS
        return real_claim(*a, **kw)

    def scan() -> dict[str, str]:
        nonlocal current
        current += timedelta(seconds=60)
        return SOURCE

    monkeypatch.setattr(source, 'source_identity', scan)
    monkeypatch.setattr(RunStore, 'claim', claim)
    monkeypatch.setattr(application, 'execute', forbidden)
    assert cli.main(args) == 3 and not (tmp_path / 'owner').exists()
    assert json.loads(capsys.readouterr().out)['reason'] == 'DEADLINE'


def test_one_cli_claim_and_exact_production_dispatch(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.skinny_card import application as card_app
    from cipher_lab_plugins.innovation1.skinny_card.config import (
        CARD,
        DEADLINE_ENV,
        GPU_UUID,
        ROOT,
        RUN_ID,
        SLOTS,
    )

    root = tmp_path / 'owner'
    seen: list[str] = []
    monkeypatch.setattr(cli, 'validate_card_root', lambda _: None)
    monkeypatch.setattr(application, 'validate_card_root', lambda _: None)
    monkeypatch.setattr(source, 'no_reparse', lambda p: root if str(p) == ROOT else p)
    monkeypatch.setattr(source, 'source_identity', lambda: dict(SOURCE))
    monkeypatch.setattr(application, 'Clocks', lambda: Clocks(
        wall=lambda: 0., cpu=lambda: 0., utc=lambda: NOW))
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 0)
    monkeypatch.setenv(DEADLINE_ENV, DEADLINE)

    def device(recipe: Any, uuid: str) -> None:
        assert recipe is CARD and uuid == GPU_UUID
        seen.append('device')

    def execute(recipe: Any, store: RunStore, res: Resources, *, launch: dict[str, str]) -> None:
        assert seen == ['device'] and recipe is CARD and set(store._model_slots) == set(SLOTS)
        assert res.meter._policy.deadline_utc == DEADLINE
        assert launch == dict(gpu_uuid=GPU_UUID, device='cuda:0',
                              gpu_memory_scope='process_allocator')
        seen.append('pair')
        # A synthetic assembly failure is sealed by the live owner's API.
        store.finalize(dict(status='FAILED', summary=dict(reason='INJECTED_SETUP_FAILURE')))

    monkeypatch.setattr(application, 'prepare_device', device)
    monkeypatch.setattr(card_app, 'execute', execute)
    args = ['run', '--config', str(CONFIG), '--root', ROOT, '--id', RUN_ID, '--gpu-uuid', GPU_UUID]
    assert cli.main(args) == 4 and seen == ['device', 'pair']
    saved: dict[str, Any] = read_saved(root, RUN_ID)
    assert saved['source'] == SOURCE and saved['recipe'] == CARD.to_dict()
    assert saved['policy']['request']['gpu_uuid'] == GPU_UUID
    assert saved['policy']['deadline_utc'] == DEADLINE
    assert saved['artifacts'] == {}


@pytest.mark.parametrize('case', ['record-hash', 'old-recipe', 'old-best', 'manifest', 'skinny',
    'pool-record', 'pool-train', 'pool-dev', 'wrong-test-sha', 'missing-test',
    'test-metadata', 'test-ciphertext'])
def test_each_archive_boundary(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.layout_probe.config import FilePin
    from cipher_lab_plugins.innovation1.skinny_card.config import SLOTS
    from cipher_lab_plugins.innovation1.skinny_card.data import CardPools

    recipe, files, _, test = _fixture(tmp_path, case)
    record_path = test.parent.parent / 'run.json'
    assert recipe.skinny_source is not None
    if case.startswith('pool-'):
        name = {'pool-record': 'run.json', 'pool-train': 'train-SKINNY.npz',
                'pool-dev': 'dev-SKINNY.npz'}[case]
        path = next(p for p in files if p.name == name and p.is_relative_to(
            recipe.skinny_source.root))
        raw = files[path]
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif case == 'missing-test':
        test.unlink()
    elif case == 'wrong-test-sha':
        raw = files[test]
        test.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif case in ('test-metadata', 'test-ciphertext'):
        with np.load(io.BytesIO(files[test]), allow_pickle=False) as fixture:
            arrays = {name: fixture[name] for name in ('ciphertext', 'labels')}
            meta = json.loads(fixture['metadata'].tobytes())
        if case == 'test-metadata':
            meta['best']['epoch'] += 1
        else:
            arrays['ciphertext'] = arrays['ciphertext'][::-1]
        # Make the old logits member deliberately undecodable, even as an NPY.
        buf = io.BytesIO()
        np.savez(buf, **arrays, metadata=np.frombuffer(_encode(meta), dtype=np.uint8))
        buf.seek(0)
        with ZipFile(buf, 'a') as z:
            z.writestr('logits.npy', b'not an NPY header')
        raw = buf.getvalue()
        test.write_bytes(raw)
        sha = sha256(raw).hexdigest()
        record = json.loads(files[record_path])
        record['artifacts']['dataset/test-SKINNY.npz'].update(bytes=len(raw), sha256=sha)
        record['summary']['test']['SKINNY']['panel'].update(bytes=len(raw), sha256=sha)
        manifest = _encode(record)
        record_path.write_bytes(manifest)
        recipe = replace(recipe, test_sha256=sha,
            archive=replace(recipe.archive, record=FilePin(len(manifest),
                                                           sha256(manifest).hexdigest())))
    pools = CardPools(recipe, lambda: None)
    seen: list[Path] = []
    path_open, zip_open = Path.open, ZipFile.open

    def guard(path: Path, *a: Any, **kw: Any) -> Any:
        assert path.name in ('run.json', 'train-SKINNY.npz', 'dev-SKINNY.npz', 'test-SKINNY.npz')
        seen.append(path)
        return path_open(path, *a, **kw)

    def member(z: ZipFile, item: Any, mode: Literal['r', 'w'] = 'r', **kw: Any) -> Any:
        assert getattr(item, 'filename', item) != 'logits.npy'
        return zip_open(z, item, mode, **kw)

    monkeypatch.setattr(Path, 'open', guard)
    monkeypatch.setattr(ZipFile, 'open', member)
    if case in ('wrong-test-sha', 'missing-test', 'test-metadata', 'test-ciphertext'):
        pools.prepare_pool()
        assert test not in seen
        pools.activate()
        with pytest.raises(replay.TestSourceError):
            pools.load_test({slot: dict(ref=f'model/{slot}.pt') for slot in SLOTS})
        assert (test in seen) == (case != 'missing-test')
    else:
        with pytest.raises(reuse.PoolSourceError):
            pools.prepare_pool()
        assert test not in seen


def test_checkpoint_exact_binding_and_fixed_buffers(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.four_single.models import geometry, load_state
    from cipher_lab_plugins.innovation1.skinny_card.config import SLOTS
    from cipher_lab_plugins.innovation1.skinny_card.data import CardPools
    from cipher_lab_plugins.innovation1.skinny_card.runtime import CardRuntime

    recipe, _, _, _ = _fixture(tmp_path)
    pools = CardPools(recipe, lambda: None)
    pools.prepare_pool()
    runtime = CardRuntime.__new__(CardRuntime)
    runtime.contract, runtime.backend, runtime._pools = recipe, 'artificial', pools.pool_provenance
    model = runtime._build(SLOTS[0])
    runtime.tables = {s: geometry(model) for s in SLOTS}
    runtime.models, runtime.optimizers, runtime.device = {}, {}, torch.device('cpu')
    expected = runtime.identity(recipe, SOURCE, 'artificial', SLOTS[0], 7, .6)
    assert expected['recipe'] == recipe.to_dict() and expected['pools'] == pools.pool_provenance()
    state = model.state_dict()
    assert set(expected) == {'schema', 'method', 'run_id', 'recipe', 'source', 'backend',
                             'slot', 'epoch', 'score', 'geometry', 'pools'}
    for key, different in dict(schema=True, method='old', run_id='other',
            recipe={**recipe.to_dict(), 'gpu_uuid': 'other'}, source={'bad/file.py': '0' * 64},
            backend='ciphers', slot=SLOTS[1], epoch=8, score=.601, geometry={}, pools={}).items():
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            load_state(model, encode({**expected, key: different}, state), expected)
    fixed = 'skinny_column'
    with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
        load_state(model, encode(expected, {**state, fixed: state[fixed] + 1}), expected)
    real_load = model.load_state_dict
    strict_calls: list[bool] = []

    def strict(value: Any, *, strict: bool = False) -> Any:
        strict_calls.append(strict)
        return real_load(value, strict=strict)

    monkeypatch.setattr(model, 'load_state_dict', strict)
    monkeypatch.setattr(runtime, '_build', lambda _: model)
    runtime.load_best(SLOTS[0], encode(expected, state), recipe, SOURCE, 7, .6, lambda: None)
    assert strict_calls == [True] and not runtime.optimizers and not model.training


def test_production_evidence_size_without_production_arrays() -> None:
    from cipher_lab_plugins.innovation1.skinny_card.config import CARD, ROW_BYTES, SLOTS, estimate

    n = estimate(CARD)
    source_map = source.source_identity()
    assert all(f'cipher_lab_plugins/innovation1/skinny_card/{module}.py' in source_map
               for module in ('__init__', 'config', 'data', 'runtime', 'application', 'report'))
    worst = -1.2345678901234567e-250
    row = dict(epoch=50, cycle=89999, groups=5000, lr=.00010000000000000002,
        ciphertext_labels_sha256='f'*64, input_sha256='f'*64,
        loss=dict.fromkeys(SLOTS, worst), updates_completed=list(SLOTS))
    assert len(_encode(row)) <= ROW_BYTES
    pending = [row] * 256
    chunks_per_epoch = (1800 + 255) // 256
    artifacts = {f'payload/batches-{i:04d}.json': dict(kind='payload', sha256='f'*64,
                    bytes=len(_encode(pending))) for i in range(50 * chunks_per_epoch)}
    dev = {slot: dict(bce=.5151515151515151, accuracy=.5151515151515151) for slot in SLOTS}
    routes = {slot: dict(local=.12345678901234567, nibble=.8765432109876543) for slot in SLOTS}
    curves = [dict(epoch=e, dev=dev, routes=routes, train_bce=dict.fromkeys(SLOTS, .52525252525))
              for e in range(1, 51)]
    best = {slot: dict(ref=f'model/{slot}.pt', sha256='f'*64, epoch=50, bce=dev[slot]['bce'],
                       dev=dev[slot], dev_digest='f'*64) for slot in SLOTS}
    progress = dict(last_dev=curves[-1], reload_routes=routes, best=best, persisted_reloads=best,
                    pending=pending, paired_completed=90000,
                    updates={slot: dict(attempted=90000, completed=90000) for slot in SLOTS})
    assert len(_encode(dict(progress=progress))) < 128 * 1024
    evidence = dict(source=source_map, recipe=CARD.to_dict(), artifacts=artifacts, summary=dict(
        pending=pending, curves=curves, progress=progress, batch_refs=list(artifacts)))
    assert len(_encode(evidence)) < cli._MAX_RUN_RECORD
    assert ((len(_encode(pending)) - 2) * 90000 // 256 + 2 * 50 * chunks_per_epoch
            <= n['batch_ledger_bound_bytes'])
