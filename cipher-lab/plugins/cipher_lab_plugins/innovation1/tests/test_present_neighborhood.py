"""PUBLIC layout and seeded CPU init; all paid work and archives are substitutes."""
from __future__ import annotations

import builtins
import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1 import application, cli, resources, source
from cipher_lab_plugins.innovation1.checkpoint import encode
from cipher_lab_plugins.innovation1.config import Task, load_config
from cipher_lab_plugins.innovation1.four_single import replay
from cipher_lab_plugins.innovation1.layout_probe.reuse import PoolSourceError, _pinned
from cipher_lab_plugins.innovation1.models import TorchRuntime
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.single_train import training
from cipher_lab_plugins.innovation1.single_train.config import learning_rate
from cipher_lab_plugins.innovation1.single_train.data import PairBatch, Provider
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import SOURCE, _archive
from plugin_isolation import forbidden
from torch import nn

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.execution import Clocks, Meter, Stopped
from cipher_lab.models.shared_m.router import PublicRouter
from cipher_lab.representations.neighborhoods import neighborhood_indices
from cipher_lab.representations.tasks import public_features

CONFIG = Path('configs/innovation1-present-neighborhood.json')
NOW = datetime(2026, 11, 26, 10, tzinfo=UTC)
DEADLINE = '2026-11-26T10:01:00Z'


def test_exact_stdlib_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                            capsys: pytest.CaptureFixture[str]) -> None:
    original = builtins.__import__

    def stdlib(name: str, *a: Any, **kw: Any) -> Any:
        assert name.split('.')[0] not in ('numpy', 'torch') and not name.endswith('scoring')
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', stdlib)
    monkeypatch.setattr(application, 'make_resources', forbidden)
    monkeypatch.setattr(RunStore, 'claim', forbidden)
    assert cli.main(['check', '--config', str(CONFIG)]) == 0
    from cipher_lab_plugins.innovation1.present_neighborhood.config import (
        NEIGHBORHOOD,
        SLOTS,
        NeighborhoodRecipe,
    )

    assert isinstance(load_config(CONFIG), NeighborhoodRecipe)
    check = json.loads(capsys.readouterr().out)
    r, n = check['recipe'], check['estimate']
    assert r == NEIGHBORHOOD.to_dict() and check['scope'] == 'STATIC_ONLY_NO_DEVICE_ADMISSION'
    assert r['tasks'] == [dict(cipher='PRESENT', rounds=7, difference=0xD00000)]
    assert r['training']['epochs'] == 10 and r['training']['batches_per_epoch'] == 1800
    assert r['training']['batch_groups'] == 5000 and r['resources']['wall_seconds'] == 10800
    assert r['selection']['early_stopping'] is False
    test = r['data']['test']
    assert (test['ref'], test['sha256'], test['digest']) == (
        'dataset/test-PRESENT.npz',
        '6e8ea523326fddebf61d997e91c75c72517fbc1edcbd63621553f85e93b05143',
        'c270e4a121e6945ba283c12f93f30cf0e663b968a62a609d7c9a270bae261ffd')
    assert r['data']['archive']['record'] == dict(bytes=77847,
        sha256='7a4a441ede8144b462cf15e5975240efd84eba1757a78f040c7eec3a3a665ddf')
    assert tuple(n[k] for k in ('train_groups_per_epoch', 'dev_groups', 'test_groups',
        'updates_each', 'optimizer_updates', 'forward_rows', 'real_blocks', 'null_words',
        'retries')) == (9000000, 1000000, 262144, 18000, 36000, 200524288, 0, 0, 0)
    assert SLOTS == ('A-default-best', 'B-structure-best')
    assert [learning_rate(NEIGHBORHOOD, c) for c in (0, 17999)] == [.001, .0001]
    bad = tmp_path / 'bad.json'
    for k, wrong in [('run_id', 'other'), ('namespace', 'other'), ('data', {}), ('geometry', {}),
                     ('training', {**r['training'], 'epochs': 10.}), ('extra', None)]:
        bad.write_text(json.dumps({**r, k: wrong}), encoding='ascii')
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            load_config(bad)


def test_public_layout_and_matched_cpu_init(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood import model
    from cipher_lab_plugins.innovation1.present_neighborhood.config import NEIGHBORHOOD, SLOTS
    from cipher_lab_plugins.innovation1.present_neighborhood.layout import Q, remap

    base = neighborhood_indices('nibble')
    q = tuple((i % 4)*16 + i//4 for i in range(64)) + (64,)
    assert Q == q and sorted(Q) == list(range(65))
    assert [Q[i] for i in (0, 1, 2, 3, 4, 17, 62, 63, 64)] == [0,16,32,48,1,20,47,63,64]
    expected = torch.full_like(base, -1)
    for i, row in enumerate(base.tolist()):
        expected[q[i]] = torch.tensor([q[j] for j in row])
    assert torch.equal(torch.tensor(remap(tuple(tuple(row) for row in base.tolist()))), expected)
    assert torch.equal(expected[:, 4], torch.arange(64))
    rng = torch.random.get_rng_state()
    a, b = (model.build(NEIGHBORHOOD, slot) for slot in SLOTS)
    assert torch.equal(rng, torch.random.get_rng_state())
    monkeypatch.setattr(PublicRouter, 'forward', forbidden)
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for name in pa:
        x, y = pa[name], pb[name]
        assert x.dtype == y.dtype == torch.float32 and x.device.type == y.device.type == 'cpu'
        assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
        assert x.requires_grad == y.requires_grad == (not name.startswith('shared.router.'))
    ba, bb = dict(a.named_buffers()), dict(b.named_buffers())
    assert ba.keys() == bb.keys()
    changed = 'shared.stems.nibble.indices'
    for name in ba:
        assert ba[name].data_ptr() != bb[name].data_ptr()
        if name != changed:
            assert torch.equal(ba[name], bb[name])
    assert torch.equal(ba[changed], base) and torch.equal(bb[changed], expected)
    card = torch.tensor(public_features('PRESENT'))
    # Intercept shared operations before any numerical forward. The unchanged
    # paired bits/readout order, real public card, and both fixed routes reach it.
    from cipher_lab_plugins.innovation1.four_single import models as pair_models

    assert [pair_models.geometry(arm)[changed] for arm in (a, b)] == [
        '8e83f4f7e30d817d7f161c67bf85a8320477bd537097d8f241db3add56d38122',
        'e3d8578406b90440c74f938c2cb110e12160b8a8342fae18ecb2a5dea0d4f4ae',
    ]
    assert [int((buf[changed] == 64).sum()) for buf in (ba, bb)] == [116, 116]
    bits = torch.zeros(4, 1, 2, 64)

    def fake_shared(shared: Any, slots: Any, public: Any, *, rounds: int,
                    nibble: Any, route_for: Any) -> Any:
        assert public is card and rounds == 7 and nibble is None
        assert slots.data_ptr() == bits.data_ptr() and slots.shape == (1, 4, 2, 64)
        route = route_for(public)
        assert route.indices.tolist() == [2, 1] and route.weights.tolist() == [.5, .5]
        return torch.zeros(4, 1)

    monkeypatch.setattr(pair_models, 'shared_logits', fake_shared)
    for arm in (a, b):
        assert arm.route_weights(card) == dict(nibble=.5, word=.5)
        assert arm(bits, card, rounds=7).shape == (4, 1)


def _fixture(tmp_path: Path) -> tuple[Any, list[str], Path, bytes]:
    from cipher_lab_plugins.innovation1.present_neighborhood.config import NEIGHBORHOOD

    archive, files, batches = _archive(tmp_path, 'early')
    base = Path(archive.archive.root) / 'data' / archive.archive.run_id
    target = base / 'dataset/test-PRESENT.npz'
    record = files[base / 'run.json']
    test = json.loads(record)['summary']['test']['PRESENT']['panel']
    recipe = replace(NEIGHBORHOOD, training=replace(archive.training, epochs=10),
                     archive=archive.archive, skinny_source=archive.skinny_source,
                     test_sha256=test['sha256'], test_digest=test['digest'])
    return recipe, batches['PRESENT'], target, record


@pytest.mark.parametrize('pin', ['record', 'test_sha256', 'test_digest'])
def test_pin_failure_does_not_open_panels(pin: str, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood.data import NeighborhoodPools

    r, _, test, _ = _fixture(tmp_path)
    if pin == 'record':
        r = replace(r, archive=replace(r.archive, record=replace(r.archive.record, sha256='0'*64)))
    else:
        r = replace(r, **{pin: '0'*64})
    monkeypatch.setattr(Provider, '__init__', forbidden)
    opened: list[Path] = []

    def pin_file(path: Path, *a: Any) -> Any:
        opened.append(path)
        assert path == test.parent.parent / 'run.json'
        return _pinned(path, *a)

    monkeypatch.setattr(replay, '_pinned', pin_file)
    with pytest.raises(PoolSourceError):
        NeighborhoodPools(r, lambda: None).prepare_pool()
    assert opened == [test.parent.parent / 'run.json']


@pytest.mark.parametrize('case', ['ok', 'disk-a', 'strict-b', 'table-b', 'bad-test', 'save-test',
                                  'late-stop', 'partial-update'])
def test_paired_loop_and_archive_barrier(case: str, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood import application as product
    from cipher_lab_plugins.innovation1.present_neighborhood.config import (
        NIBBLE_BUFFER,
        SLOTS,
        TABLE_SHA256,
        make_policy,
    )
    from cipher_lab_plugins.innovation1.present_neighborhood.data import NeighborhoodPools
    from cipher_lab_plugins.innovation1.present_neighborhood.runtime import NeighborhoodRuntime

    recipe, batches, test, record = _fixture(tmp_path / 'inputs')
    root = tmp_path / 'owner'
    store = RunStore.claim(root, recipe.run_id, output_bytes=2**30, needs_models=True,
                           model_slots=SLOTS, source_supplier=lambda: SOURCE)
    now = NOW
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: now)
    res = Resources(Meter(make_policy(recipe, clocks, DEADLINE), clocks, lambda: 0))
    if case == 'bad-test':
        raw_test = test.read_bytes()  # Artificial fixture only; preserve the registered size.
        test.write_bytes(raw_test[:-1] + bytes([raw_test[-1] ^ 1]))
    events: list[str] = []
    paid: list[tuple[str, PairBatch, float]] = []
    adams: list[object] = []
    loads: list[str] = []
    opened: list[Path] = []
    run = store._data / 'run.json'
    read_model, load_best = store.read_model, NeighborhoodRuntime.load_best
    open_path, open_zip, atomic = Path.open, ZipFile.open, store._atomic

    def cpu(self: Any, streams: Any, device: str, *, inference_only: bool) -> None:
        assert device == 'cuda:0' and inference_only
        self.recipe, self.device = streams, torch.device('cpu')
        self.models, self.optimizers = {}, {}

    def fake_adam(params: Any, **kw: Any) -> object:
        assert tuple(params) and kw['lr'] == .001
        adams.append(object())
        return adams[-1]

    class Tiny(nn.Module):
        def __init__(self, slot: str) -> None:
            super().__init__()
            self.epoch = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.register_buffer('table', torch.tensor([SLOTS.index(slot)]))

        def route_weights(self, public: Any) -> dict[str, float]:
            assert public.tolist() == list(public_features('PRESENT'))
            return dict(nibble=.5, word=.5)

    def tiny_geometry(m: Any) -> dict[str, str]:
        return {NIBBLE_BUFFER: TABLE_SHA256[int(m.table.item())]}

    def update(self: Any, slot: str, task: Task, x: PairBatch, y: Any, lr: float,
               meter: Resources) -> float:
        assert task == Task('PRESENT', 7, 0xD00000) and not events and y is x.labels
        assert slot == SLOTS[len(paid) % 2] and x.index == len(paid)//2
        assert x.digest == batches[x.index] and lr == learning_rate(recipe, x.index)
        assert self.optimizers[slot] is adams[SLOTS.index(slot)]
        if slot == SLOTS[1]:
            assert x is paid[-1][1] and lr == paid[-1][2]
            if case == 'partial-update' and x.index == 2:
                raise KeyboardInterrupt
        paid.append((slot, x, lr))
        meter.reserve('forward_rows', x.groups)
        meter.complete('forward_rows', x.groups)
        with torch.no_grad():
            self.models[slot].epoch.fill_(x.index // 9 + 1)
        return .6

    def predict(self: Any, slot: str, task: Task, x: PairBatch) -> Any:
        assert task == Task('PRESENT', 7, 0xD00000)
        epoch = int(self.models[slot].epoch.item())
        if x.split == 'test':
            assert events[:3] == [*SLOTS, 'open-test'] and not self.optimizers
            assert epoch == (3 if slot == SLOTS[0] else 7)
            return (2. * x.labels - 1.) * (-.1 if slot == SLOTS[0] else .2)
        assert x.split == 'dev' and epoch == len(paid)//18
        winners = (3, 4) if slot == SLOTS[0] else (7, 8)
        return (2. * x.labels - 1.) * (1. if epoch in winners else .5)

    def disk(ref: str, digest: str, limit: int) -> bytes:
        loads.append(ref)
        if case == 'disk-a':
            (root / 'models' / recipe.run_id / f'{SLOTS[0]}.pt').write_bytes(b'tampered fake')
        return read_model(ref, digest, limit)

    def strict(self: Any, slot: str, raw: bytes, *a: Any) -> None:
        registered = json.loads(run.read_bytes())['artifacts'][f'model/{slot}.pt']
        assert registered['sha256'] == sha256(raw).hexdigest() and not self.optimizers
        if slot == SLOTS[1] and case in ('strict-b', 'table-b'):
            payload = torch.load(io.BytesIO(raw), weights_only=True)
            if case == 'strict-b':
                payload['metadata']['slot'] = SLOTS[0]
            else:
                payload['state']['table'].zero_()
            raw = encode(payload['metadata'], payload['state'])
        load_best(self, slot, raw, *a)
        events.append(slot)

    def guard_path(path: Path, *a: Any, **kw: Any) -> Any:
        if path.is_relative_to(tmp_path / 'inputs'):
            assert path.name in ('run.json', 'train-PRESENT.npz', 'dev-PRESENT.npz', test.name)
            opened.append(path)
            if path == test:
                assert events == list(SLOTS)
                events.append('open-test')
        return open_path(path, *a, **kw)

    def guard_member(z: ZipFile, item: Any, mode: Literal['r', 'w'] = 'r', **kw: Any) -> Any:
        assert mode != 'r' or getattr(item, 'filename', item) != 'logits.npy'
        return open_zip(z, item, mode, **kw)

    def write(path: Path, content: bytes) -> None:
        nonlocal now
        if path == store._data / 'dataset/test-PRESENT.npz' and case == 'save-test':
            raise OSError
        atomic(path, content)
        if path == store._data / 'dataset/test-PRESENT.npz' and case == 'late-stop':
            now += timedelta(seconds=60)

    from cipher_lab_plugins.innovation1.present_neighborhood import runtime
    monkeypatch.setattr(Provider, '__init__', forbidden)
    monkeypatch.setattr(TorchRuntime, '__init__', cpu)
    monkeypatch.setattr(torch.optim, 'Adam', fake_adam)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(NeighborhoodRuntime, '_build', lambda self, slot: Tiny(slot))
    monkeypatch.setattr(NeighborhoodRuntime, 'prepare', lambda self, batch: (batch, batch.labels))
    monkeypatch.setattr(NeighborhoodRuntime, 'update', update)
    monkeypatch.setattr(NeighborhoodRuntime, 'predict', predict)
    monkeypatch.setattr(NeighborhoodRuntime, 'load_best', strict)
    monkeypatch.setattr(runtime, 'geometry', tiny_geometry)
    from cipher_lab_plugins.innovation1.four_single import models as pair_models
    monkeypatch.setattr(pair_models, 'geometry', tiny_geometry)
    monkeypatch.setattr(pair_models, 'FIXED_BUFFERS', frozenset({'table'}))
    monkeypatch.setattr(training, 'tensor_digest', lambda x, y: x.digest)
    monkeypatch.setattr(store, 'read_model', disk)
    monkeypatch.setattr(store, '_atomic', write)
    with monkeypatch.context() as m:
        m.setattr(Path, 'open', guard_path)
        m.setattr(ZipFile, 'open', guard_member)
        product.execute(recipe, store, res, launch={}, backend='artificial',
                        assemble=lambda r, check: (p := NeighborhoodPools(r, check),
                            NeighborhoodRuntime(r, check, p.pool_provenance, 'artificial')))
    raw = run.read_bytes()
    saved = json.loads(raw)
    assert store._sealed and store._frozen_sha == sha256(raw).hexdigest()
    assert (test.parent.parent / 'run.json').read_bytes() == record
    assert [p.name for p in opened[:3]] == ['run.json', 'train-PRESENT.npz', 'dev-PRESENT.npz']
    assert len(adams) == 2 and adams[0] is not adams[1]
    summary, counts = saved['summary'], saved['counts']
    assert counts['completed_optimizer_updates'] == len(paid)
    assert counts['attempted_real_blocks'] == counts['attempted_null_words'] == 0
    if case != 'ok':
        expected = 'STOPPED' if case in ('late-stop', 'partial-update') else 'FAILED'
        assert saved['status'] == expected
        assert ('open-test' in events) == (case in ('bad-test', 'save-test', 'late-stop'))
        assert 'test_scores' not in summary and 'paired_test' not in summary
        if case == 'late-stop':
            assert 'dataset/test-PRESENT.npz' in saved['artifacts']
        if case == 'partial-update':
            assert summary['paired_completed'] == 2 and counts['attempted_optimizer_updates'] == 6
        if case == 'disk-a':
            assert cli._finish(store, root, recipe.run_id, resources=res) == 6
            assert not (root / 'result' / recipe.run_id / 'report.md').exists()
            return
        from cipher_lab_plugins.innovation1.present_neighborhood.report import render
        with monkeypatch.context() as guard:
            guard.setattr(np, 'load', forbidden)
            guard.setattr(torch, 'load', forbidden)
            files = render(read_saved(root, recipe.run_id))
        assert 'B−A' not in files['report.md'].decode() and b'<svg' not in files['index.html']
        return
    assert saved['status'] == 'SUCCEEDED' and len(paid) == 180
    assert counts['completed_forward_rows'] == counts['attempted_forward_rows'] == 808
    assert summary['training_termination'] == dict(reason='EPOCH_CEILING', completed_epochs=10)
    assert summary['paired_completed'] == 90 and len(summary['curves']) == 10
    assert loads == [f'model/{slot}.pt' for slot in SLOTS]
    for slot, epoch in zip(SLOTS, (3, 7), strict=True):
        assert summary['updates'][slot] == dict(attempted=90, completed=90)
        row = summary['best'][slot]
        assert row == summary['persisted_reloads'][slot] and row['epoch'] == epoch
        assert row['dev'] == summary['curves'][epoch-1]['dev'][slot]
        with (root / 'models' / recipe.run_id / f'{slot}.pt').open('rb') as f:
            meta = torch.load(f, weights_only=True)['metadata']
        assert meta['pools'] == {k: summary[k] for k in ('train_panel', 'dev_panel')}
        assert meta['slot'] == slot and meta['source'] == SOURCE
        assert meta['recipe'] == saved['recipe']
        assert meta['epoch'] == epoch and meta['score'] == row['bce']
    assert summary['test_input']['digest'] == recipe.test_digest
    assert summary['test_panel']['ref'] == 'dataset/test-PRESENT.npz'
    assert summary['paired_test'] == dict(rows=4, direction='B-minus-A', delta_acc_pp=100.,
        delta_bce=summary['test_scores'][SLOTS[1]]['bce'] - summary['test_scores'][SLOTS[0]]['bce'])
    with np.load(store._data / summary['test_panel']['ref'], allow_pickle=False) as panel:
        assert set(panel.files) == {'metadata', 'ciphertext', 'labels', *SLOTS}
        assert all(panel[k].dtype != object for k in panel.files)
        assert np.array_equal(panel[SLOTS[0]], (2.*panel['labels'] - 1.) * -.1)
        assert np.array_equal(panel[SLOTS[1]], (2.*panel['labels'] - 1.) * .2)
        meta = json.loads(panel['metadata'].tobytes())
        assert meta['best'] == summary['persisted_reloads'] and meta['reused_input'] == (
            summary['test_input'])
    assert cli._finish(store, root, recipe.run_id, resources=res) == 0
    report = (root / 'result' / recipe.run_id / 'report.md').read_text(encoding='utf-8')
    assert 'B−A' in report and '不是独立确认' in report


@pytest.mark.parametrize('case', ['invalid', 'claim-deadline', 'device-deadline',
                                  'owner', 'device', 'rss', 'dispatch'])
def test_admission_failures_precede_paid_work(case: str, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood.config import (
        DEADLINE_ENV,
        GPU_UUID,
        NEIGHBORHOOD,
        ROOT,
        RUN_ID,
        SLOTS,
        validate_root,
    )

    validate_root(PureWindowsPath(ROOT))
    args = ['run', '--config', str(CONFIG), '--root', ROOT, '--id', RUN_ID, '--gpu-uuid', GPU_UUID]
    monkeypatch.setattr(Provider, '__init__', forbidden)
    if case == 'invalid':
        monkeypatch.setattr(RunStore, 'claim', forbidden)
        for option, value in (('--root', ROOT + r'\..\other'), ('--id', RUN_ID.upper()),
            ('--id', 'different'), ('--gpu-uuid', 'GPU-00000000-0000-0000-0000-000000000000')):
            bad = args.copy()
            bad[args.index(option)+1] = value
            assert cli.main(bad) == 2
        return
    root = tmp_path / 'fresh'
    now = NOW
    ticks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: now)
    hits: list[str] = []

    def rss() -> int:
        return NEIGHBORHOOD.resources.rss_bytes + 1 if case == 'rss' else 0
    def validate_owner(p: Path) -> None:
        assert p == root

    monkeypatch.setattr(cli, 'validate_neighborhood_root',
                        lambda p: validate_root(PureWindowsPath(p)))
    monkeypatch.setattr(application, 'validate_neighborhood_root', validate_owner)
    monkeypatch.setenv(DEADLINE_ENV, DEADLINE)
    monkeypatch.setattr(application, 'Clocks', lambda: ticks)
    monkeypatch.setattr(resources, 'production_rss_bytes', rss)
    monkeypatch.setattr(source, 'no_reparse', lambda p: root if str(p) == ROOT else p)

    def scan() -> dict[str, str]:
        nonlocal now
        hits.append('scan')
        if case == 'claim-deadline':
            now += timedelta(seconds=60)
        return SOURCE

    monkeypatch.setattr(source, 'source_identity', scan)
    claim = RunStore.claim

    def claim_owner(*a: Any, **kw: Any) -> RunStore:
        nonlocal now
        assert kw['model_slots'] == SLOTS
        owner = claim(*a, **kw)
        if case == 'device-deadline':
            now += timedelta(seconds=60)
        return owner

    monkeypatch.setattr(RunStore, 'claim', claim_owner)
    if case == 'owner':
        (root / 'data' / RUN_ID).mkdir(parents=True)
        (root / 'data' / RUN_ID / 'marker').write_text('prior')
    if case in ('device', 'dispatch'):
        from cipher_lab_plugins.innovation1.resources import ResourceStop

        def device(recipe: Any, uuid: str) -> None:
            assert recipe is NEIGHBORHOOD and uuid == GPU_UUID
            hits.append('device')
            if case == 'device':
                raise ResourceStop('DEVICE_ENTRY_REJECTED')

        monkeypatch.setattr(application, 'prepare_device', device)
        if case == 'dispatch':
            from cipher_lab_plugins.innovation1.present_neighborhood import application as pair

            def dispatch(recipe: Any, store: RunStore, resources: Resources, *,
                         launch: dict[str, str]) -> None:
                assert hits == ['scan', 'device'] and recipe is NEIGHBORHOOD
                assert launch['gpu_uuid'] == GPU_UUID and store._run_id == RUN_ID
                assert resources.meter._policy.deadline_utc == DEADLINE
                assert resources.snapshot()['attempted_optimizer_updates'] == 0
                hits.append('dispatch')
                store.finalize(dict(status='STOPPED', summary=dict(reason='GUARD'),
                                    counts=resources.snapshot()))

            monkeypatch.setattr(pair, 'execute', dispatch)
    else:
        monkeypatch.setattr(application, 'prepare_device', forbidden)
    assert cli.main(args) == (2 if case == 'owner' else 3)
    assert hits == {'device': ['scan', 'device'], 'rss': [],
                    'dispatch': ['scan', 'device', 'dispatch']}.get(case, ['scan'])
    if case in ('claim-deadline', 'rss'):
        assert not root.exists()
    elif case == 'owner':
        assert (root / 'data' / RUN_ID / 'marker').read_text() == 'prior'
        assert not (root / 'models').exists()
    else:
        saved = json.loads((root / 'data' / RUN_ID / 'run.json').read_bytes())
        assert saved['status'] == 'STOPPED' and not saved['artifacts']
        assert saved['policy']['deadline_utc'] == DEADLINE
        assert saved['counts']['attempted_optimizer_updates'] == 0


@pytest.mark.parametrize('deadline', [None, '', 'bad', '2026-11-26T10:00:00Z',
                                     '2026-11-26T13:00:01Z', '2026-11-26T11:00:00+01:00'])
def test_invalid_deadline_is_not_rebased(deadline: str | None) -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood.config import NEIGHBORHOOD, make_policy

    with pytest.raises(Stopped, match='DEADLINE'):
        make_policy(NEIGHBORHOOD, Clocks(utc=lambda: NOW), deadline)
