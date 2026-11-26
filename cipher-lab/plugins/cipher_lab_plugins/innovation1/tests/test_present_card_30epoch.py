"""Single PRESENT owner: public geometry, seeded CPU init and artificial lifecycle."""
from __future__ import annotations

import builtins
import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from struct import pack
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1 import cli
from cipher_lab_plugins.innovation1.checkpoint import encode
from cipher_lab_plugins.innovation1.config import load_config
from cipher_lab_plugins.innovation1.models import TorchRuntime
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.single_train import training
from cipher_lab_plugins.innovation1.single_train.config import learning_rate
from cipher_lab_plugins.innovation1.single_train.data import PairBatch, Provider
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import SOURCE
from cipher_lab_plugins.innovation1.tests.test_present_neighborhood import _fixture
from plugin_isolation import forbidden
from torch import nn

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Clocks, Meter
from cipher_lab.representations.neighborhoods import neighborhood_indices

CONFIG = Path('configs/innovation1-present-card-30epoch.json')
SLOT = 'M-structure-best'
SHA = 'e3d8578406b90440c74f938c2cb110e12160b8a8342fae18ecb2a5dea0d4f4ae'


def test_explicit_card_generates_all_entries() -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood.layout import (
        NIBBLE_RULE,
        compile_card,
        remap,
    )

    from cipher_lab.present_public import PRESENT_P_LAYER

    layer, rule = PRESENT_P_LAYER, NIBBLE_RULE
    assert (layer.state_bits, layer.block_count, layer.block_bits, layer.bit_order) == (64, 16, 4,
                                                                                      'LSB0')
    assert layer.source_to_destination == tuple((16 * i) % 63 if i < 63 else 63 for i in range(64))
    assert (rule.input_order, rule.block_offsets, rule.bit_offsets, rule.sentinel,
            rule.padding, rule.center_slot, rule.slot_order) == (
                'MSB0', (-1, 0, 1), (-1, 0, 1), 64, 'zero', 4, 'block-then-bit')
    default = tuple(tuple(row) for row in neighborhood_indices('nibble').tolist())
    table = compile_card(layer, rule)
    independent = {(i % 4) * 16 + i // 4: tuple(
        64 if j == 64 else (j % 4) * 16 + j // 4 for j in row) for i, row in enumerate(default)}
    assert table == remap(default) == tuple(independent[i] for i in range(64))
    assert sum(row.count(64) for row in table) == 116
    assert sha256(pack('<576q', *(j for row in table for j in row))).hexdigest() == SHA
    # A different bijection or slot ordering changes the output of the compiler;
    # official assembly separately pins both its inputs and the complete table.
    assert compile_card(replace(layer, source_to_destination=tuple(range(64))), rule) == default
    assert compile_card(layer, replace(rule, bit_offsets=(1, 0, -1))) != table
    for bad in (layer.source_to_destination[:-1], (0,) * 64, (True, *range(1, 64)),
                (64, *range(1, 64))):
        with pytest.raises(ValueError, match='PRESENT_COORDINATES'):
            compile_card(replace(layer, source_to_destination=bad), rule)
    with pytest.raises(ValueError, match='PRESENT_COORDINATES'):
        compile_card(replace(layer, bit_order='MSB0'), rule)
    for wrong in (dict(input_order='LSB0'), dict(sentinel=63), dict(padding='wrap'),
                  dict(center_slot=3), dict(slot_order='bit-then-block'),
                  dict(block_offsets=(-1, -1, 1))):
        with pytest.raises(ValueError, match='PRESENT_GATHER_RULE'):
            compile_card(layer, replace(rule, **wrong))


def test_exact_singleton_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood.config import (
        PRESENT_CARD,
        NeighborhoodRecipe,
        estimate,
        official,
    )

    original = builtins.__import__

    def guarded(name: str, *a: Any, **kw: Any) -> Any:
        assert name.split('.')[0] not in ('torch', 'numpy', 'psutil', 'pynvml')
        assert not name.startswith('cipher_lab.ciphers')
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(RunStore, 'claim', forbidden)
    assert cli.main(['check', '--config', str(CONFIG)]) == 0
    check = json.loads(capsys.readouterr().out)
    r, n = check['recipe'], check['estimate']
    assert load_config(CONFIG) is PRESENT_CARD and r == PRESENT_CARD.to_dict()
    assert check['scope'] == 'STATIC_ONLY_NO_DEVICE_ADMISSION'
    assert r['slots'] == [SLOT] and r['geometry']['sha256'] == {SLOT: SHA}
    assert r['training']['epochs'] == r['optimizer']['cosine_epochs'] == 30
    assert r['resources']['wall_seconds'] == 14400 and r['selection']['early_stopping'] is False
    assert tuple(n[k] for k in ('fits', 'cycles', 'optimizer_updates', 'train_groups',
                               'forward_rows', 'real_blocks', 'null_words', 'retries')) == (
                                   1, 54000, 54000, 270000000, 300262144, 0, 0, 0)
    assert [learning_rate(PRESENT_CARD, c) for c in (0, 53999)] == [.001, .0001]
    assert learning_rate(PRESENT_CARD, 17999) > .0001
    assert r['tasks'] == [dict(cipher='PRESENT', rounds=7, difference=0xD00000)]
    old = NeighborhoodRecipe().to_dict()
    assert r['data'] == old['data'] and r['seeds'] == old['seeds']
    path = tmp_path / 'mutant.json'
    mutations: tuple[dict[str, Any], ...] = (
        dict(slots=old['slots']), dict(method=old['method']), dict(data={}),
        dict(trail=1), dict(structure_card={}), dict(training={**r['training'], 'epochs': 30.0}))
    for change in mutations:
        path.write_text(json.dumps({**r, **change}), encoding='ascii')
        with pytest.raises(ValueError, match='CONFIG_EXACT'):
            load_config(path)
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        official(replace(PRESENT_CARD, training=NeighborhoodRecipe.training), PRESENT_CARD.run_id)
    with pytest.raises(ValueError, match='NEIGHBORHOOD_TABLE'):
        estimate(replace(PRESENT_CARD, gather_rule=replace(PRESENT_CARD.gather_rule,
                                                          bit_offsets=(1, 0, -1))))
    for option, bad in (('--id', old['run_id']), ('--root', old['output_root']),
                        ('--gpu-uuid', 'GPU-00000000-0000-0000-0000-000000000000')):
        args = ['run', '--config', str(CONFIG), '--id', PRESENT_CARD.run_id,
                '--root', r['output_root'], '--gpu-uuid', r['gpu_uuid']]
        args[args.index(option) + 1] = bad
        assert cli.main(args) == 2


def test_single_seeded_cpu_model_gets_one_fresh_adam(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.present_neighborhood import runtime
    from cipher_lab_plugins.innovation1.present_neighborhood.config import PRESENT_CARD

    calls: list[Any] = []
    monkeypatch.setattr(TorchRuntime, '__init__', _cpu)

    def adam(params: Any, **kw: Any) -> object:
        calls.append((tuple(params), kw))
        return object()

    monkeypatch.setattr(torch.optim, 'Adam', adam)
    rng = torch.random.get_rng_state()
    rt = runtime.NeighborhoodRuntime(PRESENT_CARD, lambda: None, lambda: {})
    assert list(rt.models) == list(rt.optimizers) == [SLOT] and len(calls) == 1
    params, options = calls[0]
    assert options == dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
    model = rt.models[SLOT]
    for (name, param), same in zip(model.named_parameters(), params, strict=True):
        assert param is same and param.device.type == 'cpu' and param.dtype == torch.float32
        assert param.requires_grad == (not name.startswith('shared.router.'))
    assert rt.routes(SLOT) == dict(nibble=.5, word=.5)
    assert rt.tables[SLOT]['shared.stems.nibble.indices'] == SHA
    assert torch.equal(rng, torch.random.get_rng_state())
    with pytest.raises(ValueError, match='MODEL_SLOT'):
        rt._build('B-structure-best')


def _cpu(self: Any, streams: Any, device: str, *, inference_only: bool) -> None:
    assert device == 'cuda:0' and inference_only
    self.recipe, self.device = streams, torch.device('cpu')
    self.models, self.optimizers = {}, {}


class OrderedShuffle:
    """Explicit index-only fake, never invokes a seeded or unseeded real RNG."""

    def permutation(self, values: Any) -> Any:
        return values[::-1].copy()

    def shuffle(self, values: Any) -> None:
        values[:] = values[::-1].copy()


@pytest.mark.parametrize('case', ['ok', 'disk', 'strict', 'bad-test', 'update', 'pool', 'storage'])
def test_singleton_owner_and_replay(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.four_single import models as pair_models
    from cipher_lab_plugins.innovation1.present_neighborhood import application, runtime
    from cipher_lab_plugins.innovation1.present_neighborhood.config import (
        NIBBLE_BUFFER,
        PRESENT_CARD,
        TASK,
        make_policy,
        validate_root,
    )
    from cipher_lab_plugins.innovation1.present_neighborhood.data import NeighborhoodPools

    monkeypatch.setattr(np.random, 'default_rng', lambda seed: OrderedShuffle())
    old, original_order, old_test, old_record = _fixture(tmp_path / 'archive-fixture')
    recipe = replace(PRESENT_CARD, training=replace(old.training, epochs=30), archive=old.archive,
                     skinny_source=old.skinny_source, test_sha256=old.test_sha256,
                     test_digest=old.test_digest)
    if case == 'pool':
        recipe = replace(recipe, test_sha256='0' * 64)
    root = tmp_path / 'owner'
    launch = dict(gpu_uuid=recipe.to_dict()['gpu_uuid'], device='cuda:0',
                  gpu_memory_scope='process_allocator')
    validate_root(PureWindowsPath(recipe.to_dict()['output_root']), recipe=recipe)
    store = RunStore.claim(root, recipe.run_id, output_bytes=2**30, needs_models=True,
                           model_slots=(SLOT,), source_supplier=lambda: SOURCE)
    deadline = '2026-11-26T14:00:00Z'
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0.,
                    utc=lambda: datetime(2026, 11, 26, 10, tzinfo=UTC))
    res = Resources(Meter(make_policy(recipe, clocks, deadline), clocks, lambda: 0))
    record_path = store._data / 'run.json'
    if case == 'bad-test':
        raw = old_test.read_bytes()
        old_test.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    events: list[str] = []
    opened: list[str] = []
    updates: list[int] = []
    devs: list[int] = []
    read_current, strict_current = store.read_model, runtime.NeighborhoodRuntime.load_best
    open_path, open_member, atomic = Path.open, ZipFile.open, store._atomic

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.epoch = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.register_buffer('table', torch.tensor([1]))

        def route_weights(self, public: Any) -> dict[str, float]:
            return dict(nibble=.5, word=.5)

    def geometry(m: Any) -> dict[str, str]:
        assert m.table.item() == 1
        return {NIBBLE_BUFFER: SHA}

    def adam(params: Any, **kw: Any) -> object:
        assert len(tuple(params)) == 1
        events.append('adam')
        return object()

    def update(self: Any, slot: str, task: Any, x: PairBatch, y: Any, lr: float,
               meter: Resources) -> float:
        cycle = len(updates)
        assert slot == SLOT and task == TASK and y is x.labels
        assert x.index == cycle and lr == learning_rate(recipe, cycle)
        if cycle < len(original_order):
            assert x.digest == original_order[cycle]
        assert list(self.optimizers) == [SLOT] and events == ['adam']
        if case == 'update' and cycle == 2:
            raise RuntimeError('artificial update failure')
        updates.append(cycle)
        meter.reserve('forward_rows', x.groups)
        meter.complete('forward_rows', x.groups)
        with torch.no_grad():
            self.models[slot].epoch.fill_(cycle // recipe.training.batches_per_epoch + 1)
        return .7

    def predict(self: Any, slot: str, task: Any, x: PairBatch) -> Any:
        assert task == TASK and slot == SLOT
        epoch = int(self.models[slot].epoch.item())
        if x.split == 'test':
            assert events == ['adam', 'disk-verified', 'strict', 'test-open']
            assert not self.optimizers and epoch == 17
            events.append('test-score')
            margin = .2
        else:
            assert x.split == 'dev' and epoch == len(devs) + 1
            devs.append(epoch)
            margin = 1. if epoch in (17, 18) else .5
        return (2. * x.labels - 1.) * margin

    def disk(ref: str, digest: str, limit: int) -> bytes:
        assert ref == f'model/{SLOT}.pt'
        if case == 'disk':
            (root / 'models' / recipe.run_id / f'{SLOT}.pt').write_bytes(b'modified artificial')
        raw = read_current(ref, digest, limit)
        events.append('disk-verified')
        return raw

    def strict(self: Any, slot: str, raw: bytes, *args: Any) -> None:
        assert events == ['adam', 'disk-verified'] and not self.optimizers and not self.models
        payload = torch.load(io.BytesIO(raw), weights_only=True)
        assert payload['metadata']['recipe'] == recipe.to_dict()
        assert payload['metadata']['geometry'] == {NIBBLE_BUFFER: SHA}
        if case == 'strict':
            # Valid container and identity, bad *fixed* buffer: strict load fails.
            payload['state']['table'].zero_()
            raw = encode(payload['metadata'], payload['state'])
        strict_current(self, slot, raw, *args)
        events.append('strict')

    def guard_path(path: Path, *a: Any, **kw: Any) -> Any:
        if path.is_relative_to(tmp_path / 'archive-fixture'):
            assert path.name in ('run.json', 'train-PRESENT.npz', 'dev-PRESENT.npz', old_test.name)
            opened.append(path.name)
            if path == old_test:
                assert events == ['adam', 'disk-verified', 'strict']
                events.append('test-open')
        return open_path(path, *a, **kw)

    def guard_zip(z: ZipFile, item: Any, mode: Literal['r', 'w'] = 'r', **kw: Any) -> Any:
        assert mode != 'r' or getattr(item, 'filename', item) != 'logits.npy'
        return open_member(z, item, mode, **kw)

    def write(path: Path, content: bytes) -> None:
        if case == 'storage' and path == store._data / 'payload/batches-0000.json':
            raise OSError('artificial payload failure')
        atomic(path, content)

    def build(self: Any, slot: str) -> Tiny:
        assert slot == SLOT
        return Tiny()

    def assemble(rec: Any, check: Any) -> Any:
        p = NeighborhoodPools(rec, check)
        with pytest.raises(ValueError, match='TEST_ORDER'):
            p.load_test({})
        return p, runtime.NeighborhoodRuntime(rec, check, p.pool_provenance, 'artificial')

    monkeypatch.setattr(Provider, '__init__', forbidden)
    monkeypatch.setattr(TorchRuntime, '__init__', _cpu)
    monkeypatch.setattr(torch.optim, 'Adam', adam)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(runtime.NeighborhoodRuntime, '_build', build)
    monkeypatch.setattr(runtime.NeighborhoodRuntime, 'prepare', lambda self, x: (x, x.labels))
    monkeypatch.setattr(runtime.NeighborhoodRuntime, 'update', update)
    monkeypatch.setattr(runtime.NeighborhoodRuntime, 'predict', predict)
    monkeypatch.setattr(runtime.NeighborhoodRuntime, 'load_best', strict)
    monkeypatch.setattr(runtime, 'geometry', geometry)
    monkeypatch.setattr(pair_models, 'geometry', geometry)
    monkeypatch.setattr(pair_models, 'FIXED_BUFFERS', frozenset({'table'}))
    monkeypatch.setattr(training, 'tensor_digest', lambda x, y: x.digest)
    monkeypatch.setattr(store, 'read_model', disk)
    monkeypatch.setattr(store, '_atomic', write)
    with monkeypatch.context() as guard:
        guard.setattr(Path, 'open', guard_path)
        guard.setattr(ZipFile, 'open', guard_zip)
        application.execute(recipe, store, res, launch=launch, backend='artificial',
                            assemble=assemble)
    raw = record_path.read_bytes()
    saved = json.loads(raw)
    assert store._frozen_sha == sha256(raw).hexdigest() and store._sealed
    assert old_record == (old_test.parent.parent / 'run.json').read_bytes()
    s, c = saved['summary'], saved['counts']
    assert not any(key.startswith('paired_') for key in s)
    assert c['attempted_real_blocks'] == c['attempted_null_words'] == 0
    assert c['completed_optimizer_updates'] == len(updates)
    assert opened == (['run.json'] if case == 'pool' else
        ['run.json', 'train-PRESENT.npz', 'dev-PRESENT.npz'] +
        ([old_test.name] if case in ('ok', 'bad-test') else []))
    if case == 'ok':
        assert saved['status'] == 'SUCCEEDED' and s['completed_batches'] == len(updates) == 270
        assert len(devs) == len(s['curves']) == 30
        assert s['last_dev'] == dict(epoch=30, dev=s['curves'][-1]['dev'])
        assert s['training_termination'] == dict(reason='EPOCH_CEILING', completed_epochs=30)
        assert s['updates'] == {SLOT: dict(attempted=270, completed=270)}
        assert c['attempted_forward_rows'] == c['completed_forward_rows'] == 1204
        assert s['best'] == s['persisted_reloads'] and s['best'][SLOT]['epoch'] == 17
        assert s['reload_routes'] == {SLOT: dict(nibble=.5, word=.5)}
        assert set(s['test_scores']) == {SLOT} and s['test_input']['digest'] == recipe.test_digest
        assert len(saved['artifacts']) == 32 and s['batch_refs'] == [
            f'payload/batches-{i:04d}.json' for i in range(30)]
        for i, ref in enumerate(s['batch_refs']):
            ledger = json.loads((store._data / ref).read_bytes())
            assert [r['cycle'] for r in ledger] == list(range(i*9, i*9 + 9))
            assert all(r['epoch'] == i + 1 and r['updates_completed'] == [SLOT] for r in ledger)
        assert s['test_panel']['ref'] == 'dataset/test-PRESENT.npz'
        with np.load(store._data / s['test_panel']['ref'], allow_pickle=False) as panel:
            assert set(panel.files) == {'metadata', 'ciphertext', 'labels', SLOT}
            assert np.array_equal(panel[SLOT], (2. * panel['labels'] - 1.) * .2)
            meta = json.loads(panel['metadata'].tobytes())
            assert meta['best'] == s['persisted_reloads']
            assert meta['reused_input'] == s['test_input']
    else:
        assert saved['status'] == 'FAILED'
        assert set(s).isdisjoint({'test_scores', 'test_panel', 'test_input'})
        assert 'dataset/test-PRESENT.npz' not in saved['artifacts']
        if case == 'update':
            assert s['completed_batches'] == 2
            assert s['updates'][SLOT] == dict(attempted=3, completed=2)
        if case == 'storage':
            assert s['progress']['completed_batches'] == 9
        if case in ('disk', 'strict'):
            assert not s.get('persisted_reloads', s.get('progress', {}).get('persisted_reloads'))
    capsys.readouterr()
    with monkeypatch.context() as guard:
        guard.setattr(np, 'load', forbidden)
        guard.setattr(torch, 'load', forbidden)
        code = cli._finish(store, root, recipe.run_id, resources=res, requested=launch)
    printed = capsys.readouterr().out
    assert json.loads(printed)['exit_code_scope'] == 'REQUESTED_NOT_NATIVE'
    assert code == (0 if case == 'ok' else 6 if case == 'disk' else 4)
    (tmp_path / 'cli.stdout.json').write_text(printed, encoding='utf-8')
    report_path = root / 'result' / recipe.run_id / 'report.md'
    if case != 'disk':
        report = report_path.read_text(encoding='utf-8')
        assert 'B−A' not in report and '不是独立确认' in report
        assert ('<svg' in report_path.with_name('index.html').read_text(encoding='utf-8')) == (
            case == 'ok')
