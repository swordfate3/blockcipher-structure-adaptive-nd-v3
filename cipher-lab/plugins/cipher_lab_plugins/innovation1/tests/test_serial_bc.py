"""Serial B/C contract tests: byte-pinned artificial archives and no real fit."""
from __future__ import annotations

import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, cast
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1 import cli, resources, source
from cipher_lab_plugins.innovation1.config import Task, load_config
from cipher_lab_plugins.innovation1.four_single import config as old
from cipher_lab_plugins.innovation1.layout_probe.reuse import PoolSourceError
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.serial_bc import config, data, models, training
from cipher_lab_plugins.innovation1.single_train.config import learning_rate
from cipher_lab_plugins.innovation1.single_train.data import PairBatch
from cipher_lab_plugins.innovation1.tests.test_four_single_replay import GPU, SOURCE, _archive
from plugin_isolation import forbidden
from torch import Tensor, nn

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.execution import Clocks, Meter, Stopped
from cipher_lab.models.shared_m.router import PublicRouter
from cipher_lab.representations.tasks import public_features, validate_public

HISTORICAL = {
    'four-task-single-pair-50epoch':
        '296a39e9c894b51f22e835d43ec72a3e9e752b92499c2e59fdada7c6f3fd9315',
    'four-task-single-pair': '694a1fc8a3dd13d0f8affb7082a03bd35c37d999ef0195ec752b6dbd3ba7fe83',
    'fresh': '96254dbc5207779e462e9411e1200b83962ad28e0f9aebbd79ca482f0c4d4361',
    'skinny-layout-early-stop': '7b7e6441bd9f555f5b104fdf1c696b8d748952b2f134ddc83c64ef4981479aac',
    'skinny-layout-fixed-bce': '58767a238dfbad1abf0ef38ca25fbc272f211cdb4aff5af214216affee0f9ef1',
    'skinny-layout-fixed-mse': '46b2f24339c037b3387472df07e82801e63ebb86da310accd0f3c63bed703773',
}


def test_static(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1 import application

    monkeypatch.setattr(source, 'no_reparse', forbidden)
    monkeypatch.setattr(application, 'make_resources', forbidden)
    monkeypatch.setenv(config.DEADLINE_ENV, 'invalid-must-not-be-read')
    for name, digest in HISTORICAL.items():
        assert cli.main(['check', '--config', f'configs/innovation1-{name}.json']) == 0
        assert sha256(capsys.readouterr().out.encode()).hexdigest() == digest
    for suffix, expected in (('b', config.SERIAL_B), ('c', config.SERIAL_C)):
        stem = f'innovation1-four-task-serial-{suffix}'
        path = f'configs/{stem}.json'
        assert expected.run_id == f'{stem}-20261124-01'
        assert cli.main(['check', '--config', path]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['scope'] == 'STATIC_ONLY_NO_DEVICE_ADMISSION'
        counts = result['estimate']
        assert [counts[k] for k in ('optimizer_updates', 'forward_rows', 'real_blocks',
                                    'fits', 'retries')] == [7200, 38648576, 0, 1, 0]
        assert counts['artifact_peak_bound_bytes'] <= expected.resources.artifact_bytes
        assert load_config(path) == expected
        assert config.make_policy(expected, Clocks()).wall_seconds == 5400
        assert [learning_rate(expected, i) for i in (0, 1799)] == [.001, .0001]
        wrong = expected.to_dict()
        wrong['optimizer']['lr_max'] *= 2
        bad = tmp_path / 'settings.json'
        bad.write_text(json.dumps(wrong))
        assert cli.main(['check', '--config', str(bad)]) == 2
        config.validate_root(PureWindowsPath(config.ROOT))
        for root, run_id in ((config.ROOT, 'wrong-id'), ('relative', expected.run_id),
                             (expected.archive.root, expected.run_id),
                             (cast(Any, expected.skinny_source).root, expected.run_id)):
            assert cli.main(['run', '--config', path, '--root', root, '--id', run_id,
                             '--gpu-uuid', GPU]) == 2


def test_serial_deadline_is_production_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1 import application

    monkeypatch.delenv(config.DEADLINE_ENV, raising=False)
    now = datetime(2026, 11, 24, 11, tzinfo=UTC)
    current = now
    clocks = Clocks(wall=lambda: 0., cpu=lambda: 0., utc=lambda: current)
    monkeypatch.setattr(application, 'Clocks', lambda: clocks)
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 0)
    first = config.SERIAL_B
    own = (now + timedelta(seconds=5400)).isoformat()
    assert application.make_resources(first).meter._policy.deadline_utc == own
    short = '2026-11-24T11:01:00Z'
    monkeypatch.setenv(config.DEADLINE_ENV, short)
    for arm in (first, config.SERIAL_C):
        meter = application.make_resources(arm)
        assert meter.meter._policy.deadline_utc == short
        meter.check()
        current = now + timedelta(seconds=60)
        with pytest.raises(Stopped, match='DEADLINE'):
            meter.check()
        current = now
    monkeypatch.setenv(config.DEADLINE_ENV, '2026-11-24T14:00:00+00:00')
    assert application.make_resources(first).meter._policy.deadline_utc == own
    for invalid in ('', '2026-11-25T12:00:00', '2026-11-25T12:00:00+01:00',
                    '2026-11-24T11:00:00Z', '2026-11-23T11:00:00Z'):
        monkeypatch.setenv(config.DEADLINE_ENV, invalid)
        with pytest.raises(Stopped, match='DEADLINE'):
            application.make_resources(first)
        application.make_resources(old.FOUR_SINGLE_50).check()


def test_bad_deadline_before_owner(monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    # Native Windows Path semantics are already checked with PureWindowsPath.
    # Bypass only that root check on POSIX, to drive the real CLI stop boundary.
    monkeypatch.setattr(cli, 'validate_serial_root', lambda root: None)
    monkeypatch.setattr(source, 'no_reparse', forbidden)
    monkeypatch.setenv(config.DEADLINE_ENV, 'broken')
    assert cli.main(['run', '--config', 'configs/innovation1-four-task-serial-b.json',
                     '--root', config.ROOT, '--id', config.SERIAL_B.run_id, '--gpu-uuid', GPU]) == 3
    assert json.loads(capsys.readouterr().out)['reason'] == 'DEADLINE'


def test_serial_source_closure() -> None:
    closure = source.source_identity()
    owner = 'cipher_lab_plugins/innovation1/serial_bc/'
    for module in ('__init__', 'config', 'data', 'models', 'training'):
        key = f'{owner}{module}.py'
        assert closure[key] == sha256((Path('plugins') / key).read_bytes()).hexdigest()


def _tiny(base: config.SerialRecipe, archive: old.ReplayRecipe) -> config.SerialRecipe:
    return replace(base, archive=archive.archive, skinny_source=archive.skinny_source,
                   training=replace(base.training, epochs=2, batch_groups=4,
                                    batches_per_epoch=9, dev_groups=4, test_groups=4,
                                    eval_batch_groups=4))


def test_balanced_index_subset() -> None:
    x = np.arange(36 * 16, dtype=np.uint16).reshape(36, 1, 2, 8).astype(np.uint8)
    y = np.tile(np.array([[0], [1]], dtype=np.uint8), (18, 1))
    panel = PairBatch(x, y, 'dev', 0, 'fixture')
    subset, rows = data.index_subset(panel, 20, 123, 'subset')
    again, same = data.index_subset(panel, 20, 123, 'subset')
    assert rows.dtype.str == '<i8' and len(set(rows)) == 20
    assert subset.digest == again.digest
    np.testing.assert_array_equal(rows, same)
    np.testing.assert_array_equal(subset.labels.ravel(), [0, 1] * 10)
    np.testing.assert_array_equal(subset.ciphertext, x[rows])
    np.testing.assert_array_equal(subset.labels, y[rows])
    assert not np.array_equal(rows, data.index_subset(panel, 20, 124, 'subset')[1])
    # Tag each source split and its within-split row independently. Numeric
    # offsets may overlap; (originating panel, row offset) identities cannot.
    for offset, split in enumerate(('train', 'dev')):
        tagged = x.copy()
        tagged[:, 0, 0, 0] = offset
        tagged[:, 0, 0, 1] = np.arange(36, dtype=np.uint8)
        source_panel = PairBatch(tagged, y, split, 0, 'fixture')
        selected, indexes = data.index_subset(source_panel, 20, 123, 'subset')
        np.testing.assert_array_equal(selected.ciphertext[:, 0, 0, 0], offset)
        np.testing.assert_array_equal(selected.ciphertext[:, 0, 0, 1], indexes)
        for i in range(0, 20, 4):
            assert selected.slice(i, i + 4).labels.sum() == 2
    with pytest.raises(ValueError):
        data.index_subset(panel, 40, 123, 'subset')


def test_verified_archive_inputs_same_order_and_no_fallback(tmp_path: Path) -> None:
    archive, original, _ = _archive(tmp_path / 'good', 'early')
    all_digests = []
    for base in (config.SERIAL_B, config.SERIAL_C):
        recipe = _tiny(base, archive)
        pools = data.SerialPools(recipe, lambda: None, archive=archive)
        dev, bindings = pools.bind(lambda task: None)
        before = {t: p.digest for t, p in dev.items()}
        assert all(e[k]['index_sha256'] and e[k]['source_digest'] for e in bindings.values()
                   for k in ('train', 'dev'))
        assert all(e['train']['source_digest'] != e['dev']['source_digest']
                   for e in bindings.values())
        batches = {t.cipher: [pools.train(t, i).digest for i in range(18)] for t in recipe.tasks}
        all_digests.append((before, bindings, batches))
    assert all_digests[0] == all_digests[1]
    assert all(path.read_bytes() == content for path, content in original.items())
    for case in ('metadata', 'skinny', 'old-recipe', 'record-hash', 'digest'):
        broken, _, _ = _archive(tmp_path / case, case)
        with pytest.raises(PoolSourceError):
            data.SerialPools(_tiny(config.SERIAL_B, broken), lambda: None, archive=broken).bind(
                lambda task: None)


def test_gate_only_uses_constant_with_fake_differentiable_scores() -> None:
    seen: list[Tensor] = []
    parameters = torch.tensor([.9, -.3, .2], requires_grad=True)

    class Gate(nn.Module):
        def forward(self, vector: Tensor) -> Tensor:
            seen.append(vector)
            return parameters + vector[:3]

    # The existing guard forbids real layers and autograd invocations. Verify
    # the fake score graph by traversing it and checking stable Top2 weights.
    router = PublicRouter(2)

    class First(nn.Module):
        weight = parameters

        def forward(self, vector: Tensor) -> Tensor:
            return vector

    router.gate = nn.Sequential(First(), Gate())
    cards = [torch.tensor(public_features(t.cipher)) for t in config.SERIAL_B.tasks]
    for index, card in enumerate(cards):
        public = router(card)
        constant = models.constant_route(router)
        assert seen[2 * index] is card
        assert torch.equal(seen[2 * index + 1], torch.full((13,), .5))
        assert constant.indices.tolist() == [0, 2]
        assert torch.equal(constant.weights, parameters[[0, 2]].add(.5).softmax(0))
        assert public.weights.requires_grad and constant.weights.requires_grad
        assert constant.weights.grad_fn is not None
    with pytest.raises(ValueError, match='unreviewed'):
        validate_public(seen[1])

    # Verify the softmax derivative with a centered finite difference on the
    # fake gate, without invoking autograd or unpoisoning any real layer.
    base = parameters
    q = constant.weights[0] * constant.weights[1]
    step = torch.tensor([1e-3, 0., 0.])
    parameters = base + step
    plus = models.constant_route(router).weights
    parameters = base - step
    minus = models.constant_route(router).weights
    torch.testing.assert_close((plus - minus) / .002, torch.stack((q, -q)),
                               rtol=0, atol=5e-5)


def test_serial_model_identity_and_strict_reload() -> None:
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.four_single.models import build, geometry

    rng = torch.random.get_rng_state()
    models_by_arm = []
    for recipe in (config.SERIAL_B, config.SERIAL_C):
        # Use the production mode-aware builder/decoder, without the CUDA constructor.
        runtime = models.SerialRuntime.__new__(models.SerialRuntime)
        runtime.contract = recipe
        runtime.models, runtime.optimizers, runtime.device = {}, {}, torch.device('cpu')
        model = runtime._build()
        assert isinstance(model, models.NoCardM) == (recipe is config.SERIAL_C)
        identity = dict(recipe=recipe.to_dict(), geometry=geometry(model), stage=4, slot=old.SLOT,
                        routing_mode=recipe.routing_mode, run_id=recipe.run_id)
        raw = encode(identity, model.state_dict())
        for key, value in (('stage', 3), ('stage', 4.), ('routing_mode', 'wrong'),
                           ('run_id', 'wrong'), ('recipe', {}), ('geometry', {})):
            with pytest.raises(ValueError, match='CHECKPOINT_'):
                runtime.restore(old.SLOT, raw, {**identity, key: value})
        runtime.restore(old.SLOT, raw, identity)
        models_by_arm.append(runtime.models[old.SLOT].state_dict())
    original = build(old.FOUR_SINGLE)
    assert torch.equal(rng, torch.random.get_rng_state())
    for state in models_by_arm:
        torch.testing.assert_close(state, original.state_dict(), rtol=0, atol=0)



@pytest.mark.parametrize('case',
                         ['ok-b', 'ok-c', 'strict', 'disk', 'test-missing', 'source', 'budget'])
def test_serial_fit(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, original, _ = _archive(tmp_path / 'archives', case)
    base = config.SERIAL_C if case == 'ok-c' else config.SERIAL_B
    recipe = _tiny(base, archive)
    run_id = recipe.run_id
    clock = Clocks()
    res = Resources(Meter(config.make_policy(recipe, clock), clock, lambda: 0))
    root = tmp_path / 'output'
    store = RunStore.claim(root, run_id, output_bytes=recipe.resources.artifact_bytes,
                           needs_models=True, model_slots=config.STAGE_SLOTS,
                           source_supplier=lambda: dict(SOURCE))
    restored = False
    updates: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    scored: list[tuple[int, str, str]] = []
    read_tests: list[str] = []
    names = [t.cipher for t in recipe.tasks]
    per_task = recipe.training.batches_per_epoch * recipe.training.epochs
    optimizer = object()
    original_open, original_member = Path.open, ZipFile.open

    def open_path(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.is_relative_to(archive.archive.root):
            assert path.suffix != '.pt'
            if path.name.startswith('test-'):
                assert restored
                read_tests.append(path.name)
        return original_open(path, *args, **kwargs)

    def open_member(self: ZipFile, member: Any, mode: Literal['r', 'w'] = 'r',
                    **kwargs: Any) -> Any:
        if getattr(member, 'filename', member) == 'logits.npy' and mode == 'r':
            forbidden()
        return original_member(self, member, mode, **kwargs)

    class Runtime:
        def __init__(self) -> None:
            self.optimizers = {old.SLOT: optimizer}

        def metadata(self) -> dict[str, Any]:
            return {old.SLOT: dict(parameters=old.PARAMETERS, geometry={'fake': 'b' * 64})}

        def prepare(self, batch: PairBatch) -> tuple[PairBatch, Any]:
            return batch, batch.labels

        def routes(self) -> dict[str, dict[str, float]]:
            return {name: {'local': .7, 'nibble': .3} for name in names}

        def update(self, slot: str, task: Task, batch: PairBatch, y: Any, lr: float,
                   meter: Resources) -> float:
            assert self.optimizers[slot] is optimizer and not restored and not read_tests
            i = len(updates)
            assert task.cipher == names[i // per_task] and batch.index == i % per_task
            assert batch.groups == 4 and lr == learning_rate(recipe, i % per_task)
            if case == 'budget' and i == per_task + 1:
                raise Stopped('WALL_LIMIT')
            updates.append(task.cipher)
            meter.reserve('forward_rows', batch.groups)
            meter.complete('forward_rows', batch.groups)
            return .5

        def predict(self, slot: str, task: Task, batch: PairBatch) -> Any:
            assert slot == old.SLOT
            scored.append((len(updates), batch.split, task.cipher))
            assert (batch.split == 'test') == restored
            # DEV degrades at each stage, so stage 4 cannot be selected on DEV.
            return np.full(batch.labels.shape, len(updates) / per_task)

        def checkpoint(self, slot: str, identity: dict[str, Any]) -> bytes:
            assert slot == old.SLOT
            assert identity['routing_mode'] == recipe.routing_mode
            assert identity['geometry'] == self.metadata()[slot]['geometry']
            checkpoints.append(identity)
            return json.dumps(identity).encode()

        def release(self) -> None:
            self.optimizers.clear()

        def restore(self, slot: str, raw: bytes, identity: dict[str, Any]) -> None:
            nonlocal restored
            assert slot == old.SLOT and not self.optimizers and not restored
            assert len(checkpoints) == 4 and json.loads(raw) == identity == checkpoints[-1]
            if case == 'strict':
                raise ValueError('injected bad strict reload')
            if case == 'source':
                monkeypatch.setattr(store, 'current_source_identity', lambda: {'changed': 'a'*64})
            restored = True

    monkeypatch.setattr(Path, 'open', open_path)
    monkeypatch.setattr(ZipFile, 'open', open_member)
    if case == 'disk':
        real_read = store.read_model

        def bad_read(ref: str, digest: str, bound: int) -> bytes:
            (root / 'models' / run_id / 'M-best.pt').write_bytes(b'damaged')
            return real_read(ref, digest, bound)

        monkeypatch.setattr(store, 'read_model', bad_read)
    store.write_snapshot(dict(recipe=recipe.to_dict()))
    training._execute(recipe, store, res, launch={},
                      pools=data.SerialPools(recipe, res.check, archive=archive),
                      runtime=cast(Any, Runtime()))
    monkeypatch.setattr(Path, 'open', original_open)
    monkeypatch.setattr(ZipFile, 'open', original_member)
    assert all(path.read_bytes() == content for path, content in original.items())
    if case == 'disk':
        saved_disk = json.loads((root / 'data' / run_id / 'run.json').read_bytes())
        assert saved_disk['status'] == 'FAILED' and not restored and not read_tests
        assert len(checkpoints) == 4
        assert all(f'payload/stage-{i}.json' in saved_disk['artifacts'] for i in range(1, 5))
        assert not any(split == 'test' for _, split, _ in scored)
        return
    saved = cast(dict[str, Any], read_saved(root, run_id))
    s, counts = saved['summary'], saved['counts']
    assert counts['completed_real_blocks'] == counts['attempted_real_blocks'] == 0
    assert counts['completed_optimizer_updates'] == len(updates)
    assert counts['completed_forward_rows'] == len(updates) * 4 + len(scored) * 4
    expected_stages = 1 if case == 'budget' else 4
    assert [x['task'] for x in s['stages']] == names[:expected_stages]
    assert [x['ref'] for x in s['stages']] == [f'model/{slot}.pt'
                                             for slot in config.STAGE_SLOTS[:expected_stages]]
    assert all(set(x['dev']) == set(x['routes']) == set(names) for x in s['stages'])
    for stage in s['stages']:
        assert saved['payloads'][stage['evidence_ref']] == {k: v for k, v in stage.items()
                                                             if k != 'evidence_ref'}
    assert scored[:expected_stages * 4] == [(s * per_task, 'dev', t)
                                           for s in range(1, expected_stages + 1) for t in names]
    if case.startswith('ok'):
        assert saved['status'] == 'SUCCEEDED'
        assert len(updates) == 4 * per_task
        assert [row for row in scored if row[1] == 'test'] == [(len(updates), 'test', t)
                                                              for t in names]
        assert read_tests == [f'test-{t}.npz' for t in names]
        assert s['persisted_reload']['stage'] == 4 and set(s['test']) == set(names)
        assert cli._finish(store, root, run_id, resources=res) == 0
        report = (root / 'result' / run_id / 'report.md').read_text(encoding='utf-8')
        assert recipe.routing_mode in report and '第四阶段权重严格重载' in report
        for item in s['test'].values():
            path = root / 'data' / run_id / item['panel']['ref']
            with np.load(io.BytesIO(path.read_bytes())) as panel:
                assert np.all(panel['logits'] == 4.)
    else:
        assert saved['status'] != 'SUCCEEDED' and not s['test']
        assert not any(split == 'test' for _, split, _ in scored)
        if case != 'test-missing':
            assert not read_tests
    rows = [row for ref in s['batch_refs'] for row in saved['payloads'][ref]] + s['pending']
    assert len(rows) == counts['attempted_optimizer_updates']
    assert sum(row['completed'] for row in rows) == len(updates)
