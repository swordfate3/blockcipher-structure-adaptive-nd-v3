"""Guarded engineering only: fake source/model work and new temporary RunStores."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


def test_locked_recipe(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.single_train.config import FIXED, estimate, load_config

    path = Path('configs/innovation1-skinny-single-train.json')
    assert load_config(path) == FIXED
    assert estimate(FIXED) == dict(fits=2, retries=0, cycles=24576, updates_each=24576,
                                  optimizer_updates=49152, real_blocks=25821184,
                                  forward_rows=28835840, train_groups=12582912,
                                  dev_groups=65536, test_groups=262144, null_words=0)
    assert (FIXED.resources.wall_seconds, FIXED.resources.cpu_seconds) == (28800, 115200)
    invalid = tmp_path / 'invalid.json'
    raw = json.loads(path.read_bytes())
    for update in (dict(extra=True), dict(run_id='other'), dict(task={'rounds': 8}),
                   dict(training={**raw['training'], 'epochs': 24.0})):
        invalid.write_text(json.dumps({**raw, **update}))
        with pytest.raises(ValueError):
            load_config(invalid)
    invalid.write_text('{"method":1,"method":2}')
    with pytest.raises(ValueError):
        load_config(invalid)


def test_artificial_policy_is_new() -> None:
    from cipher_lab_plugins.innovation1.single_train.config import FIXED, make_policy

    from cipher_lab.execution import Clocks

    tiny = replace(FIXED, training=replace(FIXED.training, epochs=2, batches_per_epoch=2,
        batch_groups=4, dev_groups=4, test_groups=4, eval_batch_groups=4))
    policy = make_policy(tiny, Clocks())
    assert (policy.real_blocks, policy.optimizer_updates, policy.forward_rows) == (48, 8, 56)


@pytest.mark.parametrize('case', ['success', 'd-fails', 'disk-tamper', 'invalid-d'])
def test_guarded_fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    import io

    import numpy as np
    import torch
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.resources import Resources
    from cipher_lab_plugins.innovation1.single_train.application import _execute
    from cipher_lab_plugins.innovation1.single_train.cli import _finish
    from cipher_lab_plugins.innovation1.single_train.config import (
        FIXED,
        RUN_ID,
        SLOTS,
        make_policy,
    )
    from cipher_lab_plugins.innovation1.single_train.data import Provider
    from cipher_lab_plugins.innovation1.single_train.weights import decode

    from cipher_lab.artifacts import RunStore, read_saved
    from cipher_lab.execution import Clocks, Meter

    tiny = replace(FIXED, training=replace(FIXED.training, epochs=2, batches_per_epoch=2,
        batch_groups=4, dev_groups=4, test_groups=4, eval_batch_groups=4))
    resources = Resources(Meter(make_policy(tiny, Clocks()), Clocks(), lambda: 11))
    store = RunStore.claim(tmp_path, RUN_ID, output_bytes=2**24, needs_models=True,
                           model_slots=SLOTS, source_supplier=lambda: {'fake/fixture.py': 'a'*64})
    run_json = tmp_path / 'data' / RUN_ID / 'run.json'
    paths = {s: tmp_path / 'models' / RUN_ID / f'{s}.pt' for s in SLOTS}
    events: list[tuple[str, str]] = []
    keys: list[tuple[int, ...]] = []
    points: list[tuple[int, ...]] = []
    draws: list[int] = []
    carrier: list[object] = []
    seen_saves: list[str] = []
    orig_finalize = RunStore.finalize
    orig_read = RunStore.read_model
    model_reads: list[str] = []

    def tracked_finalize(self: RunStore, value: dict[str, object]) -> None:
        seen_saves.append(str(value['status']))
        orig_finalize(self, value)

    def tracked_read(self: RunStore, ref: str, expected_sha256: str, max_bytes: int) -> bytes:
        model_reads.append(ref)
        return orig_read(self, ref, expected_sha256, max_bytes)

    monkeypatch.setattr(RunStore, 'finalize', tracked_finalize)
    monkeypatch.setattr(RunStore, 'read_model', tracked_read)

    def private(n: int) -> bytes:
        assert n == 8
        draws.append(len(draws)+1)
        return (0xDEADBEEF_00000000 + draws[-1]).to_bytes(8, 'big')

    def oracle(blocks: tuple[int, ...], key: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        assert rounds == 8 and len(blocks) == len(key) == 2
        keys.append(key)
        points.append(blocks)
        return 0x8000000000000000 + len(points), 0x0100000000000000 + len(points)

    class Panels(Provider):
        def panel(self, split):  # type: ignore[no-untyped-def]
            if split == 'test':
                assert [event for event in events if event[0] == 'restore'] == [
                    ('restore', slot) for slot in SLOTS]
                proof = json.loads(run_json.read_bytes())['summary']['persisted_reloads']
                assert set(proof) == set(SLOTS)
            events.append(('panel', split))
            batch = super().panel(split)
            assert batch.groups == 4
            return batch

        def train(self, cycle):  # type: ignore[no-untyped-def]
            start = len(points)
            batch = super().train(cycle)
            for (a, b), label in zip(points[start:], batch.labels[:, 0], strict=True):
                assert ((a ^ b) == 0x2000) == bool(label)
            events.append(('batch', str(cycle)))
            return batch

    class Fake:
        restored = False
        last_slot: str | None = None

        def metadata(self):  # type: ignore[no-untyped-def]
            return {'artificial': True}

        def prepare(self, batch):  # type: ignore[no-untyped-def]
            x, y = batch.tensors()
            assert x.shape == (4, 1, 2, 64) and y.shape == (4, 1)
            assert x[0, 0, 0, 0] == 1 and x[0, 0, 1, 7] == 1
            carrier[:] = [x, y]
            return x, y

        def update(self, slot, task, x, y, lr, meter):  # type: ignore[no-untyped-def]
            assert carrier[0] is x and carrier[1] is y and task.cipher == 'SKINNY'
            events.append(('update', slot))
            if slot == SLOTS[1] and self.last_slot == SLOTS[0]:
                live = json.loads(run_json.read_bytes())
                if live['counts']['completed_optimizer_updates'] == 1:
                    assert live['summary']['updates'][SLOTS[1]]['completed'] == 0
                    assert live['summary']['paired_completed'] == 0
                    assert lr == .001
            self.last_slot = slot
            meter.reserve('forward_rows', 4)
            if case == 'd-fails' and slot == SLOTS[1]:
                raise RuntimeError('ARTIFICIAL_FAILURE')
            meter.complete('forward_rows', 4)
            return .5

        def predict(self, slot, task, x):  # type: ignore[no-untyped-def]
            assert x is carrier[0] and task.rounds == 8
            # M always ties; D improves in epoch two. TEST uses restored state.
            y = carrier[1].numpy()  # type: ignore[attr-defined]
            if self.restored:
                z = 1 if slot == SLOTS[0] else -1
            else:
                z = 1 if slot == SLOTS[0] or events.count(('update', slot)) == 2 else 2
            return (2 * y - 1) * z

        def template(self, slot: str) -> dict[str, torch.Tensor]:
            state = {'weight': torch.tensor([.5], dtype=torch.float32)}
            if slot == SLOTS[0]:
                state.update({f'stems.{stem}.indices': torch.tensor([0, 1])
                              for stem in ('local', 'word', 'nibble')})
            return state

        def checkpoint(self, slot, metadata):  # type: ignore[no-untyped-def]
            state = self.template(slot)
            if case == 'invalid-d' and slot == SLOTS[1]:
                metadata = dict(metadata, run_id='not-this-run')
            return encode(metadata, state)

        def release(self) -> None:
            if case == 'disk-tamper':
                paths[SLOTS[1]].write_bytes(b'tampered')

        def load_best(self, slot, blob, recipe, source, epoch, bce, check):  # type: ignore[no-untyped-def]
            assert blob == paths[slot].read_bytes()
            decode(blob, recipe, source, 'artificial', slot, epoch, bce, self.template(slot))
            events.append(('restore', slot))
            self.restored = True
            check()

    _execute(tiny, store, resources, 'artificial', lambda r, _: (Panels(tiny, oracle, private, r),
                                                                Fake()))
    assert keys == [(k, 0xDEADBEEF) for k in draws]
    assert len(keys) == len(set(keys))
    record = json.loads(run_json.read_bytes())
    s = record['summary']
    assert record['counts']['completed_null_words'] == 0
    if case == 'success':
        assert seen_saves == ['SUCCEEDED'] and record['status'] == 'SUCCEEDED'
        assert events.count(('update', SLOTS[0])) == events.count(('update', SLOTS[1])) == 4
        assert s['paired_completed'] == 4 and len(s['curves']) == len(s['batch_refs']) == 2
        assert [s['best'][slot]['epoch'] for slot in SLOTS] == [1, 2]
        assert len(draws) == 24 and record['counts']['completed_forward_rows'] == 56
        assert record['counts']['completed_optimizer_updates'] == 8
        assert model_reads == [f'model/{slot}.pt' for slot in SLOTS]
        assert s['m_minus_d']['accuracy'] == 1.0
        path = tmp_path / 'data' / RUN_ID / s['test_panel']['ref']
        test_bytes = path.read_bytes()
        with np.load(io.BytesIO(test_bytes), allow_pickle=False) as z:
            assert z['ciphertext'].shape == (4, 1, 2, 8)
            assert json.loads(z['metadata'].tobytes())['weights'] == s['persisted_reloads']
        code, line = _finish(store, tmp_path, resources.meter)
        assert code == 0 and json.loads(line)['exit_code_scope'] == 'REQUESTED_NOT_NATIVE'
        assert read_saved(tmp_path, RUN_ID)['status'] == 'SUCCEEDED'
    else:
        assert record['status'] == 'FAILED' and ('panel', 'test') not in events
        assert 'dataset/test-SKINNY.npz' not in record['artifacts']
        if case == 'd-fails':
            assert seen_saves == ['FAILED']
            assert s['location']['slot'] == SLOTS[1] and s['paired_completed'] == 0
            assert record['counts']['attempted_optimizer_updates'] == 2
            assert record['counts']['completed_optimizer_updates'] == 1
            assert record['counts']['attempted_forward_rows'] == 8
            assert record['counts']['completed_forward_rows'] == 4
        else:
            assert model_reads == [f'model/{slot}.pt' for slot in SLOTS]
            assert ('restore', SLOTS[0]) in events and ('restore', SLOTS[1]) not in events
            assert seen_saves == ([] if case == 'disk-tamper' else ['FAILED'])
        if case == 'disk-tamper':
            assert s['reason'] == 'HASH_MISMATCH'
            frozen = run_json.read_bytes()
            code, line = _finish(store, tmp_path, resources.meter)
            result = json.loads(line)
            assert code == 6 and result['scientific_status'] == 'FAILED'
            assert result['scientific_reason'] == 'HASH_MISMATCH'
            assert result['artifact_integrity'] == 'FAILED'
            assert result['reason'] == 'SAVED_ARTIFACT_INTEGRITY_FAILED'
            assert result['report_status'] == 'NOT_ATTEMPTED'
            assert not list((tmp_path / 'result' / RUN_ID).iterdir())
            assert run_json.read_bytes() == frozen and seen_saves == []
            # A valid-looking, changed schema-2 record is not owner-sealed evidence.
            record['summary']['reason'] = 'IMPERSONATED'
            run_json.write_text(json.dumps(record))
            _, line = _finish(store, tmp_path, resources.meter)
            result = json.loads(line)
            assert result['scientific_status'] == 'UNKNOWN'
            assert result['reason'] == 'OWNER_RECORD_UNVERIFIED'
            assert result['artifact_integrity'] == 'NOT_CHECKED'


@pytest.mark.parametrize('defect', ['geometry', 'shape', 'dtype', 'nan', 'slot', 'source',
                                    'backend', 'epoch', 'score', 'recipe', 'archive'])
def test_weight_rejections(monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    import torch
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.single_train.config import FIXED, SLOTS
    from cipher_lab_plugins.innovation1.single_train.weights import decode, identity

    template = {'weight': torch.tensor([0., 1.], dtype=torch.float32),
                **{f'stems.{stem}.indices': torch.tensor([0, 1])
                   for stem in ('local', 'word', 'nibble')}}
    source = {'fake/fixture.py': 'a'*64}
    slot, backend = SLOTS[0], 'ciphers'
    metadata = identity(FIXED, source, backend, slot, 1, .125)
    state = dict(template)
    state.update({'geometry': {'stems.word.indices': torch.tensor([1, 0])},
                  'shape': {'weight': torch.tensor([[0., 1.]], dtype=torch.float32)},
                  'dtype': {'weight': torch.tensor([0., 1.], dtype=torch.float64)},
                  'nan': {'weight': torch.tensor([0., float('nan')], dtype=torch.float32)}}
                 .get(defect, {}))
    mismatches: dict[str, dict[str, Any]] = {'slot': {'slot': SLOTS[1]}, 'source': {'source': {}},
                     'backend': {'backend': 'demo'}, 'epoch': {'epoch': 2},
                     'score': {'score': .25}, 'recipe': {'recipe': {}}}
    metadata.update(mismatches.get(defect, {}))
    load = torch.load
    calls = []

    def guarded_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', guarded_load)
    blob = encode(metadata, state) if defect != 'archive' else b'not a zip'
    with pytest.raises(ValueError, match='ARCHIVE_SCHEMA|CHECKPOINT_INVALID'):
        decode(blob, FIXED, source, backend, slot, 1, .125, template)
    assert calls == ([] if defect == 'archive' else [{'map_location': 'cpu', 'weights_only': True}])


def test_pair_adapter_wires_two_models_without_pooling(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    import torch
    from cipher_lab_plugins.innovation1.models import TorchRuntime
    from cipher_lab_plugins.innovation1.single_train import models
    from cipher_lab_plugins.innovation1.single_train.config import ARCHITECTURE, SLOTS, TASK
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    from cipher_lab.representations.dedicated import author_pair_layout

    made: list[str] = []
    opt: list[dict[str, object]] = []
    seen: list[tuple[object, object]] = []

    class Model:
        def __init__(self, slot: str) -> None:
            self.slot = slot
            self.training = True

        def to(self, **kwargs: object) -> Model:
            assert str(kwargs['device']) == 'cuda:0' and kwargs['dtype'] is torch.float32
            return self

        def parameters(self) -> tuple[str]:
            return (self.slot,)

    def construct(recipe: object, slot: str) -> Model:
        assert recipe is ARCHITECTURE
        made.append(slot)
        return Model(slot)

    def adam(parameters: tuple[str], **kwargs: object) -> object:
        assert parameters == (SLOTS[len(opt)],)
        opt.append(kwargs)
        return object()

    def m(model: Model, x: torch.Tensor, card: object, *, rounds: int) -> str:
        assert model.slot == SLOTS[0] and card == 'public-card' and rounds == 8
        seen.append((model, x))
        return 'M'

    def d(model: Model, x: torch.Tensor, layout: object) -> str:
        assert model.slot == SLOTS[1] and model.training and layout is author_pair_layout
        seen.append((model, x))
        return 'D'

    class FakeTensor:
        def __init__(self, value: torch.Tensor):
            self.value = value

        def __getattr__(self, name: str) -> object:
            return getattr(self.value, name)

        def reshape(self, *shape: int) -> FakeTensor:
            return FakeTensor(self.value.reshape(*shape))

        def to(self, device: object) -> torch.Tensor:
            assert str(device) == 'cuda:0'
            return self.value

    original_tensors = PairBatch.tensors

    def fake_tensors(batch: PairBatch) -> tuple[FakeTensor, FakeTensor]:
        bits, labels = original_tensors(batch)
        return FakeTensor(bits), FakeTensor(labels)

    monkeypatch.setattr(PairBatch, 'tensors', fake_tensors)
    monkeypatch.setattr(models, 'construct', construct)
    monkeypatch.setattr(torch.optim, 'Adam', adam)
    monkeypatch.setattr(torch, 'tensor', lambda *a, **kw: 'public-card')
    monkeypatch.setattr(models, 'shared_logits', m)
    monkeypatch.setattr(models, 'dedicated_logits', d)
    r = models.Runtime(lambda: None)
    batch = PairBatch(np.arange(64, dtype=np.uint8).reshape(4, 1, 2, 8),
                      np.array([[0], [1], [0], [1]], dtype=np.uint8), 'train', 0)
    x, y = r.prepare(batch)
    assert x.shape == (1, 4, 2, 64) and y.shape == (4, 1)
    assert torch.equal(x.reshape(4, 1, 2, 64), original_tensors(batch)[0])
    assert r._forward(SLOTS[0], TASK, x) == 'M' and r._forward(SLOTS[1], TASK, x) == 'D'
    assert seen == [(r.models[s], x) for s in SLOTS] and made == list(SLOTS)
    assert opt == [dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)] * 2
    assert models.Runtime.update is TorchRuntime.update
    assert models.Runtime.predict is TorchRuntime.predict


@pytest.mark.parametrize('mode', ['pass', 'reject', 'slow-import'])
def test_artificial_device_admission(monkeypatch: pytest.MonkeyPatch, mode: str,
                                     tmp_path: Path) -> None:
    import builtins
    import time

    from cipher_lab_plugins.innovation1 import application, resources
    from cipher_lab_plugins.innovation1.single_train import application as new
    from cipher_lab_plugins.innovation1.single_train.config import (
        ARCHITECTURE,
        DEVICE,
        FIXED,
        GPU_PCI,
        GPU_UUID,
        RUN_ID,
        make_policy,
    )
    from cipher_lab_plugins.innovation1.single_train.models import Runtime

    from cipher_lab.artifacts import RunStore
    from cipher_lab.execution import Clocks, Meter

    clocks = Clocks()
    own_meter = Meter(make_policy(FIXED, clocks), clocks, lambda: 11)
    own = resources.Resources(own_meter)
    events: list[str] = []
    now = [1.0]

    def sample() -> dict[str, object]:
        events.append('sample')
        return dict(uuid=GPU_UUID, pci='0000:3b:00.0', bytes=0)

    gate = resources.GpuGate(GPU_UUID, GPU_PCI, 20 * 2**30, sample, lambda: now[0])

    def admission(recipe, device, uuid, bus, *, observer):  # type: ignore[no-untyped-def]
        assert recipe is ARCHITECTURE and (device, uuid, bus) == (DEVICE, GPU_UUID, GPU_PCI)
        assert own.gpu is None
        events.append('admission')
        if mode == 'reject':
            raise resources.ResourceStop(detail_code='DEVICE_ENTRY_REJECTED')
        gate.check()
        observer(dict(stage='torch-import', event='begin', gate_last=gate.last,
                      gate_latest=gate.latest, elapsed_seconds=0., duration_seconds=0.))
        if mode == 'slow-import':
            now[0] += 5.001
        observer(dict(stage='torch-import', event='success', gate_last=gate.last,
                      gate_latest={**gate.latest, 'raw': 'DO NOT PERSIST'},
                      elapsed_seconds=now[0]-1, duration_seconds=now[0]-1,
                      exception='DO NOT PERSIST'))
        events.append('attested')
        return resources.Resources(Meter(make_policy(FIXED, clocks), clocks, lambda: 11), gate)

    def fake_init(self, check):  # type: ignore[no-untyped-def]
        assert events[-1] == 'attested' and own.gpu is gate and own.meter is own_meter
        check()
        events.append('fake-runtime')

    original_import = builtins.__import__

    def check_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name in ('torch', 'cipher_lab.ciphers', 'models', 'data'):
            assert 'attested' in events
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(application, 'production_resources', admission)
    monkeypatch.setattr(Runtime, '__init__', fake_init)
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(builtins, '__import__', check_import)
    evidence: dict[str, Any] = {}
    if mode == 'pass':
        provider, model = new._production(own, evidence)
        assert provider.recipe is FIXED and provider.meter is own
        assert type(model) is Runtime
        assert events == ['admission', 'sample', 'attested', 'fake-runtime']
    else:
        store = RunStore.claim(tmp_path, RUN_ID, output_bytes=2**20,
                               source_supplier=lambda: {'fake/fixture.py': 'a'*64})
        new._execute(FIXED, store, own, 'artificial', new._production)
        assert own.gpu is None and 'fake-runtime' not in events
        saved = json.loads((tmp_path / 'data' / RUN_ID / 'run.json').read_bytes())
        assert saved['status'] == 'STOPPED' and saved['summary']['reason'] == 'GPU_UNVERIFIED'
        detail = 'GPU_PROOF_STALE' if mode == 'slow-import' else 'DEVICE_ENTRY_REJECTED'
        assert saved['summary']['detail_code'] == detail
        evidence = saved['summary']['device_setup']
    if mode != 'reject':
        seconds = 5.001 if mode == 'slow-import' else 0.
        sample_event = evidence['stages']['torch-import']
        assert sample_event['event'] == 'success'
        assert sample_event['duration_seconds'] == pytest.approx(seconds)
        assert sample_event['proof_age_seconds_at_check'] == pytest.approx(seconds)
        assert sample_event['last_sample'] == dict(uuid=GPU_UUID, pci='0000:3b:00.0', bytes=0,
                                                   sampled_at=1.)
        assert 'DO NOT PERSIST' not in json.dumps(evidence)
        assert events.count('sample') == 1


def test_report_failure_never_reseals_science(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.single_train.cli import _finish
    from cipher_lab_plugins.innovation1.single_train.config import FIXED, RUN_ID, make_policy

    from cipher_lab.artifacts import RunStore
    from cipher_lab.execution import Clocks, Meter, Stopped

    store = RunStore.claim(tmp_path, RUN_ID, output_bytes=2**20,
                           source_supplier=lambda: {'fake/fixture.py': 'a'*64})
    store.finalize(dict(status='SUCCEEDED', summary=dict(reason='COMPLETE')))
    path = tmp_path / 'data' / RUN_ID / 'run.json'
    frozen = path.read_bytes()
    clocks = Clocks()
    meter = Meter(make_policy(FIXED, clocks), clocks, lambda: 11)
    n = 0

    def check() -> None:
        nonlocal n
        n += 1
        if n >= 4:
            raise Stopped('WALL_LIMIT')

    monkeypatch.setattr(meter, 'check', check)
    code, line = _finish(store, tmp_path, meter)
    result = json.loads(line)
    assert code == 3 and result['scientific_status'] == 'SUCCEEDED'
    assert result['command_status'] == result['report_status'] == 'STOPPED'
    assert result['report_files_written'] is True and path.read_bytes() == frozen


@pytest.mark.parametrize('target', ['core-dir', 'core-file', 'owner-dir', 'owner-file'])
def test_source_does_not_follow_windows_reparse(monkeypatch: pytest.MonkeyPatch,
                                                target: str) -> None:
    import stat
    from types import SimpleNamespace

    from cipher_lab_plugins.innovation1.single_train import source
    from cipher_lab_plugins.innovation1.source import source_identity as original

    from cipher_lab import artifacts

    old = original()
    current = source.source_identity()
    assert {name: current[name] for name in old} == old
    assert len(current) == len(old) + 12
    core = Path(artifacts.__file__).absolute().parent
    owner = Path(source.__file__).absolute().parent.parent
    flagged = {'core-dir': core / 'models', 'core-file': core / 'cli.py',
               'owner-dir': owner, 'owner-file': owner / 'models.py'}[target]
    lstat = Path.lstat
    read = Path.read_bytes

    def fake_lstat(path: Path) -> Any:
        info = lstat(path)
        if path == flagged:
            return SimpleNamespace(st_mode=info.st_mode,
                                   st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return info

    def guarded_read(path: Path) -> bytes:
        assert path != flagged and flagged not in path.parents, 'read through reparse point'
        return read(path)

    monkeypatch.setattr(Path, 'lstat', fake_lstat)
    monkeypatch.setattr(Path, 'read_bytes', guarded_read)
    with pytest.raises(artifacts.ArtifactError, match='SOURCE_UNAVAILABLE'):
        source.source_identity()
