"""Guarded artificial fixtures only; never open historical artifacts or provision CUDA."""
from __future__ import annotations

import builtins
import io
import json
import os
import runpy
from copy import deepcopy
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest
from cipher_lab_plugins.innovation1 import application
from cipher_lab_plugins.innovation1.batch import Batch
from cipher_lab_plugins.innovation1.checkpoint import identity
from cipher_lab_plugins.innovation1.config import TASKS, Recipe
from cipher_lab_plugins.innovation1.contracts import SLOTS as OLD_SLOTS
from cipher_lab_plugins.innovation1.resources import GpuGate, Resources, ResourceStop
from cipher_lab_plugins.innovation1.source import source_identity as old_source

from cipher_lab import artifacts
from cipher_lab.execution import Clocks, Meter, Stopped


def _forbidden(*a: object, **kw: object) -> NoReturn:
    raise AssertionError('not part of the fixed, fake-only protocol')


def _panel(n: int) -> Batch:
    # Each row is unique across chunks. Indices 1–3 deliberately change the bytes.
    x = np.full((n, 4, 2, 8), 255, np.uint8)
    x[:, 0, :, :] = 0
    x[:, 0, 0, 0] = np.arange(n) % 256
    x[:, 0, 0, 1] = np.arange(n) // 256
    x[:, 0, 1, 7] = np.arange(n) % 2
    return Batch(x, (np.arange(n) % 2).astype(np.uint8).reshape(n, 1),
                 Recipe().tasks[-1], 'test', 0)


def test_selection_and_meter_count_real_rows_before_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1.single_pair import evaluate as e

    panel = _panel(1024)
    unpack = np.unpackbits
    expected = unpack(panel.ciphertext[:, :1], axis=-1).astype(np.float32)
    unpacked = []

    def checked_unpack(x: Any, **kwargs: Any) -> Any:
        assert x.shape == (512, 1, 2, 8)
        assert np.shares_memory(x, panel.ciphertext)
        assert kwargs == {'axis': -1, 'bitorder': 'big'}
        unpacked.append(x)
        return unpack(x, **kwargs)

    monkeypatch.setattr(np, 'unpackbits', checked_unpack)
    monkeypatch.setattr(Batch, 'tensors', _forbidden)
    monkeypatch.setattr(Batch, 'slice', _forbidden)
    monkeypatch.setattr(e, 'GROUPS', 1024)
    run_policy = e.make_policy(Clocks())
    assert run_policy.forward_rows == 524288
    resources = Resources(Meter(run_policy, Clocks(), lambda: 0))
    ledger = e.Forwards(resources)
    seen: list[tuple[str, Any]] = []

    class Runtime:
        def prepare(self, bits: Any) -> Any:
            return bits

        def predict(self, slot: str, bits: Any) -> Any:
            at = len(seen)
            assert resources.snapshot()['attempted_forward_rows'] == (at + 1) * 512
            assert resources.snapshot()['completed_forward_rows'] == at * 512
            np.testing.assert_array_equal(bits, expected[at // 2 * 512:][:512])
            if at % 2:
                assert bits is seen[-1][1]
            assert slot == e.SLOTS[at % 2]
            seen.append((slot, bits))
            z = 2 * bits[:, 0, 1, -1:] - 1
            return z if slot == 'M-best' else -z

    scores = e.score(panel, Runtime(), ledger, {'M-best': .506466, 'D-SKINNY-best': .635044})
    assert len(unpacked) == 2 and len(seen) == 4
    assert ledger.by_slot['M-best'] == {'attempted': 1024, 'completed': 1024}
    assert ledger.by_slot['D-SKINNY-best'] == ledger.by_slot['M-best']
    assert scores['M-best']['accuracy'] == 1
    assert scores['M-best']['bce'] == pytest.approx(0.3132616875)
    assert scores['M-best']['vs_four_pair_pp'] == pytest.approx(49.3534)
    assert scores['D-SKINNY-best']['accuracy'] == 0
    assert scores['D-SKINNY-best']['vs_four_pair_pp'] == pytest.approx(-63.5044)


def test_pair_heads_do_not_pool_container_slots() -> None:
    from cipher_lab_plugins.innovation1.single_pair import models as m

    # Fake linear and genuinely nonlinear heads use different hidden values for every row.
    ids = np.arange(1., 9.)
    bits = np.broadcast_to(ids[:, None, None, None], (8, 1, 2, 64))
    packed = m.computation_slots(bits)
    public = object()
    calls: list[str] = []

    def fake_repr(x: Any, card: object, *, rounds: int) -> Any:
        assert x is packed and card is public and rounds == 8
        assert x.shape == (2, 4, 2, 64)
        np.testing.assert_array_equal(x[:, :, 0, 0].reshape(-1), ids)
        return SimpleNamespace(canonical=x[:, :, :1], channel=x[:, :, 1:])

    def stem(raw: Any, channel: Any) -> Any:
        calls.append('stem')
        np.testing.assert_array_equal(raw, channel)
        return raw

    def route(card: object) -> Any:
        assert card is public
        return SimpleNamespace(indices=np.array([2, 0]), weights=[.25, .75])

    model = SimpleNamespace(representation=fake_repr, router=route,
        spec=SimpleNamespace(channels=1),
        stems={'nibble': stem, 'local': stem, 'word': _forbidden}, suffix=lambda x: x + 1,
        readout=SimpleNamespace(forward=_forbidden, base=lambda x: 3 * x[:, :1] + 5,
                                residual=lambda x: x[:, :1] ** 2 + 7))
    result = m.shared_logits(model, packed, public, rounds=8)
    h = ids[:, None] + 1
    np.testing.assert_array_equal(result, (3*h+5) + (h**2+7))
    fake_original_m = 3 * h.reshape(2, 4).mean(1) + 5 + (h**2+7).reshape(2, 4).mean(1)
    np.testing.assert_array_equal(result.reshape(2, 4).mean(1), fake_original_m)
    assert calls == ['stem', 'stem']

    def layout(x: Any, *, rounds: int) -> Any:
        assert x is packed and rounds == 8
        return x[:, :, 0, 0].reshape(8, 1)

    d = SimpleNamespace(rounds=8, encoder=lambda x: x ** 2, head=lambda x: x - 5,
                        forward=_forbidden)
    result = m.dedicated_logits(d, packed, layout)
    np.testing.assert_array_equal(result, ids[:, None]**2 - 5)
    np.testing.assert_array_equal(result.reshape(2, 4).mean(1),
                                  (ids**2).reshape(2, 4).mean(1) - 5)
    for bad in (bits[:, 0], bits[:3], np.zeros((8, 4, 2, 64))):
        with pytest.raises(ValueError, match='SINGLE_PAIR_SHAPE'):
            m.computation_slots(bad)


def test_bounded_pinned_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.single_pair import evaluate as e

    assert e.INPUT_SHA == '5d70f6ce4e5e0fc73cdc9f9107496f9aa600e3e953ec66164fc5cf639bfa64c7'
    folder = tmp_path / 'data' / e.INPUT_ID
    folder.mkdir(parents=True)
    target = folder / 'run.json'
    target.write_bytes(b'{}')  # identity checked before even decoding
    with pytest.raises(ValueError, match='INPUT_RUN_SHA'):
        e.read_record(tmp_path, e.INPUT_ID, 4, e.INPUT_SHA)
    target.write_bytes(b' ' * 6)
    with pytest.raises(ValueError, match='RUN_JSON_SIZE'):
        e.read_record(tmp_path, e.INPUT_ID, 4, sha256(b' ' * 6).hexdigest())

    def bounded_open(path: Path, mode: str) -> Any:
        assert mode == 'rb' and path == target

        class Stream(io.BytesIO):
            def read(self, size: int | None = -1) -> bytes:
                assert size == 5
                return super().read(size)

        return Stream(b' ' * 6)

    monkeypatch.setattr(Path, 'open', bounded_open)
    with pytest.raises(ValueError, match='RUN_JSON_SIZE'):
        e.read_record(tmp_path, e.INPUT_ID, 4, e.INPUT_SHA)


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> tuple[Any, list[str]]:
    from cipher_lab_plugins.innovation1.single_pair import evaluate as e

    trace: list[str] = []
    recipe = Recipe()
    source = old_source()
    backend = 'demo' if fault == 'backend' else 'ciphers'
    inputs = {'model/M-best.pt': b'm', 'model/D-SKINNY-best.pt': b'd',
              'dataset/test-SKINNY.npz': b'test'}
    refs = {ref: dict(kind=ref.split('/')[0], bytes=len(raw), sha256=sha256(raw).hexdigest())
            for ref, raw in inputs.items()}
    # These extra references deliberately have no files at all.
    for ref in ('model/D-SIMON-best.pt', 'dataset/dev-SKINNY.npz',
                'payload/predictions-SKINNY.json'):
        refs[ref] = dict(kind=ref.split('/')[0], bytes=5, sha256='0' * 64)
    best: dict[str, Any] = {
        slot: dict(ref=f'model/{slot}.pt', sha256=refs[f'model/{slot}.pt']['sha256'],
                   metadata=identity(recipe, source, backend, slot, 2, .5))
        for slot in e.SLOTS
    }
    if fault == 'checkpoint-metadata':
        checkpoint = best['D-SKINNY-best']
        # Frozen identity() exposes vars(task); detach before injecting a fault.
        checkpoint['metadata'] = deepcopy(checkpoint['metadata'])
        checkpoint['metadata']['task']['difference'] = 32
    if fault == 'checkpoint-binding':
        best['M-best']['sha256'] = '0' * 64
    if fault == 'source':
        source = {**source, next(iter(source)): '0'*64}
    raw_recipe = recipe.to_dict()
    if fault == 'recipe':
        raw_recipe['tasks'][-1]['rounds'] = 9
    if fault == 'task':
        refs.pop('dataset/test-SKINNY.npz')
    record = dict(schema=2, run_id=e.INPUT_ID, source=source,
        environment=artifacts._environment_identity(), recipe=raw_recipe, policy={}, counts={},
        status='RUNNING' if fault == 'unsealed' else 'SUCCEEDED', artifacts=refs,
        summary=dict(method=recipe.method, backend_identity=backend, synthetic=backend == 'demo',
            reason='COMPLETE', persisted_reloads=list(OLD_SLOTS), best=best,
            compare={'SKINNY': {'shared': {'accuracy': .506466},
                                'dedicated': {'accuracy': .635044}}}))
    blob = artifacts._encode(record)
    raw_path = root / 'data' / e.INPUT_ID / 'run.json'
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(blob)
    if fault != 'unpinned':
        monkeypatch.setattr(e, 'INPUT_SHA', sha256(blob).hexdigest())
    for ref, raw in inputs.items():
        path = artifacts._location(root, e.INPUT_ID, ref)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(b'?' * len(raw) if fault == 'binary-sha' and raw == b'test' else raw)
    read_binary = artifacts.read_binary

    def read(root: Path, run_id: str, ref: str, digest: str, maximum: int) -> bytes:
        assert run_id == e.INPUT_ID and ref in inputs and 'gpu' not in trace
        trace.append(ref)
        return read_binary(root, run_id, ref, digest, maximum)

    monkeypatch.setattr(e, 'read_binary', read)
    monkeypatch.setattr(artifacts, 'read_saved', _forbidden)
    monkeypatch.setattr(e, 'production_rss_bytes', lambda: 100)
    monkeypatch.setattr(e, 'GROUPS', 512)
    from cipher_lab_plugins.innovation1.single_pair import models as m

    fake_wall = 1.
    probes: list[float] = []

    def sample() -> dict[str, Any]:
        probes.append(fake_wall)
        assert len(probes) == 1  # no refresh of stale proof
        return {'uuid': e.GPU_UUID, 'pci': e.GPU_PCI, 'bytes': 0}

    def setup(*a: Any, observer: Any) -> Any:
        trace.append('gpu')
        assert tuple(trace[:3]) == tuple(inputs)
        if fault == 'device':
            raise ResourceStop(detail_code='DEVICE_IDENTITY_MISMATCH')
        assert a == (Recipe(), 'cuda:0', e.GPU_UUID, e.GPU_PCI)
        observer({'stage': 'returned', 'event': 'success', 'attestation': {'logical_device': 0}})
        gate = GpuGate(e.GPU_UUID, e.GPU_PCI, e.GPU, sample, lambda: fake_wall)
        gate.check()
        return SimpleNamespace(gpu=gate)

    def decode(*a: Any) -> Any:
        nonlocal fake_wall
        trace.append('decode')
        if fault == 'stale':
            fake_wall += 5.001
        assert 'gpu' in trace and a[0] == b'test'
        n = 256 if fault == 'shape' else 512
        return _panel(n)

    def restore(recipe: Any, task: Any, raw: Any, expected: Any, check: Any) -> Any:
        trace.append('restore')
        assert task == Recipe().tasks[-1] and list(raw) == list(e.SLOTS)
        assert all(expected[s]['slot'] == s for s in e.SLOTS)
        check()
        if fault == 'checkpoint':
            raise ValueError('CHECKPOINT_INVALID')
        return SimpleNamespace(prepare=lambda x: x, predict=predict)

    def predict(slot: str, bits: Any) -> Any:
        trace.append(slot)
        if slot == 'D-SKINNY-best' and fault == 'forward':
            raise RuntimeError('must not expose exception detail')
        return 2 * bits[:, 0, 1, -1:] - 1

    from cipher_lab_plugins.innovation1 import dataset

    monkeypatch.setattr(application, 'production_resources', setup)
    monkeypatch.setattr(dataset, 'decode_panel', decode)
    monkeypatch.setattr(m, 'restore', restore)
    return e, trace


@pytest.mark.parametrize('fault,reason', [
    ('unpinned', 'INPUT_RUN_SHA'), ('unsealed', 'INPUT_RUN_IDENTITY'),
    ('source', 'SOURCE_IDENTITY'), ('backend', 'INPUT_RUN_IDENTITY'),
    ('recipe', 'INPUT_RECIPE'), ('task', 'INPUT_BINDING'),
    ('checkpoint-binding', 'INPUT_BINDING'), ('binary-sha', 'INPUT_ARTIFACT'),
    ('device', 'GPU_UNVERIFIED'), ('stale', 'GPU_UNVERIFIED'), ('shape', 'PANEL_SHAPE'),
    ('checkpoint-metadata', 'CHECKPOINT_IDENTITY'), ('checkpoint', 'CHECKPOINT_INVALID'),
    ('forward', 'FORWARD_FAILED'), ('', 'COMPLETE'),
])
def test_saved_summary_and_no_unadmitted_forward(
    fault: str, reason: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / 'inputs'
    assert Recipe().tasks is TASKS
    before_tasks = tuple(TASKS)
    before = tuple(vars(task).copy() for task in before_tasks)
    e, trace = _fixture(root, monkeypatch, fault)
    original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    output = tmp_path / 'output'
    result, _ = e.run(root, output)
    data = output / 'data' / e.OUTPUT_ID / 'run.json'
    saved = json.loads(data.read_bytes())
    assert saved['summary']['reason'] == reason
    assert result['scientific_status'] == saved['status']
    assert result['scientific_reason'] == reason
    assert saved['artifacts'] == {} and not (output / 'models').exists()
    reports = output / 'result' / e.OUTPUT_ID
    expected_paths = {data, reports / 'report.md', reports / 'index.html', reports / 'source.json'}
    assert {p for p in output.rglob('*') if p.is_file()} == expected_paths
    assert json.loads((reports / 'source.json').read_bytes())['run_sha256'] == sha256(
        data.read_bytes()).hexdigest()
    assert original == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    counts = saved['counts']
    assert all(counts[f'{state}_{key}'] == 0 for state in ('attempted', 'completed')
               for key in ('real_blocks', 'null_words', 'optimizer_updates'))
    assert saved['policy']['forward_rows'] == 524288
    assert saved['recipe']['chunk_groups'] == 512 and saved['recipe']['retries'] == 0
    if fault not in ('', 'forward'):
        assert counts['attempted_forward_rows'] == counts['completed_forward_rows'] == 0
        assert not set(e.SLOTS) & set(trace)
        if fault == 'stale':
            assert 'restore' not in trace
            assert saved['summary']['detail_code'] == 'GPU_PROOF_STALE'
        if fault not in ('shape', 'checkpoint-metadata', 'checkpoint', 'device', 'stale'):
            assert 'gpu' not in trace
    else:
        assert trace[-2:] == list(e.SLOTS)
        assert counts['attempted_forward_rows'] == 1024
        assert counts['completed_forward_rows'] == (512 if fault else 1024)
    if fault:
        assert 'scores' not in saved['summary']
        assert result['exit_code'] == (3 if fault in ('device', 'stale') else 4)
    else:
        assert result['exit_code'] == 0
        assert saved['summary']['scores']['M-best']['accuracy'] == 1
        assert saved['summary']['m_minus_d_pp'] == 0
        assert '50.6466' in (reports / 'report.md').read_text(encoding='utf-8')
        assert '63.5044' in (reports / 'report.md').read_text(encoding='utf-8')
        html = (reports / 'index.html').read_text(encoding='utf-8')
        assert '<svg' in html and 'aria-label="保存的单对与四对准确率"' in html
    assert 'must not expose' not in data.read_text(encoding='utf-8')
    assert Recipe().tasks is TASKS
    assert all(actual is expected for actual, expected in zip(TASKS, before_tasks, strict=True))
    assert tuple(vars(task) for task in TASKS) == before


def test_new_output_id_coexists_with_sealed_old_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    e, _ = _fixture(tmp_path / 'inputs', monkeypatch, '')
    assert e.OUTPUT_ID == 'innovation1-skinny-single-pair-20261119-02'
    old = tmp_path / 'output' / 'data' / 'innovation1-skinny-single-pair-20261119-01'
    old.mkdir(parents=True)
    sealed = json.dumps({'schema': 2, 'run_id': old.name, 'status': 'FAILED'}).encode()
    (old / 'run.json').write_bytes(sealed)
    reports = tmp_path / 'output' / 'result' / old.name
    reports.mkdir(parents=True)
    (reports / 'report.md').write_bytes(b'sealed old failure')
    result, _ = e.run(tmp_path / 'inputs', tmp_path / 'output')
    assert result['scientific_status'] == 'SUCCEEDED' and result['exit_code'] == 0
    assert (old / 'run.json').read_bytes() == sealed
    assert (reports / 'report.md').read_bytes() == b'sealed old failure'
    new = tmp_path / 'output' / 'data' / e.OUTPUT_ID / 'run.json'
    assert json.loads(new.read_bytes())['status'] == 'SUCCEEDED'
    assert (tmp_path / 'output' / 'result' / e.OUTPUT_ID / 'report.md').is_file()


def test_source_closure_adds_exact_nested_bytes_only() -> None:
    from cipher_lab_plugins.innovation1.single_pair import evaluate as e

    new_src, old_src = e.source_identity(), old_source()
    assert old_src == {k: v for k, v in new_src.items() if k in old_src}
    nested = Path(e.__file__).parent
    assert {k: v for k, v in new_src.items() if k not in old_src} == {
        p.relative_to(nested.parents[2]).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in nested.rglob('*.py')
    }


def test_help_and_report_stdlib_only(monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    import sys

    from cipher_lab_plugins.innovation1.single_pair import cli, report

    monkeypatch.setattr(cli, 'sys', SimpleNamespace(platform='win32', dont_write_bytecode=False))
    original_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split('.')[0] not in ('torch', 'numpy', 'scipy')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    monkeypatch.setattr(sys, 'argv', ['cipher_lab', 'evaluate', '--plugin',
        'cipher_lab_plugins.innovation1.single_pair.cli', '--help'])
    with pytest.raises(SystemExit) as result:
        runpy.run_module('cipher_lab', run_name='__main__')
    assert result.value.code == 0 and '--run-root' in capsys.readouterr().out
    files = report.render(dict(run_id='new', run_ref='data/new/run.json', run_sha256='0' * 64,
        status='FAILED', counts={}, summary={'reason': 'FAULT <unsafe>'}))
    assert b'&lt;unsafe&gt;' in files['index.html']
    assert '<svg' in files['index.html'].decode()
    assert '保存的单对与四对准确率' not in files['index.html'].decode()
    assert b'FAULT <unsafe>' in files['report.md']
    assert 'stdout' in files['report.md'].decode() and 'flush' in files['report.md'].decode()


def test_reparse_parent_and_explicit_artifact_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from cipher_lab_plugins.innovation1.single_pair import evaluate as e

    lstat = Path.lstat
    parent = tmp_path / 'data'
    parent.mkdir()
    target = parent / 'selected.npz'
    target.write_bytes(b'fake')
    flagged = parent

    def pretend_reparse(path: Path) -> Any:
        if path == flagged:
            return SimpleNamespace(st_file_attributes=0x400)
        if flagged == parent and path == target:
            pytest.fail('must reject reparse parent before touching descendant')
        return lstat(path)

    monkeypatch.setattr(Path, 'lstat', pretend_reparse)
    with pytest.raises(ValueError, match='UNSAFE_PATH'):
        e.no_reparse(target)
    flagged = target
    with pytest.raises(ValueError, match='UNSAFE_PATH'):
        e.no_reparse(target)


@pytest.mark.parametrize('resource_stop', [False, True])
def test_report_closeout_does_not_refinalize(
    resource_stop: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from cipher_lab_plugins.innovation1.single_pair import report

    e, _ = _fixture(tmp_path / 'inputs', monkeypatch, '')
    original_finalize = artifacts.RunStore.finalize
    finalized: list[str] = []
    renders: list[str] = []
    expired = False

    def rss() -> int:
        if expired:
            raise Stopped('RSS_UNAVAILABLE')
        return 100

    def finalize(store: artifacts.RunStore, value: Any) -> None:
        finalized.append(value['status'])
        original_finalize(store, value)

    def render(saved: Any) -> dict[str, bytes]:
        nonlocal expired
        renders.append(saved['run_sha256'])
        expired = resource_stop
        raise ValueError('injected report failure')

    monkeypatch.setattr(e, 'production_rss_bytes', rss)
    monkeypatch.setattr(artifacts.RunStore, 'finalize', finalize)
    monkeypatch.setattr(report, 'render', render)
    result, _ = e.run(tmp_path / 'inputs', tmp_path / 'output')
    assert finalized == ['SUCCEEDED'] and len(renders) == 1
    assert result['scientific_status'] == 'SUCCEEDED'
    assert result['command_status'] == ('STOPPED' if resource_stop else 'FAILED')
    assert result['report_reason'] == 'REPORT_FAILED'
    assert result['exit_code'] == (3 if resource_stop else 6)
    assert not list((tmp_path / 'output' / 'result' / e.OUTPUT_ID).iterdir())


def test_restore_with_original_inference_only_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1 import models as old_models
    from cipher_lab_plugins.innovation1.single_pair import models

    recipe, task = Recipe(), Recipe().tasks[-1]
    expected = {'M-best': {'format': 'M'}, 'D-SKINNY-best': {'format': 'D'}}
    trace = []

    class Model:
        def requires_grad_(self, enabled: bool) -> None:
            assert enabled is False
            trace.append('frozen')

    class Runtime:
        def __init__(self, r: Recipe, device: str, *, inference_only: bool) -> None:
            assert r is recipe and device == 'cuda:0' and inference_only is True
            trace.append('construct-empty')
            self.models: dict[str, Model] = {}

        def restore(self, slot: str, raw: bytes, metadata: object) -> None:
            assert metadata is expected[slot] and raw == slot.encode()
            trace.append(slot)
            self.models[slot] = Model()

    def adapter(runtime: Runtime, selected: object) -> object:
        assert selected is task and tuple(runtime.models) == tuple(expected)
        trace.append('adapter')
        return object()

    monkeypatch.setattr(old_models, 'TorchRuntime', Runtime)
    monkeypatch.setattr(models, 'SinglePairRuntime', adapter)
    models.restore(recipe, task, {s: s.encode() for s in expected}, expected,
                   lambda: trace.append('check'))
    assert trace == ['construct-empty', 'check', 'M-best', 'frozen', 'check',
                     'check', 'D-SKINNY-best', 'frozen', 'check', 'adapter']



def test_failed_finalize_does_not_retry_and_reports_partial_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    e, _ = _fixture(tmp_path / 'inputs', monkeypatch, 'forward')
    attempts = []

    def fail(_store: artifacts.RunStore, value: Any) -> None:
        attempts.append(value['status'])
        raise artifacts.ArtifactError('WRITE_FAILED')

    monkeypatch.setattr(artifacts.RunStore, 'finalize', fail)
    result, _ = e.run(tmp_path / 'inputs', tmp_path / 'output')
    assert attempts == ['FAILED'] and result['scientific_status'] == 'UNKNOWN'
    assert result['scientific_reason'] is None
    assert result['attempted_terminal_reason'] == 'FORWARD_FAILED'
    assert result['report_status'] == 'FAILED' and result['report_reason'] == 'FINALIZE_FAILED'
    assert result['command_resources']['attempted_forward_rows'] == 1024
    assert result['command_resources']['completed_forward_rows'] == 512
    assert result['forward_rows_by_slot']['D-SKINNY-best'] == {'attempted': 512, 'completed': 0}


def test_missing_nested_source_cannot_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from cipher_lab_plugins.innovation1.single_pair import evaluate as e

    nested = Path(e.__file__).parent
    walk = os.walk

    def incomplete(top: Path, **kw: Any) -> Any:
        for path, dirs, files in walk(top, **kw):
            yield path, dirs, [name for name in files
                               if Path(path) != nested or name != 'report.py']

    monkeypatch.setattr(os, 'walk', incomplete)
    monkeypatch.setattr(e, 'production_rss_bytes', lambda: 100)
    monkeypatch.setattr(e, 'read_binary', _forbidden)
    result, _ = e.run(tmp_path / 'input', tmp_path / 'output')
    assert result['closeout_reason'] == 'CLAIM_FAILED'
    assert not (tmp_path / 'output').exists()



def test_cli_refuses_bytecode_enabled_before_evaluator_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from cipher_lab_plugins.innovation1.single_pair import cli

    original_import = builtins.__import__

    def block_evaluation(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name != 'evaluate', 'bytecode refusal must precede evaluator import'
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(cli, 'sys', SimpleNamespace(platform='win32', dont_write_bytecode=False))
    monkeypatch.setattr(cli, 'Path', PureWindowsPath)
    monkeypatch.setattr(builtins, '__import__', block_evaluation)
    with pytest.raises(SystemExit) as error:
        cli.main(['evaluate', '--run-root', r'G:\lxy\input', '--root', r'G:\lxy\output'])
    assert error.value.code == 2
    out = capsys.readouterr()
    assert out.out == '' and 'python -B' in out.err


@pytest.mark.parametrize('expires_after_serialization', [False, True])
def test_cli_pre_stdout_gate_and_flush_are_not_native_exit(
    expires_after_serialization: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from cipher_lab_plugins.innovation1.single_pair import cli

    input_root, output_root = tmp_path / 'input', tmp_path / 'output'
    e, _ = _fixture(input_root, monkeypatch, '')
    original_run = e.run
    trace: list[str] = []
    saved_bytes: bytes | None = None
    output = output_root / 'data' / e.OUTPUT_ID / 'run.json'
    printed: dict[str, Any] = {}

    def run(inp: PureWindowsPath, out: PureWindowsPath) -> Any:
        assert inp == PureWindowsPath(r'G:\lxy\input') and out == PureWindowsPath(r'G:\lxy\output')
        return original_run(input_root, output_root)

    def fake_rss() -> int:
        if trace == ['serialize']:
            trace.append('sample')
            return e.RSS + 1 if expires_after_serialization else 200
        return 100

    def serialize(result: Any, **kwargs: Any) -> str:
        nonlocal saved_bytes
        if saved_bytes is None:
            saved_bytes = output.read_bytes()
        trace.append('serialize')
        return json.dumps(result, **kwargs)

    def write(line: str, *, flush: bool = False) -> None:
        trace.append('write')
        assert flush is True
        printed.update(json.loads(line))

    monkeypatch.setattr(cli, 'sys', SimpleNamespace(platform='win32', dont_write_bytecode=True))
    monkeypatch.setattr(cli, 'Path', PureWindowsPath)
    monkeypatch.setattr(cli, 'json', SimpleNamespace(dumps=serialize))
    monkeypatch.setattr(cli, 'print', write, raising=False)
    monkeypatch.setattr(e, 'production_rss_bytes', fake_rss)
    monkeypatch.setattr(e, 'run', run)
    code = cli.main(['evaluate', '--run-root', r'G:\lxy\input', '--root', r'G:\lxy\output'])
    assert trace == ['serialize', 'sample', 'serialize', 'write']
    assert output.read_bytes() == saved_bytes
    assert printed['command_resources_scope'] == 'PRE_STDOUT_SAMPLE'
    assert printed['exit_code_scope'] == 'REQUESTED_NOT_NATIVE'
    assert printed['scientific_status'] == 'SUCCEEDED'
    assert printed['report_status'] == 'SAVED'
    assert printed['exit_code'] == code == (3 if expires_after_serialization else 0)
    if expires_after_serialization:
        assert printed['command_status'] == 'STOPPED' and printed['closeout_reason'] == 'RSS_LIMIT'
    else:
        assert printed['command_status'] == 'SUCCEEDED'
        assert printed['command_resources']['rss_bytes'] == 200
