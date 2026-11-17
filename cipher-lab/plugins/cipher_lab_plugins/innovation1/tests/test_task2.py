"""Artificial-only contracts for orchestration; real science remains poisoned."""
from __future__ import annotations

import builtins
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1.config import Recipe, Statistics, Training

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved


def small() -> Recipe:
    return replace(Recipe(), training=Training(epochs=2, batches_per_epoch=1, batch_groups=2,
                   dev_groups=2, test_groups=2, eval_batch_groups=2),
                   statistics=Statistics(replicates=20))


def owner(tmp_path: Path, recipe: Recipe) -> Any:
    from cipher_lab_plugins.innovation1.checkpoint import SLOTS
    from cipher_lab_plugins.innovation1.source import source_identity
    return RunStore.claim(tmp_path, 'manual', output_bytes=recipe.resources.artifact_bytes,
                          needs_models=True, model_slots=SLOTS, source_supplier=source_identity)


def setup(tmp_path: Path) -> tuple[Any, ...]:
    from cipher_lab_plugins.innovation1.resources import manual_resources
    from cipher_lab_plugins.innovation1.synthetic import ManualProvider, ManualRuntime
    r = small()
    resources = manual_resources(r)
    store = owner(tmp_path, r)
    provider = ManualProvider(r, resources)
    runtime = ManualRuntime(r)
    return r, provider, runtime, store, resources


def test_manual_full_same_batch_best_disk_late_test(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    result = run(r, p, m, s, resources)
    assert result['status'] == 'SUCCEEDED'
    saved: Any = read_saved(tmp_path, 'manual')
    assert saved['summary']['synthetic'] is True
    assert saved['counts']['completed_optimizer_updates'] == 16
    assert saved['counts']['completed_forward_rows'] == 80
    assert saved['counts']['completed_real_blocks'] == 256  # SIMULATED
    assert len(m.restored) == 5 and all(e['epoch'] == 1 for e in m.restored)
    assert len(p.train_calls) == 8 and len(p.test_calls) == 4
    assert all(len({x[0] for x in pair}) == 1 for pair in m.consumed.values())
    assert all(len({x[1] for x in pair}) == 1 for pair in m.consumed.values())
    batches = [v for k, v in saved['payloads'].items() if 'batches' in k]
    assert sum(len(b) for b in batches) == 8
    assert len(saved['summary']['compare']) == 4
    assert len(saved['summary']['curves']) == 2


@pytest.mark.parametrize('failure', ['dedicated', 'mutation', 'disk', 'wrong-sha'])
def test_failure_keeps_partial_and_never_builds_test(tmp_path: Path, failure: str,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    if failure == 'dedicated':
        original = m.update
        def update(slot: str, *a: Any, **kw: Any) -> float:
            if slot != 'M-best':
                raise RuntimeError('private error must never be saved')
            return float(original(slot, *a, **kw))
        monkeypatch.setattr(m, 'update', update)
    elif failure == 'mutation':
        original = m.update
        def mutate(slot: str, task: Any, x: Any, y: Any, *a: Any) -> float:
            value = original(slot, task, x, y, *a)
            if slot == 'M-best':
                x[0, 0, 0, 0] = 1 - x[0, 0, 0, 0]
            return float(value)
        monkeypatch.setattr(m, 'update', mutate)
    else:
        original_read = s.read_model
        def bad_read(ref: str, digest: str, maximum: int) -> bytes:
            if failure == 'disk':
                (tmp_path / 'models/manual/M-best.pt').write_bytes(b'tampered')
            else:
                digest = '0' * 64
            return bytes(original_read(ref, digest, maximum))
        monkeypatch.setattr(s, 'read_model', bad_read)
    result = run(r, p, m, s, resources)
    assert result['status'] == 'FAILED'
    assert not p.test_calls
    if failure == 'dedicated':
        saved: Any = read_saved(tmp_path, 'manual')
        assert saved['counts']['attempted_optimizer_updates'] == 2
        assert saved['counts']['completed_optimizer_updates'] == 1
        assert saved['summary']['pending'][0]['shared_completed']
        assert not saved['summary']['pending'][0]['dedicated_completed']
        assert 'private error' not in json.dumps(saved)


def test_checkpoint_strict_identity_state_and_weights_only() -> None:
    from cipher_lab_plugins.innovation1.checkpoint import decode, encode, identity
    r = small()
    meta = identity(r, {'fake.py': 'a' * 64}, 'demo', 'M-best', 1, 2.0)
    state = {'weight': torch.tensor([1.], dtype=torch.float32),
             'num_batches_tracked': torch.tensor(2, dtype=torch.int64)}
    raw = encode(meta, state)
    assert torch.equal(decode(raw, meta, state)['weight'], state['weight'])
    for field, value in [('synthetic', 1), ('schema', True), ('family', 'dedicated'),
                         ('backend', 'ciphers'), ('epoch', True)]:
        with pytest.raises(ValueError):
            decode(encode({**meta, field: value}, state), meta, state)
    for bad in ({'weight': torch.ones(2)}, {'weight': torch.ones(1).double()},
                {**state, 'extra': torch.zeros(1)}):
        with pytest.raises(ValueError):
            decode(encode(meta, bad), meta, state)
    with pytest.raises(ValueError):
        decode(b'pickle!', meta, state)


def test_dataset_exact_roundtrip_identity_and_corruption(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.dataset import decode_panel, encode_panel
    r, p, _, _, _ = setup(tmp_path)
    batch = p.dev(r.tasks[0])
    raw = encode_panel(batch, r, {'fake.py': 'b' * 64}, 'demo')
    restored = decode_panel(raw, r, r.tasks[0], 'dev', {'fake.py': 'b' * 64}, 'demo')
    assert restored.digest == batch.digest
    with pytest.raises(ValueError):
        decode_panel(raw, r, r.tasks[1], 'dev', {'fake.py': 'b' * 64}, 'demo')
    with pytest.raises(ValueError):
        decode_panel(raw, r, r.tasks[0], 'test', {'fake.py': 'b' * 64}, 'demo')
    with pytest.raises(ValueError):
        decode_panel(b'bad', r, r.tasks[0], 'dev', {}, 'ciphers')


def test_report_saved_only_escaped_and_store_owned(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1.report import render
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    run(r, p, m, s, resources)
    saved: Any = read_saved(tmp_path, 'manual')
    saved['summary']['reason'] = '<script>unsafe</script>'
    def forbidden(*a: Any, **kw: Any) -> Any:
        raise AssertionError('render must not compute/read')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    files = render(saved)
    assert set(files) == {'report.md', 'index.html', 'source.json'}
    assert b'<svg' in files['index.html'] and b'<script>' not in files['index.html']
    s.write_report_files(files)


def test_source_includes_core_plugin_parents_not_other_owners() -> None:
    from cipher_lab_plugins.innovation1.source import source_identity
    value = source_identity()
    assert 'cipher_lab/models/dedicated/feistel.py' in value
    assert 'cipher_lab_plugins/__init__.py' in value
    assert 'cipher_lab_plugins/innovation1/training.py' in value
    assert not any('/tests/' in k or 'speck_study' in k for k in value)


def test_resources_cached_gpu_missing_stale_and_limits(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.resources import GpuGate, ResourceStop, manual_resources

    from cipher_lab.execution import Stopped
    now = [0.0]
    calls: list[int] = []
    def sample() -> dict[str, Any]:
        calls.append(1)
        return {'uuid': 'GPU-manual', 'pci': '0000:01:00.0', 'bytes': 2}
    gate = GpuGate('GPU-manual', '0000:01:00.0', 20, sample, lambda: now[0])
    for _ in range(100):
        gate.check()
    assert len(calls) == 1
    now[0] = 1.0
    gate.check()
    assert len(calls) == 2
    now[0] = 9.
    with pytest.raises(ResourceStop):
        gate.check()
    resources = manual_resources(small())
    with pytest.raises(Stopped):
        resources.reserve('optimizer_updates', 17)
    assert resources.snapshot()['attempted_optimizer_updates'] == 0


def test_evaluate_saved_only_and_sha_failure(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.evaluate import evaluate
    from cipher_lab_plugins.innovation1.resources import manual_resources
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    run(r, p, m, s, resources)
    saved: Any = read_saved(tmp_path, 'manual')
    cp = 'model/M-best.pt'
    ds = 'dataset/test-SIMECK.npz'
    cp_sha = saved['artifacts'][cp]['sha256']
    ds_sha = saved['artifacts'][ds]['sha256']
    from cipher_lab_plugins.innovation1.source import source_identity
    out = RunStore.claim(tmp_path, 'evaluation', output_bytes=10**7,
                         source_supplier=source_identity)
    result = evaluate(tmp_path, 'manual', saved['run_sha256'], cp, cp_sha, ds, ds_sha,
                      out, manual_resources(r), backend='demo')
    assert result['status'] == 'SUCCEEDED'
    assert result['counts']['completed_real_blocks'] == 0
    assert result['counts']['completed_optimizer_updates'] == 0
    assert result['counts']['completed_forward_rows'] == 2
    out2 = RunStore.claim(tmp_path, 'bad-eval', output_bytes=10**7)
    with pytest.raises((ValueError, ArtifactError)):
        evaluate(tmp_path, 'manual', saved['run_sha256'], cp, '0'*64, ds, ds_sha,
                 out2, manual_resources(r), backend='demo')


def test_different_dev_bests_and_real_replacement_before_test(tmp_path: Path,
                                                           monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    def logits(slot: str, task: Any, x: Any) -> Any:
        epoch = int(m.states[slot]['manual_steps']) // (4 if slot == 'M-best' else 1)
        # M macro improves; SIMECK D worsens; other Ds improve.
        return np.full((x.shape[0], 1), (3 - epoch) if slot != 'D-SIMECK-best' else epoch)
    monkeypatch.setattr(m, 'predict', logits)
    result = run(r, p, m, s, resources)
    assert result['status'] == 'SUCCEEDED'
    assert result['summary']['best']['M-best']['metadata']['epoch'] == 2
    assert result['summary']['best']['D-SIMECK-best']['metadata']['epoch'] == 1
    assert result['summary']['best']['D-PRESENT-best']['metadata']['epoch'] == 2
    assert len(m.restored) == 5


def test_fifth_reload_failure_and_source_change_stop_before_test(tmp_path: Path,
                                                               monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    original = m.restore
    def restore(slot: str, raw: bytes, metadata: Any) -> None:
        if slot == 'D-SKINNY-best':
            raise ValueError('strict buffer failed')
        original(slot, raw, metadata)
    monkeypatch.setattr(m, 'restore', restore)
    result = run(r, p, m, s, resources)
    assert result['status'] == 'FAILED' and len(m.restored) == 4 and not p.test_calls


def test_changed_source_identity_is_stop_no_sampling(tmp_path: Path, monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    monkeypatch.setattr(s, 'current_source_identity', lambda: {'changed.py': '0' * 64})
    result = run(r, p, m, s, resources)
    assert result['status'] == 'STOPPED'
    assert result['summary']['reason'] == 'SOURCE_CHANGED'
    assert not p.train_calls and not p.panels and not p.test_calls


def test_resource_failure_after_shared_keeps_completed_step(tmp_path: Path,
                                                          monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run

    from cipher_lab.execution import Stopped
    r, p, m, s, resources = setup(tmp_path)
    original = m.update
    def update(*args: Any) -> float:
        result = original(*args)
        monkeypatch.setattr(resources.meter, '_rss_bytes', lambda: 10**30)
        return float(result)
    monkeypatch.setattr(m, 'update', update)
    result = run(r, p, m, s, resources)
    assert result['status'] == 'STOPPED'
    assert result['summary']['reason'] == 'RSS_LIMIT'
    assert result['counts']['completed_optimizer_updates'] == 1
    assert result['counts']['attempted_optimizer_updates'] == 1
    assert not p.test_calls
    with pytest.raises(Stopped):
        resources.check()


def test_checkpoint_rejects_bool_int_even_if_expected_mutated() -> None:
    from cipher_lab_plugins.innovation1.checkpoint import decode, encode, identity
    meta = identity(small(), {'fake.py': 'a'*64}, 'demo', 'M-best', 1, 0.5)
    state = {'manual_steps': torch.tensor(1, dtype=torch.int64)}
    # Caller-provided expected metadata must itself be a valid schema, not just equal bytes.
    mutated = {**meta, 'schema': True, 'epoch': True, 'synthetic': 1}
    with pytest.raises(ValueError):
        decode(encode(mutated, state), mutated, state)


@pytest.mark.parametrize('bad', ['missing', 'over', 'nan-clock', 'slow', 'wrong-pci'])
def test_gpu_fail_closed_public_artificial_samples(bad: str) -> None:
    from cipher_lab_plugins.innovation1.resources import GpuGate, ResourceStop
    now = [0.0]
    def sample() -> dict[str, Any]:
        if bad == 'slow':
            now[0] = 2.
        return dict(uuid='GPU-manual', pci='bad' if bad == 'wrong-pci' else '0000:01:00.0',
                    bytes=None if bad == 'missing' else 21 if bad == 'over' else 1)
    if bad == 'nan-clock':
        now[0] = float('nan')
    gate = GpuGate('GPU-manual', '0000:01:00.0', 20, sample, lambda: now[0])
    with pytest.raises(ResourceStop):
        gate.check()
    with pytest.raises(ResourceStop):
        gate.check()  # failure irreversible; no retry to renew evidence


def test_disk_only_read_has_no_content_cache_and_no_finalize_retry(tmp_path: Path,
                                                                 monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    reads: list[str] = []
    original_read = s.read_model
    def read(ref: str, digest: str, maximum: int) -> bytes:
        reads.append(ref)
        return bytes(original_read(ref, digest, maximum))
    monkeypatch.setattr(s, 'read_model', read)
    assert run(r, p, m, s, resources)['status'] == 'SUCCEEDED'
    assert reads == [f'model/{slot}.pt' for slot in (
        'M-best', 'D-SIMECK-best', 'D-PRESENT-best', 'D-SIMON-best', 'D-SKINNY-best')]


def test_output_budget_seal_is_not_overwritten(tmp_path: Path, monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    # Test-owned failing disk budget, not a production resource probe.
    original_write = s.write_model
    calls: list[Any] = []
    original_finalize = s.finalize
    def write(*a: Any, **k: Any) -> Any:
        s._limit = s._total() + 1
        return original_write(*a, **k)
    def finalize(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return original_finalize(*a, **k)
    monkeypatch.setattr(s, 'write_model', write)
    monkeypatch.setattr(s, 'finalize', finalize)
    result = run(r, p, m, s, resources)
    assert result['reason'] == 'ARTIFACT_FAILED'
    assert not calls and not p.test_calls


def test_evaluate_does_not_relabel_unverified_or_wrong_family(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.evaluate import evaluate
    from cipher_lab_plugins.innovation1.resources import manual_resources
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    run(r, p, m, s, resources)
    saved: Any = read_saved(tmp_path, 'manual')
    cp = 'model/D-PRESENT-best.pt'
    ds = 'dataset/test-SIMECK.npz'
    for index, backend in enumerate(('UNVERIFIED', 'demo', 'ciphers')):
        out = RunStore.claim(tmp_path, f'reject-{index}', output_bytes=10**7)
        with pytest.raises(ValueError):
            evaluate(tmp_path, 'manual', saved['run_sha256'], cp, saved['artifacts'][cp]['sha256'],
                     ds, saved['artifacts'][ds]['sha256'], out, manual_resources(r),
                     backend=backend)


def test_model_metadata_and_saved_report_include_parameter_evidence(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.report import render
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    result = run(r, p, m, s, resources)
    assert result['summary']['models']['M-best']['parameters'] == 0
    assert result['summary']['models']['M-best']['manual_state_only'] is True
    saved: Any = read_saved(tmp_path, 'manual')
    assert b'parameters' in render(saved)['report.md']


def test_evaluate_rejects_changed_current_source_before_forward(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.evaluate import evaluate
    from cipher_lab_plugins.innovation1.resources import manual_resources
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    run(r, p, m, s, resources)
    saved: Any = read_saved(tmp_path, 'manual')
    cp, ds = 'model/M-best.pt', 'dataset/test-SIMECK.npz'
    out = RunStore.claim(tmp_path, 'different-source', output_bytes=10**7,
                         source_supplier=lambda: {'manual/changed.py': 'a'*64})
    with pytest.raises(ValueError):
        evaluate(tmp_path, 'manual', saved['run_sha256'], cp, saved['artifacts'][cp]['sha256'],
                 ds, saved['artifacts'][ds]['sha256'], out, manual_resources(r), backend='demo')


def test_evaluate_explicit_forward_budget_and_batch(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.evaluate import evaluate
    from cipher_lab_plugins.innovation1.resources import manual_resources
    from cipher_lab_plugins.innovation1.source import source_identity
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    run(r, p, m, s, resources)
    saved: Any = read_saved(tmp_path, 'manual')
    cp, ds = 'model/M-best.pt', 'dataset/test-SIMECK.npz'
    out = RunStore.claim(tmp_path, 'limited-eval', output_bytes=10**7,
                         source_supplier=source_identity)
    with pytest.raises(ValueError, match='EVALUATION_BUDGET'):
        evaluate(tmp_path, 'manual', saved['run_sha256'], cp, saved['artifacts'][cp]['sha256'],
                 ds, saved['artifacts'][ds]['sha256'], out, manual_resources(r), backend='demo',
                 max_forward_rows=1, batch_groups=2)


def test_no_production_resources_or_gpu_attestation_escape_guard() -> None:
    from cipher_lab_plugins.innovation1.device import actual_attestor
    from cipher_lab_plugins.innovation1.resources import gpu_sample, production_rss_bytes
    for probe in (lambda: gpu_sample('GPU-manual'), production_rss_bytes, actual_attestor):
        with pytest.raises(AssertionError, match='UNAUTHORIZED_SCIENCE'):
            probe()


def test_best_storage_failure_preserves_preceding_update_counts(tmp_path: Path,
                                                              monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, p, m, s, resources = setup(tmp_path)
    original = s.write_model
    def fail(*args: Any, **kwargs: Any) -> Any:
        s._failed('WRITE_FAILED')  # artificial disk failure using original owner's seal
        raise ArtifactError('WRITE_FAILED')
    monkeypatch.setattr(s, 'write_model', fail)
    result = run(r, p, m, s, resources)
    assert result['status'] == 'FAILED'
    saved: Any = read_saved(tmp_path, 'manual')
    assert saved['counts']['completed_optimizer_updates'] == 8
    assert saved['counts']['completed_forward_rows'] == 32
    assert not p.test_calls
    assert callable(original)
