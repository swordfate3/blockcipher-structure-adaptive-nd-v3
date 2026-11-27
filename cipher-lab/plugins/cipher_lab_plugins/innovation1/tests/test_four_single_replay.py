"""Exercise the replay owner with only tiny, artificial, byte-pinned panels."""
from __future__ import annotations

import builtins
import io
import json
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pytest
from cipher_lab_plugins.innovation1.config import Task, canonical, load_config
from cipher_lab_plugins.innovation1.four_single import config, training
from cipher_lab_plugins.innovation1.four_single.data import encode_panel
from cipher_lab_plugins.innovation1.layout_probe.config import FilePin
from cipher_lab_plugins.innovation1.resources import Resources
from cipher_lab_plugins.innovation1.single_train.config import BACKEND, learning_rate
from cipher_lab_plugins.innovation1.single_train.data import FixedPairPool, PairBatch, Provider
from cipher_lab_plugins.innovation1.tests.test_innovation1_fixed_bce import _fixture
from plugin_isolation import forbidden

from cipher_lab.artifacts import RunStore, _encode
from cipher_lab.execution import Clocks, Meter

CONFIG = Path('configs/innovation1-four-task-single-pair-50epoch.json')
GPU = 'GPU-00000000-0000-0000-0000-000000000000'
SOURCE = {'artificial/archive.py': 'a' * 64}


def test_strict_recipe_counts_and_original_lr(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.cli import main

    recipe = load_config(CONFIG)
    assert recipe == config.FOUR_SINGLE_50 and isinstance(recipe, config.ReplayRecipe)
    old = config.FOUR_SINGLE
    n = config.estimate(recipe)
    assert (n['optimizer_updates'], n['forward_rows'], n['real_blocks']) == (
        360_000, 2_001_048_576, 0)
    assert n['panel_bound_bytes'] <= 31 * 2**20
    assert n['artifact_peak_bound_bytes'] < 2**30
    assert config.make_policy(recipe, Clocks()).wall_seconds == 28 * 3600
    assert recipe.resources == replace(old.resources, wall_seconds=28 * 3600)
    assert recipe.streams == old.streams
    for k in ('model', 'seeds', 'tasks'):
        assert recipe.to_dict()[k] == old.to_dict()[k]
    assert all(learning_rate(recipe, c, cosine_epochs=recipe.cosine_epochs)
               == learning_rate(old, c) for c in range(18000))
    assert [learning_rate(recipe, c, cosine_epochs=recipe.cosine_epochs)
            for c in (0, 17999, 18000, 89999)] == [.001, .0001, .0001, .0001]
    assert main(['check', '--config', str(CONFIG)]) == 0
    assert {p.name for p in Path('configs').glob('innovation1-*.json') if p != CONFIG} == {
        'innovation1-fresh.json', 'innovation1-four-task-single-pair.json',
        'innovation1-skinny-layout-early-stop.json', 'innovation1-skinny-layout-fixed-bce.json',
        'innovation1-skinny-layout-fixed-mse.json', 'innovation1-present-prefix2-bridge.json',
        'innovation1-present-s2-window.json', 'innovation1-present-mask-aux.json',
        'innovation1-present-single-pair-mask-10m.json',
        'innovation1-present-card-fresh-1m.json', 'innovation1-present-card-mask-1m.json',
        'innovation1-present-card-mask-positions.json', 'innovation1-present-card-swap.json',
        'innovation1-present-card-swap-mask.json', 'innovation1-present-card-dual-train.json'}
    for path in Path('configs').glob('innovation1-*.json'):
        if path.name == 'innovation1-present-prefix2-bridge.json':
            from cipher_lab_plugins.innovation1.prefix_bridge.cli import main as bridge_main

            assert bridge_main(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-s2-window.json':
            from cipher_lab_plugins.innovation1.s2_window.cli import main as window_main

            assert window_main(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-mask-aux.json':
            from cipher_lab_plugins.innovation1.mask_aux.cli import main as mask_main

            assert mask_main(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-single-pair-mask-10m.json':
            from cipher_lab_plugins.innovation1.mask_aux_single.cli import main as pair_main

            assert pair_main(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-card-fresh-1m.json':
            from cipher_lab_plugins.innovation1.present_card_fresh.cli import main as card_main

            assert card_main(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-card-mask-1m.json':
            from cipher_lab_plugins.innovation1.present_card_mask.cli import main as card_mask_main

            assert card_mask_main(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-card-mask-positions.json':
            from cipher_lab_plugins.innovation1.present_card_positions.cli import main as positions

            assert positions(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-card-swap.json':
            from cipher_lab_plugins.innovation1.present_card_swap.cli import main as swap_main

            assert swap_main(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-card-swap-mask.json':
            from cipher_lab_plugins.innovation1.present_card_swap_mask.cli import main as swap_mask

            assert swap_mask(['check', '--config', str(path)]) == 0
        elif path.name == 'innovation1-present-card-dual-train.json':
            from cipher_lab_plugins.innovation1.present_card_dual_train.cli import main as dual

            assert dual(['check', '--config', str(path)]) == 0
        else:
            load_config(path)
    value = recipe.to_dict()
    value['selection']['early_stopping']['patience'] = 11
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='CONFIG_EXACT'):
        load_config(bad)


def test_cli_rejects_other_write_roots_and_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1 import cli, source

    monkeypatch.setattr(source, 'no_reparse', forbidden)
    config.validate_replay_root(PureWindowsPath(config.REPLAY_ROOT))
    recipe = config.FOUR_SINGLE_50
    assert recipe.skinny_source is not None
    for root, run_id in ((recipe.archive.root, recipe.run_id),
                         (recipe.skinny_source.root, recipe.run_id),
                         (recipe.archive.root + r'\child', recipe.run_id),
                         (r'G:\lxy\shared-m-cipher-nd-runs', recipe.run_id),
                         (config.REPLAY_ROOT + r'\..\elsewhere', recipe.run_id),
                         (config.REPLAY_ROOT, recipe.archive.run_id)):
        assert main_run(cli.main, root, run_id) == 2
        if root != config.REPLAY_ROOT:
            with pytest.raises(ValueError, match='REPLAY_OUTPUT_ROOT'):
                config.validate_replay_root(PureWindowsPath(root))


def main_run(main: Callable[[list[str]], int], root: str, run_id: str) -> int:
    return main(['run', '--config', str(CONFIG), '--root', root, '--id', run_id, '--gpu-uuid', GPU])


def test_replay_build_preserves_initial_state_and_rng() -> None:
    import torch
    from cipher_lab_plugins.innovation1.four_single.models import build, geometry

    rng = torch.random.get_rng_state()
    old, replay = build(config.FOUR_SINGLE), build(config.FOUR_SINGLE_50)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert geometry(old) == geometry(replay)
    torch.testing.assert_close(old.state_dict(), replay.state_dict(), rtol=0, atol=0)


def _batch(groups: int, split: str, namespace: str, seed: int) -> PairBatch:
    x = (np.arange(groups*16, dtype=np.uint16) + seed).astype(np.uint8).reshape(-1, 1, 2, 8)
    y = np.tile(np.array([0, 1], dtype=np.uint8), groups//2).reshape(-1, 1)
    return PairBatch(x, y, split, 0, namespace)


def _archive(tmp_path: Path, case: str) -> tuple[config.ReplayRecipe, dict[Path, bytes],
                                                dict[str, list[str]]]:
    from cipher_lab_plugins.innovation1.layout_probe.reuse import load_reused_pool

    skinny, prior, original = _fixture(tmp_path)
    old = replace(config.FOUR_SINGLE, run_id='archive-four', skinny_source=skinny.pool_source,
                  training=replace(skinny.training, epochs=10))
    recipe = replace(config.FOUR_SINGLE_50, run_id='tiny-replay', skinny_source=old.skinny_source,
                     training=replace(old.training, epochs=50))
    root = tmp_path / 'archive'
    record = root / 'data/archive-four/run.json'
    store = RunStore.claim(root, 'archive-four', output_bytes=2**20, needs_models=True,
                           model_slots=(config.SLOT,), source_supplier=lambda: SOURCE)
    replay = {'SKINNY': [prior.train(c).digest for c in range(11*9)]}
    _, _, provenance = load_reused_pool(skinny, lambda: None)
    best = dict(ref=store.write_model(config.SLOT, b'old model must never be opened'),
                sha256=sha256(b'old model must never be opened').hexdigest(), epoch=3,
                macro_dev_bce=.6, routes={})
    tests: dict[str, Any] = {}
    pools: dict[str, Any] = {'SKINNY': provenance}
    for i, task in enumerate(old.tasks):
        panels = {}
        for split, groups in (('train', 36), ('dev', 4), ('test', 4)):
            if task.cipher == 'SKINNY' and split != 'test':
                continue
            batch = _batch(groups, split, old.namespace, i)
            scored = split == 'test'
            raw = encode_panel(batch, task, old, SOURCE, BACKEND,
                               logits=np.full((groups, 1), 17., np.float64) if scored else None,
                               best=best if scored else None)
            if (split, task.cipher) in (('dev', 'SIMON'), ('test', 'PRESENT')):
                with np.load(io.BytesIO(raw)) as saved:
                    metadata = json.loads(saved['metadata'].tobytes())
                    arrays = {k: saved[k] for k in saved.files if k != 'metadata'}
                if (split, case) in (('dev', 'metadata'), ('test', 'test-best')):
                    if split == 'dev':
                        metadata['source'] = {'artificial/other.py': 'b' * 64}
                    else:
                        metadata['best']['epoch'] += 1
                if (split, case) == ('dev', 'digest'):
                    arrays['ciphertext'] = arrays['ciphertext'][::-1].copy()
                if scored:
                    # Even a non-loadable old logits member is immaterial to this replay.
                    arrays['logits'] = np.array([object()])
                buf = io.BytesIO()
                np.savez(buf, metadata=np.frombuffer(canonical(metadata).encode(), np.uint8),
                         **arrays)
                raw = buf.getvalue()
            ref = store.write_dataset(f'{split}-{task.cipher}', raw)
            path = record.parent / ref
            panels[split] = dict(ref=ref, sha256=sha256(raw).hexdigest(), digest=batch.digest,
                                origin='fresh', bytes=len(raw), groups=groups,
                                namespace=batch.namespace)
            if split == 'train':
                from types import SimpleNamespace

                previous = FixedPairPool(replace(old, training=recipe.training), forbidden,
                    forbidden, SimpleNamespace(check=lambda: None), streams=old.streams,
                    task=task, shuffle_namespace=old.namespace)
                previous._use_train(batch)
                previous._dev = _batch(4, 'dev', old.namespace, i)
                previous.panel('dev')
                previous.activate()
                replay[task.cipher] = [previous.train(c).digest for c in range(11*9)]
            if (split, task.cipher, case) in (('dev', 'SIMON', 'tamper'),
                                             ('test', 'PRESENT', 'test-tamper')):
                path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            if (split, task.cipher, case) in (('dev', 'SIMON', 'missing'),
                                             ('test', 'PRESENT', 'test-missing')):
                path.unlink()
            else:
                original[path] = path.read_bytes()
        tests[task.cipher] = dict(panel=panels.pop('test'))
        if panels:
            pools[task.cipher] = panels
    model_path = root / 'models/archive-four/M-best.pt'
    original[model_path] = model_path.read_bytes()
    evidence = dict(method=old.method, backend_identity=BACKEND, synthetic=False,
        scientific_validation=False, pool_status='READY', reason='COMPLETE', pools=pools,
        models={config.SLOT: dict(parameters=config.PARAMETERS)}, best=best,
        persisted_reload=best, test=tests)
    store.finalize(dict(status='SUCCEEDED', recipe=old.to_dict(), summary=evidence))
    manifest = json.loads(record.read_bytes())
    if case == 'manifest':
        manifest['artifacts']['dataset/train-SIMECK.npz']['kind'] = 'payload'
    if case == 'skinny':
        manifest['summary']['pools']['SKINNY']['dev_panel']['digest'] = '0' * 64
    if case == 'old-best':
        manifest['summary']['persisted_reload']['sha256'] = '0' * 64
    if case == 'old-recipe':
        manifest['recipe']['optimizer']['eps'] = .01
    raw = _encode(manifest)
    recipe = replace(recipe, archive=replace(recipe.archive, root=str(root), run_id='archive-four',
                                            record=FilePin(len(raw), sha256(raw).hexdigest())))
    record.write_bytes(raw if case != 'record-hash' else b' ' + raw[1:])
    original[record] = record.read_bytes()
    return recipe, original, replay


@pytest.mark.parametrize('case', ['early', 'ceiling', 'threshold', 'missing', 'tamper', 'metadata',
    'digest', 'manifest', 'skinny', 'old-recipe', 'old-best', 'record-hash', 'disk', 'strict',
    'test-missing', 'test-tamper', 'test-best', 'test-score', 'finalize'])
def test_replay_fit(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.innovation1 import application
    from cipher_lab_plugins.innovation1.cli import _finish
    from cipher_lab_plugins.innovation1.four_single import replay as replay_module
    from cipher_lab_plugins.innovation1.layout_probe.reuse import _pinned

    recipe, original, expected_batches = _archive(tmp_path, case)
    root = tmp_path / 'new'
    clocks = Clocks()
    res = Resources(Meter(config.make_policy(recipe, clocks), clocks, lambda: 0))
    new_source = {'artificial/new.py': 'b' * 64}
    store = RunStore.claim(root, recipe.run_id, output_bytes=2**30, needs_models=True,
                           model_slots=(config.SLOT,), source_supplier=lambda: new_source)
    record = root / 'data' / recipe.run_id / 'run.json'
    tasks = [t.cipher for t in recipe.tasks]
    updates, restored, finalized = 0, False, 0
    test_reads: list[str] = []
    test_scores: list[str] = []
    epochs = {'ceiling': 50, 'threshold': 38}.get(case, 36)
    checkpoint_epochs = []

    def tracked_pin(path: Path, *args: Any) -> Any:
        if path.name.startswith('test-'):
            assert restored
            test_reads.append(path.name)
        assert path.name != 'M-best.pt'
        return _pinned(path, *args)

    def open_member(self: ZipFile, member: Any, mode: Literal['r', 'w'] = 'r',
                    **kwargs: Any) -> Any:
        assert mode != 'r' or getattr(member, 'filename', member) != 'logits.npy'
        return original_member(self, member, mode, **kwargs)

    original_member = ZipFile.open
    original_open = Path.open

    def open_path(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.is_relative_to(recipe.archive.root):
            assert path.suffix != '.pt'
            if path.name.startswith('test-'):
                assert restored
        return original_open(path, *args, **kwargs)

    original_import = builtins.__import__

    def no_oracle_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name != 'secrets' and not name.startswith('cipher_lab.ciphers')
        return original_import(name, *args, **kwargs)

    class Runtime:
        def __init__(self, *args: object) -> None:
            pass

        def metadata(self) -> dict[str, Any]:
            return {config.SLOT: dict(parameters=config.PARAMETERS, geometry={'test': 'b' * 64})}

        def prepare(self, batch: PairBatch) -> tuple[PairBatch, np.ndarray[Any, Any]]:
            assert not batch.labels.flags.writeable and not batch.ciphertext.flags.writeable
            return batch, batch.labels

        def update(self, slot: str, task: Task, batch: PairBatch, y: Any, lr: float,
                   resources: Resources) -> float:
            nonlocal updates
            assert slot == config.SLOT and task.cipher == tasks[updates % 4]
            if batch.index < 99:
                assert batch.digest == expected_batches[task.cipher][batch.index]
            expected_lr = (learning_rate(replace(config.FOUR_SINGLE, training=replace(
                recipe.training, epochs=10)), batch.index) if batch.index < 90 else .0001)
            assert lr == expected_lr and not test_reads
            updates += 1
            resources.reserve('forward_rows', batch.groups)
            resources.complete('forward_rows', batch.groups)
            return .5

        def predict(self, slot: str, task: Task, batch: PairBatch) -> Any:
            epoch = updates // 36
            if batch.split == 'test':
                assert restored
                test_scores.append(task.cipher)
                if case == 'test-score' and task.cipher == 'PRESENT':
                    raise ValueError('injected scoring failure')
                a = 2.
            elif case == 'ceiling':
                a = epoch / 20
            elif case == 'threshold':
                # A meaningful improvement resets wait on the second post-warmup epoch.
                a = .01 if epoch >= 26 and task.cipher == 'SKINNY' else 0.
            else:
                # Warmup anchors the minimum, not the last warmup value.
                # After warmup there can be a true best below the stopping threshold.
                a = -.01 if epoch == 24 else .00001 if epoch >= 30 else 0.
            return (2 * batch.labels.astype(np.float64) - 1) * a

        def routes(self) -> dict[str, dict[str, float]]:
            return {t: {'word': .75, 'nibble': .25} for t in tasks}

        def checkpoint(self, slot: str, meta: dict[str, Any]) -> bytes:
            assert slot == config.SLOT and meta['pools']['SIMECK']['train']['origin'] == 'reused'
            checkpoint_epochs.append(meta['epoch'])
            return canonical(meta).encode()

        def release(self) -> None:
            assert updates == 36*epochs and not test_reads
            if case == 'disk':
                (root / 'models' / recipe.run_id / 'M-best.pt').write_bytes(b'tampered')

        def restore(self, slot: str, raw: bytes, meta: dict[str, Any]) -> None:
            nonlocal restored
            assert not test_reads and json.loads(raw) == meta
            if case == 'strict':
                raise ValueError('CHECKPOINT_INVALID')
            restored = True

    finalize, atomic = store.finalize, store._atomic

    def counted_finalize(value: Any) -> None:
        nonlocal finalized
        finalized += 1
        finalize(value)

    def fail_terminal_write(path: Path, content: bytes) -> None:
        if path == record and json.loads(content)['status'] == 'SUCCEEDED':
            raise OSError('injected finalize failure')
        atomic(path, content)

    monkeypatch.setattr(store, 'finalize', counted_finalize)
    if case == 'finalize':
        monkeypatch.setattr(store, '_atomic', fail_terminal_write)
    monkeypatch.setattr(training, 'Runtime', Runtime)
    monkeypatch.setattr(training, 'official', lambda recipe, run_id: recipe)
    monkeypatch.setattr(training, 'assemble', forbidden)
    monkeypatch.setattr(application, 'official_four_single', lambda recipe, run_id: recipe)
    monkeypatch.setattr(application, 'prepare_device', lambda *_: None)
    # The local archive seam replaces only the production path admission check.
    checked_roots: list[Path] = []
    monkeypatch.setattr(application, 'validate_replay_root', checked_roots.append)
    monkeypatch.setattr(replay_module, '_pinned', tracked_pin)
    monkeypatch.setattr(ZipFile, 'open', open_member)
    monkeypatch.setattr(Path, 'open', open_path)
    monkeypatch.setattr(FixedPairPool, 'prepare_pool', forbidden)
    monkeypatch.setattr(Provider, '_generate', forbidden)
    with monkeypatch.context() as m:
        m.setattr(builtins, '__import__', no_oracle_import)
        application.execute(recipe, store, res, gpu_uuid=GPU, run_id=recipe.run_id)
    assert checked_roots == [root]
    monkeypatch.setattr(Path, 'open', original_open)
    assert all(path.read_bytes() == raw for path, raw in original.items())
    saved = json.loads(record.read_bytes())
    summary = saved['summary']
    assert saved['policy']['request']['gpu_uuid'] == GPU
    assert not any(ref.split('/')[-1].startswith(('train-', 'dev-'))
                   for ref in saved['artifacts'])
    if case in ('missing', 'tamper', 'metadata', 'digest', 'manifest', 'skinny', 'old-recipe',
                'old-best', 'record-hash'):
        assert saved['status'] == 'FAILED' and summary['reason'] == 'POOL_SOURCE_INVALID'
        assert updates == 0 and not restored and not test_reads
    elif case in ('disk', 'strict'):
        assert saved['status'] == 'FAILED' and not restored and not test_reads
    elif case == 'test-score':
        assert saved['status'] == 'FAILED' and test_scores == tasks[:2]
        assert not summary['test']
    elif case.startswith('test-'):
        assert restored and test_reads == ['test-SIMECK.npz', 'test-PRESENT.npz']
        assert saved['status'] == 'FAILED' and summary['reason'] == 'TEST_SOURCE_INVALID'
        assert not test_scores and not summary['test']
        assert _finish(store, root, recipe.run_id, resources=res) == 4
    elif case == 'finalize':
        assert saved['status'] == 'FAILED' and summary['reason'] == 'WRITE_FAILED'
        assert finalized == 1
    else:
        assert saved['status'] == 'SUCCEEDED' and test_scores == tasks
        assert summary['training_termination'] == dict(
            reason='EPOCH_CEILING' if epochs == 50 else 'EARLY_STOPPING', completed_epochs=epochs)
        assert checkpoint_epochs[-1] == summary['best']['epoch'] == {
            'early': 30, 'ceiling': 50, 'threshold': 26}[case]
        for kind in ('attempted', 'completed'):
            assert saved['counts'][f'{kind}_optimizer_updates'] == 36*epochs
            assert saved['counts'][f'{kind}_forward_rows'] == 160*epochs + 16
            assert saved['counts'][f'{kind}_real_blocks'] == 0
        monkeypatch.setattr(ZipFile, 'open', original_member)
        for name, info in summary['test'].items():
            assert info['scores']['accuracy'] == 1.
            with np.load(record.parent / info['panel']['ref'], allow_pickle=False) as panel:
                np.testing.assert_array_equal(panel['logits'], 2*(2*panel['labels'].astype(int)-1))
                meta = json.loads(panel['metadata'].tobytes())
                assert meta['source'] == new_source and meta['recipe'] == recipe.to_dict()
                assert meta['best'] == summary['persisted_reload']
                assert meta['reused_input']['ref'] == f'dataset/test-{name}.npz'
                assert meta['reused_input']['source_best']['epoch'] == 3
                assert meta['reused_input']['source_record']['record'] == {
                    'bytes': recipe.archive.record.bytes, 'sha256': recipe.archive.record.sha256}
        assert _finish(store, root, recipe.run_id, resources=res) == 0
        report = (root / 'result' / recipe.run_id / 'report.md').read_text(encoding='utf-8')
        assert '复用' in report and '独立确认' in report and f'{epochs}' in report
        assert '磁盘重载后的路由权重' in report
    assert finalized == (0 if case == 'disk' else 1)
