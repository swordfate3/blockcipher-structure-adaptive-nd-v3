"""Residual CLI assembly/estimate/report tests; fake ledgers and manual forwards only.

No production cipher, private RNG, actual optimizer/backward, real forward or
production RSS may run here. Autouse `poison` replaces BitMLP.forward and forbids
production entries before any test body.
"""
from __future__ import annotations

import builtins
import html as html_module
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import cipher_lab_plugins.prediction_common.cli as cli
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter
from cipher_lab_plugins.prediction_common.application import prepare_training
from cipher_lab_plugins.prediction_common.cli import _estimate
from cipher_lab_plugins.residual.cli import load_experiment
from cipher_lab_plugins.residual.cli import main as plugin_main
from cipher_lab_plugins.residual.config import ResidualPredictionRecipe
from cipher_lab_plugins.residual.tests.test_residual_config import poison

__all__ = ['poison']

PROJECT = Path(__file__).resolve().parents[4]


def ledger_meter(real_blocks: int = 1000, null_words: int = 1000,
                 updates: int = 100, forward_rows: int = 10000) -> Meter:
    return Meter(
        Policy(2, 60., 60., '2031-01-01T00:00:00Z', real_blocks, null_words,
               1000000, 10000000, updates, forward_rows),
        Clocks(lambda: 0., lambda: 0., lambda: datetime(2030, 1, 1, tzinfo=UTC)),
        lambda: 0,
    )


def write_policy(tmp_path: Path, *, real_blocks: int = 300000,
                 optimizer_updates: int = 20000, forward_rows: int = 5000000) -> Path:
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({
        'schema': 2, 'wall_seconds': 120, 'cpu_seconds': 120,
        'deadline_utc': '2099-01-01T00:00:00Z', 'real_blocks': real_blocks,
        'null_words': 100, 'rss_stop_bytes': 2000000000,
        'output_bytes': 20000000, 'optimizer_updates': optimizer_updates,
        'forward_rows': forward_rows,
    }))
    return path


def experiment(name: str) -> ResidualPredictionRecipe:
    loaded = load_experiment(PROJECT / 'configs' / name)
    assert isinstance(loaded, ResidualPredictionRecipe)
    return loaded


def formal(name: str, rounds: int, sample_seed: int, shuffle_seed: int) -> dict[str, Any]:
    return {
        'schema': 1, 'method': 'bit-residual-paired-v1',
        'reference': 'simon64-zero-master-v1',
        'model': {'hidden_width': 128},
        'training': {'train': 65536, 'dev': 8192, 'test': 65536, 'batch_size': 256,
                     'epochs': 32, 'learning_rate': 0.001, 'init_seed': 17,
                     'sample_seed': sample_seed, 'shuffle_seed': shuffle_seed},
        'rounds': rounds, 'target_bit': 0, 'label_mode': 'cipher',
    }


# ---------------------------------------------------------------------------
# Static estimates and complete recipes


def test_cipher_estimate_doubles_encryptions_and_counts_both_arms() -> None:
    recipe = experiment('residual-paired-r6.json')
    estimate = _estimate(recipe)
    assert estimate['planned_real_blocks'] == 2 * (65536 + 8192 + 65536) == 278528
    assert estimate['planned_null_words'] == 0
    assert estimate['planned_optimizer_updates'] == 2 * 32 * 65536 // 256 == 16384
    assert estimate['planned_forward_rows'] == 2 * (32 * (65536 + 8192) + 139264) == 4997120
    basis = str(estimate['basis'])
    assert 'arm' in basis and 'reference' in basis
    units = cast(Mapping[str, object], estimate['units'])
    assert 'zero-master-key' in str(units['real_blocks'])
    assert 'either arm' in str(units['forward_rows'])


def test_null_estimate_charges_one_real_reference_and_one_null_label() -> None:
    recipe = experiment('residual-paired-r6.json')
    null = replace(recipe, label_mode='null')
    estimate = _estimate(null)
    assert estimate['planned_real_blocks'] == 139264
    assert estimate['planned_null_words'] == 139264
    assert estimate['planned_optimizer_updates'] == 16384
    assert estimate['planned_forward_rows'] == 4997120


def test_example_estimate_is_strictly_tiny() -> None:
    estimate = _estimate(experiment('residual-paired-example.json'))
    assert estimate['planned_real_blocks'] == 12
    assert estimate['planned_null_words'] == 0
    assert estimate['planned_optimizer_updates'] == 2
    assert estimate['planned_forward_rows'] == 20


def test_old_method_estimates_are_unchanged() -> None:
    from cipher_lab.cli import _estimate
    from cipher_lab.config import load_experiment
    prediction = _estimate(load_experiment(PROJECT / 'configs/prediction-example.json'))
    assert prediction['planned_real_blocks'] == 6
    assert prediction['planned_null_words'] == 0
    assert prediction['planned_optimizer_updates'] == 1
    assert prediction['planned_forward_rows'] == 10
    shared = _estimate(load_experiment(PROJECT / 'configs/shared-m-example.json'))
    assert shared['planned_real_blocks'] == 192
    assert shared['planned_null_words'] == 0
    assert shared['planned_optimizer_updates'] == 4
    assert shared['planned_forward_rows'] == 40


@pytest.mark.parametrize(('name', 'rounds', 'sample_seed', 'shuffle_seed'), [
    ('residual-paired-r3.json', 3, 196901, 296901),
    ('residual-paired-r4.json', 4, 196902, 296902),
    ('residual-paired-r6.json', 6, 196903, 296903),
])
def test_formal_residual_recipes_are_complete_and_exact(name: str, rounds: int,
                                                        sample_seed: int,
                                                        shuffle_seed: int) -> None:
    path = PROJECT / 'configs' / name
    raw = json.loads(path.read_text())
    expected = formal(name, rounds, sample_seed, shuffle_seed)
    assert raw == expected
    assert experiment(name).to_dict() == expected
    assert 'key_seed' not in path.read_text() and 'private' not in path.read_text()


def test_example_recipe_is_complete_strict_and_not_authorization() -> None:
    raw = json.loads((PROJECT / 'configs/residual-paired-example.json').read_text())
    assert raw['training']['train'] == raw['training']['dev'] == raw['training']['test'] == 2
    assert raw['training']['batch_size'] == 2 and raw['training']['epochs'] == 1
    assert raw['model'] == {'hidden_width': 128}
    assert raw['method'] == 'bit-residual-paired-v1'
    assert raw['reference'] == 'simon64-zero-master-v1'
    assert 'key_seed' not in raw and 'key_seed' not in json.dumps(raw)


# ---------------------------------------------------------------------------
# CLI identity/dispatch for the new method


def test_check_subprocess_imports_no_science_and_writes_nothing(tmp_path: Path) -> None:
    policy = write_policy(tmp_path)
    arguments = ['check', '--config', str(PROJECT / 'configs/residual-paired-r6.json'),
                 '--policy', str(policy)]
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('numpy', 'torch', 'cipher_lab.data', 'cipher_lab.ciphers',
                   'cipher_lab.execution', 'cipher_lab.artifacts', 'cipher_lab.models',
                   'cipher_lab.training', 'cipher_lab.evaluation')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('scientific import during check')
sys.meta_path.insert(0, Block())
from cipher_lab_plugins.residual.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    done = subprocess.run([sys.executable, '-c', script, *arguments], cwd=tmp_path,
                          capture_output=True, text=True, check=False,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                              (str(PROJECT / 'src'), str(PROJECT / 'plugins'))),
                                   PYTHONDONTWRITEBYTECODE='1'))
    assert done.returncode == 0, done.stderr
    output = json.loads(done.stdout)
    estimate = output['estimate']
    assert estimate['planned_real_blocks'] == 278528
    assert estimate['planned_forward_rows'] == 4997120
    assert output['planned_calls_fit_policy'] == {
        'real_blocks': True, 'null_words': True,
        'optimizer_updates': True, 'forward_rows': True}
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('backend', ['demo', 'simon'])
def test_run_claims_models_and_dispatches_saved_residual_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], backend: str) -> None:
    import cipher_lab.execution as execution
    from cipher_lab.demo import demo_rss_bytes

    seen: list[tuple[object, str]] = []

    def fake_prepare(recipe: object, meter: object, store: RunStore,
                     selected: str) -> Any:
        seen.append((recipe, selected))

        def run() -> None:
            store.finalize({'status': 'SUCCEEDED', 'summary': {
                'method': 'bit-residual-paired-v1', 'backend_identity': selected,
                'scientific_validation': False, 'synthetic': selected == 'demo',
                'reason': '', 'arms': {}, 'paired': {}, 'diagnostics': {},
                'pairing': {}}})
        return run

    monkeypatch.setattr(
        'cipher_lab_plugins.prediction_common.application.prepare_training', fake_prepare)
    if backend == 'simon':
        monkeypatch.setattr(execution, 'production_rss_bytes', demo_rss_bytes)
    policy = write_policy(tmp_path)
    root = tmp_path / 'out'
    code = plugin_main([
        'run', '--config', str(PROJECT / 'configs/residual-paired-example.json'),
        '--policy', str(policy), '--root', str(root), '--id', 'pair',
        '--backend', backend])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output['scientific_status'] == 'SUCCEEDED'
    assert output['report_status'] == 'SAVED'
    assert output['backend_identity'] == backend
    assert (root / 'models' / 'pair').is_dir()  # residual always needs models
    assert seen and seen[0][1] == backend
    assert (root / 'result' / 'pair' / 'index.html').is_file()
    saved = read_saved(root, 'pair')
    assert cast(Mapping[str, object], saved['summary'])['method'] == 'bit-residual-paired-v1'


def test_residual_rejects_models_free_or_unknown_backends_before_claim(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(tmp_path)
    code = plugin_main(['run', '--config',
                     str(PROJECT / 'configs/residual-paired-example.json'),
                     '--policy', str(policy), '--root', str(tmp_path / 'out'),
                     '--id', 'pair', '--backend', 'ciphers'])
    assert code == 2
    assert not (tmp_path / 'out').exists()
    assert 'invalid method/backend combination' in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Application assembly: fake demo vs explicit production adapters


def test_demo_assembly_uses_only_public_fake_reference(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.residual import training as paired

    def terminal(recipe: object, data: object, meter: object, store: object, *,
                 backend_identity: str) -> dict[str, object]:
        assert backend_identity == 'demo'
        reference = data.reference('train')  # type: ignore[attr-defined]
        labels = data.labels('train')  # type: ignore[attr-defined]
        assert reference.shape == labels.shape == (2, 1)
        assert set(reference.flatten().tolist()) <= {0., 1.}
        return {'status': 'SUCCEEDED', 'seen': (recipe, data)}

    monkeypatch.setattr(paired, 'run_paired_prediction', terminal)
    store = RunStore.claim(tmp_path, 'demo', output_bytes=10000000, needs_models=True)
    invoke = prepare_training(experiment('residual-paired-example.json'),
                              ledger_meter(), store, 'demo')
    result = cast(Mapping[str, object], invoke())
    assert result['status'] == 'SUCCEEDED'


def test_simon_assembly_passes_zero_master_key_and_real_rounds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    from cipher_lab_plugins.residual import training as paired

    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_encrypt(points: tuple[int, ...], key: tuple[int, ...],
                     rounds: int) -> tuple[int, ...]:
        calls.append((key, rounds))
        return tuple(point ^ rounds for point in points)

    monkeypatch.setattr('cipher_lab.ciphers.simon.encrypt_many', fake_encrypt)
    monkeypatch.setattr(secrets, 'token_bytes', lambda size: b'\x11' * size)

    def terminal(recipe: object, data: object, meter: object, store: object, *,
                 backend_identity: str) -> dict[str, object]:
        assert backend_identity == 'simon'
        reference = data.reference('train')  # type: ignore[attr-defined]
        assert reference.shape == (2, 1)
        assert set(reference.flatten().tolist()) <= {0., 1.}
        return {'status': 'SUCCEEDED'}

    monkeypatch.setattr(paired, 'run_paired_prediction', terminal)
    store = RunStore.claim(tmp_path, 'simon', output_bytes=10000000, needs_models=True)
    recipe = experiment('residual-paired-example.json')
    prepare_training(recipe, ledger_meter(), store, 'simon')()
    zero = [rounds for key, rounds in calls if key == (0, 0, 0, 0)]
    assert zero and set(zero) == {recipe.rounds}
    assert any(key != (0, 0, 0, 0) for key, _ in calls)  # private key stayed private
    assert all(len(key) == 4 for key, _ in calls)


# ---------------------------------------------------------------------------
# Full demo CLI run with fake optimizer/step and manual forward (ledger only)


def test_demo_paired_run_end_to_end_with_fake_training(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.residual import training as paired

    def fake_step(model: object, inputs: torch.Tensor, labels: torch.Tensor,
                  context: object, optimizer: object, meter: Meter) -> float:
        meter.reserve('optimizer_updates', 1)
        meter.reserve('forward_rows', len(inputs))
        meter.complete('forward_rows', len(inputs))
        meter.complete('optimizer_updates', 1)
        return .7

    monkeypatch.setattr(paired, 'make_optimizer', lambda *args: object())
    monkeypatch.setattr(paired, 'train_step', fake_step)
    policy = write_policy(tmp_path, real_blocks=100, optimizer_updates=10, forward_rows=100)
    root = tmp_path / 'out'
    code = plugin_main(['run', '--config', str(PROJECT / 'configs/residual-paired-example.json'),
                     '--policy', str(policy), '--root', str(root), '--id', 'pair',
                     '--backend', 'demo'])
    output = json.loads(capsys.readouterr().out)
    assert code == 0, output
    assert output['report_status'] == 'SAVED'
    saved = read_saved(root, 'pair')
    assert saved['status'] == 'SUCCEEDED'
    summary = cast(Mapping[str, object], saved['summary'])
    assert summary['method'] == 'bit-residual-paired-v1'
    assert set(cast(Mapping[str, object], summary['arms'])) == {'direct', 'residual'}
    assert set(cast(Mapping[str, object], summary['paired'])) == {'train', 'dev', 'test'}
    assert set(cast(Mapping[str, object], summary['diagnostics'])) == {'reference_constant'}
    counts = cast(Mapping[str, int], saved['counts'])
    assert summary['meter'] is not None
    meter = cast(Mapping[str, int], summary['meter'])
    assert meter['completed_real_blocks'] == 12  # 6 labels + 6 public references
    assert meter['completed_null_words'] == 0
    assert meter['completed_forward_rows'] == 20
    assert counts['updates'] == 2
    assert counts['train_presentations'] == 4
    assert counts['dev_evaluations'] == 2
    models = sorted(path.name for path in (root / 'models' / 'pair').glob('*.pt'))
    assert models == ['direct-e1.pt', 'residual-e1.pt']
    assert (root / 'data' / 'pair' / 'dataset' / 'test.npz').is_file()
    html = (root / 'result' / 'pair' / 'index.html').read_text()
    assert '残差配对' in html and '绝对bit' in html and '<script>' not in html


# ---------------------------------------------------------------------------
# Saved-only residual renderer


def arms_evidence() -> dict[str, object]:
    def metrics(acc: float, bce: float) -> dict[str, object]:
        return {'rows': 400, 'mean_bce': bce, 'tasks': {'SIMON': {
            'rows': 400, 'acc': acc, 'bce': bce,
            'baseline_acc': .5, 'baseline_bce': .7}}}

    def arm(acc_direct: float, acc_residual: float, bce: float) -> dict[str, object]:
        curve = []
        for epoch in (1, 2):
            curve.append({'epoch': epoch, 'train_bce': bce + .2 - epoch * .05,
                          'dev': metrics(acc_direct if epoch == 1 else acc_residual,
                                         bce - epoch * .01)})
        return {'selected': {'epoch': 2, 'updates': 2, 'selected_dev_bce': bce},
                'learning_curve': curve,
                'metrics': {'train': metrics(acc_direct, bce + .1),
                            'dev': metrics(acc_residual, bce),
                            'test': metrics(acc_residual, bce + .05)}}

    return {'direct': arm(.75, .76, .70), 'residual': arm(.75, .85, .72)}


def paired_split() -> dict[str, object]:
    return {'n': 400, 'b': 4, 'c': 0, 'both_correct': 296, 'both_wrong': 96,
            'sum_d': 4, 'sum_d2': 4, 'delta': 0.01, 'p_one_sided': 0.0625,
            'normal_95_interval': None,
            'loss_difference': {'direction': 'residual-minus-direct', 'sum': 1.5,
                                'sum_squares': -0.25, 'mean': 0.00375}}


def diagnostics_split() -> dict[str, object]:
    def binary(acc: float, bce: float) -> dict[str, float | int]:
        return {'rows': 400, 'acc': acc, 'bce': bce,
                'baseline_acc': .5, 'baseline_bce': .7}

    return {'train_residual_constant': 0, 'train_residual_probability': 0.5,
            'fixed_flip': binary(.5, .6931471805599453),
            'residual_probability': binary(.5, .6931471805599453)}


def saved_residual() -> dict[str, object]:
    summary: dict[str, object] = {
        'method': 'bit-residual-paired-v1', 'backend_identity': 'demo',
        'synthetic': True, 'scientific_validation': False, 'reason': '<script>x</script>',
        'arms': arms_evidence(),
        'paired': {split: paired_split() for split in ('train', 'dev', 'test')},
        'diagnostics': {'reference_constant': {
            'train_residual_rows': 400, 'train_residual_ones': 200,
            'train_residual_constant': 0, 'train_residual_probability': 0.5,
            'splits': {split: diagnostics_split() for split in ('train', 'dev', 'test')}}},
        'pairing': {'initial_state_sha256': {'direct': 'a' * 64, 'residual': 'a' * 64},
                    'shuffle_seed': 11,
                    'panels': {split: {'rows': 400, 'x_sha256': 'b' * 64,
                                       'y_sha256': 'c' * 64, 'h_sha256': 'd' * 64}
                               for split in ('train', 'dev', 'test')}},
        'refs': {'test': 'dataset/test.npz', 'direct': 'payload/direct.json'},
        'meter': {'completed_forward_rows': 20, 'completed_real_blocks': 12},
        'selected': {}, 'metrics': {}, 'learning_curve': [], 'recipe_sha256': 'e' * 64,
        'source': {'src': 'f' * 64},
    }
    return {'run_id': 'pair', 'run_ref': 'data/pair/run.json', 'run_sha256': '1' * 64,
            'status': 'SUCCEEDED', 'recipe': {'method': 'bit-residual-paired-v1'},
            'counts': {'updates': 2}, 'summary': summary, 'source': {},
            'policy': {'schema': 2}}


def test_residual_report_consumes_saved_values_with_zero_io_or_science(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab_plugins.residual.report as report
    from cipher_lab_plugins.residual.report import render_residual_report

    saved = saved_residual()
    before = deepcopy(saved)
    original = builtins.__import__

    def no_science(name: str, globals_: Mapping[str, object] | None = None,
                   locals_: Mapping[str, object] | None = None,
                   fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        if name.startswith(('torch', 'numpy', 'cipher_lab.data', 'cipher_lab.ciphers',
                            'cipher_lab.models', 'cipher_lab.execution',
                            'cipher_lab.evaluation', 'cipher_lab_plugins.residual.training')):
            pytest.fail('report imported science or training')
        return original(name, globals_, locals_, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', no_science)
        patch.setattr(Path, 'read_bytes', lambda _self: pytest.fail('report I/O'))
        files = render_residual_report(cast(Mapping[str, object], saved))
    assert set(files) == {'report.md', 'index.html', 'source.json'}
    html = files['index.html'].decode()
    markdown = files['report.md'].decode()
    for text in (html, markdown):
        assert '残差配对' in text and '绝对bit' in text
        assert 'b=4' in text and 'c=0' in text and '0.0625' in text
        assert '1.000pp' in text
        assert 'direct epoch 1' in text and 'residual epoch 2' in text
        assert '单位幅度' in text and 'sigmoid(+1)≈0.7311' in text
        assert 'sigmoid(-1)≈0.2689' in text and 'Jeffreys' in text
        assert '空占位' in text and 'arms/paired' in text
        assert '不是技术 sealed' in text
        assert '<script>' not in text
    assert html.count('<polyline') >= 4
    assert html.count('<svg') >= 3
    assert json.loads(files['source.json'])['run_sha256'] == '1' * 64
    assert saved == before
    assert report.render_residual_report(
        cast(Mapping[str, object], saved))['index.html'] == files['index.html']


def test_paired_report_prints_all_saved_fields_without_recompute() -> None:
    from cipher_lab_plugins.residual.report import render_residual_report

    saved = saved_residual()
    summary = cast(dict[str, object], saved['summary'])
    # Deliberately inconsistent sentinels: n/b/c do not derive sum_d/sum_d2,
    # so a renderer that recomputes instead of reading saved values will fail.
    sentinel: dict[str, object] = {
        'n': 424242, 'b': 41, 'c': 7, 'both_correct': 555, 'both_wrong': 777,
        'sum_d': 9037, 'sum_d2': 77123, 'delta': 0.12345, 'p_one_sided': 0.987654,
        'normal_95_interval': [1.5, 2.5],
        'loss_difference': {'direction': 'residual-minus-direct', 'sum': 3.5,
                            'sum_squares': 9.25, 'mean': 0.007}}
    summary['paired'] = {split: dict(sentinel) for split in ('train', 'dev', 'test')}
    files = render_residual_report(cast(Mapping[str, object], saved))
    expected = ('n=424242', 'b=41', 'c=7', 'both_correct=555', 'both_wrong=777',
                'sum_d=9037', 'sum_d2=77123', 'delta(residual-direct)=12.345pp',
                'p=0.987654', '1.5', '2.5', 'direction=residual-minus-direct',
                'sum=3.5', 'sum_squares=9.25', 'mean=0.007')
    for name in ('index.html', 'report.md'):
        # Renderer entity-escapes punctuation for safe dual rendering; decode it
        # here so the assertions inspect presented content, not escaping choices.
        text = html_module.unescape(files[name].decode())
        for field in expected:
            assert field in text, (name, field)
        # Recomputed b-c=34 / b+c=48 must not replace saved sum_d/sum_d2.
        assert 'sum_d=34' not in text and 'sum_d2=48' not in text
        # details JSON must carry the saved paired mapping, not only derived text.
        assert '"paired"' in text


def test_residual_report_missing_panels_are_not_zero_filled() -> None:
    from cipher_lab_plugins.residual.report import render_residual_report

    saved = saved_residual()
    summary = cast(dict[str, object], saved['summary'])
    summary['arms'] = {}
    summary['paired'] = {}
    summary['diagnostics'] = {}
    files = render_residual_report(cast(Mapping[str, object], saved))
    text = files['index.html'].decode()
    assert '未保存（不补 0）' in text
    assert 'ACC=0<' not in text


def test_saved_report_dispatch_and_escape_for_residual(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = RunStore.claim(tmp_path, 'pair', output_bytes=10000000, needs_models=True)
    saved = saved_residual()
    summary = cast(dict[str, object], saved['summary'])
    store.finalize({'status': 'SUCCEEDED', 'summary': summary, 'counts': {'updates': 2},
                    'recipe': {'method': 'bit-residual-paired-v1'}})
    code = cli._saved_report(store, tmp_path, 'pair', 'demo')
    output = json.loads(capsys.readouterr().out)
    assert code == 0 and output['report_status'] == 'SAVED'
    assert output['backend_identity'] == 'demo'
    html = (tmp_path / 'result' / 'pair' / 'index.html').read_text()
    assert '<script>' not in html and '&lt;script&gt;' in html
    source = json.loads((tmp_path / 'result' / 'pair' / 'source.json').read_text())
    assert source['run_sha256'] == read_saved(tmp_path, 'pair')['run_sha256']


def test_residual_renderer_ignores_legacy_empty_top_level_metrics() -> None:
    from cipher_lab_plugins.residual.report import render_residual_report

    with_arms = render_residual_report(cast(Mapping[str, object], saved_residual()))
    saved = saved_residual()
    summary = cast(dict[str, object], saved['summary'])
    summary['metrics'] = {'train': {'tasks': {'SIMON': {'acc': 0.999}}}}
    summary['learning_curve'] = [{'epoch': 99, 'train_bce': -1.}]
    summary['selected'] = {'epoch': 99}
    assert render_residual_report(
        cast(Mapping[str, object], saved))['index.html'] == with_arms['index.html']
