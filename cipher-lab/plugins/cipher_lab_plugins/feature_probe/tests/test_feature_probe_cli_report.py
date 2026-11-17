"""Feature-probe CLI/estimate/Chinese report tests; fake ledgers and manual forwards only.

Autouse `poison` from feature_probe_fakes forbids real forward/backward/optimizer,
production cipher/private RNG/RSS before any test body or owner. The end-to-end
demo run keeps the real three-arm scheduling and only replaces train_step and
make_optimizer, so a network forward cannot slip through the CLI.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import cipher_lab_plugins.prediction_common.cli as cli
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter
from cipher_lab_plugins.feature_probe.cli import load_experiment
from cipher_lab_plugins.feature_probe.cli import main as plugin_main
from cipher_lab_plugins.feature_probe.config import FeatureProbeRecipe
from cipher_lab_plugins.feature_probe.tests.feature_probe_fakes import poison
from cipher_lab_plugins.prediction_common.cli import _estimate

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


def write_policy(tmp_path: Path, *, real_blocks: int = 100,
                 optimizer_updates: int = 10, forward_rows: int = 100) -> Path:
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({
        'schema': 2, 'wall_seconds': 120, 'cpu_seconds': 120,
        'deadline_utc': '2099-01-01T00:00:00Z', 'real_blocks': real_blocks,
        'null_words': 100, 'rss_stop_bytes': 2000000000,
        'output_bytes': 20000000, 'optimizer_updates': optimizer_updates,
        'forward_rows': forward_rows,
    }))
    return path


def experiment(name: str) -> FeatureProbeRecipe:
    loaded = load_experiment(PROJECT / 'configs' / name)
    assert isinstance(loaded, FeatureProbeRecipe)
    return loaded


def formal() -> dict[str, Any]:
    return {
        'schema': 1, 'method': 'bit-public-feature-probe-v1',
        'reference': 'simon64-zero-master-v1',
        'representation': 'simon-r3-residual13-v1',
        'model': {'hidden_width': 128},
        'training': {'train': 16384, 'dev': 4096, 'test': 16384, 'batch_size': 256,
                     'epochs': 32, 'learning_rate': 0.001, 'init_seed': 31,
                     'sample_seed': 196905, 'shuffle_seed': 296905},
        'rounds': 3, 'target_bit': 0, 'label_mode': 'cipher',
    }


def tiny() -> dict[str, Any]:
    return {
        'schema': 1, 'method': 'bit-public-feature-probe-v1',
        'reference': 'simon64-zero-master-v1',
        'representation': 'simon-r3-residual13-v1',
        'model': {'hidden_width': 2},
        'training': {'train': 2, 'dev': 2, 'test': 2, 'batch_size': 2, 'epochs': 1,
                     'learning_rate': 0.001, 'init_seed': 0, 'sample_seed': 1,
                     'shuffle_seed': 4294967295},
        'rounds': 3, 'target_bit': 0, 'label_mode': 'cipher',
    }


# ---------------------------------------------------------------------------
# Static estimates and complete recipes


def test_probe_estimate_counts_three_arms_and_public_reference() -> None:
    estimate = _estimate(experiment('feature-probe-r3.json'))
    assert estimate['planned_real_blocks'] == 2 * (16384 + 4096 + 16384) == 73728
    assert estimate['planned_null_words'] == 0
    assert estimate['planned_optimizer_updates'] == 3 * 32 * 16384 // 256 == 6144
    assert estimate['planned_forward_rows'] == 3 * (32 * (16384 + 4096)
                                                    + 36864) == 2076672
    basis = str(estimate['basis']).lower()
    assert 'three fixed arms' in basis and 'reference' in basis and 'test' in basis
    units = cast(Mapping[str, object], estimate['units'])
    assert 'zero-master-key' in str(units['real_blocks'])
    assert 'either arm' not in str(units['forward_rows'])
    assert 'any arm' in str(units['forward_rows'])


def test_probe_example_estimate_is_strictly_tiny() -> None:
    estimate = _estimate(experiment('feature-probe-example.json'))
    assert estimate['planned_real_blocks'] == 12
    assert estimate['planned_null_words'] == 0
    assert estimate['planned_optimizer_updates'] == 3
    assert estimate['planned_forward_rows'] == 30


def test_probe_null_estimate_charges_reference_and_null_labels() -> None:
    from dataclasses import replace

    recipe = experiment('feature-probe-example.json')
    estimate = _estimate(replace(recipe, label_mode='null'))
    assert estimate['planned_real_blocks'] == 6  # public reference only
    assert estimate['planned_null_words'] == 6
    assert estimate['planned_optimizer_updates'] == 3
    assert estimate['planned_forward_rows'] == 30


def test_old_method_estimates_are_unchanged() -> None:
    from cipher_lab.cli import _estimate
    from cipher_lab.config import load_experiment
    prediction = _estimate(load_experiment(PROJECT / 'configs/prediction-example.json'))
    assert prediction['planned_real_blocks'] == 6
    assert prediction['planned_optimizer_updates'] == 1
    assert prediction['planned_forward_rows'] == 10
    shared = _estimate(load_experiment(PROJECT / 'configs/shared-m-example.json'))
    assert shared['planned_real_blocks'] == 192
    assert shared['planned_optimizer_updates'] == 4
    assert shared['planned_forward_rows'] == 40
    from cipher_lab_plugins.prediction_common.cli import _estimate as estimate_owned
    from cipher_lab_plugins.residual.cli import load_experiment as load_residual
    residual = estimate_owned(load_residual(PROJECT / 'configs/residual-paired-example.json'))
    assert residual['planned_real_blocks'] == 12
    assert residual['planned_optimizer_updates'] == 2
    assert residual['planned_forward_rows'] == 20


def test_formal_probe_recipe_matches_spec_exactly_and_is_not_authorization() -> None:
    path = PROJECT / 'configs/feature-probe-r3.json'
    raw = json.loads(path.read_text())
    assert raw == formal()
    assert experiment('feature-probe-r3.json').to_dict() == formal()
    text = path.read_text()
    assert 'key_seed' not in text and 'private' not in text
    # The approved matrix is one three-round, width-128 recipe: no extra arms,
    # rounds, seeds or cells may be smuggled into the config.
    assert 'arms' not in raw and set(raw) == set(formal())


def test_example_recipe_is_complete_strict_and_not_authorization() -> None:
    path = PROJECT / 'configs/feature-probe-example.json'
    raw = json.loads(path.read_text())
    assert raw == tiny()
    assert experiment('feature-probe-example.json').to_dict() == tiny()
    assert raw['model'] == {'hidden_width': 2}
    assert 'key_seed' not in path.read_text()


# ---------------------------------------------------------------------------
# CLI check / backend identity


def test_check_subprocess_imports_no_science_and_writes_nothing(tmp_path: Path) -> None:
    policy = write_policy(tmp_path, real_blocks=300000, optimizer_updates=20000,
                          forward_rows=5000000)
    arguments = ['check', '--config', str(PROJECT / 'configs/feature-probe-r3.json'),
                 '--policy', str(policy)]
    script = """
import importlib.abc, secrets, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('numpy', 'torch', 'cipher_lab.data', 'cipher_lab.ciphers',
                   'cipher_lab.execution', 'cipher_lab.artifacts', 'cipher_lab.models',
                   'cipher_lab.training', 'cipher_lab.evaluation')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('scientific import during check')
sys.meta_path.insert(0, Block())
def poison(*a, **k):
    raise AssertionError('PRIVATE production call')
secrets.token_bytes = poison
from cipher_lab_plugins.feature_probe.cli import main
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
    assert estimate['planned_real_blocks'] == 73728
    assert estimate['planned_optimizer_updates'] == 6144
    assert estimate['planned_forward_rows'] == 2076672
    assert output['planned_calls_fit_policy'] == {
        'real_blocks': True, 'null_words': True,
        'optimizer_updates': True, 'forward_rows': True}
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


def test_probe_rejects_ciphers_backend_before_claim(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(tmp_path)
    code = plugin_main(['run', '--config',
                     str(PROJECT / 'configs/feature-probe-example.json'),
                     '--policy', str(policy), '--root', str(tmp_path / 'out'),
                     '--id', 'probe', '--backend', 'ciphers'])
    assert code == 2
    assert not (tmp_path / 'out').exists()
    assert 'invalid method/backend combination' in capsys.readouterr().err


def test_probe_check_main_estimates_without_side_effects(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(tmp_path, real_blocks=300000, optimizer_updates=20000,
                          forward_rows=5000000)
    code = plugin_main(['check', '--config',
                     str(PROJECT / 'configs/feature-probe-example.json'),
                     '--policy', str(policy)])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output['estimate']['planned_forward_rows'] == 30


# ---------------------------------------------------------------------------
# Application assembly keeps one data owner and the declared backend


def test_demo_probe_assembly_uses_one_data_owner_and_public_reference(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.feature_probe import training as feature_probe

    seen: list[tuple[object, str]] = []

    def terminal(recipe: object, data: object, meter: object, store: object, *,
                 backend_identity: str) -> dict[str, object]:
        seen.append((recipe, backend_identity))
        labels = data.labels('train')  # type: ignore[attr-defined]
        reference = data.reference('train')  # type: ignore[attr-defined]
        assert labels.shape == reference.shape == (2, 1)
        assert set(reference.flatten().tolist()) <= {0., 1.}
        return {'status': 'SUCCEEDED'}

    monkeypatch.setattr(feature_probe, 'run_feature_probe', terminal)
    store = RunStore.claim(tmp_path, 'demo', output_bytes=10000000, needs_models=True)
    from cipher_lab_plugins.prediction_common.application import prepare_training

    invoke = prepare_training(experiment('feature-probe-example.json'),
                              ledger_meter(), store, 'demo')
    assert cast(Mapping[str, object], invoke())['status'] == 'SUCCEEDED'
    assert seen == [(experiment('feature-probe-example.json'), 'demo')]


def test_simon_probe_assembly_uses_public_zero_master_reference(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    from cipher_lab_plugins.feature_probe import training as feature_probe

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
        assert data.reference('train').shape == (2, 1)  # type: ignore[attr-defined]
        return {'status': 'SUCCEEDED'}

    monkeypatch.setattr(feature_probe, 'run_feature_probe', terminal)
    store = RunStore.claim(tmp_path, 'simon', output_bytes=10000000, needs_models=True)
    from cipher_lab_plugins.prediction_common.application import prepare_training

    recipe = experiment('feature-probe-example.json')
    prepare_training(recipe, ledger_meter(), store, 'simon')()
    zero = [rounds for key, rounds in calls if key == (0, 0, 0, 0)]
    assert zero and set(zero) == {recipe.rounds}
    assert any(key != (0, 0, 0, 0) for key, _ in calls)


# ---------------------------------------------------------------------------
# Full demo CLI run: real scheduling, fake optimizer/step, manual forward


def test_demo_probe_run_end_to_end_with_fake_optimizer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.feature_probe import training as feature_probe

    def fake_step(model: object, inputs: torch.Tensor, labels: torch.Tensor,
                  context: object, optimizer: object, meter: Meter) -> float:
        meter.reserve('optimizer_updates', 1)
        meter.reserve('forward_rows', len(inputs))
        meter.complete('forward_rows', len(inputs))
        meter.complete('optimizer_updates', 1)
        return .7

    monkeypatch.setattr(feature_probe, 'make_optimizer', lambda *args: object())
    monkeypatch.setattr(feature_probe, 'train_step', fake_step)
    policy = write_policy(tmp_path)
    root = tmp_path / 'out'
    code = plugin_main(['run', '--config',
                     str(PROJECT / 'configs/feature-probe-example.json'),
                     '--policy', str(policy), '--root', str(root), '--id', 'probe',
                     '--backend', 'demo'])
    output = json.loads(capsys.readouterr().out)
    assert code == 0, output
    assert output['report_status'] == 'SAVED'
    assert output['backend_identity'] == 'demo'
    saved = read_saved(root, 'probe')
    assert saved['status'] == 'SUCCEEDED'
    summary = cast(Mapping[str, object], saved['summary'])
    assert summary['method'] == 'bit-public-feature-probe-v1'
    assert set(cast(Mapping[str, object], summary['arms'])) == {'A', 'B', 'C'}
    assert set(cast(Mapping[str, object], summary['comparisons'])) == {'C-A', 'C-B'}
    assert set(cast(Mapping[str, object], summary['arm_resources'])) == {'A', 'B', 'C'}
    meter = cast(Mapping[str, int], summary['meter'])
    assert meter['completed_real_blocks'] == 12  # 6 private + 6 public references
    assert meter['completed_null_words'] == 0
    assert meter['completed_forward_rows'] == 30
    counts = cast(Mapping[str, int], saved['counts'])
    assert counts['updates'] == 3
    assert counts['train_presentations'] == 6
    assert counts['dev_evaluations'] == 3
    models = sorted(path.name for path in (root / 'models' / 'probe').glob('*.pt'))
    assert models == ['A-e1.pt', 'B-e1.pt', 'C-e1.pt']
    assert (root / 'data' / 'probe' / 'dataset' / 'test.npz').is_file()
    for name in ('report.md', 'index.html', 'source.json'):
        assert (root / 'result' / 'probe' / name).is_file()
    html = (root / 'result' / 'probe' / 'index.html').read_text()
    assert '直接预测' in html and '公开组合' in html and '<script>' not in html
    assert json.loads((root / 'result' / 'probe' / 'source.json').read_text())[
        'run_sha256'] == saved['run_sha256']


def test_simon_probe_run_dispatches_models_and_saved_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    import cipher_lab.execution as execution
    from cipher_lab.demo import demo_rss_bytes

    seen: list[tuple[object, str]] = []

    def fake_prepare(recipe: object, meter: object, store: RunStore,
                     selected: str) -> Any:
        seen.append((recipe, selected))

        def run() -> None:
            store.finalize({'status': 'SUCCEEDED', 'summary': {
                'method': 'bit-public-feature-probe-v1', 'backend_identity': selected,
                'scientific_validation': False, 'synthetic': selected == 'demo',
                'reason': '', 'arms': {}, 'comparisons': {}, 'pairing': {},
                'arm_resources': {}}})
        return run

    monkeypatch.setattr(
        'cipher_lab_plugins.prediction_common.application.prepare_training', fake_prepare)
    monkeypatch.setattr(execution, 'production_rss_bytes', demo_rss_bytes)
    policy = write_policy(tmp_path)
    root = tmp_path / 'out'
    code = plugin_main(['run', '--config',
                     str(PROJECT / 'configs/feature-probe-example.json'),
                     '--policy', str(policy), '--root', str(root), '--id', 'probe',
                     '--backend', 'simon'])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output['report_status'] == 'SAVED'
    assert output['backend_identity'] == 'simon'
    assert (root / 'models' / 'probe').is_dir()  # probe always needs models
    assert seen and seen[0][1] == 'simon'
    assert (root / 'result' / 'probe' / 'index.html').is_file()
    # A missing/partial saved summary must still render without inventing values.
    assert '未保存' in (root / 'result' / 'probe' / 'index.html').read_text()


# ---------------------------------------------------------------------------
# Saved-only Chinese renderer


def _task_metrics(acc: float, bce: float) -> dict[str, object]:
    return {'rows': 400, 'acc': acc, 'bce': bce,
            'baseline_acc': .5, 'baseline_bce': .7}


def _split_metrics(acc: float, bce: float) -> dict[str, object]:
    return {'rows': 400, 'mean_bce': bce,
            'tasks': {'SIMON': _task_metrics(acc, bce)}}


def _arm(acc: tuple[float, float, float], bce: float) -> dict[str, object]:
    curve = [{'epoch': 1, 'train_bce': bce + .2,
              'dev': _split_metrics(acc[1] - .02, bce + .05)},
             {'epoch': 2, 'train_bce': bce + .1,
              'dev': _split_metrics(acc[1], bce)}]
    return {'arm_id': '', 'input_representation': 'raw64-msb0-v1',
            'output_mode': 'direct', 'model_spec': {'hidden_width': 2},
            'parameter_count': 24961,
            'selected': {'epoch': 2, 'updates': 2, 'selected_dev_bce': bce,
                         'sha256': 'e' * 64},
            'learning_curve': curve,
            'metrics': {'train': _split_metrics(acc[0], bce + .1),
                        'dev': _split_metrics(acc[1], bce),
                        'test': _split_metrics(acc[2], bce + .05)}}


def _comparison(direction: str) -> dict[str, object]:
    return {'n': 400, 'b': 4, 'c': 0, 'both_correct': 296, 'both_wrong': 96,
            'sum_d': 4, 'sum_d2': 4, 'delta': 0.01, 'p_one_sided': 0.0625,
            'normal_95_interval': None,
            'loss_difference': {'direction': direction, 'sum': 1.5,
                                'sum_squares': -0.25, 'mean': 0.00375}}


def saved_probe() -> dict[str, object]:
    arms = {'A': _arm((.75, .76, .70), .70), 'B': _arm((.74, .77, .72), .71),
            'C': _arm((.80, .84, .81), .66)}
    summary: dict[str, object] = {
        'method': 'bit-public-feature-probe-v1', 'backend_identity': 'demo',
        'synthetic': True, 'scientific_validation': False,
        'reason': '<script>x</script>',
        'arms': arms,
        'comparisons': {'C-A': {split: _comparison('C-A')
                                for split in ('train', 'dev', 'test')},
                        'C-B': {split: _comparison('C-B')
                                for split in ('train', 'dev', 'test')}},
        'pairing': {
            'initial_state_sha256': {'A': 'a' * 64, 'B': 'a' * 64, 'C': 'b' * 64},
            'shared_layers_sha256': {'A': 'c' * 64, 'B': 'c' * 64, 'C': 'c' * 64},
            'shuffle_seed': 296905, 'init_seed': 31,
            'initialization_rule': 'layer-streams-v1',
            'panels': {split: {'rows': 400, 'x_sha256': 'd' * 64, 'y_sha256': 'e' * 64,
                               'h_sha256': 'f' * 64}
                       for split in ('train', 'dev', 'test')}},
        'arm_resources': {'A': {'wall_seconds': 12., 'cpu_seconds': 11.,
                                'wall_limit_seconds': 600., 'cpu_limit_seconds': 600.},
                          'B': {'wall_seconds': 13., 'cpu_seconds': 12.,
                                'wall_limit_seconds': 600., 'cpu_limit_seconds': 600.},
                          'C': {'wall_seconds': 14., 'cpu_seconds': 13.,
                                'wall_limit_seconds': 600., 'cpu_limit_seconds': 600.}},
        'refs': {'test': 'dataset/test.npz', 'A': 'payload/A.json'},
        'meter': {'completed_forward_rows': 30, 'completed_real_blocks': 12},
        'selected': {}, 'metrics': {}, 'learning_curve': [], 'recipe_sha256': 'e' * 64,
        'source': {'src': 'f' * 64},
    }
    return {'run_id': 'probe', 'run_ref': 'data/probe/run.json',
            'run_sha256': '1' * 64, 'status': 'SUCCEEDED',
            'recipe': {'method': 'bit-public-feature-probe-v1',
                       'label_mode': 'cipher'},
            'counts': {'updates': 3}, 'summary': summary, 'source': {},
            'policy': {'schema': 2}}


FORBIDDEN_IN_BODY = ('ACC', 'BCE', 'DEV', 'TEST', 'direct', 'residual', 'fit', 'p=')


def _body(html: str) -> str:
    # Everything above the folded appendix is ordinary-reader prose.
    return html.split('<h2>附录')[0]


def test_probe_report_consumes_saved_values_with_zero_io_or_science(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab_plugins.feature_probe.report as report
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    saved = saved_probe()
    before = deepcopy(saved)
    original = builtins.__import__

    def no_science(name: str, globals_: Mapping[str, object] | None = None,
                   locals_: Mapping[str, object] | None = None,
                   fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        blocked = ('torch', 'numpy', 'cipher_lab.data', 'cipher_lab.ciphers',
                   'cipher_lab.models', 'cipher_lab.execution',
                   'cipher_lab.evaluation', 'cipher_lab_plugins.feature_probe.training',
                   'cipher_lab_plugins.prediction_common.application')
        if name.startswith(blocked):
            pytest.fail('report imported science or training')
        return original(name, globals_, locals_, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', no_science)
        patch.setattr(Path, 'read_bytes', lambda _self: pytest.fail('report I/O'))
        files = render_feature_probe_report(cast(Mapping[str, object], saved))
    assert set(files) == {'report.md', 'index.html', 'source.json'}
    html = files['index.html'].decode()
    markdown = files['report.md'].decode()
    for text in (html, markdown):
        assert '直接预测' in text and '原始输入学习纠错' in text
        assert '13个公开组合学习纠错' in text
        assert '70.000%' in text and '72.000%' in text and '81.000%' in text
        assert '1.000 个百分点' in text
        assert '丙对而甲错' in text and '甲对而丙错' in text
        assert '三轮' in text and '公开组合' in text
        assert '不是' in text  # explicit limits, never an unqualified success claim
        body = _body(text)
        for token in FORBIDDEN_IN_BODY:
            assert token not in body, token
    assert html.count('<polyline') >= 6
    assert html.count('<svg') >= 3
    assert json.loads(files['source.json'])['run_sha256'] == '1' * 64
    assert saved == before
    assert report.render_feature_probe_report(
        cast(Mapping[str, object], saved))['index.html'] == files['index.html']


def test_probe_report_prints_saved_four_cells_without_recompute() -> None:
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    saved = saved_probe()
    summary = cast(dict[str, object], saved['summary'])
    sentinel: dict[str, object] = {
        'n': 424242, 'b': 41, 'c': 7, 'both_correct': 555, 'both_wrong': 777,
        'sum_d': 9037, 'sum_d2': 77123, 'delta': 0.12345, 'p_one_sided': 0.987654,
        'normal_95_interval': [1.5, 2.5],
        'loss_difference': {'direction': 'C-A', 'sum': 3.5, 'sum_squares': 9.25,
                            'mean': 0.007}}
    summary['comparisons'] = {'C-A': {split: dict(sentinel)
                                      for split in ('train', 'dev', 'test')},
                              'C-B': {split: dict(sentinel)
                                      for split in ('train', 'dev', 'test')}}
    files = render_feature_probe_report(cast(Mapping[str, object], saved))
    for name in ('index.html', 'report.md'):
        # Renderer entity-escapes punctuation; inspect presented content.
        text = html_module.unescape(files[name].decode())
        assert '424242' in text and '555' in text and '777' in text
        assert '41' in text and '7' in text
        assert '12.345 个百分点' in text
        assert '0.987654' in text  # saved test statistic is shown verbatim
        # Recomputed b-c=34 / b+c=48 must not replace saved sum_d/sum_d2.
        assert 'sum_d=34' not in text and 'sum_d2=48' not in text
        assert '9037' in text and '77123' in text


def test_probe_report_missing_arms_are_not_zero_filled() -> None:
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    saved = saved_probe()
    summary = cast(dict[str, object], saved['summary'])
    summary['arms'] = {}
    summary['comparisons'] = {}
    summary['pairing'] = {}
    summary['arm_resources'] = {}
    files = render_feature_probe_report(cast(Mapping[str, object], saved))
    html = files['index.html'].decode()
    assert '未保存（不补 0）' in html
    assert '0.000%' not in html


def test_probe_report_pairing_requires_all_relevant_arm_evidence() -> None:
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    sha_a = 'a' * 64
    sha_c = 'c' * 64
    cases: list[tuple[dict[str, object], dict[str, object], str, str]] = [
        ({}, {}, '证据不完整（未保存）', '证据不完整（未保存）'),
        ({'A': sha_a}, {}, '证据不完整（未保存）', '证据不完整（未保存）'),
        ({'C': sha_c}, {'C': sha_c}, '证据不完整（未保存）', '证据不完整（未保存）'),
        ({'A': sha_a, 'B': sha_a}, {'A': sha_a, 'B': sha_a},
         '相同', '证据不完整（未保存）'),
        ({'A': 'not-a-checksum', 'B': 'not-a-checksum'}, {},
         '证据不完整（未保存）', '证据不完整（未保存）'),
        ({'A': sha_a, 'B': sha_a, 'C': sha_c}, {'A': sha_a, 'B': sha_a, 'C': sha_c},
         '相同', '不同'),
        ({'A': sha_a, 'B': sha_a, 'C': sha_a}, {'A': sha_a, 'B': sha_a, 'C': sha_a},
         '相同', '相同'),
    ]
    for identities, shared, ab_text, shared_text in cases:
        saved = saved_probe()
        summary = cast(dict[str, object], saved['summary'])
        summary['pairing'] = {'initial_state_sha256': identities,
                              'shared_layers_sha256': shared,
                              'shuffle_seed': 1, 'init_seed': 2,
                              'initialization_rule': 'layer-streams-v1', 'panels': {}}
        html = render_feature_probe_report(
            cast(Mapping[str, object], saved))['index.html'].decode()
        # None == None and a lone value can never be presented as agreement.
        assert f'甲与乙整份初始参数校验值：{ab_text}' in html, (identities, shared)
        assert f'三条路线共有的隐藏层/输出层初值校验值：{shared_text}' in html, (
            identities, shared)


def test_probe_report_states_fixed_main_comparison_rule_and_threshold() -> None:
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    body = _body(render_feature_probe_report(
        cast(Mapping[str, object], saved_probe()))['index.html'].decode())
    # Fixed C-A verdict rule is part of the report, not only a raw appendix field.
    assert '0.025' in body
    assert '大于 0' in body
    assert '单侧尾概率' in body and '配对精确检验' in body
    assert '不是“方法无效”的概率' in body
    assert '探索' in body and '跨密钥' in body
    assert '不自动' in body and '不重新计算' in body
    assert '次要比较' in body
    assert '训练集或开发集' in body  # never picked for a significant-looking result
    assert '0.0625' not in body  # the saved statistic itself stays in the appendix


@pytest.mark.parametrize('missing', ['all', 'delta', 'p'])
def test_probe_report_cannot_adjudicate_without_saved_comparison(missing: str) -> None:
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    saved = saved_probe()
    summary = cast(dict[str, object], saved['summary'])
    comparisons = cast(dict[str, object], summary['comparisons'])
    if missing == 'all':
        summary['comparisons'] = {}
    else:
        test = cast(dict[str, object], cast(dict[str, object],
                    comparisons['C-A'])['test'])
        test.pop('delta' if missing == 'delta' else 'p_one_sided')
    body = _body(render_feature_probe_report(
        cast(Mapping[str, object], saved))['index.html'].decode())
    assert '不能按规则裁决' in body
    assert '不自动' in body


@pytest.mark.parametrize(('mode', 'present', 'absent'), [
    ('cipher', ('两次加密', '私有答案', '公开参考'),
     ('独立的空标签随机抽样',)),
    ('null', ('一次公开参考加密', '独立的空标签随机抽样', '与明文、密钥都无关'),
     ('两次加密',)),
    ('missing', ('没有写明 label_mode', '不推定'),
     ('两次加密', '一次公开参考加密')),
])
def test_probe_report_distinguishes_answer_sources(mode: str,
                                                   present: tuple[str, ...],
                                                   absent: tuple[str, ...]) -> None:
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    saved = saved_probe()
    recipe = cast(dict[str, object], saved['recipe'])
    if mode == 'missing':
        recipe.pop('label_mode')
    else:
        recipe['label_mode'] = mode
    html = render_feature_probe_report(cast(Mapping[str, object], saved))[
        'index.html'].decode()
    # Renderer entity-escapes punctuation/underscores for safe dual rendering.
    text = html_module.unescape(html)
    for token in present:
        assert token in text, (mode, token)
    for token in absent:
        assert token not in text, (mode, token)
    body = _body(html)
    for token in FORBIDDEN_IN_BODY:
        assert token not in body, token


@pytest.mark.parametrize(('backend', 'synthetic', 'present', 'absent'), [
    ('demo', True,
     ('模拟', '不执行真实密码', '私有答案', '公开参考', '两次加密'),
     ('答案来自真实加密', '调用方声明的生产适配器')),
    ('simon', False,
     ('调用方声明', '本页不认证', '两次加密'),
     ('模拟', '不执行真实密码')),
    ('UNVERIFIED', None,
     ('无法确认', '不推定'),
     ('答案来自真实加密', '不执行真实密码', '调用方声明的生产适配器')),
])
def test_probe_report_cipher_label_path_matches_saved_backend(
        backend: str, synthetic: bool | None, present: tuple[str, ...],
        absent: tuple[str, ...]) -> None:
    from cipher_lab_plugins.feature_probe.report import render_feature_probe_report

    saved = saved_probe()
    summary = cast(dict[str, object], saved['summary'])
    summary['backend_identity'] = backend
    summary['synthetic'] = synthetic
    html = render_feature_probe_report(cast(Mapping[str, object], saved))[
        'index.html'].decode()
    text = html_module.unescape(html)
    # The demo path is simulated counting, not a real cipher; the declared
    # production adapter is not certified by this page; an unverified identity
    # never claims real encryption. The top-level disclaimer alone is not used
    # to paper over an internally contradictory method paragraph.
    for token in present:
        assert token in text, (backend, token)
    for token in absent:
        assert token not in text, (backend, token)
    for token in FORBIDDEN_IN_BODY:
        assert token not in _body(html), token


def test_saved_report_dispatch_and_escape_for_probe(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = RunStore.claim(tmp_path, 'probe', output_bytes=10000000, needs_models=True)
    saved = saved_probe()
    summary = cast(Mapping[str, object], saved['summary'])
    store.finalize({'status': 'SUCCEEDED', 'summary': summary,
                    'counts': {'updates': 3},
                    'recipe': {'method': 'bit-public-feature-probe-v1'}})
    code = cli._saved_report(store, tmp_path, 'probe', 'demo')
    output = json.loads(capsys.readouterr().out)
    assert code == 0 and output['report_status'] == 'SAVED'
    assert output['backend_identity'] == 'demo'
    html = (tmp_path / 'result' / 'probe' / 'index.html').read_text()
    assert '<script>' not in html and '&lt;script&gt;' in html
    source = json.loads((tmp_path / 'result' / 'probe' / 'source.json').read_text())
    assert source['run_sha256'] == read_saved(tmp_path, 'probe')['run_sha256']
