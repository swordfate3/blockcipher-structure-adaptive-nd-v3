"""Frontier CLI/config/saved-report wiring; every ledger is artificial.

Autouse `poison` from frontier_fakes forbids production forward/backward/
optimizer/cipher/private RNG/production RSS before any test body. No production
science runs here: the end-to-end CLI is a clean-source subprocess whose script
installs the same forward+optimizer poison *before* importing the plugin CLI.
"""
from __future__ import annotations

import builtins
import hashlib
import html as html_module
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import cipher_lab_plugins.prediction_common.cli as cli
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter
from cipher_lab_plugins.frontier.cli import load_experiment
from cipher_lab_plugins.frontier.cli import main as plugin_main
from cipher_lab_plugins.frontier.config import FrontierRecipe
from cipher_lab_plugins.frontier.tests.frontier_fakes import plan, plan_path, poison
from cipher_lab_plugins.prediction_common.cli import _estimate

__all__ = ['poison']

PROJECT = Path(__file__).resolve().parents[4]
# Captured at import (before the autouse poison patches subprocess.Popen): the
# parent test process still needs to launch one guarded clean-source child.
_POPEN = subprocess.Popen


def run_child(arguments: list[str], *, cwd: Path, env: dict[str, str]
              ) -> tuple[int, str, str]:
    process = _POPEN(arguments, cwd=cwd, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, env=env)
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr
R4_PLAN = plan_path(4)
R5_PLAN = plan_path(5)

AUDITED = {
    4: ('simon-r4-residual89-v1', 89,
        'bb87a12a789b9713f0d79875ac85b560b05fe938817550182ec6d90b5bae2968',
        '21038a42d331559d5f3b5252489849f1ae10b05571541fcc328c942108754dff'),
    5: ('simon-r5-factored-v1', 8098,
        '43f03e294b5b7849ae23b6d592a01ca89c9f609f652d5dbe50be1e0139f9b409',
        '3554d0ea9fa0a38ba2811bbc69dcbbf8019ba1a7f9ac0e72e41ad9c2d1dd4841'),
}


def ledger_meter(real_blocks: int = 1000, null_words: int = 1000,
                 updates: int = 100, forward_rows: int = 10000) -> Meter:
    return Meter(
        Policy(2, 60., 60., '2031-01-01T00:00:00Z', real_blocks, null_words,
               1000000, 10000000, updates, forward_rows),
        Clocks(lambda: 0., lambda: 0., lambda: datetime(2030, 1, 1, tzinfo=UTC)),
        lambda: 0,
    )


def write_policy(tmp_path: Path, *, real_blocks: int = 300000,
                 optimizer_updates: int = 20000,
                 forward_rows: int = 5000000) -> Path:
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({
        'schema': 2, 'wall_seconds': 120, 'cpu_seconds': 120,
        'deadline_utc': '2099-01-01T00:00:00Z', 'real_blocks': real_blocks,
        'null_words': 100, 'rss_stop_bytes': 2000000000,
        'output_bytes': 64 * 1024 * 1024, 'optimizer_updates': optimizer_updates,
        'forward_rows': forward_rows,
    }))
    return path


def experiment(name: str) -> FrontierRecipe:
    loaded = load_experiment(PROJECT / 'configs' / name)
    assert isinstance(loaded, FrontierRecipe)
    return loaded


def formal(rounds: int) -> dict[str, Any]:
    representation, dimension, order_sha, plan_sha = AUDITED[rounds]
    return {
        'schema': 1, 'method': 'bit-public-feature-frontier-v1',
        'reference': 'simon64-zero-master-v1',
        'identity': {'rounds': rounds, 'target_bit': 0,
                     'representation': representation, 'dimension': dimension,
                     'order_sha256': order_sha, 'plan_sha256': plan_sha},
        'model': {'hidden_width': 128},
        'training': {'train': 16384, 'dev': 4096, 'test': 16384, 'batch_size': 256,
                     'epochs': 32, 'learning_rate': 0.001, 'init_seed': 31,
                     'sample_seed': 196905 + rounds, 'shuffle_seed': 296905 + rounds},
        'rounds': rounds, 'target_bit': 0, 'label_mode': 'cipher',
    }


def tiny() -> dict[str, Any]:
    representation, dimension, order_sha, plan_sha = AUDITED[4]
    return {
        'schema': 1, 'method': 'bit-public-feature-frontier-v1',
        'reference': 'simon64-zero-master-v1',
        'identity': {'rounds': 4, 'target_bit': 0,
                     'representation': representation, 'dimension': dimension,
                     'order_sha256': order_sha, 'plan_sha256': plan_sha},
        'model': {'hidden_width': 2},
        'training': {'train': 2, 'dev': 2, 'test': 2, 'batch_size': 2, 'epochs': 1,
                     'learning_rate': 0.001, 'init_seed': 31, 'sample_seed': 196909,
                     'shuffle_seed': 296909},
        'rounds': 4, 'target_bit': 0, 'label_mode': 'cipher',
    }


# ---------------------------------------------------------------------------
# Config dispatch and static estimates


def test_load_experiment_dispatches_frontier_and_old_methods_unchanged() -> None:
    from cipher_lab.training_config import PredictionRecipe
    from cipher_lab_plugins.feature_probe.config import FeatureProbeRecipe

    assert isinstance(load_experiment(PROJECT / 'configs/frontier-r4.json'),
                      FrontierRecipe)
    assert isinstance(load_experiment(PROJECT / 'configs/frontier-example.json'),
                      FrontierRecipe)
    from cipher_lab.config import load_experiment as load_base
    from cipher_lab_plugins.feature_probe.cli import load_experiment as load_probe
    assert isinstance(load_base(PROJECT / 'configs/prediction-example.json'),
                      PredictionRecipe)
    assert isinstance(load_probe(PROJECT / 'configs/feature-probe-r3.json'),
                      FeatureProbeRecipe)


def test_frontier_estimate_counts_three_arms_and_public_reference() -> None:
    estimate = _estimate(experiment('frontier-r4.json'))
    assert estimate['planned_real_blocks'] == 2 * (16384 + 4096 + 16384) == 73728
    assert estimate['planned_null_words'] == 0
    assert estimate['planned_optimizer_updates'] == 3 * 32 * 16384 // 256 == 6144
    assert estimate['planned_forward_rows'] == 3 * (32 * (16384 + 4096)
                                                    + 36864) == 2076672
    basis = str(estimate['basis']).lower()
    assert 'three fixed arms' in basis and 'reference' in basis and 'test' in basis
    units = cast(Mapping[str, object], estimate['units'])
    assert 'zero-master-key' in str(units['real_blocks'])
    assert 'any arm' in str(units['forward_rows'])


def test_frontier_example_estimate_is_strictly_tiny() -> None:
    estimate = _estimate(experiment('frontier-example.json'))
    assert estimate['planned_real_blocks'] == 12
    assert estimate['planned_null_words'] == 0
    assert estimate['planned_optimizer_updates'] == 3
    assert estimate['planned_forward_rows'] == 30


def test_frontier_null_estimate_charges_reference_and_null_labels() -> None:
    from dataclasses import replace

    null = replace(experiment('frontier-example.json'), label_mode='null')
    estimate = _estimate(null)
    assert estimate['planned_real_blocks'] == 6
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
    from cipher_lab_plugins.feature_probe.cli import load_experiment as load_probe
    from cipher_lab_plugins.prediction_common.cli import _estimate as estimate_owned
    probe = estimate_owned(load_probe(PROJECT / 'configs/feature-probe-example.json'))
    assert probe['planned_real_blocks'] == 12
    assert probe['planned_optimizer_updates'] == 3
    assert probe['planned_forward_rows'] == 30


# ---------------------------------------------------------------------------
# Frozen, audited formal configs; no invented r6 admission


@pytest.mark.parametrize('rounds', [4, 5])
def test_formal_frontier_config_is_the_frozen_audited_identity(rounds: int) -> None:
    path = PROJECT / 'configs' / f'frontier-r{rounds}.json'
    raw = json.loads(path.read_text())
    expected = formal(rounds)
    assert raw == expected
    loaded = experiment(path.name)
    assert loaded.to_dict() == expected
    # The audited identity must equal the frozen plan bytes, not a new SHA claim.
    frozen = plan(rounds)
    assert loaded.identity == frozen.identity
    assert 'key_seed' not in path.read_text()
    assert loaded.label_mode == 'cipher'


def test_no_r6_config_is_invented_without_admission() -> None:
    assert not (PROJECT / 'configs/frontier-r6.json').exists()
    assert not (PROJECT / 'configs/frontier-r6-example.json').exists()


def test_example_recipe_is_complete_strict_and_not_authorization() -> None:
    raw = json.loads((PROJECT / 'configs/frontier-example.json').read_text())
    assert raw == tiny()
    assert experiment('frontier-example.json').to_dict() == tiny()
    assert raw['model'] == {'hidden_width': 2}
    assert 'key_seed' not in raw and 'key_seed' not in json.dumps(raw)


# ---------------------------------------------------------------------------
# CLI check: pure stdlib bounded plan read, no artifacts


def test_check_requires_plan_for_frontier_and_rejects_it_for_old_methods(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(tmp_path)
    missing = plugin_main(['check', '--config', str(PROJECT / 'configs/frontier-r4.json'),
                        '--policy', str(policy)])
    assert missing == 2
    old = plugin_main(['check', '--config', str(PROJECT / 'configs/prediction-example.json'),
                    '--policy', str(policy), '--public-plan', str(R4_PLAN)])
    assert old == 2
    errors = capsys.readouterr().err
    assert 'frontier requires --public-plan' in errors
    assert 'invalid or unreadable settings' in errors


def test_evaluate_has_no_public_plan_flag(tmp_path: Path) -> None:
    policy = write_policy(tmp_path)
    with pytest.raises(SystemExit):
        plugin_main(['evaluate', '--checkpoint', str(tmp_path / 'c.pt'),
                  '--checkpoint-sha256', 'a' * 64,
                  '--dataset', str(tmp_path / 'd.npz'),
                  '--dataset-sha256', 'b' * 64, '--policy', str(policy),
                  '--root', str(tmp_path / 'out'), '--id', 'eval',
                  '--public-plan', str(R4_PLAN)])
    assert not (tmp_path / 'out').exists()


def test_check_rejects_missing_mismatched_and_oversized_plans(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(tmp_path)
    absent = plugin_main(['check', '--config', str(PROJECT / 'configs/frontier-r4.json'),
                       '--policy', str(policy),
                       '--public-plan', str(tmp_path / 'absent.json')])
    assert absent == 2
    mismatched = plugin_main(['check', '--config', str(PROJECT / 'configs/frontier-r4.json'),
                           '--policy', str(policy), '--public-plan', str(R5_PLAN)])
    assert mismatched == 2
    oversized = tmp_path / 'oversized-plan.json'
    oversized.write_bytes(b'x' * (16 * 1024 * 1024 + 1))
    too_big = plugin_main(['check', '--config', str(PROJECT / 'configs/frontier-r4.json'),
                        '--policy', str(policy), '--public-plan', str(oversized)])
    assert too_big == 2
    assert not any(tmp_path.iterdir()) or all(
        path.name in {'policy.json', 'oversized-plan.json'} for path in tmp_path.iterdir())
    capsys.readouterr()


def test_check_subprocess_imports_no_science_and_writes_nothing(tmp_path: Path) -> None:
    policy = write_policy(tmp_path, real_blocks=300000, optimizer_updates=20000,
                          forward_rows=5000000)
    plan_copy = tmp_path / 'r4-public-plan.json'
    plan_copy.write_bytes(R4_PLAN.read_bytes())
    arguments = ['check', '--config', str(PROJECT / 'configs/frontier-r4.json'),
                 '--policy', str(policy), '--public-plan', str(plan_copy)]
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
from cipher_lab_plugins.frontier.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    code, stdout, stderr = run_child(
        [sys.executable, '-c', script, *arguments], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                              (str(PROJECT / 'src'), str(PROJECT / 'plugins'))),
                 PYTHONDONTWRITEBYTECODE='1'))
    assert code == 0, stderr
    output = json.loads(stdout)
    estimate = output['estimate']
    assert estimate['planned_real_blocks'] == 73728
    assert estimate['planned_optimizer_updates'] == 6144
    assert estimate['planned_forward_rows'] == 2076672
    assert output['planned_calls_fit_policy'] == {
        'real_blocks': True, 'null_words': True,
        'optimizer_updates': True, 'forward_rows': True}
    assert output['recipe']['identity'] == formal(4)['identity']
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


# ---------------------------------------------------------------------------
# CLI run: plan is required, read before claim, and identity is recorded


def test_run_requires_plan_and_rejects_old_method_plan_and_ciphers(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    policy = write_policy(tmp_path)
    absent = plugin_main(['run', '--config', str(PROJECT / 'configs/frontier-example.json'),
                       '--policy', str(policy), '--root', str(tmp_path / 'a'),
                       '--id', 'frontier', '--backend', 'demo'])
    assert absent == 2 and not (tmp_path / 'a').exists()
    old = plugin_main(['run', '--config', str(PROJECT / 'configs/prediction-example.json'),
                    '--policy', str(policy), '--root', str(tmp_path / 'b'),
                    '--id', 'old', '--backend', 'demo', '--public-plan', str(R4_PLAN)])
    assert old == 2 and not (tmp_path / 'b').exists()
    ciphers = plugin_main(['run', '--config', str(PROJECT / 'configs/frontier-example.json'),
                        '--policy', str(policy), '--public-plan', str(R4_PLAN),
                        '--root', str(tmp_path / 'c'), '--id', 'frontier',
                        '--backend', 'ciphers'])
    assert ciphers == 2 and not (tmp_path / 'c').exists()
    capsys.readouterr()


def test_engineering_limit_is_reported_before_any_prediction_data(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """output_bound must run before PredictionData.__init__: no engineering science."""
    from dataclasses import replace

    from cipher_lab_plugins.prediction_common import data as prediction

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('PredictionData constructed before output_bound')

    monkeypatch.setattr(prediction.PredictionData, '__init__', forbidden)
    policy = write_policy(tmp_path)
    recipe = experiment('frontier-example.json')
    training = replace(recipe.training, epochs=33)  # beyond the fixed <=32 bound
    over = replace(recipe, training=training)
    monkeypatch.setattr('cipher_lab_plugins.frontier.cli.parse_frontier', lambda value: over)
    root = tmp_path / 'out'
    code = plugin_main(['run', '--config', str(PROJECT / 'configs/frontier-example.json'),
                     '--policy', str(policy), '--public-plan', str(R4_PLAN),
                     '--root', str(root), '--id', 'frontier', '--backend', 'demo'])
    output = json.loads(capsys.readouterr().out)
    assert code == 4
    assert output['scientific_status'] == 'FAILED'
    saved = read_saved(root, 'frontier')
    summary = cast(Mapping[str, object], saved['summary'])
    assert summary['reason'] == 'ENGINEERING_LIMIT'
    # Identity was committed before the bound was evaluated; no panel/model exists.
    assert not (root / 'data' / 'frontier' / 'dataset').exists()
    assert not list((root / 'models' / 'frontier').iterdir())  # claimed, empty


def test_plan_identity_and_config_reference_are_saved(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.prediction_common import application as application

    seen: dict[str, object] = {}

    def fake_prepare(recipe: object, meter: object, store: RunStore,
                     backend: str, plan: object = None, identity: object = None) -> Any:
        seen['plan'] = plan
        seen['identity'] = identity

        def run() -> None:
            store.finalize({'status': 'SUCCEEDED', 'summary': {
                'method': 'bit-public-feature-frontier-v1',
                'backend_identity': backend, 'scientific_validation': False,
                'synthetic': backend == 'demo', 'reason': '', 'arms': {}}})
        return run

    monkeypatch.setattr(application, 'prepare_training', fake_prepare)
    policy = write_policy(tmp_path)
    root = tmp_path / 'out'
    code = plugin_main(['run', '--config', str(PROJECT / 'configs/frontier-example.json'),
                     '--policy', str(policy), '--public-plan', str(R4_PLAN),
                     '--root', str(root), '--id', 'frontier', '--backend', 'demo'])
    assert code == 0
    saved = read_saved(root, 'frontier')
    payloads = cast(Mapping[str, Mapping[str, object]], saved['payloads'])
    identity = payloads['payload/plan-identity.json']
    assert identity['plan_sha256'] == hashlib.sha256(R4_PLAN.read_bytes()).hexdigest()
    assert identity['config_ref'] == 'frontier-example.json'
    assert identity['plan_ref'] == R4_PLAN.name
    assert cast(Mapping[str, object], identity['identity'])['dimension'] == 89
    assert seen['plan'] is not None


# ---------------------------------------------------------------------------
# Clean-source fake CLI: check -> run -> read_saved -> explicit evaluate ABC


GUARD = '''
import secrets, subprocess, sys
import torch
from torch import Tensor

CALLS = []


def forbidden(*args, **kwargs):
    CALLS.append('forbidden')
    raise AssertionError('FORBIDDEN production science/subprocess')


def manual(model, inputs, *args, **kwargs):
    bias = model.state_dict().get('layers.4.bias', torch.zeros(1))
    return bias.detach().reshape(1, 1).expand(len(inputs), 1).clone()


secrets.token_bytes = forbidden
secrets.randbits = forbidden
secrets.randbelow = forbidden
torch.optim.Adam.__init__ = forbidden
torch.optim.Adam.step = forbidden
torch.Tensor.backward = forbidden
torch.autograd.backward = forbidden
torch.nn.Linear.forward = forbidden
torch.nn.functional.linear = forbidden
torch.optim.Adam.zero_grad = forbidden
for cls in (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.GroupNorm):
    cls.forward = forbidden
for name in ('conv1d', 'conv2d', 'conv3d', 'group_norm'):
    setattr(torch.nn.functional, name, forbidden)
subprocess.Popen = forbidden
from cipher_lab.models import BitMLP
from cipher_lab_plugins.feature_probe.model import FeatureBitMLP
from cipher_lab.models import SharedM
from cipher_lab_plugins.frontier.model import FrontierBitMLP
for cls in (BitMLP, FeatureBitMLP, SharedM, FrontierBitMLP):
    cls.forward = manual
import cipher_lab.execution as execution
from cipher_lab.demo import demo_rss_bytes
execution.production_rss_bytes = demo_rss_bytes
import importlib
for family in ('simon', 'simeck', 'present', 'skinny'):
    module = importlib.import_module('cipher_lab.ciphers.' + family)
    for name in ('encrypt_many', 'encrypt_batch', 'encrypt'):
        if hasattr(module, name):
            setattr(module, name, forbidden)
PROBES = []


def _probe(name, call):
    try:
        call()
    except AssertionError:
        PROBES.append(name)
    else:
        raise AssertionError('poison missing: ' + name)


_probe('secrets.token_bytes', lambda: secrets.token_bytes(1))
_probe('secrets.randbits', lambda: secrets.randbits(8))
_probe('secrets.randbelow', lambda: secrets.randbelow(8))
_probe('torch.nn.functional.linear', lambda: torch.nn.functional.linear(None, None))
_probe('torch.Tensor.backward', lambda: torch.Tensor.backward(None))
_probe('torch.optim.Adam.step', lambda: torch.optim.Adam.step(None))
_probe('subprocess.Popen', lambda: subprocess.Popen(None))
PROBE_CALLS = len(CALLS)
import cipher_lab_plugins.frontier.training as frontier
def fake_step(model, inputs, labels, context, optimizer, meter):
    meter.reserve('optimizer_updates', 1)
    meter.reserve('forward_rows', len(inputs))
    meter.complete('forward_rows', len(inputs))
    meter.complete('optimizer_updates', 1)
    return .7
frontier.make_optimizer = lambda *args: object()
frontier.train_step = fake_step
'''


def _clean_copy(tmp_path: Path) -> Path:
    root = tmp_path / 'clean'
    shutil.copytree(PROJECT / 'src', root / 'src',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(PROJECT / 'plugins', root / 'plugins',
                    ignore=shutil.ignore_patterns('__pycache__', 'tests'))
    shutil.copyfile(PROJECT / '.gitignore', root / '.gitignore')
    (root / 'plans').mkdir()
    shutil.copyfile(R4_PLAN, root / 'plans' / R4_PLAN.name)
    shutil.copyfile(PROJECT / 'configs/frontier-example.json',
                    root / 'frontier-example.json')
    return root


def _child_script() -> str:
    return GUARD + '''
import json, sys
from pathlib import Path
from cipher_lab.artifacts import read_saved
from cipher_lab.cli import main

config, policy, plan, root_text, run_id = sys.argv[1:6]
root = Path(root_text)
result = {}

def call(arguments):
    from io import StringIO
    from contextlib import redirect_stdout
    buffer = StringIO()
    with redirect_stdout(buffer):
        code = main([arguments[0], '--plugin', 'cipher_lab_plugins.frontier.cli',
                     *arguments[1:]])
    return code, json.loads(buffer.getvalue())

code, checked = call(['check', '--config', config, '--policy', policy,
                      '--public-plan', plan])
result['check'] = {'code': code, 'estimate': checked['estimate']}
code, ran = call(['run', '--config', config, '--policy', policy,
                  '--public-plan', plan, '--root', str(root), '--id', run_id,
                  '--backend', 'demo'])
result['run'] = {'code': code, 'output': ran}
saved = read_saved(root, run_id)
summary = saved['summary']
counts = saved['counts']
result['saved'] = {
    'status': saved['status'],
    'run_sha256': saved['run_sha256'],
    'meter': summary['meter'],
    'counts': {'updates': counts['updates'],
               'train_presentations': counts['train_presentations'],
               'dev_evaluations': counts['dev_evaluations']},
    'plan_identity': summary.get('plan_identity'),
    'output_bound': summary.get('output_bound'),
    'selected': {arm: summary['arms'][arm]['selected']['sha256'] for arm in 'ABC'},
    'models': sorted(path.name for path in (root / 'models' / run_id).iterdir()),
}
evaluates = []
for arm in 'ABC':
    checkpoint = root / 'models' / run_id / (arm + '-best.pt')
    dataset = root / 'data' / run_id / 'dataset' / 'test.npz'
    code, output = call([
        'evaluate',
        '--checkpoint', str(checkpoint),
        '--checkpoint-sha256', __import__('hashlib').sha256(
            checkpoint.read_bytes()).hexdigest(),
        '--dataset', str(dataset),
        '--dataset-sha256', __import__('hashlib').sha256(dataset.read_bytes()).hexdigest(),
        '--policy', policy, '--root', str(root), '--id', 'eval-' + arm])
    evaluated = read_saved(root, 'eval-' + arm)
    evaluates.append({'arm': arm, 'code': code, 'status': evaluated['status'],
                      'report': output.get('report_status'),
                      'rows': evaluated['summary']['metrics']['tasks']['SIMON']['rows']})
result['evaluates'] = evaluates
result['poison'] = {'probes': PROBES, 'probe_calls': PROBE_CALLS,
                    'calls_during_work': len(CALLS) - PROBE_CALLS}
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
'''


def test_clean_source_fake_cli_check_run_saved_and_three_evaluates(tmp_path: Path) -> None:
    clean = _clean_copy(tmp_path)
    policy = clean / 'policy.json'
    policy.write_text(json.dumps({
        'schema': 2, 'wall_seconds': 120, 'cpu_seconds': 120,
        'deadline_utc': '2099-01-01T00:00:00Z', 'real_blocks': 100,
        'null_words': 100, 'rss_stop_bytes': 2000000000,
        'output_bytes': 64 * 1024 * 1024, 'optimizer_updates': 100,
        'forward_rows': 100}))
    script = clean / 'child.py'
    script.write_text(_child_script())
    root = clean / 'out'
    code, stdout, stderr = run_child(
        [sys.executable, str(script), str(clean / 'frontier-example.json'),
         str(policy), str(clean / 'plans' / R4_PLAN.name), str(root), 'frontier'],
        cwd=clean,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                      (str(clean / 'src'), str(clean / 'plugins'))),
                 PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1'))
    assert code == 0, stderr
    result = json.loads(stdout)
    poison = result['poison']
    assert poison['probes'] == [
        'secrets.token_bytes', 'secrets.randbits', 'secrets.randbelow',
        'torch.nn.functional.linear', 'torch.Tensor.backward',
        'torch.optim.Adam.step', 'subprocess.Popen']
    assert poison['probe_calls'] == len(poison['probes'])  # every probe really fired
    assert poison['calls_during_work'] == 0  # no production science after probes
    assert result['check']['code'] == 0
    assert result['check']['estimate']['planned_real_blocks'] == 12
    assert result['check']['estimate']['planned_optimizer_updates'] == 3
    assert result['check']['estimate']['planned_forward_rows'] == 30
    assert result['run']['code'] == 0
    assert result['run']['output']['scientific_status'] == 'SUCCEEDED'
    assert result['run']['output']['report_status'] == 'SAVED'
    saved = result['saved']
    assert saved['status'] == 'SUCCEEDED'
    assert saved['meter']['completed_real_blocks'] == 12
    assert saved['meter']['completed_null_words'] == 0
    assert saved['meter']['completed_optimizer_updates'] == 3
    assert saved['meter']['completed_forward_rows'] == 30
    assert saved['counts'] == {'updates': 3, 'train_presentations': 6,
                               'dev_evaluations': 3}
    assert saved['models'] == ['A-best.pt', 'B-best.pt', 'C-best.pt']
    assert saved['plan_identity'] == formal(4)['identity']
    bound = saved['output_bound']
    assert isinstance(bound, dict) and bound['peak_bytes'] <= 128 * 1024 * 1024
    assert bound['metadata_bytes'] == 16 * 1024 * 1024
    assert set(saved['selected']) == {'A', 'B', 'C'}
    assert len(result['evaluates']) == 3
    for evaluated in result['evaluates']:
        assert evaluated['code'] == 0, evaluated
        assert evaluated['status'] == 'SUCCEEDED'
        assert evaluated['report'] == 'SAVED'
        assert evaluated['rows'] == 2  # each explicit fake evaluate sees the saved panel
    for name in ('report.md', 'index.html', 'source.json'):
        assert (root / 'result' / 'frontier' / name).is_file()
    html = (root / 'result' / 'frontier' / 'index.html').read_text()
    assert '公开组合' in html and '<script>' not in html
    source = json.loads((root / 'result' / 'frontier' / 'source.json').read_text())
    assert source['run_sha256'] == saved['run_sha256']


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


def _comparison(direction: str, p: float = 0.0625) -> dict[str, object]:
    return {'n': 400, 'b': 4, 'c': 0, 'both_correct': 296, 'both_wrong': 96,
            'sum_d': 4, 'sum_d2': 4, 'delta': 0.01, 'p_one_sided': p,
            'normal_95_interval': None,
            'loss_difference': {'direction': direction, 'sum': 1.5,
                                'sum_squares': -0.25, 'mean': 0.00375}}


def _identity() -> dict[str, object]:
    return cast(dict[str, object], formal(4)['identity'])


def saved_frontier() -> dict[str, object]:
    arms = {'A': _arm((.75, .76, .70), .70), 'B': _arm((.74, .77, .72), .71),
            'C': _arm((.80, .84, .81), .66)}
    summary: dict[str, object] = {
        'method': 'bit-public-feature-frontier-v1', 'backend_identity': 'demo',
        'synthetic': True, 'scientific_validation': False,
        'reason': '<script>x</script>',
        'plan_identity': _identity(),
        'output_bound': {'peak_bytes': 1200000, 'metadata_bytes': 16 * 1024 * 1024},
        'arms': arms,
        'comparisons': {'C-A': {split: _comparison('C-A')
                                for split in ('train', 'dev', 'test')},
                        'C-B': {split: _comparison('C-B')
                                for split in ('train', 'dev', 'test')}},
        'pairing': {
            'initial_state_sha256': {'A': 'a' * 64, 'B': 'a' * 64, 'C': 'b' * 64},
            'shared_layers_sha256': {'A': 'c' * 64, 'B': 'c' * 64, 'C': 'c' * 64},
            'shuffle_seed': 296909, 'init_seed': 31,
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
    return {'run_id': 'frontier', 'run_ref': 'data/frontier/run.json',
            'run_sha256': '1' * 64, 'status': 'SUCCEEDED',
            'recipe': {'method': 'bit-public-feature-frontier-v1',
                       'label_mode': 'cipher', 'rounds': 4, 'target_bit': 0},
            'counts': {'updates': 3}, 'summary': summary, 'source': {},
            'policy': {'schema': 2}}


def test_frontier_report_consumes_saved_values_with_zero_io_or_science(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab_plugins.frontier.report as report
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    before = deepcopy(saved)
    original = builtins.__import__

    def no_science(name: str, globals_: Mapping[str, object] | None = None,
                   locals_: Mapping[str, object] | None = None,
                   fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        if name.startswith(('torch', 'numpy', 'cipher_lab.data', 'cipher_lab.ciphers',
                            'cipher_lab.models', 'cipher_lab.execution',
                            'cipher_lab.evaluation', 'cipher_lab_plugins.frontier.training')):
            pytest.fail('report imported science or training')
        return original(name, globals_, locals_, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', no_science)
        patch.setattr(Path, 'read_bytes', lambda _self: pytest.fail('report I/O'))
        files = render_frontier_report(cast(Mapping[str, object], saved))
    assert set(files) == {'report.md', 'index.html', 'source.json'}
    html = files['index.html'].decode()
    markdown = files['report.md'].decode()
    for text in (html, markdown):
        # Ordinary-reader body: question, three methods, 0-100 cards, curves.
        assert '公开组合' in text and '直接预测' in text and '原始输入' in text
        assert '测试集猜对比例' in text
        assert '0.025' in text and '/3' in text and '探索' in text
        assert '55%' in text and '不是' in text and '校准' in text
        assert '描述' in text and '多数类' in text
        assert '下溢' not in text  # saved p is nonzero here
        assert '不是跨密钥' in text or '不构成跨密钥' in text
        assert '未保存（不补 0）' not in text  # all values are saved here
        assert '<script>' not in text
    assert html.count('<polyline') >= 4
    assert html.count('<svg') >= 3
    assert json.loads(files['source.json'])['run_sha256'] == '1' * 64
    assert saved == before
    assert report.render_frontier_report(
        cast(Mapping[str, object], saved))['index.html'] == files['index.html']


def test_frontier_report_prints_saved_fields_without_recompute() -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    summary = cast(dict[str, object], saved['summary'])
    sentinel: dict[str, object] = {
        'n': 424242, 'b': 41, 'c': 7, 'both_correct': 555, 'both_wrong': 777,
        'sum_d': 9037, 'sum_d2': 77123, 'delta': 0.12345, 'p_one_sided': 0.987654,
        'normal_95_interval': [1.5, 2.5],
        'loss_difference': {'direction': 'C-A', 'sum': 3.5,
                            'sum_squares': 9.25, 'mean': 0.007}}
    summary['comparisons'] = {key: {split: dict(sentinel)
                                    for split in ('train', 'dev', 'test')}
                              for key in ('C-A', 'C-B')}
    files = render_frontier_report(cast(Mapping[str, object], saved))
    expected = ('n=424242', 'b=41', 'c=7', 'both_correct=555', 'both_wrong=777',
                'sum_d=9037', 'sum_d2=77123', 'delta=12.345pp', 'p=9.876540e-01')
    for name in ('index.html', 'report.md'):
        text = html_module.unescape(files[name].decode())
        for field in expected:
            assert field in text, (name, field)
        assert 'sum_d=34' not in text and 'sum_d2=48' not in text
        assert '"comparisons"' in text


def _set_test_p(saved: dict[str, object], value: object, *, remove: bool = False) -> None:
    summary = cast(dict[str, object], saved['summary'])
    comparisons = cast(dict[str, object], summary['comparisons'])
    stats = cast(dict[str, object], cast(dict[str, object], comparisons['C-A'])['test'])
    if remove:
        del stats['p_one_sided']
    else:
        stats['p_one_sided'] = value


@pytest.mark.parametrize('value', [1e-8, 4.9e-7, 0.123456789])
def test_frontier_report_keeps_small_saved_p_nonzero(value: float) -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    _set_test_p(saved, value)
    files = render_frontier_report(cast(Mapping[str, object], saved))
    for name in ('index.html', 'report.md'):
        text = html_module.unescape(files[name].decode())
        # Saved nonzero p survives as scientific notation, never rounded to 0.
        assert f'{value:.6e}' in text, (name, value)
        assert 'p=0.000000；' not in text
        # No underflow claim anywhere when the saved probability is nonzero.
        assert '下溢' not in text, name


def test_frontier_report_zero_p_is_the_only_underflow_case() -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    _set_test_p(saved, 0.0)
    files = render_frontier_report(cast(Mapping[str, object], saved))
    for name in ('index.html', 'report.md'):
        text = html_module.unescape(files[name].decode())
        assert '0.000000e+00' in text, name
        assert '下溢' in text, name
        assert '不是概率精确为 0' in text, name


def test_frontier_report_missing_p_is_not_zero_and_claims_no_underflow() -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    _set_test_p(saved, None, remove=True)
    files = render_frontier_report(cast(Mapping[str, object], saved))
    for name in ('index.html', 'report.md'):
        text = html_module.unescape(files[name].decode())
        assert '未保存（不补 0）' in text, name
        assert '下溢' not in text, name
        assert '0.000000e+00' not in text, name


def test_frontier_report_missing_values_are_not_zero_filled() -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    summary = cast(dict[str, object], saved['summary'])
    summary['arms'] = {}
    summary['comparisons'] = {}
    summary['pairing'] = {}
    files = render_frontier_report(cast(Mapping[str, object], saved))
    text = html_module.unescape(files['index.html'].decode())
    assert '未保存（不补 0）' in text
    assert '猜对比例 0.000%' not in text
    assert 'ACC=0<' not in text


def test_frontier_report_pairing_requires_all_relevant_evidence() -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    summary = cast(dict[str, object], saved['summary'])
    pairing = cast(dict[str, object], summary['pairing'])
    shared = cast(dict[str, object], pairing['shared_layers_sha256'])
    del shared['C']
    files = render_frontier_report(cast(Mapping[str, object], saved))
    text = html_module.unescape(files['index.html'].decode())
    assert '证据不完整（未保存）' in text
    assert '甲与乙整份初始参数校验值：相同' in text
    assert '三条路线共有的隐藏层/输出层初值校验值：相同' not in text
    assert '预定初始化规则' in text
    assert '甲/乙初始参数逐字相同' not in text


@pytest.mark.parametrize('case', ['missing', 'invalid', 'different', 'consistent'])
def test_frontier_pairing_is_judged_only_from_complete_valid_sha(case: str) -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    summary = cast(dict[str, object], saved['summary'])
    pairing = cast(dict[str, object], summary['pairing'])
    identities = cast(dict[str, object], pairing['initial_state_sha256'])
    if case == 'missing':
        del identities['A']
    elif case == 'invalid':
        identities['A'] = 'Z' * 64
    elif case == 'different':
        identities['B'] = 'b' * 64
    else:
        identities['C'] = 'a' * 64  # complete and consistent with A/B
    files = render_frontier_report(cast(Mapping[str, object], saved))
    text = html_module.unescape(files['index.html'].decode())
    # The predefined rule is named as a rule, never as an unconditional fact.
    assert '预定初始化规则' in text
    assert '本案' in text or '本次' in text
    assert '甲/乙初始参数逐字相同' not in text
    if case == 'consistent':
        assert '甲与乙整份初始参数校验值：相同' in text
        assert '三条路线共有的隐藏层/输出层初值校验值：相同' in text
    elif case == 'different':
        assert '甲与乙整份初始参数校验值：不同' in text
    else:
        assert '甲与乙整份初始参数校验值：证据不完整（未保存）' in text
    if case != 'consistent':
        assert '甲与乙整份初始参数校验值：相同' not in text


def test_frontier_report_marks_engineering_limit_as_not_started() -> None:
    from cipher_lab_plugins.frontier.report import render_frontier_report

    saved = saved_frontier()
    saved['status'] = 'FAILED'
    summary = cast(dict[str, object], saved['summary'])
    summary['reason'] = 'ENGINEERING_LIMIT'
    files = render_frontier_report(cast(Mapping[str, object], saved))
    text = html_module.unescape(files['index.html'].decode())
    assert '工程未启动' in text
    assert 'ENGINEERING_LIMIT' in text
    assert '不是' in text and ('阴性' in text or '下限' in text)


def test_frontier_saved_report_dispatch_and_escape(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = RunStore.claim(tmp_path, 'frontier', output_bytes=100000000,
                           needs_models=True)
    saved = saved_frontier()
    summary = cast(dict[str, object], saved['summary'])
    store.finalize({'status': 'SUCCEEDED', 'summary': summary, 'counts': {'updates': 3},
                    'recipe': {'method': 'bit-public-feature-frontier-v1'}})
    code = cli._saved_report(store, tmp_path, 'frontier', 'demo')
    output = json.loads(capsys.readouterr().out)
    assert code == 0 and output['report_status'] == 'SAVED'
    html = (tmp_path / 'result' / 'frontier' / 'index.html').read_text()
    assert '<script>' not in html and '&lt;script&gt;' in html
    source = json.loads((tmp_path / 'result' / 'frontier' / 'source.json').read_text())
    assert source['run_sha256'] == read_saved(tmp_path, 'frontier')['run_sha256']
