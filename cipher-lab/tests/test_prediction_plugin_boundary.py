"""Static/fake-only migration contract; never execute scientific entry points."""
import ast
import builtins
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('owner,config', [('residual', 'residual-paired-example.json'),
    ('feature_probe', 'feature-probe-example.json'), ('frontier', 'frontier-example.json')])
def test_explicit_plugin_check(owner: str, config: str, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'schema': 2, 'wall_seconds': 120, 'cpu_seconds': 120,
        'deadline_utc': '2099-01-01T00:00:00Z', 'real_blocks': 100, 'null_words': 100,
        'optimizer_updates': 100, 'forward_rows': 1000, 'rss_stop_bytes': 1000000,
        'output_bytes': 100000000}))
    original = builtins.__import__
    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split('.')[0] in {'torch', 'numpy', 'secrets'} or name.startswith(
                ('cipher_lab.ciphers', 'cipher_lab.execution', 'cipher_lab.models')):
            raise AssertionError('science import during check')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    from cipher_lab.cli import main
    arguments = ['check', '--plugin', f'cipher_lab_plugins.{owner}.cli', '--config',
                 str(ROOT / 'configs' / config), '--policy', str(policy)]
    if owner == 'frontier':
        arguments += ['--public-plan', str(ROOT / 'plugins/cipher_lab_plugins/frontier/tests'
                                           / 'fixtures/frontier/r4-public-plan.json')]
    assert main(arguments) == 0
    assert not capsys.readouterr().err


@pytest.mark.parametrize('config', ['residual-paired-example.json',
    'feature-probe-example.json', 'frontier-example.json'])
def test_core_rejects_owned_methods(config: str) -> None:
    from cipher_lab.config import load_experiment
    with pytest.raises(ValueError):
        load_experiment(ROOT / 'configs' / config)


def test_source_supplier_fresh(tmp_path: Path) -> None:
    from cipher_lab.artifacts import RunStore
    source = {'cipher_lab_plugins/residual/cli.py': 'a' * 64}
    store = RunStore.claim(tmp_path, 'fake', output_bytes=100000,
                           source_supplier=lambda: dict(source))
    assert store.source_snapshot() == source
    source['cipher_lab_plugins/residual/cli.py'] = 'b' * 64
    assert store.current_source_identity() == source
    assert store.source_snapshot() != store.current_source_identity()


def test_core_has_no_owned_imports() -> None:
    banned = ('residual', 'feature_probe', 'frontier', 'simon_r3', 'simon_anf',
              'simon_frontier', 'frontier_inputs', 'span_compression', 'three_arm')
    for path in (ROOT / 'src/cipher_lab').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not any(p in banned for p in (node.module or '').split('.')), path


@pytest.mark.parametrize('owner,method', [
    ('residual', 'bit-residual-paired-v1'),
    ('feature_probe', 'bit-public-feature-probe-v1'),
    ('frontier', 'bit-public-feature-frontier-v1')])
def test_plugin_saved_evaluate_has_no_training_or_generation_imports(owner: str, method: str,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fake loaders stop before forward and still exercise strict owner identity gate."""
    import importlib
    from types import SimpleNamespace

    import torch
    from cipher_lab_plugins.prediction_common import bundles, checkpoints, evaluation
    from test_training_steps import make_meter

    from cipher_lab.artifacts import RunStore, read_saved
    original = builtins.__import__
    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(('cipher_lab.training', 'cipher_lab.ciphers',
                'cipher_lab.data.prediction', 'cipher_lab.data.differential',
                'cipher_lab_plugins.prediction_common.data')) or name.endswith('.training'):
            raise AssertionError('independent evaluate imported training/generation')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(evaluation, 'read_source', lambda *args: b'artificial')
    monkeypatch.setattr(checkpoints, 'load_checkpoint', lambda *args: SimpleNamespace(
        public_model_spec=SimpleNamespace(method='bit-prediction-v1'), task_declarations=()))
    monkeypatch.setattr(bundles, 'decode_bundle', lambda *args: SimpleNamespace(
        method='bit-prediction-v1', task_declarations=()))
    module = importlib.import_module(f'cipher_lab_plugins.{owner}.evaluation')
    store = RunStore.claim(tmp_path, 'wrong', output_bytes=1000000)
    module.run_evaluation(Path('fake'), 'a'*64, Path('fake'), 'b'*64, make_meter(), store)
    saved = read_saved(tmp_path, 'wrong')
    assert saved['status'] == 'FAILED'
    assert saved['summary']['meter']['attempted_forward_rows'] == 0  # type: ignore[index]
    assert torch.is_grad_enabled()


@pytest.mark.parametrize('name', ['/absolute/x.py', '../escape.py', 'a//x.py',
    'a/../x.py', 'a/./x.py', 'a\\x.py', 'a/private.key', 'a/x.py/extra.py'])
def test_unsafe_supplier_keys_rejected_before_output(name: str, tmp_path: Path) -> None:
    from cipher_lab.artifacts import ArtifactError, RunStore
    with pytest.raises(ArtifactError):
        RunStore.claim(tmp_path, 'unsafe', output_bytes=100000,
                       source_supplier=lambda: {name: 'a'*64})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('key', ['cipher_lab/legacy.py',
    'cipher_lab_plugins/frontier/cli.py', 'plugins/study/runtime/start-main.ps1'])
def test_source_key_compatibility_and_saved_reader(key: str, tmp_path: Path) -> None:
    from cipher_lab.artifacts import RunStore, read_saved
    store = RunStore.claim(tmp_path, 'keys', output_bytes=100000,
                           source_supplier=lambda: {key: 'a'*64})
    store.finalize({'status': 'SUCCEEDED'})
    assert read_saved(tmp_path, 'keys')['source'] == {key: 'a'*64}


@pytest.mark.parametrize('command', ['check', 'run'])
def test_core_rejects_public_plan_flag_before_claim_or_scientific_import(command: str,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = builtins.__import__
    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(('cipher_lab.artifacts', 'cipher_lab.execution', 'cipher_lab.data',
                'cipher_lab.models', 'cipher_lab.ciphers')) or name.split('.')[0] in {
                    'torch', 'numpy', 'secrets'}:
            raise AssertionError('base flag rejection crossed execution boundary')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    from cipher_lab.cli import main
    args = [command, '--config', 'missing.json', '--policy', 'missing.json',
            '--public-plan', 'missing.json']
    if command == 'run':
        args += ['--backend', 'demo', '--root', str(tmp_path), '--id', 'never']
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and not list(tmp_path.iterdir())
