"""Task7 CLI boundaries; only two tiny demo cases execute real Adam (2+4)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from cipher_lab import cli
from cipher_lab.artifacts import read_saved

PROJECT = Path(__file__).resolve().parents[1]


def settings(tmp_path: Path, shared: bool = False, *, null: bool = False) -> list[str]:
    name = 'shared-m' if shared else 'prediction'
    recipe = json.loads((PROJECT / f'configs/{name}-example.json').read_text())
    if shared:
        recipe['model'] = dict(stem_width=2, channels=2, norm_groups=1,
                               suffix_kernels=[3], head_hidden=2, router_hidden=2)
    else:
        recipe['model'] = {'hidden_width': 2}
        recipe['training']['train'] = 4
        if null:
            recipe['label_mode'] = 'null'
    config, policy = tmp_path / 'recipe.json', tmp_path / 'policy.json'
    config.write_text(json.dumps(recipe))
    policy.write_text(json.dumps(dict(schema=2, optimizer_updates=4, forward_rows=100,
        wall_seconds=60, cpu_seconds=60, deadline_utc='2099-01-01T00:00:00Z',
        real_blocks=192, null_words=8, rss_stop_bytes=1000000, output_bytes=20000000)))
    return ['--config', str(config), '--policy', str(policy)]


def subprocess_cli(script: str, args: list[str],
                   tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-c', script, *args], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(PROJECT / 'src'), PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        capture_output=True, text=True, check=False, timeout=60)


POISON = '''
import importlib.abc, sys, torch
# Import Torch first: its own internals may import secrets; application may not.
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'cipher_lab.training' or fullname.startswith((
            'cipher_lab.training.', 'cipher_lab.ciphers',
            'cipher_lab.data.prediction', 'cipher_lab.data.differential',
            'cipher_lab.data.cubes', 'cipher_experiments'))) or fullname == 'secrets':
            raise AssertionError('PRIVATE forbidden import')
sys.modules.pop('secrets', None)
def no_optimizer(*a, **k):
    raise AssertionError('PRIVATE optimizer construction')
torch.optim.Adam = no_optimizer
sys.meta_path.insert(0, Block())
import cipher_lab.execution as execution
execution.production_rss_bytes = lambda: 0  # simulated probe; never read real RSS
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''


@pytest.mark.parametrize('shared', [False, True])
def test_training_check_static_subprocess(tmp_path: Path, shared: bool) -> None:
    args = settings(tmp_path, shared)
    before = sorted(tmp_path.rglob('*'))
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('torch', 'numpy', 'cipher_lab.data', 'cipher_lab.ciphers',
                                'cipher_lab.training.')):
            raise AssertionError('PRIVATE scientific import')
sys.meta_path.insert(0, Block())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    completed = subprocess_cli(script, ['check', *args], tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert sorted(tmp_path.rglob('*')) == before


@pytest.mark.parametrize('shared', [False, True])
def test_demo_run_then_explicit_evaluate_no_new_training_or_binary_copy(
    tmp_path: Path, shared: bool, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(1)
    args = settings(tmp_path, shared, null=not shared)
    root = tmp_path / 'out'
    # No production randomness, cipher or RSS, including null demo labels.
    import secrets

    import cipher_lab.execution as execution
    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError('PRIVATE production call')
    monkeypatch.setattr(secrets, 'token_bytes', forbidden)
    monkeypatch.setattr(execution, 'production_rss_bytes', forbidden)
    calls = 0
    original = torch.optim.Adam.step
    def step(self: Any, *a: Any, **kw: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(self, *a, **kw)
    monkeypatch.setattr(torch.optim.Adam, 'step', step)
    start = time.monotonic()
    assert cli.main(['run', *args, '--root', str(root), '--id', 'train',
                     '--backend', 'demo']) == 0
    elapsed = time.monotonic() - start
    assert elapsed < 60
    assert calls == (4 if shared else 2)
    output = json.loads(capsys.readouterr().out)
    assert output['backend_identity'] == 'demo'
    saved = cast(dict[str, Any], read_saved(root, 'train'))
    assert saved['summary']['synthetic'] is True
    assert saved['summary']['meter']['completed_optimizer_updates'] == calls
    print(f'CLI demo shared={shared} Adam={calls} elapsed={elapsed:.3f}s')
    refs = saved['summary']['refs']
    checkpoint = root / 'models/train/best.pt'
    dataset = root / 'data/train/dataset/test.npz'
    assert refs['best'] == 'model/best.pt' and refs['test'] == 'dataset/test.npz'
    before = {p: p.read_bytes() for p in (root / 'data/train').rglob('*') if p.is_file()}
    eval_args = ['evaluate', '--checkpoint', str(checkpoint), '--checkpoint-sha256',
                 hashlib.sha256(checkpoint.read_bytes()).hexdigest(), '--dataset', str(dataset),
                 '--dataset-sha256', hashlib.sha256(dataset.read_bytes()).hexdigest(),
                 '--policy', args[3], '--root', str(root), '--id', 'eval']
    # Independent subprocess proves no training/provider/private/cipher imports.
    completed = subprocess_cli(POISON, eval_args, tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evaluated = cast(dict[str, Any], read_saved(root, 'eval'))
    assert evaluated['summary']['metrics'] == saved['summary']['metrics']['test']
    assert evaluated['summary']['backend_identity'] == 'UNVERIFIED'
    assert evaluated['summary']['synthetic'] is None
    assert evaluated['summary']['dataset_role'] == 'explicit-saved-panel'
    for unit in ('real_blocks', 'null_words', 'optimizer_updates'):
        assert evaluated['summary']['meter'][f'attempted_{unit}'] == 0
    assert evaluated['summary']['meter']['completed_forward_rows'] == (8 if shared else 2)
    assert not (root / 'models/eval').exists()
    assert evaluated['artifacts'] == {}  # references, not duplicate binaries
    assert str(tmp_path) not in json.dumps(evaluated)
    assert {p: p.read_bytes() for p in before} == before
    assert calls == (4 if shared else 2)
    html = (root / 'result/eval/index.html').read_text()
    assert 'explicit-saved-panel' in html and '<svg' in html


@pytest.mark.parametrize('shared,backend', [(False, 'ciphers'), (True, 'simon')])
def test_illegal_backend_combination_never_claims(
    tmp_path: Path, shared: bool, backend: str,
) -> None:
    args = settings(tmp_path, shared)
    assert cli.main(['run', *args, '--root', str(tmp_path / 'out'), '--id', 'x',
                     '--backend', backend]) == 2
    assert not (tmp_path / 'out').exists()


def test_evaluate_requires_explicit_sources_and_claims_before_load(tmp_path: Path) -> None:
    args = settings(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(['evaluate', '--policy', args[3], '--root', str(tmp_path), '--id', 'x'])
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('torch', 'cipher_lab.training.')):
            raise AssertionError('PRIVATE setup import')
sys.meta_path.insert(0, Block())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    eval_args = ['evaluate', '--checkpoint', 'PRIVATE.pt', '--checkpoint-sha256', 'a'*64,
                 '--dataset', 'PRIVATE.npz', '--dataset-sha256', 'b'*64, '--policy', args[3],
                 '--root', str(tmp_path / 'out'), '--id', 'x']
    completed = subprocess_cli(script, eval_args, tmp_path)
    assert completed.returncode == 6, completed.stdout + completed.stderr
    assert 'PRIVATE' not in completed.stdout + completed.stderr
    assert cast(dict[str, Any], read_saved(tmp_path / 'out', 'x'))[
        'summary']['reason'] == 'BACKEND_SETUP_FAILED'
    completed = subprocess_cli(script, eval_args, tmp_path)
    assert completed.returncode == 2
    assert json.loads(completed.stdout)['reason'] == 'CLAIM_FAILED'


def public_sources(tmp_path: Path) -> tuple[Path, str, Path, str]:
    """Artificial tensors + untrained model, ZERO optimizer updates."""
    from cipher_lab.data.bundles import DatasetBundle, PredictionTask, encode_bundle
    from cipher_lab.evaluation.checkpoints import Selection, encode_checkpoint
    from cipher_lab.evaluation.metrics import PublicModelSpec, TrainBaseline
    from cipher_lab.models import BitMLP
    from cipher_lab.training_config import MLPSpec
    task = PredictionTask('SIMON', 3, 63, 'null')
    spec = PublicModelSpec(MLPSpec(2), (task,))
    content = encode_checkpoint(BitMLP(MLPSpec(2)), spec, (task,), Selection(1, 1, .7),
                                {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
    bundle = encode_bundle(DatasetBundle(spec.method, (task,), {
        'SIMON': (torch.zeros(2, 64), torch.tensor([[0.], [1.]]))}))
    checkpoint, dataset = tmp_path / 'private-user-model.pt', tmp_path / 'private-panel.npz'
    checkpoint.write_bytes(content)
    dataset.write_bytes(bundle)
    return (checkpoint, hashlib.sha256(content).hexdigest(),
            dataset, hashlib.sha256(bundle).hexdigest())


def test_explicit_other_panel_allowed_but_task_mismatch_and_sha_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import cipher_lab.execution as execution
    from cipher_lab.data.bundles import DatasetBundle, PredictionTask, encode_bundle
    monkeypatch.setattr(execution, 'production_rss_bytes', lambda: 0)
    args = settings(tmp_path)
    cp, csha, ds, dsha = public_sources(tmp_path)
    root = tmp_path / 'out'
    base = ['evaluate', '--checkpoint', str(cp), '--checkpoint-sha256', csha,
            '--dataset', str(ds), '--dataset-sha256', dsha, '--policy', args[3],
            '--root', str(root)]
    assert cli.main([*base, '--id', 'other']) == 0
    capsys.readouterr()
    saved = cast(dict[str, Any], read_saved(root, 'other'))
    assert saved['summary']['dataset_role'] == 'explicit-saved-panel'
    assert saved['summary']['source_refs']['checkpoint']['path'] == 'explicit/checkpoint'
    assert cp.name not in json.dumps(saved) and str(tmp_path) not in json.dumps(saved)
    for name, task in [('round', PredictionTask('SIMON', 4, 63, 'null')),
                       ('bit', PredictionTask('SIMON', 3, 62, 'null')),
                       ('mode', PredictionTask('SIMON', 3, 63, 'cipher'))]:
        content = encode_bundle(DatasetBundle('bit-prediction-v1', (task,), {
            'SIMON': (torch.zeros(2, 64), torch.ones(2, 1))}))
        ds.write_bytes(content)
        bad = list(base)
        bad[bad.index('--dataset-sha256')+1] = hashlib.sha256(content).hexdigest()
        assert cli.main([*bad, '--id', name]) == 4
        capsys.readouterr()
        failed = cast(dict[str, Any], read_saved(root, name))
        assert failed['summary']['meter']['attempted_forward_rows'] == 0
    assert cli.main([*base, '--id', 'wrongsha']) == 4
    capsys.readouterr()
    failed = cast(dict[str, Any], read_saved(root, 'wrongsha'))
    assert failed['summary']['reason'] == 'EVALUATION_FAILED'
    assert failed['summary']['meter']['attempted_forward_rows'] == 0
    # Occupied identity is immutable and never evaluated again.
    before = (root / 'data/other/run.json').read_bytes()
    assert cli.main([*base, '--id', 'other']) == 2
    assert (root / 'data/other/run.json').read_bytes() == before


@pytest.mark.parametrize('kind', ['oversize', 'symlink', 'sha', 'directory'])
def test_raw_source_bounds_before_decoder(tmp_path: Path, kind: str) -> None:
    from cipher_lab.evaluation.application import MAX_BYTES, read_source
    path = tmp_path / 'raw'
    path.write_bytes(b'public')
    digest = hashlib.sha256(b'public').hexdigest()
    if kind == 'oversize':
        with path.open('wb') as stream:
            stream.truncate(MAX_BYTES + 1)  # sparse, never allocate giant bytes
    elif kind == 'symlink':
        link = tmp_path / 'link'
        link.symlink_to(path)
        path = link
    elif kind == 'directory':
        path = tmp_path
    else:
        digest = '0'*64
    with pytest.raises(ValueError):
        read_source(path, digest)


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('failure', ['setup', 'missing', 'report'])
def test_training_cli_failure_boundaries_without_optimizer(
    tmp_path: Path, shared: bool, failure: str,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import cipher_lab.training.application as application
    import cipher_lab.training.report as report
    from cipher_lab.artifacts import RunStore
    args = settings(tmp_path, shared)
    root = tmp_path / 'out'
    def prepare(recipe: Any, meter: Any, store: RunStore, backend: str) -> Any:
        assert (root / 'models/x').is_dir()
        initial = read_saved(root, 'x')
        assert cast(dict[str, Any], initial['summary'])['backend_identity'] == 'demo'
        if failure == 'setup':
            raise RuntimeError('PRIVATE setup details')
        def invoke() -> None:
            if failure == 'report':
                store.finalize({'status': 'SUCCEEDED', 'summary': {
                    'method': recipe.method, 'backend_identity': backend}})
            else:
                raise RuntimeError('PRIVATE method details')
        return invoke
    monkeypatch.setattr(application, 'prepare_training', prepare)
    if failure == 'report':
        def broken(*a: Any) -> Any:
            raise RuntimeError('PRIVATE report details')
        monkeypatch.setattr(report, 'render_training_report', broken)
    assert cli.main(['run', *args, '--root', str(root), '--id', 'x',
                     '--backend', 'demo']) == {'setup': 4, 'missing': 7, 'report': 6}[failure]
    out = capsys.readouterr()
    assert 'PRIVATE' not in out.out + out.err
    saved = read_saved(root, 'x')
    assert saved['status'] == {'setup': 'FAILED', 'missing': 'RUNNING',
                               'report': 'SUCCEEDED'}[failure]


@pytest.mark.parametrize('shared', [False, True])
def test_training_snapshot_io_failure_never_defaults_to_integral(
    tmp_path: Path, shared: bool, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import cipher_lab.artifacts as artifacts
    import cipher_lab.integral.report as integral_report
    import cipher_lab.training.application as application

    args = settings(tmp_path, shared)
    root = tmp_path / 'out'
    run_path = root / 'data/x/run.json'
    replace = os.replace
    failed_bytes: list[bytes] = []
    io_failures = 0
    forbidden_calls: list[str] = []
    integral_calls: list[object] = []
    render_integral = integral_report.render_report

    def replace_once(src: Any, dst: Any) -> None:
        nonlocal io_failures
        if Path(dst) == run_path:
            proposed = json.loads(Path(src).read_bytes())
            if proposed['recipe'] and io_failures == 0:
                initial = cast(dict[str, Any], read_saved(root, 'x'))
                assert initial['status'] == 'RUNNING'
                assert initial['summary']['backend_identity'] == 'demo'
                assert initial['recipe'] == {} and initial['policy'] == {}
                io_failures += 1
                raise OSError('PRIVATE one-shot recipe/policy write failure')
        replace(src, dst)
        if Path(dst) == run_path and json.loads(run_path.read_bytes())['status'] == 'FAILED':
            failed_bytes.append(run_path.read_bytes())

    def forbidden(*a: Any, **kw: Any) -> Any:
        forbidden_calls.append('setup/finalize/optimizer')
        raise AssertionError('PRIVATE setup/finalize/optimizer must not run')

    def integral(saved: Any) -> dict[str, bytes]:
        integral_calls.append(saved)
        return render_integral(saved)

    monkeypatch.setattr(os, 'replace', replace_once)
    monkeypatch.setattr(application, 'prepare_training', forbidden)
    monkeypatch.setattr(artifacts.RunStore, 'finalize', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(integral_report, 'render_report', integral)
    code = cli.main(['run', *args, '--root', str(root), '--id', 'x', '--backend', 'demo'])
    output = capsys.readouterr()
    saved = cast(dict[str, Any], read_saved(root, 'x'))
    assert io_failures == 1
    assert saved['status'] == 'FAILED'
    assert saved['summary'] == {'backend_identity': 'demo', 'scientific_validation': False,
                                'reason': 'WRITE_FAILED'}
    assert saved['recipe'] == {} and saved['policy'] == {}
    assert failed_bytes == [run_path.read_bytes()]
    assert forbidden_calls == []
    assert code == 6
    public = json.loads(output.out)
    assert public['scientific_status'] == 'FAILED'
    assert public['report_status'] == 'FAILED' and public['report_failed'] is True
    assert public['reason'] == 'REPORT_METHOD_UNKNOWN'
    assert 'report_path' not in public
    assert 'PRIVATE' not in output.out + output.err
    assert integral_calls == []
    assert list((root / 'result/x').iterdir()) == []


@pytest.mark.parametrize('location', ['summary', 'recipe'])
@pytest.mark.parametrize('method', [None, 'future-method-v1', 'integral-empirical-v1',
    'bit-prediction-v1', 'shared-m-four-task-v1', 'explicit-evaluation-v1'])
def test_saved_report_dispatch_requires_known_stored_method(
    tmp_path: Path, location: str, method: str | None,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import cipher_lab.evaluation.application as evaluation
    import cipher_lab.integral.report as integral
    import cipher_lab.training.report as training
    from cipher_lab.artifacts import RunStore

    store = RunStore.claim(tmp_path, 'dispatch', output_bytes=20000000)
    identity: dict[str, object] = {} if method is None else {'method': method}
    store.finalize({'status': 'FAILED', location: identity})
    run_path = tmp_path / 'data/dispatch/run.json'
    before = run_path.read_bytes()
    calls: list[str] = []

    def renderer(name: str) -> Any:
        def render(saved: dict[str, Any]) -> dict[str, bytes]:
            calls.append(name)
            source = {'schema': 1, 'run_id': saved['run_id'],
                      'run_ref': saved['run_ref'], 'run_sha256': saved['run_sha256']}
            return {'report.md': b'public', 'index.html': b'public',
                    'source.json': json.dumps(source).encode()}
        return render

    monkeypatch.setattr(integral, 'render_report', renderer('integral'))
    monkeypatch.setattr(training, 'render_training_report', renderer('training'))
    monkeypatch.setattr(evaluation, 'render_evaluation_report', renderer('evaluation'))
    code = cli._saved_report(store, tmp_path, 'dispatch', 'demo')
    public = json.loads(capsys.readouterr().out)
    assert run_path.read_bytes() == before
    assert public['scientific_status'] == 'FAILED'
    assert public['backend_identity'] == 'UNVERIFIED'
    expected = {'integral-empirical-v1': 'integral', 'bit-prediction-v1': 'training',
                'shared-m-four-task-v1': 'training', 'explicit-evaluation-v1': 'evaluation'}
    if method in expected:
        assert calls == [expected[method]]
        assert code == 4 and public['report_status'] == 'SAVED'
    else:
        assert code == 6 and public['report_status'] == 'FAILED'
        assert public['reason'] == 'REPORT_METHOD_UNKNOWN'
        assert calls == []
        assert list((tmp_path / 'result/dispatch').iterdir()) == []


def test_training_claim_precedes_production_import(tmp_path: Path) -> None:
    args = settings(tmp_path, shared=True)
    root = tmp_path / 'out'
    script = '''
import importlib.abc, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.argv.pop(1)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('cipher_lab.ciphers'):
            assert (root / 'data/x/run.json').is_file()
            assert (root / 'models/x').is_dir()
            raise AssertionError('PRIVATE no production import')
sys.meta_path.insert(0, Block())
from cipher_lab.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    completed = subprocess_cli(script, [str(root), 'run', *args, '--root', str(root),
                                        '--id', 'x', '--backend', 'ciphers'], tmp_path)
    assert completed.returncode == 4, completed.stdout + completed.stderr
    assert 'PRIVATE' not in completed.stdout + completed.stderr
    saved = cast(dict[str, Any], read_saved(root, 'x'))
    assert saved['summary']['backend_identity'] == 'ciphers'
    assert saved['summary']['reason'] == 'BACKEND_SETUP_FAILED'


def test_provider_setup_stop_retains_partial_four_unit_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    args = settings(tmp_path, shared=True)
    path = Path(args[3])
    policy = json.loads(path.read_text())
    policy['real_blocks'] = 8
    path.write_text(json.dumps(policy))
    root = tmp_path / 'out'
    assert cli.main(['run', *args, '--root', str(root), '--id', 'partial',
                     '--backend', 'demo']) == 3
    capsys.readouterr()
    saved = cast(dict[str, Any], read_saved(root, 'partial'))
    summary = saved['summary']
    assert summary['reason'] == 'REAL_BLOCKS_LIMIT'
    assert summary['backend_identity'] == 'demo'
    assert summary['meter']['attempted_real_blocks'] == 8
    assert summary['meter']['completed_real_blocks'] == 8
    assert summary['meter']['attempted_optimizer_updates'] == 0
    assert summary['meter']['attempted_forward_rows'] == 0
