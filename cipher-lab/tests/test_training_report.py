"""Saved mappings only: report cannot decode binary or run models."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cipher_lab.training.report import render_training_report


@pytest.mark.parametrize('method', ['bit-prediction-v1', 'shared-m-four-task-v1'])
def test_saved_report_exact_files_escaping_method_and_result_figures(method: str) -> None:
    saved: dict[str, object] = {
        'run_id': '<script>[attack](bad)</script>', 'run_ref': 'data/report/run.json',
        'run_sha256': 'a'*64, 'status': 'SUCCEEDED',
        'recipe': {'rounds': 3, 'target_bit': 0, 'label_mode': 'null'},
        'counts': {'updates': 2, 'panels': {'train/SIMON': 2, 'test/SIMON': 2}},
        'summary': {'method': method, 'backend_identity': 'demo', 'synthetic': True,
            'scientific_validation': False,
            'task_declarations': [{'cipher': 'SIMON', 'rounds': 3}],
            'selected': {'epoch': 1, 'selected_dev_bce': .7},
            'learning_curve': [{'epoch': 1, 'dev': {'mean_bce': .7}}],
            'metrics': {'test': {'tasks': {'SIMON': {'rows': 2, 'acc': .5,
                'bce': .7, 'baseline_acc': .5, 'baseline_bce': .693}}}},
            'meter': {'completed_optimizer_updates': 2, 'completed_forward_rows': 10,
                      'attempted_real_blocks': 0, 'attempted_null_words': 0,
                      'wall_elapsed_seconds': .2, 'cpu_elapsed_seconds': .1},
            'refs': {'test': 'dataset/test.npz', 'best': 'model/best.pt'}},
        'payloads': {}, 'artifacts': {'model/best.pt': {'sha256': 'b'*64}},
    }
    before = copy.deepcopy(saved)
    files = render_training_report(saved)
    assert set(files) == {'report.md', 'index.html', 'source.json'}
    assert all(type(v) is bytes for v in files.values()) and saved == before
    html, markdown = files['index.html'].decode(), files['report.md'].decode()
    for value in (html, markdown):
        assert '<script>' not in value and '[attack](bad)' not in value
        assert 'ACC' in value and 'BCE' in value and 'SIMON' in value
        assert '合成' in value and '科学' in value and '成本' in value
        assert '3' in value and '.7' in value and 'optimizer_updates' in value
        assert value.count('<svg') == 2
        assert '0.5' in value and 'TRAIN' in value and 'TEST' in value
        assert '位预测' in value if method == 'bit-prediction-v1' else '四任务' in value
    assert json.loads(files['source.json']) == {
        'schema': 1, 'run_id': saved['run_id'], 'run_ref': saved['run_ref'],
        'run_sha256': saved['run_sha256']}


def test_storage_failure_report_recovers_saved_prediction_identity(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from cipher_lab.artifacts import ArtifactError, RunStore, read_saved

    def forbidden_forward(*args: object, **kwargs: object) -> object:
        pytest.fail('saved report must not run a model forward')

    monkeypatch.setattr(torch.nn.Module, '_call_impl', forbidden_forward)
    store = RunStore.claim(tmp_path, 'failed-report', output_bytes=100000)
    store.write_snapshot({
        'recipe': {'method': 'bit-prediction-v1'},
        'summary': {'method': 'bit-prediction-v1', 'synthetic': True,
                    'backend_identity': 'demo', 'scientific_validation': False},
    })
    original = store._write_all
    writes = 0

    def fail_once(fd: int, content: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError('synthetic recoverable storage failure')
        original(fd, content)

    monkeypatch.setattr(store, '_write_all', fail_once)
    with pytest.raises(ArtifactError, match='^WRITE_FAILED$'):
        store.write_snapshot({'counts': {'updates': 1}})
    assert writes == 2  # failed write, then real _failed() commits the compact record
    saved = read_saved(tmp_path, 'failed-report')
    assert saved['status'] == 'FAILED'
    assert saved['summary'] == {'reason': 'WRITE_FAILED', 'backend_identity': 'demo',
                                'scientific_validation': False}
    assert saved['recipe'] == {'method': 'bit-prediction-v1'}
    before = copy.deepcopy(saved)
    files = render_training_report(saved)
    assert saved == before and writes == 2
    for name in ('report.md', 'index.html'):
        text = files[name].decode()
        assert '位预测 MLP 训练报告' in text
        assert '固定目标 bit；延后 TEST 标签；BCE 选模' in text
        assert '释放 TEST 标签' in text
        assert '合成工程演示，不是真实密码科学结果。' in text
        assert 'backend=demo' in text and 'FAILED' in text
        assert '四任务 shared-M' not in text and '每宏batch四任务各一步' not in text


@pytest.mark.parametrize(('summary', 'recipe'), [
    ({}, {}),
    ({'method': 'unrecognized'}, {}),
    ({}, {'method': 'unrecognized'}),
    ({'method': None}, {'method': None}),
])
def test_unknown_method_does_not_default_to_shared(summary: dict[str, object],
        recipe: dict[str, object]) -> None:
    saved: dict[str, object] = {
        'run_id': 'unknown', 'run_ref': 'data/unknown/run.json', 'run_sha256': 'a'*64,
        'status': 'FAILED', 'summary': summary, 'recipe': recipe,
    }
    files = render_training_report(saved)
    for name in ('report.md', 'index.html'):
        text = files[name].decode()
        assert 'UNKNOWN' in text
        assert '四任务 shared-M' not in text and '位预测 MLP' not in text
        assert '每宏batch四任务各一步' not in text and '固定目标 bit' not in text
        assert 'Adam/BCE' not in text and '预建 TEST 首次评分' not in text


@pytest.mark.parametrize('method', ['bit-prediction-v1', 'shared-m-four-task-v1'])
def test_recipe_method_fallback_and_summary_precedence(method: str) -> None:
    other = ('shared-m-four-task-v1' if method == 'bit-prediction-v1'
             else 'bit-prediction-v1')
    saved: dict[str, object] = {
        'run_id': 'identity', 'run_ref': 'data/identity/run.json', 'run_sha256': 'a'*64,
        'summary': {'synthetic': True}, 'recipe': {'method': method},
    }
    title = ('位预测 MLP 训练报告' if method == 'bit-prediction-v1'
             else '四任务 shared-M 训练报告')
    assert title in render_training_report(saved)['index.html'].decode()
    saved['summary'] = {'method': method, 'synthetic': True}
    saved['recipe'] = {'method': other}
    assert title in render_training_report(saved)['index.html'].decode()


def test_report_import_has_no_tensor_dependencies(tmp_path: Path) -> None:
    code = '''
import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy') or fullname.startswith(
                ('cipher_lab.models', 'cipher_lab.data', 'cipher_lab.evaluation')):
            raise RuntimeError('forbidden report dependency')
sys.meta_path.insert(0, Guard())
from cipher_lab.training.report import render_training_report
files = render_training_report({'run_id':'x','run_ref':'data/x/run.json',
                                'run_sha256':'a'*64, 'status':'RUNNING'})
assert b'UNKNOWN' in files['report.md']
'''
    result = subprocess.run([sys.executable, '-c', code], check=False,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_report_write_failure_does_not_change_finalized_training_status(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from test_prediction_training import ArtificialPrediction, fake_step, recipe
    from test_training_steps import make_meter

    from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
    from cipher_lab.training import prediction
    monkeypatch.setattr(prediction, 'train_step', fake_step)
    store = RunStore.claim(tmp_path, 'report', output_bytes=10000000, needs_models=True)
    result = prediction.run_prediction(recipe(), ArtificialPrediction(), make_meter(), store,
                                       backend_identity='demo')
    assert result['status'] == 'SUCCEEDED'
    saved = read_saved(tmp_path, 'report')
    files = render_training_report(saved)
    store.write_report_files(files)
    assert set(p.name for p in (tmp_path / 'result/report').iterdir()) == set(files)
    before = (tmp_path / 'data/report/run.json').read_bytes()
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError('PRIVATE')
    monkeypatch.setattr(store, '_atomic', fail)
    with pytest.raises(ArtifactError):
        store.write_report_files(files)
    assert (tmp_path / 'data/report/run.json').read_bytes() == before
    assert read_saved(tmp_path, 'report')['status'] == 'SUCCEEDED'
