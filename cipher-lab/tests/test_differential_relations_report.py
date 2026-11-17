"""Saved-only report tests: no cipher, no model forward, no file or science reads."""
import builtins
import importlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from cipher_lab_plugins.differential_relations import report as report_module
from cipher_lab_plugins.differential_relations.report import render_report


def _relation(rows: int, ones: int, accuracy: float | None, bce: float | None,
              *, mcc: float | None = 0.1) -> dict[str, object]:
    return {'rows': rows, 'ones': ones, 'bce': bce, 'accuracy': accuracy,
            'balanced_accuracy': accuracy, 'recall_0': accuracy, 'recall_1': 0.5,
            'mcc': mcc, 'undefined_reasons': [] if mcc is not None else ['MCC_ZERO_DENOMINATOR']}


def _metrics(accuracies: tuple[float, ...], bce: float) -> dict[str, object]:
    return {'relations': [_relation(1000, 500, a, bce) for a in accuracies],
            'mean_bce': bce}


def _curve(values: tuple[float, ...]) -> list[dict[str, object]]:
    return [{'step': (i + 1) * 1024, 'bce': value} for i, value in enumerate(values)]


def _fit(fit_id: str, split: str, lr: float, seed: int, rows: int, updates: int,
         step: int, curve: tuple[float, ...], test_acc: float, test_bce: float,
         train_acc: float = 0.51, dev_acc: float = 0.52) -> dict[str, object]:
    return {'fit_id': fit_id, 'split': split, 'train_rows': rows,
            'learning_rate': lr, 'init_seed': seed, 'updates': updates,
            'selected_step': step, 'dev_curve': _curve(curve),
            'final_metrics': {
                'train': _metrics((train_acc,) * 4, 0.69),
                'dev': _metrics((dev_acc,) * 4, 0.68),
                'test': _metrics((test_acc,) * 4, test_bce)}}


def _baseline(acc: float) -> dict[str, object]:
    return {'relations': [_relation(65536, 30000, acc, 0.69) for _ in range(4)]}


def _saved() -> dict[str, object]:
    return {
        'run_id': 'toy-relations',
        'status': 'SUCCEEDED',
        'recipe': {'method': 'shared-m-differential-relations-v1',
                   'cipher': 'SIMON64/128', 'rounds': 6, 'alpha': 64,
                   'train_small': 262144, 'train_large': 1048576,
                   'test': 262144, 'plugin_source': {'commit': 'abc123'}},
        'counts': {'saved_fits': 5, 'optimizer_updates': 163840},
        'summary': {
            'method': 'shared-m-differential-relations-v1',
            'completion': 'COMPLETE',
            'reason': 'all five fits saved and test evaluated once',
            'fits': [
                _fit('F1', 'small', 0.001, 31, 262144, 32768, 30720,
                     (0.70, 0.69, 0.688), 0.53, 0.67),
                _fit('F2', 'small', 0.0003, 31, 262144, 32768, 31744,
                     (0.71, 0.695), 0.52, 0.675),
                _fit('F3', 'large', 0.001, 31, 1048576, 32768, 32768,
                     (0.69, 0.68, 0.67), 0.55, 0.66),
                _fit('F4', 'large', 0.001, 37, 1048576, 32768, 31744,
                     (0.695, 0.685), 0.54, 0.665),
                _fit('F5', 'large', 0.001, 31, 1048576, 32768, 30720,
                     (0.70, 0.69), 0.505, 0.693),
            ],
            'baselines': {
                'small': {'train_rows': 262144, 'constant': _baseline(0.50),
                          'stump': _baseline(0.51)},
                'large': {'train_rows': 1048576, 'constant': _baseline(0.50),
                          'stump': _baseline(0.52)},
            },
            'main_comparisons': {
                'model_bce': 0.66,
                'comparisons': {
                    'constant': {'baseline_bce': 0.69, 'mean_difference': 0.03,
                                 'relative_improvement': 0.0435, 'ci_low': 0.02,
                                 'ci_high': 0.04, 'confidence': 0.975,
                                 'uncertain': False, 'reasons': []},
                    'stump': {'baseline_bce': 0.68, 'mean_difference': 0.02,
                              'relative_improvement': 0.0294, 'ci_low': 0.01,
                              'ci_high': 0.03, 'confidence': 0.975,
                              'uncertain': False, 'reasons': []}},
                'main_signal': True,
            },
        },
        'run_ref': 'data/toy-relations/run.json',
        'run_sha256': 'a' * 64,
    }


def _assert_self_contained(html: str) -> None:
    low = html.lower()
    assert '<script' not in low
    assert '<link' not in low
    assert '<style' not in low
    assert re.search(r'<\w+[^>]*\bsrc\s*=', html, re.IGNORECASE) is None
    assert '@import' not in low
    assert 'url(' not in low
    for match in re.findall(r'https?://[^"\'\s>]+', html):
        assert match == 'http://www.w3.org/2000/svg'


def _body(saved: Mapping[str, object]) -> str:
    payload = render_report(saved)
    assert isinstance(payload, bytes)
    return payload.decode('utf-8')


def test_complete_report_starts_with_scope_main_bce_relation_diffs_and_single_key() -> None:
    html = _body(_saved())
    assert html.startswith('<!DOCTYPE html>')
    opening = html.split('</header>', 1)[0]
    assert 'COMPLETE' in opening
    assert '主BCE判断' in opening
    assert 'main_signal=True' in opening
    assert '单key' in opening and '不是跨key' in opening
    assert 'R1' in opening and 'ACC差(saved)=未保存' in opening
    assert '+0.050000' not in opening
    assert 'F3' in html and 'main-candidate' in html
    assert 'F5' in html and '置乱' in html and '不是泛化差距' in html
    assert html.count('<svg') == 3
    _assert_self_contained(html)


def test_partial_report_marks_missing_f3_and_does_not_invent_baseline_chart() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    summary['completion'] = 'PARTIAL'
    summary['fits'] = cast(list[dict[str, object]], summary['fits'])[:2]
    summary.pop('main_comparisons')
    summary.pop('baselines')
    html = _body(saved)
    assert 'PARTIAL' in html
    assert 'F3 未保存' in html
    assert '未完成' in html
    assert '未保存' in html
    assert html.count('<svg') == 2
    assert 'F3 TEST 逐关系指标未保存' in html
    _assert_self_contained(html)


def test_all_train_constant_reports_no_fit_or_test_and_no_fabricated_result() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    summary['completion'] = 'ALL_TRAIN_CONSTANT'
    summary['fits'] = []
    summary['reason'] = 'four TRAIN relations constant'
    summary.pop('main_comparisons')
    summary.pop('baselines')
    html = _body(saved)
    assert 'ALL_TRAIN_CONSTANT' in html
    assert '未生成 TEST' in html
    assert '无保存 DEV 曲线数据' in html
    assert '不能判断' in html
    assert html.count('<svg') == 2
    _assert_self_contained(html)


def test_undefined_relation_metrics_show_undefined_not_zero() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    fits = cast(list[dict[str, object]], summary['fits'])
    metrics = cast(dict[str, object], fits[2]['final_metrics'])
    test = cast(dict[str, object], metrics['test'])
    relations = cast(list[dict[str, object]], test['relations'])
    relations[0]['accuracy'] = None
    relations[0]['bce'] = None
    relations[0]['mcc'] = None
    test['mean_bce'] = None
    html = _body(saved)
    assert 'undefined' in html
    assert '<td>undefined</td>' in html
    _assert_self_contained(html)


def test_empty_and_single_point_dev_curves_render_only_saved_points() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    fits = cast(list[dict[str, object]], summary['fits'])
    fits[0]['dev_curve'] = []
    fits[1]['dev_curve'] = [{'step': 1024, 'bce': 0.777}]
    fits[2]['dev_curve'] = [{'step': 1024, 'bce': 0.71}, {'step': 2048, 'bce': 0.7}]
    html = _body(saved)
    assert 'data-fit="F1" data-points="0"' in html
    assert '0.777' in html
    assert 'NaN' not in html and 'nan' not in html
    assert '无保存点' in html
    _assert_self_contained(html)


def test_f5_marks_scrambled_train_dev_targets_and_real_test() -> None:
    html = _body(_saved())
    assert 'TRAIN/DEV 目标为整行置乱标签' in html
    assert 'TEST 目标为真实标签' in html
    assert '不是泛化差距' in html
    assert 'F5/TRAIN' in html or 'F5 · TRAIN' in html


def test_poisoned_text_and_unknown_structures_are_escaped() -> None:
    saved = _saved()
    poison = '<script>alert(1)</script><img src=x onerror="x">'
    saved['run_id'] = poison
    saved['summary'] = dict(cast(Mapping[str, object], saved['summary']))
    summary = cast(dict[str, object], saved['summary'])
    summary['reason'] = poison
    summary['unknown_structure'] = {'<b>': poison, 'rows': [1, 2]}
    recipe = cast(dict[str, object], saved['recipe'])
    recipe['plugin_source'] = {'note': poison}
    html = _body(saved)
    assert poison not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert '&lt;img src=x' in html
    assert 'unknown_structure' in html
    _assert_self_contained(html)


def test_f5_target_version_keeps_test_real_and_train_dev_scrambled() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    fits = cast(list[dict[str, object]], summary['fits'])
    fits[4]['target_version'] = 'row-permutation-v1'
    html = _body(saved)
    assert 'F5/TRAIN（目标：置乱标签（row-permutation-v1））' in html
    assert 'F5/DEV（目标：置乱标签（row-permutation-v1））' in html
    assert 'F5/TEST（目标：真实标签）' in html
    assert 'F5/TEST（目标：置乱标签' not in html


def test_partial_completion_never_shows_saved_main_signal_as_success() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    summary['completion'] = 'PARTIAL'
    html = _body(saved)
    assert 'main_signal=True' in html
    assert '不能展示为主signal成功' in html
    assert '达到其保存的信号门槛' not in html


def test_missing_f3_never_shows_saved_main_signal_as_success() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    fits = cast(list[dict[str, object]], summary['fits'])
    fits.pop(2)
    html = _body(saved)
    assert 'main_signal=True' in html
    assert '不能展示为主signal成功' in html
    assert 'F3 未保存' in html
    assert '达到其保存的信号门槛' not in html


def _accuracy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline in ('constant', 'stump'):
        for index in range(4):
            rows.append({'baseline': baseline, 'relation': f'R{index + 1}',
                         'rows': 262144, 'model_only_correct': 100,
                         'baseline_only_correct': 90,
                         'accuracy_improvement': 0.012345,
                         'p': 0.001, 'adjusted_p': 0.008, 'alpha': 0.05,
                         'accuracy_signal': True})
    return rows


def test_curve_ymax_label_is_at_top_of_y_axis() -> None:
    html = _body(_saved())
    ymax = re.search(r'data-axis="ymax"[^>]*y="([\d.]+)"', html)
    ymin = re.search(r'data-axis="ymin"[^>]*y="([\d.]+)"', html)
    assert ymax is not None and ymin is not None
    assert float(ymax.group(1)) < 60
    assert float(ymin.group(1)) > 300
    assert float(ymax.group(1)) < float(ymin.group(1))


def test_relation_diff_without_saved_comparison_lists_originals_and_missing_diff() -> None:
    saved = _saved()
    html = _body(saved)
    opening = html.split('</header>', 1)[0]
    assert '+0.050000' not in opening
    assert 'ACC差(saved)=未保存' in opening
    assert 'BCE差=未保存' in opening
    assert '模型ACC=0.55' in opening
    assert '基线ACC=0.5' in opening


def test_relation_diff_reads_saved_accuracy_improvement_without_subtraction() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    summary['accuracy_comparisons'] = _accuracy_rows()
    html = _body(saved)
    opening = html.split('</header>', 1)[0]
    assert 'ACC差(saved)=0.012345' in opening
    assert '+0.050000' not in opening
    assert 'BCE差=未保存' in opening


def test_baseline_dev_metrics_are_not_used_as_test_metrics() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    baselines = cast(dict[str, object], summary['baselines'])
    large = cast(dict[str, object], baselines['large'])
    large['constant'] = {'dev': _baseline(0.99)}
    large['stump'] = {'dev': _baseline(0.98)}
    html = _body(saved)
    assert html.count('<svg') == 2
    assert 'large/constant 基线逐关系指标未保存' in html
    assert 'large/stump 基线逐关系指标未保存' in html
    assert '对比图' not in html or '不绘制图形' in html


def test_relation_table_shows_saved_undefined_reasons_and_missing_reason_text() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    fits = cast(list[dict[str, object]], summary['fits'])
    metrics = cast(dict[str, object], fits[2]['final_metrics'])
    test = cast(dict[str, object], metrics['test'])
    relations = cast(list[dict[str, object]], test['relations'])
    relations[0]['accuracy'] = None
    relations[0]['mcc'] = None
    relations[0]['undefined_reasons'] = ['MCC_ZERO_DENOMINATOR']
    relations[1]['accuracy'] = None
    relations[1].pop('undefined_reasons')
    html = _body(saved)
    assert '<th>undefined_reasons</th>' in html
    assert 'MCC_ZERO_DENOMINATOR' in html
    assert '原因未保存（指标 undefined）' in html


def test_saved_accuracy_comparisons_are_a_separate_secondary_table() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    summary['accuracy_comparisons'] = _accuracy_rows()
    html = _body(saved)
    assert '逐关系准确率次级比较' in html
    assert 'McNemar' in html
    assert html.count('<th>p</th>') == 1
    assert html.count('<th>adjusted_p</th>') == 1
    assert html.count('<th>accuracy_signal</th>') == 1
    assert '0.001' in html and '0.008' in html
    assert '不与主BCE表混同' in html


def test_unknown_comparison_and_baseline_structures_are_escaped() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    poison = '<script>alert(9)</script>'
    summary['main_comparisons'] = [{'weird': poison}]
    summary['baselines'] = [poison]
    html = _body(saved)
    assert poison not in html
    assert '&lt;script&gt;alert(9)&lt;/script&gt;' in html
    assert '结构未识别' in html
    assert html.count('<svg') == 2
    _assert_self_contained(html)


def test_render_reads_no_files_no_science_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = _saved()
    before = json.dumps(saved, sort_keys=True)
    original_import = builtins.__import__

    def no_science(name: str, globals: dict[str, object] | None = None,
                   locals: dict[str, object] | None = None,
                   fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        blocked = ('numpy', 'torch', 'cipher_lab.ciphers', 'cipher_lab.data',
                   'cipher_lab.training', 'cipher_lab.models',
                   'cipher_lab_plugins.differential_relations.data',
                   'cipher_lab_plugins.differential_relations.metrics',
                   'cipher_lab_plugins.differential_relations.checkpoint',
                   'cipher_lab_plugins.differential_relations.config',
                   'cipher_lab_plugins.differential_relations.resources')
        if any(name == item or name.startswith(item + '.') for item in blocked):
            raise AssertionError(f'report imported science module {name}')
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', no_science)
        patch.setattr(Path, 'read_bytes', lambda p: pytest.fail('report read bytes'))
        patch.setattr(Path, 'read_text', lambda p, encoding=None: pytest.fail('report read text'))
        patch.setattr(Path, 'write_bytes',
                      lambda p, data: pytest.fail('report wrote bytes'))
        patch.setattr(Path, 'write_text',
                      lambda p, data, encoding=None: pytest.fail('report wrote text'))
        patch.setattr(builtins, 'open', lambda *args, **kwargs: pytest.fail('report opened file'))
        importlib.reload(sys.modules[report_module.__name__])
        first = render_report(saved)
        second = render_report(saved)
    assert first == second
    assert json.dumps(saved, sort_keys=True) == before
    assert first.startswith(b'<!DOCTYPE html>')
    _assert_self_contained(first.decode('utf-8'))


def test_missing_f3_test_and_missing_baseline_metric_do_not_draw_comparison() -> None:
    saved = _saved()
    summary = cast(dict[str, object], saved['summary'])
    baselines = cast(dict[str, object], summary['baselines'])
    large = cast(dict[str, object], baselines['large'])
    large.pop('constant')
    large.pop('stump')
    html = _body(saved)
    assert '基线逐关系 accuracy 未保存' in html
    assert html.count('<svg') == 2
    assert '不能判断' in html
    _assert_self_contained(html)
