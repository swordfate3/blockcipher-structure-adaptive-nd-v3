from __future__ import annotations

import pytest
from cipher_lab_plugins.direct_block.cube_compare.config import recipe


def test_failed_report_escapes_saved_values_and_hides_test() -> None:
    from cipher_lab_plugins.direct_block.cube_compare import report

    html = report.render_report({
        'status': 'STOPPED', 'run_id': '<unsafe>', 'recipe': recipe(),
        'counts': {'completed_real_blocks': 18, 'rss_bytes': None},
        'summary': {'test': {'accuracy': {'mlp': 0.875}}, 'reason': 'DEADLINE'},
    }).decode()
    assert '<unsafe>' not in html and '&lt;unsafe&gt;' in html
    assert 'DEADLINE' in html and '18' in html and 'null' in html
    assert '0.875' not in html and 'No final TEST' in html
    assert '<svg' in html


def test_cube_fix1_ticks_sparse_with_full_saved_50_epoch_curves() -> None:
    import re

    from cipher_lab_plugins.direct_block.cube_compare import report

    points = [{'epoch': e, 'train_bce': 1 / e, 'full_bce': 2 / e, 'active_bce': 3 / e}
              for e in range(1, 51)]
    html = report.render_report({'summary': {'arms': {'mlp': {'dev': points}}}}).decode()
    ticks = re.findall(r'y="184" text-anchor="middle">(\d+)</text>', html)
    assert 2 <= len(ticks) <= 10 and ticks[0] == '1' and ticks[-1] == '50'
    assert html.count('<circle ') == 150
    assert [len(line.split()) for line in re.findall(r'<polyline points="([^"]+)"', html)] == [50]*3


def test_cube_fix1_legacy_recipe_overrides_summary_mode_hint() -> None:
    from cipher_lab_plugins.direct_block.cube_compare import report

    saved: dict[str, object] = {'status': 'FAILED', 'recipe': recipe(),
                                'summary': {'mode': 'saved-panels'}}
    html = report.render_report(saved).decode()
    assert 'Public preflight' in html and 'generate TEST labels' in html
    assert 'previously reported' not in html


def test_lstm_saved_report_uses_single_topology_and_reference_only() -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, report

    saved: dict[str, object] = {
        'status': 'SUCCEEDED', 'run_id': 'manual', 'run_ref': 'data/manual/run.json',
        'run_sha256': 'f' * 64, 'recipe': config.lstm_recipe(),
        'summary': {'arms': {'lstm': {'epochs_completed': 18, 'stop_reason': 'EARLY_STOPPING',
                                      'dev': [{'epoch': 1, 'active_bce': .5}]}},
                    'test': {'accuracy': {'lstm': .75, 'majority': .5},
                             'comparisons': {'lstm_minus_majority': {'delta': .25}},
                             'families': [{'family_id': 1, 'majority': 32, 'lstm': 48},
                                          {'family_id': 3, 'majority': 32, 'lstm': 48}]}},
    }
    files = report.report_files(saved)
    html, md = (files[name].decode() for name in ('index.html', 'report.md'))
    assert 'lstm: BCE by saved epoch' in html and '75.0000%' in html
    assert 'Both DEV best' not in html and 'Bonferroni' not in html and '97.5%' in html
    for text in (html, md):
        assert 'LSTM fit' in text and 'Configured data source' in text
        assert '49.9969%' in text and '49.8363%' in text and 'descriptive' in text
        assert 'befe14e5fb467412965d98cc0f121a827370d82860d9399e8cf4df145ff9ef72' in text
        assert 'previously reported' in text and 'sharedm_minus_mlp' not in text
        assert 'Per-family rows are in the saved record.' in text
        assert 'lstm_minus_majority' in text and 'Planned comparison:' not in text


@pytest.mark.parametrize(('status', 'summary'), [
    ('FAILED', {}),
    ('STOPPED', {'test': {'accuracy': {'lstm': .75}, 'families': [{'lstm': 48}]}}),
    ('SUCCEEDED', {}),
])
def test_lstm_report_without_committed_test_is_plan_and_reference_only(
    status: str, summary: dict[str, object],
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, report

    files = report.report_files({
        'status': status, 'run_id': 'manual', 'run_ref': 'data/manual/run.json',
        'run_sha256': 'f' * 64, 'recipe': config.lstm_recipe(), 'summary': summary,
    })
    for name in ('index.html', 'report.md'):
        text = files[name].decode()
        assert 'No final TEST' in text and '75.0000%' not in text and '0.75' not in text
        assert 'Per-family rows are in the saved record.' not in text
        assert 'Planned comparison:' in text and '97.5%' in text
        assert 'Configured data source' in text
        assert 'Historical saved results (descriptive only)' in text
        assert '49.9969%' in text and '49.8363%' in text
