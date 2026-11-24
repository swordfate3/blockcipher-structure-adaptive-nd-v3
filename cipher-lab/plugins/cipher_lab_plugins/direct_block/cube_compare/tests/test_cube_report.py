from __future__ import annotations

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
