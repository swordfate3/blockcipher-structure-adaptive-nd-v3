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
