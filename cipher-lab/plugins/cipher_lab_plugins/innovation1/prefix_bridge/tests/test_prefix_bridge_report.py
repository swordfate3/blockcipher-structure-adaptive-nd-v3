from __future__ import annotations

from cipher_lab_plugins.innovation1.prefix_bridge.report import report_files


def test_report_failed_no_invented_scores_and_separate_exit_evidence() -> None:
    files = report_files({
        'status': 'FAILED', 'run_id': 'fake', 'summary': {'phase': 'TEST_SCORE'}, 'recipe': {},
        'run_ref': 'data/fake/run.json', 'run_sha256': 'a' * 64,
    })
    page = files['report.md'].decode()
    for text in ('13', '组', 'S2-only', 'oracle', '不证明', 'native', 'SSH', 'UNKNOWN'):
        assert text in page
    assert 'TEST 未提交' in page
