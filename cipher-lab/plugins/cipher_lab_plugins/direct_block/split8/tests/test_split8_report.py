from __future__ import annotations

import json


def test_split8_failed_report_unknown_recipe_has_no_science_claims() -> None:
    from cipher_lab_plugins.direct_block.split8 import report

    saved = {'status': 'FAILED', 'run_id': 'manual', 'run_ref': 'data/manual/run.json',
             'run_sha256': 'a' * 64, 'summary': {'reason': '<b>failure</b>'}, 'recipe': {},
             'counts': {}}
    files = report.report_files(saved)
    assert set(files) == {'report.md', 'index.html', 'source.json'}
    doc = files['report.md'].decode()
    assert '无完整 TEST' in doc and 'FAILED' in doc
    for forbidden in ('SIMON32/64', '已抽取', '已拟合', '524288'):
        assert forbidden not in doc
    assert '&lt;b&gt;failure&lt;/b&gt;' in files['index.html'].decode()
    assert json.loads(files['source.json'])['run_sha256'] == 'a' * 64
