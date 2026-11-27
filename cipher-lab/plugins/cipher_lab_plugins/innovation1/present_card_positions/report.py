"""Write the saved matched panel comparison."""
from __future__ import annotations

import html
import json
from typing import Any, cast

from . import config


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary = record['summary']
    lines = ['# PRESENT-80 七轮遮挡位置对照', '', f"运行 {record['run_id']}。", '',
             '同一合成密钥及面板 → 三组遮挡训练 → DEV 选模 → 共享完整 TEST。', '',
             '| 位置 | 正确密文对 | TEST 准确率 | TEST BCE | 最佳轮 | 完成轮 |',
             '|---|---:|---:|---:|---:|---:|']
    for slot in config.SLOTS:
        arm, test = summary['arms'][slot], summary['test'][slot]
        lines.append(f"| {slot} | {test['correct_pairs']}/{test['pairs']} | "
                     f"{test['accuracy']:.4%} | {test['bce']:.8f} | "
                     f"{arm['selected']['epoch']} | {arm['epochs_completed']} |")
    lines += ['', '配对正确性', '', '    ' + json.dumps(summary['paired'], sort_keys=True), '',
              '实验覆盖一把合成密钥及固定训练初值。']
    markdown = '\n'.join(lines) + '\n'
    page = ('<!doctype html><meta charset="utf-8"><title>PRESENT-80</title>'
            f'<pre>{html.escape(markdown)}</pre>')
    source = {key: record[key] for key in ('run_ref', 'run_sha256', 'run_id')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps({'schema': 1, **source}).encode()}
