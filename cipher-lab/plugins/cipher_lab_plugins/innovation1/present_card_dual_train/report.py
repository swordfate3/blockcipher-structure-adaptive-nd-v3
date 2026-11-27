"""Report the held-out comparison from the saved run record."""
from __future__ import annotations

import html
import json
from typing import Any, cast

from . import config


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary = record['summary']
    paired = summary['paired'][config.CONTRAST]
    lines = [
        '# PRESENT-80 七轮双序平均训练对照', '', f"运行 `{record['run_id']}`。", '',
        '共享训练和验证面板 → 两组从相同初值出发 → 按双序 DEV BCE 选模',
        '→ 回读两份最佳权重 → 在同一份 TEST 上评估两种打分方式。', '',
        '| 训练组 | 原序准确率 | 原序 BCE | 双序准确率 | 双序 BCE | 最佳轮 | 完成轮 | 拟合秒数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for slot in config.SLOTS:
        arm = summary['arms'][slot]
        original, dual = summary['test_single'][slot], summary['test'][slot]
        lines.append(f"| {slot} | {original['accuracy']:.4%} | {original['bce']:.8f} | "
                     f"{dual['accuracy']:.4%} | {dual['bce']:.8f} | "
                     f"{arm['selected']['epoch']} | {arm['epochs_completed']} | "
                     f"{arm['fit_elapsed_seconds']:.2f} |")
    delta = paired['percentage_points']
    lines += [
        '', f'双序推理下，双序训练组相对对照组的准确率变化为 {delta:+.4f} 个百分点。',
        f"仅双序训练组判对 {paired['joint_only_correct']:,} 对。"
        f"仅对照组判对 {paired['control_only_correct']:,} 对。", '',
        '两组使用同一把合成密钥及相同模型初值。结果属于七轮 PRESENT-80 单对分类。',
    ]
    markdown = '\n'.join(lines) + '\n'
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>PRESENT-80</title>'
            f'<pre>{html.escape(markdown)}</pre></html>')
    source = {key: record[key] for key in ('run_ref', 'run_sha256', 'run_id')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps({'schema': 1, **source}).encode()}
