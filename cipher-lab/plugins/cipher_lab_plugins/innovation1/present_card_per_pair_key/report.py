"""Render the saved original-order comparison for the two key policies."""
from __future__ import annotations

import html
import json
from typing import Any, cast

from . import config


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary = record['summary']
    paired = summary['paired'][config.CONTRAST]
    lines = ['# PRESENT-80 七轮逐对换钥对照', '', f"运行 {record['run_id']}。", '',
             '同明文及标签的两份 TRAIN（固定钥与逐对钥）→ 共同原序 DEV BCE 选模',
             '→ 两份磁盘最佳权重严格回读并激活 → 共同逐对换钥 TEST 原序评分。', '',
             '| 组别 | TEST 正确密文对 | TEST 准确率 | TEST BCE | 最佳轮 | 完成轮 |',
             '|---|---:|---:|---:|---:|---:|']
    bars = []
    for slot in config.SLOTS:
        arm, test = summary['arms'][slot], summary['test'][slot]
        accuracy = test['accuracy']
        lines.append(f"| {slot} | {test['correct_pairs']}/{test['pairs']} | "
                     f"{accuracy:.4%} | {test['bce']:.8f} | "
                     f"{arm['selected']['epoch']} | {arm['epochs_completed']} |")
        bars.append(f'<div><label>{slot} <meter min="0" max="1" value="{accuracy:.10f}">'
                    f'{accuracy:.4%}</meter></label> {accuracy:.4%}</div>')
    delta = f"{paired['percentage_points']:+.4f}"
    lines += ['', f'TEST 准确率差（per-pair-key 减 fixed-key）{delta} 个百分点。',
              '', '| 两组均正确 | 仅 per-pair-key 正确 | 仅 fixed-key 正确 | 两组均错误 |',
              '|---:|---:|---:|---:|',
              f"| {paired['both_correct']} | {paired['joint_only_correct']} | "
              f"{paired['control_only_correct']} | {paired['both_wrong']} |"]
    lines += ['', '两组使用相同初始化和训练交换规则。结果限于合成样本、一次初始化与七轮。']
    markdown = '\n'.join(lines) + '\n'
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>PRESENT-80</title>'
            '<figure><figcaption>原序 TEST 准确率</figcaption>' + ''.join(bars) + '</figure>'
            f'<pre>{html.escape(markdown)}</pre></html>')
    source = {key: record[key] for key in ('run_ref', 'run_sha256', 'run_id')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps({'schema': 1, **source}).encode()}
