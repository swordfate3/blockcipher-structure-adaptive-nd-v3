"""Render the saved original-order comparison and swap sensitivity metrics."""
from __future__ import annotations

import html
import json
from typing import Any, cast

from . import config


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary = record['summary']
    paired = summary['paired'][config.CONTRAST]
    lines = ['# PRESENT-80 七轮密文对交换与掩码对照', '', f"运行 {record['run_id']}。", '',
             '共享 TRAIN、DEV → 两组独立拟合（原序 DEV BCE 选模）→ 两份磁盘最佳权重回读',
             '→ DEV 双序诊断（标签保持不变）→ 单份共享 TEST 原序评分。', '',
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
    lines += ['', '| 组别 | 目标 | 拟合秒数 | 辅助对数 | 辅助目标数 | 辅助 BCE |',
              '|---|---|---:|---:|---:|---:|']
    for slot in config.SLOTS:
        arm = summary['arms'][slot]
        aux = arm.get('auxiliary_train', {})
        bce = f"{aux['bce']:.8f}" if aux else '—'
        lines.append(f"| {slot} | {arm['objective']} | {arm['fit_elapsed_seconds']:.3f} | "
                     f"{aux.get('rows', '—')} | {aux.get('targets', '—')} | {bce} |")
    delta = f"{paired['percentage_points']:+.4f}"
    lines += ['', f'TEST 准确率差（swap-50-mask 减 swap-50）{delta} 个百分点。',
              '', '| 两组均正确 | 仅 swap-50-mask 正确 | 仅 swap-50 正确 | 两组均错误 |',
              '|---:|---:|---:|---:|',
              f"| {paired['both_correct']} | {paired['joint_only_correct']} | "
              f"{paired['control_only_correct']} | {paired['both_wrong']} |", '',
              '| 组别 | 原序 DEV 准确率 | 原序 DEV BCE | 交换 DEV 准确率 | 交换 DEV BCE | '
              '判决翻转数 | 判决翻转率 | 平均绝对 logit 差 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for slot in config.SLOTS:
        dev = summary['dev_diagnostics'][slot]
        original, swapped = dev['original'], dev['swapped']
        lines.append(f"| {slot} | {original['accuracy']:.4%} | {original['bce']:.8f} | "
                     f"{swapped['accuracy']:.4%} | {swapped['bce']:.8f} | "
                     f"{dev['flip_count']}/{dev['pairs']} | {dev['flip_fraction']:.4%} | "
                     f"{dev['mean_abs_logit_delta']:.8f} |")
    lines += ['', '结果仅覆盖一把合成密钥、一个固定初始化和七轮 PRESENT-80。']
    markdown = '\n'.join(lines) + '\n'
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>PRESENT-80</title>'
            '<figure><figcaption>原序 TEST 准确率</figcaption>' + ''.join(bars) + '</figure>'
            f'<pre>{html.escape(markdown)}</pre></html>')
    source = {key: record[key] for key in ('run_ref', 'run_sha256', 'run_id')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps({'schema': 1, **source}).encode()}
