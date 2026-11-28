"""Render the frozen cycle summary."""
from __future__ import annotations

import html
import json
from typing import Any, cast


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    summary = cast('dict[str, Any]', saved['summary'])
    lines = ['# SIMON32/64 单比特预测逐轮对照', '',
             f"运行 {saved['run_id']}。最后实测 {summary['last_completed_round']} 轮。", '']
    if summary['stop_reason'] == 'NEAR_CHANCE':
        lines.append(f"两套面板在第 {summary['last_completed_round']} 轮达到近随机停点。")
    else:
        lines.append('实验达到轮数上限。')
    lines += ['', '每轮共享训练与验证数据 → 完成六次拟合并严格回读权重 → 评估两套 TEST。',
              '', '| 轮数 | 面板 | 模型 | TEST 准确率 | TEST BCE |',
              '|---:|---|---|---:|---:|']
    names = {'correct': '正确接线', 'shuffled': '打乱接线', 'mlp': '普通 MLP', 'constant': '常量'}
    for depth, record in summary['rounds'].items():
        for panel, scores in record['test'].items():
            for arm, score in {**scores['arms'], 'constant': scores['constant']}.items():
                lines.append(f"| {depth} | {panel[-1]} | {names[arm]} | {score['accuracy']:.4%} | "
                             f"{score['bce']:.8f} |")
    lines += ['', '| 轮数 | 面板 | 正确接线减打乱接线的 TEST BCE |', '|---:|---|---:|']
    for depth, record in summary['rounds'].items():
        for panel, scores in record['test'].items():
            delta = scores['comparisons']['correct_minus_shuffled']['bce']
            lines.append(f'| {depth} | {panel[-1]} | {delta:+.8f} |')
    lines += ['', '两套正确接线组的准确率均为 49%–51%，且各自的 BCE 较常量基线改善'
              '不超过 0.005 时停止。实验覆盖实测缩减轮数、密文最高位及两把固定合成密钥。']
    markdown = '\n'.join(lines) + '\n'
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>SIMON32/64 单比特预测</title>'
            f'<pre>{html.escape(markdown)}</pre></html>')
    source = {key: saved[key] for key in ('run_id', 'run_ref', 'run_sha256')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps({'schema': 1, **source}).encode()}
