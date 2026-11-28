"""Render the saved full-word experiment summary."""
from __future__ import annotations

import html
import json
from typing import Any, cast


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    summary = cast('dict[str, Any]', saved['summary'])
    names = {'correct': '公开接线', 'shuffled': '打乱接线', 'constant': '训练集常量'}
    stop = {'POSITIVE_CONTROL_FAILED': '三轮正控未通过', 'NEAR_CHANCE': '达到近随机停点',
            'ROUND_CEILING': '达到六轮上限'}
    lines = ['# SIMON32/64 完整 32 位预测的近随机边界', '',
             f"运行 {saved['run_id']}。{stop[summary['stop_reason']]}。", '',
             '四份权重严格回读后，各轮在两套 TEST 上作完整 32 位预测。', '',
             '| 轮数 | 面板 | 组别 | 正确位 | 准确率 | 同时 95% 区间 | 高字 | 低字 | BCE |',
             '|---:|---:|---|---:|---:|---|---:|---:|---:|']
    for depth, record in summary['rounds'].items():
        for panel, scores in record['test'].items():
            for arm, score in scores.items():
                ci = (f"[{score['ci_low']:.4%}, {score['ci_high']:.4%}]"
                      if arm != 'constant' else '—')
                lines.append(f"| {depth} | {panel[-1]} | {names[arm]} | "
                             f"{score['correct_bits']:,}/{score['rows'] * 32:,} | "
                             f"{score['accuracy']:.4%} | {ci} | {score['high_accuracy']:.4%} | "
                             f"{score['low_accuracy']:.4%} | {score['bce']:.8f} |")
    lines += ['', '停点要求两套面板、两种接线的全 32 位准确率区间均落入 49%–51%。',
              '结果适用于本次受测缩减轮数、两把固定合成密钥、这两个模型和所用预算。']
    md = '\n'.join(lines) + '\n'
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>SIMON32/64 完整位预测</title>'
            f'<pre>{html.escape(md)}</pre></html>')
    source = {'schema': 1, **{key: saved[key] for key in ('run_id', 'run_ref', 'run_sha256')}}
    return {'report.md': md.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(source).encode()}
