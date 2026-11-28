"""Render only the closed experiment record."""
from __future__ import annotations

import html
import json
from typing import Any, cast


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    if saved['status'] != 'SUCCEEDED':
        raise ValueError('report requires a successful terminal record')
    summary = cast('dict[str, Any]', saved['summary'])
    lines = ['# 三轮 SIMON32/64 单比特结构对照', '', f"运行 `{saved['run_id']}`。", '',
             '共享 TRAIN → 各组按 DEV BCE 选模 → 六份最佳权重严格磁盘回读 → 共享 TEST。',
             '', '两次重复同时更换合成密钥、明文和初始化。'
             '结果限于三轮、目标位 0 与这两个探索面板。']
    deltas = []
    for panel, result in summary['test'].items():
        lines += ['', f'## {panel}', '',
                  '| 组别 | TEST 正确数 | TEST 准确率 | TEST BCE | 参数 |'
                  ' 最佳轮 | 完成轮 | 拟合秒 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, score in result['arms'].items():
            arm = summary['arms'][f'{panel}-{name}']
            lines.append(f"| {name} | {score['correct']}/{score['rows']} | "
                         f"{score['accuracy']:.4%} | "
                         f"{score['bce']:.8f} | {arm['parameters']} | {arm['selected']['epoch']} | "
                         f"{arm['epochs_completed']} | {arm['fit_elapsed_seconds']:.3f} |")
        baseline = result['constant']
        lines += ['', f"TRAIN 常量概率 {baseline['probability']:.8f}，"
                  f"TEST 多数类准确率 {baseline['accuracy']:.4%}，BCE {baseline['bce']:.8f}。", '',
                  '| 比较 | TEST BCE 差 | 均正确 | 仅正确接线正确 | 仅对照正确 | 均错误 |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, comparison in result['comparisons'].items():
            pair = comparison['paired']
            lines.append(f"| {name} | {comparison['bce']:+.8f} | {pair['both_correct']} | "
                         f"{pair['left_only']} | {pair['right_only']} | {pair['both_wrong']} |")
        deltas.append((panel, result['comparisons']['correct_minus_shuffled']['bce']))
    outcome = '达到' if summary['confirmation_candidate'] else '未达到'
    lines += ['', '主比较为正确接线减打乱接线的 TEST BCE。两次面板上正确接线的 BCE '
              '均须低于打乱接线、普通 MLP 和 TRAIN 常量基线，才达到独立确认条件。',
              f'本轮{outcome}该条件。']
    markdown = '\n'.join(lines) + '\n'
    max_magnitude = max(0.01, *(abs(delta) for _, delta in deltas))
    bars = ''.join(f'<li>{html.escape(panel)} <meter min="{-max_magnitude}" '
                   f'max="{max_magnitude}" value="{delta}">{delta:+.8f}</meter> '
                   f'{delta:+.8f}</li>' for panel, delta in deltas)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SIMON32/64</title>'
            '<figure><figcaption>正确接线 − 打乱接线的 TEST BCE（越小越好）</figcaption>'
            f'<ul>{bars}</ul></figure><pre>{html.escape(markdown)}</pre></html>')
    source = {key: saved[key] for key in ('run_id', 'run_ref', 'run_sha256')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps({'schema': 1, **source}).encode()}
