"""Render only saved public evidence; never rebuild labels, models, or metrics."""
from __future__ import annotations

import html
import json
from collections.abc import Mapping

from . import config


def report_files(saved: Mapping[str, object]) -> dict[str, bytes]:
    seal = {'schema': 1, **{key: saved[key] for key in ('run_id', 'run_ref', 'run_sha256')}}
    status = saved['status']
    recipe, summary = saved['recipe'], saved['summary']
    known = isinstance(recipe, dict) and recipe.get('method') == config.METHOD
    heading = 'SIMON32/64 split8' if known else '实验运行记录（配方未识别）'
    lines = [f'# {heading}', '', f'状态：{status}。',
             f'来源：`{seal["run_ref"]}`，SHA-256 `{seal["run_sha256"]}`。', '']
    if isinstance(summary, dict) and isinstance(summary.get('reason'), str):
        lines.extend([f'原因：{summary["reason"]}。', ''])
    bars: list[str] = []
    scores = summary.get('test') if isinstance(summary, dict) else None
    if known and status == 'SUCCEEDED':
        if not isinstance(scores, dict) or set(scores) != {'r1', 'r2', 'r3'}:
            raise ValueError('incomplete saved results')
        lines.extend([
            '方法：同一合成 key 与共享明文划分 → 各轮四个独立 2 位 MLP → '
            '12 个 DEV best 全部校验回读 → TEST 按列拼接高 8 位。', '',
            '| 轮数 | 主指标：正确位/总位 | 位准确率 | TRAIN 多数位基线 | 整 8 位全对（附诊断） |',
            '|---|---:|---:|---:|---:|',
        ])
        for key in ('r1', 'r2', 'r3'):
            s = scores[key]
            acc = s['bit_accuracy']
            if type(acc) not in (int, float) or not 0 <= acc <= 1:
                raise ValueError('invalid saved accuracy')
            lines.append(f'| {key} | {s["bit_correct"]}/{s["bit_denominator"]} | {acc:.4%} | '
                         f'{s["majority_bit_correct"]}/{s["majority_bit_denominator"]} | '
                         f'{s["eight_all_correct"]}/{s["eight_all_denominator"]} |')
            bars.append(f'<label>{key} <meter min="0" max="1" value="{acc}"></meter> '
                        f'{acc:.4%}</label><br>')
        lines.extend(['', '仅单合成 key、固定 seed、1–3 轮探索；未预测完整 32 位密文，'
                      '不代表跨 key、完整轮数或对真实系统的结论。', ''])
    else:
        lines.extend(['无完整 TEST；不提供终分或科学结论。', ''])
    lines.extend(['## 保存的账本', '', '```json',
                  json.dumps(saved['counts'], ensure_ascii=False, indent=2), '```', ''])
    md = '\n'.join(lines)
    rendered = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                f'<title>{html.escape(heading)}</title><h1>{html.escape(heading)}</h1>'
                + ''.join(bars) + f'<pre>{html.escape(md)}</pre></html>')
    return {'report.md': md.encode(), 'index.html': rendered.encode(),
            'source.json': (json.dumps(seal, sort_keys=True) + '\n').encode()}
