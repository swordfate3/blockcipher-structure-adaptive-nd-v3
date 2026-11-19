"""Pure report from the sealed RunStore record, never load models or sample data."""
from __future__ import annotations

import html
import json
from typing import Any

from .config import SLOTS


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    summary = saved['summary']
    text = ['# 创新 1：SKINNY64/64 单对从零训练', '',
            f"保存科学状态：{saved['status']}；原因：{summary.get('reason', 'UNKNOWN')}。", '',
            'SKINNY 8 轮 / Δ0x2000。每组单对、M/D 共享同批密文与标签，先 M 后 D。',
            'TRAIN → 每轮固定 DEV（各选最低 BCE，平分取早）→ 磁盘双 best 严格加载 → 新 TEST。',
            '不读取历史权重、数据或 TEST。单 seed；跨配方比较不能单独归因于对数。',
            '保存科学终态不是命令终态；命令输出是请求返回码，原生退出以独立监督为准。']
    scores = summary.get('test_scores', {}) if saved['status'] == 'SUCCEEDED' else {}
    bars = []
    if scores:
        text.extend(['', '|模型|选中轮|TEST ACC|TEST BCE|', '|---|---:|---:|---:|'])
        for i, slot in enumerate(SLOTS):
            score = scores[slot]
            acc = score['accuracy']
            epoch = summary['best'][slot]['epoch']
            text.append(f"|{slot}|{epoch}|{acc:.4%}|{score['bce']:.6f}|")
            y = 22 + 24 * i
            bars.append(f'<text x="5" y="{y}">{html.escape(slot)}</text>'
                        f'<rect x="180" y="{y-12}" width="{400*acc:.3f}" height="14" '
                        f'fill="#1769aa"/><text x="590" y="{y}">{acc:.4%}</text>')
        diff = summary['m_minus_d']
        text.append(f"同样本 M−D：ACC {100*diff['accuracy']:+.4f} pp，BCE {diff['bce']:+.6f}。")
    evidence = {k: saved[k] for k in ('recipe', 'policy', 'summary', 'counts', 'artifacts')}
    text.extend(['', '## 保存证据', '', '```json',
                 json.dumps(evidence, indent=2, ensure_ascii=False), '```', ''])
    markdown = '\n'.join(text)
    flow = ('<svg role="img" aria-label="方法流程" viewBox="0 0 740 40" width="740">'
            '<text x="10" y="25">新单对 TRAIN → M 后 D → DEV → 磁盘双 best → 新 TEST</text></svg>')
    chart = ('<svg role="img" aria-label="保存的 TEST ACC" viewBox="0 0 740 66" width="740">'
             + ''.join(bars) + '</svg>') if bars else ''
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>SKINNY 单对从零训练</title>' + flow + chart + '<pre>'
            + html.escape(markdown) + '</pre></html>')
    source = {k: saved[k] for k in ('run_id', 'run_ref', 'run_sha256')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(dict(schema=1, **source), sort_keys=True).encode()}
