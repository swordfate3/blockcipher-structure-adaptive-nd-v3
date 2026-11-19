"""Pure rendering of the new saved summary; no model access or metric recomputation."""
from __future__ import annotations

import html
import json
from typing import Any


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    summary = saved['summary']
    parts = ['# 创新1：SKINNY 固定第一对只读对照', '',
             f"保存科学状态：{saved['status']}；原因：{summary['reason']}。", '',
             '原 SKINNY 8 轮 / Δ0x2000 TEST → 每组仅 index=0 → 两个冻结模型逐对打分 → 汇总。',
             'M 和 D 均用原模型的逐对 logit；与原四对平均聚合的分解在实数算术下等价，'
             '不主张浮点逐位相同。未重训、未重新加密、未调阈值，也不是新独立 TEST 或速度基准。',
             '保存的计数截至科学终态写入前；command_resources 是 stdout 写入前的协作采样，'
             '不含 stdout 写入、flush 或进程退出。完整 wall 和原生退出由外部独立监督。']
    bars = []
    if saved['status'] == 'SUCCEEDED':
        parts.extend(['', '|模型|单对 ACC %|单对 BCE|原保存四对 ACC %|单对−四对 pp|',
                      '|---|---:|---:|---:|---:|'])
        for row, (slot, score) in enumerate(summary['scores'].items()):
            parts.append(f"|{slot}|{score['accuracy']:.4%}|{score['bce']:.6f}|"
                         f"{score['old_four_pair_accuracy']:.4%}|{score['vs_four_pair_pp']:+.4f}|")
            for col, (key, label, color) in enumerate((
                    ('accuracy', '单对', '#1769aa'),
                    ('old_four_pair_accuracy', '原四对', '#696969'))):
                y = 20 + 24 * (2 * row + col)
                value = score[key]
                width = 400 * value
                bars.append(f'<text x="10" y="{y}">{html.escape(slot)} {label}</text>'
                            f'<rect x="220" y="{y-12}" width="{width:.4f}" height="14" '
                            f'fill="{color}"/><text x="630" y="{y}">{value:.4%}</text>')
        parts.extend(['', f"单对 M−D：{summary['m_minus_d_pp']:+.4f} pp。"])
    evidence = {k: saved[k] for k in ('recipe', 'policy', 'source', 'summary', 'counts')
                if k in saved}
    parts.extend(['', '## 保存证据', '', '```json',
                  json.dumps(evidence, indent=2, ensure_ascii=False), '```', ''])
    markdown = '\n'.join(parts)
    flow = ('<svg role="img" aria-label="方法流程" viewBox="0 0 740 40" width="740">'
            '<text x="10" y="25">封存 TEST → 固定第一对 → 冻结 M / D 逐对打分 → 汇总</text></svg>')
    chart = ('<svg role="img" aria-label="保存的单对与四对准确率" viewBox="0 0 740 105" '
             'width="740">' + ''.join(bars) + '</svg>') if bars else ''
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>SKINNY 固定第一对只读对照</title>' + flow + chart + '<pre>'
            + html.escape(markdown) + '</pre></html>')
    source = {k: saved[k] for k in ('run_id', 'run_ref', 'run_sha256')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(dict(schema=1, **source), sort_keys=True).encode()}
