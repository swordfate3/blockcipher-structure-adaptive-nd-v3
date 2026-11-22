"""Render the saved four-task observations, without reloading model or datasets."""
from __future__ import annotations

from html import escape
from typing import Any

from ..config import canonical


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    summary = saved['summary']
    best = summary.get('best', {})
    tests = summary.get('test', {})
    rows, bars = [], []
    for i, task in enumerate(saved['recipe']['tasks']):
        name = task['cipher']
        if name not in tests:
            rows.append(f'|{name}|未保存|未保存|未保存|')
            continue
        scores = tests[name]['scores']
        acc = 100 * scores['accuracy']
        rows.append(f"|{name}|{scores['groups']}|{scores['bce']:.6f}|{acc:.4f}|")
        y = 22 + 28 * i
        bars.append(f'<text x="0" y="{y}">{escape(name)}</text>'
                    f'<rect x="88" y="{y-14}" width="{acc*3:.3f}" height="16" '
                    'fill="#2865d1"/>')
    flow = ('先核验 SKINNY 旧池，再固定并保存其他三池 → READY → 四任务轮换更新一个 M → '
            '四项 DEV BCE 等权选模（并列取早） → 磁盘严格重载 → 四项独立 TEST')
    evidence = canonical({key: saved[key] for key in ('summary', 'counts', 'source', 'artifacts')})
    text = (f'# 创新1 · 四任务单对共享 M\n\n状态：{saved["status"]}。'
            f'选中轮：{best.get("epoch", "未保存")}。\n\n{flow}\n\n'
            '|任务|TEST 行数|BCE|准确率 %|\n|---|---:|---:|---:|\n'
            + '\n'.join(rows) + '\n\n仅报告已保存值；部分 TEST 不代表完整执行。'
            '不是行扫/列扫因果对照，不做跨配置配对比较或置信区间。'
            '资源门禁为协作式；CUDA 上限只约束进程分配器。\n\n'
            f'## 保存证据（含选中/各轮路由）\n\n```json\n{evidence}\n```\n')
    svg = ('<svg viewBox="0 0 420 120" role="img" aria-label="已保存 TEST 准确率">'
           + ''.join(bars) + '</svg>')
    html = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>四任务单对共享 M</title><h1>四任务单对共享 M</h1>'
            f'<p aria-label="方法流程">{escape(flow)}</p>{svg}<pre>{escape(text)}</pre></html>')
    provenance = dict(schema=1, run_id=saved['run_id'], run_ref=saved['run_ref'],
                      run_sha256=saved['run_sha256'])
    return {'report.md': text.encode(), 'index.html': html.encode(),
            'source.json': canonical(provenance).encode()}
