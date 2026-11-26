"""Render only saved owner evidence; failed/partial results are not comparisons."""
from __future__ import annotations

from html import escape
from typing import Any

from ..config import canonical
from ..four_single.report import _report_files
from .config import SLOTS, TEST_REF


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    status, summary = saved['status'], saved['summary']
    progress = summary.get('progress', {})
    stop = summary.get('training_termination', progress.get('training_termination', {}))
    completed = summary.get('paired_completed', progress.get('paired_completed', '未保存'))
    epochs = stop.get('completed_epochs', '未保存')
    title = '创新1 · PRESENT-80 七轮单对：默认小块 / P 层结构小块'
    text = (f'# {title}\n\n状态：{status}；终态原因：{summary.get("reason", "未保存")}。'
            f'停训：{stop.get("reason", "未保存")}；轮数：{epochs}。'
            f'成对完成批次：{completed}。\n\n'
            '预定流程：只读原 PRESENT TRAIN/DEV → 同一批更新两版（仅小块取数表不同，'
            '固定小块与字各 0.5）→ 固定 10 轮，各取实际最低 DEV BCE（并列取早）→ '
            '两个当前权重由磁盘严格重载 → 同一原 TEST 密文及标签重新评分。旧 logits 不参与。\n\n')
    curves = summary.get('curves', [])
    if not curves and progress.get('last_dev') is not None:
        curves = [progress['last_dev']]
    text += '## 已保存 DEV\n\n'
    if curves:
        text += '|轮|版本|DEV BCE|DEV 准确率 %|\n|---:|---|---:|---:|\n'
        for point in curves:
            for slot, dev in point['dev'].items():
                text += f'|{point["epoch"]}|{slot}|{dev["bce"]:.8g}|{100*dev["accuracy"]:.4f}|\n'
        text += '\n'
    else:
        text += '未保存。\n\n'
    bars = []
    if status == 'SUCCEEDED':
        text += ('## TEST（不参与选模）\n\n'
                 '|版本|选中轮|选中 DEV BCE|TEST BCE|TEST 准确率 %|\n'
                 '|---|---:|---:|---:|---:|\n')
        for i, slot in enumerate(SLOTS):
            best, scores = summary['best'][slot], summary['test_scores'][slot]
            acc = 100 * scores['accuracy']
            text += (f'|{slot}|{best["epoch"]}|{best["bce"]:.8g}|{scores["bce"]:.8g}|'
                     f'{acc:.4f}|\n')
            y = 24 + 28 * i
            bars.append(f'<text x="0" y="{y}">{escape(slot)} {acc:.4f}%</text>'
                        f'<rect x="230" y="{y-14}" width="{acc*2.4:.3f}" height="16" '
                        'fill="#2865d1"/>')
        paired = summary['paired_test']
        text += (f'\n同一面板 {paired["rows"]:,} 对；B−A（结构减默认）准确率差 '
                 f'{paired["delta_acc_pp"]:+.4f} 个百分点，BCE 差 {paired["delta_bce"]:+.8g}。\n\n')
    else:
        if TEST_REF in saved['artifacts']:
            text += '已登记 TEST 产物（见保存证据）。'
        text += '未发布完整 TEST 比较。\n\n'
    counts = saved['counts']
    updates = counts.get('completed_optimizer_updates', '未保存')
    elapsed = counts.get('wall_elapsed_seconds', '未保存')
    text += f'已完成更新（两版合计）：{updates}；保存的累计墙钟时间（秒）：{elapsed}。\n\n'
    text += ('这是同一旧 TEST、一个配对初值、单个减轮任务的探索，不是独立确认、'
             '跨种子或跨算法结论。计数及耗时取保存记录；协作截止不是硬限。\n\n')
    if progress:
        text += '压缩失败封存：进度可能早于已登记产物，以登记路径和 SHA 为准。\n\n'
    public = {'test_panel', 'test_input', 'test_scores', 'paired_test'}
    evidence = {k: saved[k] for k in ('counts', 'artifacts', 'source')}
    evidence['summary'] = {k: v for k, v in summary.items()
                           if status == 'SUCCEEDED' or k not in public}
    text += f'## 保存证据\n\n```json\n{canonical(evidence)}\n```\n'
    svg = ('<svg viewBox="0 0 480 75" role="img" aria-label="已保存 TEST 准确率">'
           + ''.join(bars) + '</svg>') if bars else ''
    html = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            f'<title>{title}</title><h1>{title}</h1>{svg}<pre>{escape(text)}</pre></html>')
    return _report_files(saved, text, html)
