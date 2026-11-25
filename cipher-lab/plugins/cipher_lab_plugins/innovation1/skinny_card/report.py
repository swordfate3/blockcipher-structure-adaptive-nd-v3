"""Pure presentation of the owner's saved comparison; never re-score or re-rank."""
from __future__ import annotations

from html import escape
from typing import Any

from ..config import canonical
from ..four_single.report import _report_files
from .config import SLOTS


def _routes(value: dict[str, float]) -> str:
    return ', '.join(f'{stem}: {weight:.6g}' for stem, weight in value.items()) or '未保存'


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    status, summary = saved['status'], saved['summary']
    # Compact owner seals may have only the saved progress subrecord and counts.
    progress = summary.get('progress', {})
    stop = summary.get('training_termination', progress.get('training_termination', {}))
    epochs = stop.get('completed_epochs', '未保存')
    completed = summary.get('paired_completed', progress.get('paired_completed', '未保存'))
    title = '创新1 · SKINNY 完整固定池：有卡 / 无卡'
    text = (f'# {title}\n\n状态：{status}；终态原因：{summary.get("reason", "未保存")}。'
            f'停训：{stop.get("reason", "未保存")}；轮数：{epochs}。\n\n'
            '预定流程：原 SKINNY TRAIN/DEV → 同一批更新两个新模型（仅 Top2 打分输入不同） → '
            '共同 DEV 早停，各取实际最低 DEV BCE（并列取早） → 两个当前权重由磁盘严格重载 → '
            '同一原 TEST 密文及标签重新评分。旧 logits 不参与。\n\n'
            f'成对完成批次：{completed}。'
            '计数按保存记录；协作截止不是硬限。\n\n')
    rows = ['|轮|版本|DEV BCE|DEV 准确率 %|路由|', '|---:|---|---:|---:|---|']
    curves = summary.get('curves', [])
    if not curves and progress.get('last_dev') is not None:
        curves = [progress['last_dev']]
    for point in curves:
        for slot, dev in point['dev'].items():
            weights = point.get('routes', {}).get(slot, {})
            rows.append(f'|{point["epoch"]}|{slot}|{dev["bce"]:.8g}|{100*dev["accuracy"]:.4f}|'
                        f'{_routes(weights)}|')
    text += ('## 已保存 DEV\n\n' + ('\n'.join(rows) if len(rows) > 2 else '未保存。') + '\n\n')
    bars = []
    if status == 'SUCCEEDED':
        text += ('## TEST（不参与选模）\n\n'
                 '|版本|选中轮|选中 DEV BCE|TEST BCE|TEST 准确率 %|重载后路由|\n'
                 '|---|---:|---:|---:|---:|---|\n')
        for i, slot in enumerate(SLOTS):
            best = summary['best'][slot]
            scores = summary['test_scores'][slot]
            acc = 100 * scores['accuracy']
            text += (f'|{slot}|{best["epoch"]}|{best["bce"]:.8g}|{scores["bce"]:.8g}|'
                     f'{acc:.4f}|{_routes(summary["reload_routes"][slot])}|\n')
            y = 24 + 28 * i
            bars.append(f'<text x="0" y="{y}">{escape(slot)} {acc:.4f}%</text>'
                        f'<rect x="215" y="{y-14}" width="{acc*2.5:.3f}" height="16" '
                        'fill="#2865d1"/>')
        paired = summary['paired_test']
        text += (f'\n同一面板 {paired["rows"]:,} 对；C−B 准确率差 '
                 f'{paired["delta_acc_pp"]:+.4f} 个百分点，BCE 差 {paired["delta_bce"]:+.8g}。\n\n')
    else:
        if 'dataset/test-SKINNY.npz' in saved['artifacts']:
            text += '已登记 TEST 产物（见保存证据）。'
        text += '未发布完整 TEST 比较。\n\n'
    text += ('两种打分输入在单任务内均恒定；本设计预定复用历史 TEST，仅设一个配对初值，'
             '不是独立确认、跨算法切换或跨种子结论。\n\n')
    if progress:
        text += ('压缩失败封存：进度为最近提交的快照，可能早于已登记的产物；'
                 '产物身份以登记的路径和 SHA 为准。\n\n')
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
