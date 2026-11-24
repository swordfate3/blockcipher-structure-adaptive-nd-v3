"""Render saved observations, without reloading models or datasets."""
from __future__ import annotations

from html import escape
from typing import Any

from ..config import canonical
from ..serial_bc.config import INDEPENDENT_METHOD
from ..serial_bc.config import METHOD as SERIAL_METHOD
from .config import REPLAY_METHOD


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    summary = saved['summary']
    method = saved['recipe']['method']
    independent = method == INDEPENDENT_METHOD
    title = '创新1 · 单算法独立 B/C' if independent else '创新1 · 四任务单对共享 M'
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
    termination, route_table = '', ''
    if method == REPLAY_METHOD:
        flow = ('只读校验旧 TRAIN/DEV（含 SKINNY 原归档链） → 四任务轮换更新一个新 M → '
                '宏 DEV BCE 早停及选模（并列取早） → 磁盘严格重载 → '
                '验证并复用原四份固定 TEST 的密文与标签，以新 M 重新评分。'
                '旧 logits 不用于评分；不是全新独立确认，不作配对推断。')
        stop = summary.get('training_termination', {})
        termination = (f'终态原因：{summary.get("reason", "未保存")}。'
                       f'停训原因：{stop.get("reason", "未保存")}；'
                       f'停训轮：{stop.get("completed_epochs", "未保存")}。')
        routes = summary.get('persisted_reload', {}).get('routes', {})
        if routes:
            route_table = '## 磁盘重载后的路由权重\n\n|任务|选中分支及权重|\n|---|---|\n'
            for name, weights in routes.items():
                entries = ', '.join(f'{stem}: {weight:.6g}' for stem, weight in weights.items())
                route_table += f'|{name}|{entries}|\n'
            route_table += '\n'
    if method == SERIAL_METHOD or independent:
        best = summary.get('persisted_reload', {})
        if independent:
            flow = ('只核验本算法归档 → 原 TRAIN/DEV 分割内分层固定抽样 → '
                    '新 M、新 Adam 独立拟合 → 仅末轮评分本任务 DEV，不选模 → '
                    '严格重载磁盘末轮权重 → 复用本算法原 TEST，仅以新权重评分。'
                    '旧 logits 不参与；不是全新独立确认。每次拟合的路由输入恒定，'
                    '不检验跨任务换卡或接续训练遗忘。')
        else:
            flow = ('原 TRAIN/DEV 分割内分层固定抽样 → SIMECK／PRESENT／SIMON／SKINNY 串行，'
                    '共用一个 M 与 Adam → 每阶段保存四项 DEV 及路由，DEV 不选模 → '
                    '第四阶段权重严格重载 → 复用原四份 TEST，仅以新权重评分。'
                    '旧 logits 不参与；不是全新独立确认，也不能与历史轮换实验归因比较。')
        mode = saved['recipe']['routing']['mode']
        termination = f'路由模式：{mode}；终态原因：{summary.get("reason", "未保存")}。'
        route_table = ('## 阶段 DEV 与路由\n\n|阶段|训练任务|DEV 任务|BCE|准确率 %|分支及权重|\n'
                       '|---:|---|---|---:|---:|---|\n')
        for stage in summary.get('stages', []):
            for name, value in stage['dev'].items():
                routes = ', '.join(f'{stem}: {w:.6g}' for stem, w in stage['routes'][name].items())
                route_table += (f"|{stage['stage']}|{stage['task']}|{name}|{value['bce']:.6f}|"
                                f"{value['accuracy']*100:.4f}|{routes}|\n")
        route_table += '\n'
    evidence = canonical({key: saved[key] for key in ('summary', 'counts', 'source', 'artifacts')})
    selected = (f'最终重载阶段：{best.get("stage", "未保存")}。'
                if method == SERIAL_METHOD or independent
                else f'选中轮：{best.get("epoch", "未保存")}。')
    text = (f'# {title}\n\n状态：{saved["status"]}。'
            f'{selected}{termination}\n\n{flow}\n\n'
            '|任务|TEST 行数|BCE|准确率 %|\n|---|---:|---:|---:|\n'
            + '\n'.join(rows) + '\n\n仅报告已保存值；部分 TEST 不代表完整执行。'
            '不是行扫/列扫因果对照，不做跨配置配对比较或置信区间。'
            '资源门禁为协作式；CUDA 上限只约束进程分配器。\n\n'
            f'{route_table}## 保存证据（含选中/各轮路由）\n\n```json\n{evidence}\n```\n')
    svg = ('<svg viewBox="0 0 420 120" role="img" aria-label="已保存 TEST 准确率">'
           + ''.join(bars) + '</svg>')
    heading = title if independent else '四任务单对共享 M'
    html = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            f'<title>{heading}</title><h1>{heading}</h1>'
            f'<p aria-label="方法流程">{escape(flow)}</p>{svg}<pre>{escape(text)}</pre></html>')
    provenance = dict(schema=1, run_id=saved['run_id'], run_ref=saved['run_ref'],
                      run_sha256=saved['run_sha256'])
    return {'report.md': text.encode(), 'index.html': html.encode(),
            'source.json': canonical(provenance).encode()}
