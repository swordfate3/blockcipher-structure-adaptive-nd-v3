"""Render saved observations, without reloading models or datasets."""
from __future__ import annotations

from html import escape
from typing import Any

from ..config import canonical
from ..serial_bc.config import INDEPENDENT_METHOD, VOLUME_METHOD
from ..serial_bc.config import METHOD as SERIAL_METHOD
from .config import REPLAY_METHOD
from .interleave_config import FULL_METHOD, INTERLEAVE_METHODS


def _volume_text(saved: dict[str, Any]) -> str:
    summary, recipe = saved['summary'], saved['recipe']
    tr = recipe['training']
    rows = []
    observations = saved['verified_dev_observations']
    for observation in observations:
        dev = observation['dev']
        routes = ', '.join(f'{stem}: {weight:.6g}'
                           for stem, weight in observation['routes'][observation['task']].items())
        rows.append(f"|{observation['updates']}|{observation['epoch']}|{observation['task']}|"
                    f"{dev['groups']}|{dev['bce']:.6f}|{dev['accuracy']*100:.4f}|"
                    f"{observation['mean_online_train_bce']:.6f}|{routes}|")
    evidence = canonical({**{key: saved[key] for key in ('counts', 'source', 'artifacts')},
                          'summary': {k: v for k, v in summary.items() if k != 'dev_observations'},
                          'verified_dev_observations': observations})
    completed = saved['counts'].get('completed_optimizer_updates', '未保存')
    return ('# 创新1 · SKINNY TRAIN 池大小对照\n\n'
            f'状态：{saved["status"]}；终态原因：{summary.get("reason", "未保存")}。\n\n'
            f'固定 TRAIN：{tr["batch_groups"] * tr["batches_per_epoch"]:,} 对；'
            f'计划轮数：{tr["epochs"]}；已记录完成更新：{completed}。'
            '原 TRAIN/DEV 分割不变。只核验 SKINNY 原归档 → '
            '以真实公开卡从零更新同一 M 架构 → 按完成更新数在共同 DEV 评分。'
            '主指标为末点 DEV BCE；未到达或未保存末点时不以中间值替代。'
            '在线 TRAIN BCE 为到该点的累计更新均值，不是固定 TRAIN 重新评分。'
            '不早停、不选模、不保存模型权重，TEST 不打开、不评分。\n\n'
            '|更新|轮|任务|DEV 行数|DEV BCE|DEV 准确率 %|在线 TRAIN BCE|分支及权重|\n'
            '|---:|---:|---|---:|---:|---:|---:|---|\n'
            + ('\n'.join(rows) if rows else '|—|—|—|—|—|—|—|未保存|') + '\n\n'
            '仅报告已登记且经校验的 DEV 观测；本对照使用一个配对初值和已用于历史观察的固定 DEV，'
            '仅作探索，不证明跨种子或全新样本泛化。资源门禁为协作式。\n\n'
            f'## 保存证据\n\n```json\n{evidence}\n```\n')


def _report_files(saved: dict[str, Any], text: str, html: str) -> dict[str, bytes]:
    provenance = dict(schema=1, run_id=saved['run_id'], run_ref=saved['run_ref'],
                      run_sha256=saved['run_sha256'])
    return {'report.md': text.encode(), 'index.html': html.encode(),
            'source.json': canonical(provenance).encode()}


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    summary = saved['summary']
    method = saved['recipe']['method']
    if method == VOLUME_METHOD:
        text = _volume_text(saved)
        html = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
                '<title>SKINNY TRAIN 池大小对照</title>'
                f'<h1>SKINNY TRAIN 池大小对照</h1><pre>{escape(text)}</pre></html>')
        return _report_files(saved, text, html)
    independent = method == INDEPENDENT_METHOD
    interleave = method in INTERLEAVE_METHODS
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
    if method == REPLAY_METHOD or interleave:
        flow = ('只读校验旧 TRAIN/DEV（含 SKINNY 原归档链） → 四任务轮换更新一个新 M → '
                '宏 DEV BCE 早停及选模（并列取早） → 磁盘严格重载 → '
                '验证并复用原四份固定 TEST 的密文与标签，以新 M 重新评分。'
                '旧 logits 不用于评分；不是全新独立确认，不作配对推断。')
        stop = summary.get('training_termination', {})
        termination = (f'终态原因：{summary.get("reason", "未保存")}。'
                       f'停训原因：{stop.get("reason", "未保存")}；'
                       f'停训轮：{stop.get("completed_epochs", "未保存")}。')
        if interleave:
            pool = ('TRAIN 900 万对和 DEV 100 万对的全量池' if method == FULL_METHOD
                    else 'TRAIN 90 万对与 DEV 10 万对子集')
            flow = (f'复用每项历史 {pool} → 四密码按批轮换更新同一 M → '
                    '四项 DEV BCE 等权选模和早停 → 磁盘严格重载同一 M → 核验并评分四份旧 TEST。'
                    '结果来自已有 TEST，是一次探索性评估。实际取数表记录在各轮和重载路由中。')
            progress = summary.get('progress', {})
            dev_epochs = progress.get('completed_dev_epochs', '未知')
            updates = saved['counts'].get('completed_optimizer_updates', '未知')
            termination += f'完整 DEV 轮数：{dev_epochs}。已保存更新：{updates}。'
            if summary.get('counts_scope') == 'PRE_OWNER_SEAL':
                termination += '计数在终态写盘前采样。最终预算校验还需要命令和原生退出证据。'
            else:
                termination += '这些进度与计数仅反映已提交记录，为实绩下界。'
        routes = summary.get('persisted_reload', {}).get('routes', {})
        if routes:
            route_table = '## 磁盘重载后的路由权重\n\n|任务|选中分支及权重|\n|---|---|\n'
            for name, weights in routes.items():
                entries = ', '.join(f'{stem}: {weight:.6g}' for stem, weight in weights.items())
                if interleave:
                    lookup = summary['persisted_reload'].get('nibble_lookup', {}).get(name)
                    entries += f'（小块表 {lookup}）' if lookup else '（未选小块）'
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
    return _report_files(saved, text, html)
