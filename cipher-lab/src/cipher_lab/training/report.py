"""Pure Chinese training reports from read_saved; no tensors, I/O or recomputation."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from typing import cast


def _map(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    # Markdown punctuation also encoded: the same text is safe in both renderers.
    return ''.join(f'&#{ord(c)};' if c in '\\`*_{}[]()!#+|~\n\r' else c
                   for c in escape(str(value), quote=True))


def _json(value: object) -> str:
    return _text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _number(value: object) -> float | None:
    if type(value) in (int, float) and math.isfinite(cast(float, value)):
        return float(cast(float, value))
    return None


def render_training_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Exactly three files for original RunStore.write_report_files, never write here."""
    summary = _map(saved.get('summary'))
    method = summary.get('method', _map(saved.get('recipe')).get('method'))
    prediction = method == 'bit-prediction-v1'
    shared = method == 'shared-m-four-task-v1'
    title = ('位预测 MLP 训练报告' if prediction else
             '四任务 shared-M 训练报告' if shared else 'UNKNOWN 方法训练报告')
    status = saved.get('status', 'UNKNOWN')
    if status == 'RUNNING':
        status = 'UNKNOWN（RUNNING，无最终记录）'
    synthetic = summary.get('synthetic') is True or summary.get('backend_identity') == 'demo'
    identity = ('合成工程演示，不是真实密码科学结果。' if synthetic
                else '后端由调用方声明，密码实现及 KAT 未核验；不认证科学有效性。')
    warning = ('限制：scientific_validation=false；执行成功不等于预测有效。'
               '未做真实密码/KAT/GPU/生产RSS/跨key验证，无显著性或泛化结论。'
               'null 是独立标签对照而非报警；TRAIN 基线仅拟合 TRAIN 标签。'
               '资源检查为协作式；本报告只展示保存结果，不解码模型/数据或重算指标。')
    flow = ('TRAIN/DEV → Adam/BCE/clip1 → DEV 严格下降最早平局 → 保存 best → SHA 严格加载'
            + (' → 释放 TEST 标签 → 最佳模型 TRAIN/DEV/TEST' if prediction
               else ' → 预建 TEST 首次评分 → 最佳模型 TRAIN/DEV/TEST'))
    caption = ('固定目标 bit；延后 TEST 标签；BCE 选模' if prediction else
               '每宏batch四任务各一步；四任务 DEV BCE 等权均值选模')
    if not prediction and not shared:
        flow = 'UNKNOWN：未保存可识别的方法，不推断训练流程'
        caption = '方法未知，仅展示保存结果'
    method_svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 100" '
                  'role="img" aria-label="训练方法流程图">'
                  '<rect width="1200" height="100" fill="#eef3f8"/>'
                  f'<text x="12" y="42" font-size="14">{_text(flow)}</text>'
                  f'<text x="12" y="75" font-size="15">{caption}</text></svg>')
    rows: list[str] = []
    bars: list[str] = []
    metrics = _map(summary.get('metrics'))
    for split in ('train', 'dev', 'test'):
        tasks = _map(_map(metrics.get(split)).get('tasks'))
        for task, value in tasks.items():
            m = _map(value)
            rows.append(f'{split.upper()} / {_text(task)}：实际 rows={_text(m.get("rows"))}；'
                        f'ACC={_text(m.get("acc"))}；BCE={_text(m.get("bce"))}；'
                        f'TRAIN基线 ACC={_text(m.get("baseline_acc"))}；'
                        f'BCE={_text(m.get("baseline_bce"))}')
            acc = _number(m.get('acc'))
            if acc is not None and 0 <= acc <= 1:
                y = 25 + 30 * len(bars)
                bars.append(f'<text x="8" y="{y}">{_text(split.upper())}/'
                            f'{_text(task)} ACC={acc:g}</text>'
                            f'<rect x="400" y="{y-16}" width="{acc*400:.2f}" '
                            'height="18" fill="#276785"/>')
    result_svg = ('<svg xmlns="http://www.w3.org/2000/svg" '
                  f'viewBox="0 0 850 {max(70, 35+30*len(bars))}" role="img" '
                  'aria-label="已保存的逐任务 ACC 结果图">'
                  + (''.join(bars) or '<text x="8" y="30">无已保存指标，不绘制虚假成绩</text>')
                  + '</svg>')
    details = {'公开任务 algorithm/rounds/target/difference':
               summary.get('task_declarations', []),
               'recipe': saved.get('recipe', {}), '实际数据量': saved.get('counts', {}),
               '成本/四单位/墙钟/CPU/RSS': summary.get('meter', {}),
               'selected': summary.get('selected', {}),
               'learning_curve': summary.get('learning_curve', []),
               '来源引用': summary.get('refs', {}), 'source': saved.get('source', {}),
               'policy': saved.get('policy', {})}
    heading = f'{title}：{_text(saved.get("run_id", "未知"))}'
    body = (f'<h1>{heading}</h1><p>执行状态：{_text(status)}；'
            f'backend={_text(summary.get("backend_identity", "UNVERIFIED"))}</p>'
            f'<p>{identity}</p><p>{warning}</p><h2>方法</h2>{method_svg}'
            f'<h2>实际结果 ACC / BCE</h2>'
            + ''.join(f'<p>{row}</p>' for row in rows)
            + f'<h2>保存结果图</h2>{result_svg}<h2>任务、数据与成本</h2>'
            '<p>四单位：real_blocks / null_words / optimizer_updates / forward_rows</p>'
            f'<pre>{_json(details)}</pre><p>公开原因：{_text(summary.get("reason", ""))}</p>'
            '<p>来源 source.json 绑定已冻结 run SHA；不提供模型/数据下载。</p>')
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>body{{max-width:1200px;margin:auto;padding:1em;'
            'font-family:sans-serif;line-height:1.7}svg{width:100%}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><body>'
            + body + '</body></html>')
    source = {'schema': 1, 'run_id': saved['run_id'], 'run_ref': saved['run_ref'],
              'run_sha256': saved['run_sha256']}
    return {'report.md': (f'# {heading}\n\n' + body + '\n').encode(),
            'index.html': html.encode(),
            'source.json': (json.dumps(source, ensure_ascii=False, sort_keys=True,
                                       indent=2) + '\n').encode()}
