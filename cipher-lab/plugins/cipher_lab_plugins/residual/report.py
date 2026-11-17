"""Saved-only residual paired report: absolute-bit arms, paired exact test, curves.

Consumes summary arms/paired/diagnostics/pairing written by the paired method.
No tensors, model/data access, label generation, metric recomputation or file I/O.
Legacy empty top-level metrics/learning_curve/selected are deliberate placeholders
and are never presented as results.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from cipher_lab.training.report import _json, _map, _number, _text

_SPLITS = ('train', 'dev', 'test')
_ARMS = ('direct', 'residual')
_ARM_LABEL = {'direct': 'direct（直接绝对bit）', 'residual': 'residual（残差→绝对bit）'}
_COLOR = {('direct', 'train_bce'): '#1f4e79', ('direct', 'dev'): '#4a90c4',
          ('residual', 'train_bce'): '#9c2f2f', ('residual', 'dev'): '#d27676'}


def _arm(summary: Mapping[str, object], mode: str) -> Mapping[str, object]:
    return _map(_map(summary.get('arms')).get(mode))


def _metric(summary: Mapping[str, object], mode: str, split: str) -> Mapping[str, object]:
    arm = _arm(summary, mode)
    task = _map(_map(_map(arm.get('metrics')).get(split)).get('tasks')).get('SIMON')
    return _map(task)


def _value(metric: Mapping[str, object], name: str) -> str:
    number = _number(metric.get(name))
    return '未保存（不补 0）' if number is None else f'{number:.6f}'


def _method_svg() -> str:
    flow = ('同一私有key/面板 → 两臂相同初值与shuffle → direct学y、residual学y XOR h → '
            '各自DEV最低BCE/最早平局并持久保存 → 两臂均严格SHA加载 → 释放TEST → '
            '同一次前向同时得到绝对bit指标与配对统计')
    note = ('公开h(x)使用零 MASTER key 的真实轮密钥调度并照常计费，不是全零轮密钥捷径；'
            '两个模型不共享参数或梯度。TEST 在双严格加载后自动释放并评分，不是技术 sealed。')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" '
            'role="img" aria-label="残差配对方法流程图">'
            '<rect width="1200" height="120" fill="#eef3f8"/>'
            f'<text x="12" y="38" font-size="14">{_text(flow)}</text>'
            f'<text x="12" y="70" font-size="13">{_text(note)}</text>'
            '<text x="12" y="100" font-size="13">结果只读保存值；不重新训练、不重新前向、'
            '不重新加密或重算指标。</text></svg>')


def _paired_svg(summary: Mapping[str, object]) -> str:
    paired = _map(summary.get('paired'))
    bars: list[str] = []
    for index, split in enumerate(_SPLITS):
        top = 24 + index * 100
        bars.append(f'<text x="8" y="{top + 12}" font-size="14">'
                    f'{_text(split.upper())}</text>')
        direct = _number(_metric(summary, 'direct', split).get('acc'))
        residual = _number(_metric(summary, 'residual', split).get('acc'))
        for offset, mode, value in ((0, 'direct', direct), (22, 'residual', residual)):
            y = top + offset
            bars.append(f'<text x="90" y="{y + 12}" font-size="13">'
                        f'{_text(mode)}</text>')
            if value is not None and 0 <= value <= 1:
                bars.append(f'<rect x="180" y="{y}" width="{value * 420:.2f}" height="16" '
                            'fill="#276785" opacity="0.85"/>')
                bars.append(f'<text x="610" y="{y + 12}" font-size="13">'
                            f'ACC={value:.6f}</text>')
            else:
                bars.append(f'<text x="180" y="{y + 12}" font-size="13">'
                            '未保存（不补 0）</text>')
        stats = _map(paired.get(split))
        first, second = _paired_parts(split, stats)
        bars.append(f'<text x="8" y="{top + 62}" font-size="12">{_text(first)}</text>')
        if second:
            bars.append(f'<text x="8" y="{top + 78}" font-size="12">{_text(second)}</text>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 1600 {max(70, 24 + 100 * len(_SPLITS))}" role="img" '
            'aria-label="保存的绝对bit ACC 与配对检验">'
            + (''.join(bars) or '<text x="8" y="30">无已保存配对指标</text>') + '</svg>')


def _paired_parts(split: str, stats: Mapping[str, object]) -> tuple[str, str]:
    """Two display lines built only from saved fields; never derive b-c or b+c."""
    if not stats:
        return f'{split.upper()}：配对统计未保存（不补 0）', ''
    n = stats.get('n')
    b, c = stats.get('b'), stats.get('c')
    delta = _number(stats.get('delta'))
    p = _number(stats.get('p_one_sided'))
    interval = stats.get('normal_95_interval')
    loss = _map(stats.get('loss_difference'))
    delta_text = '未保存' if delta is None else f'{delta * 100:.3f}pp'
    p_text = '未保存' if p is None else f'{p:.6g}'
    interval_text = ('未报告（b,c 未同时≥25）' if interval is None
                     else f'{interval}（描述性正态近似，非确认）')
    first = (f'{split.upper()}：n={n} b={b} c={c} '
             f'both_correct={stats.get("both_correct", "未保存")} '
             f'both_wrong={stats.get("both_wrong", "未保存")} '
             f'sum_d={stats.get("sum_d", "未保存")} '
             f'sum_d2={stats.get("sum_d2", "未保存")} '
             f'delta(residual-direct)={delta_text} '
             f'单侧配对精确检验 p={p_text}；正态近似区间={interval_text}')
    second = (f'loss_difference：direction={loss.get("direction", "未保存")} '
              f'sum={loss.get("sum", "未保存")} '
              f'sum_squares={loss.get("sum_squares", "未保存")} '
              f'mean={loss.get("mean", "未保存")}')
    return first, second


def _paired_text(split: str, stats: Mapping[str, object]) -> str:
    first, second = _paired_parts(split, stats)
    return first if not second else f'{first}；{second}'


def _curve_svg(summary: Mapping[str, object]) -> str:
    series: list[tuple[str, str, list[tuple[int, float]]]] = []
    for mode in _ARMS:
        curve = _arm(summary, mode).get('learning_curve')
        entries = curve if isinstance(curve, list) else []
        for field in ('train_bce', 'dev'):
            points: list[tuple[int, float]] = []
            for entry in entries:
                node = _map(entry)
                epoch = node.get('epoch')
                raw = (node.get('train_bce') if field == 'train_bce'
                       else _map(node.get('dev')).get('mean_bce'))
                value = _number(raw)
                if type(epoch) is int and value is not None:
                    points.append((epoch, value))
            series.append((mode, field, points))
    values = [value for _, _, points in series for _, value in points]
    if not values:
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 120" '
                'role="img" aria-label="学习曲线"><rect width="900" height="120" '
                'fill="#f7f7f7"/><text x="12" y="40">无已保存学习曲线'
                '（不重算、不补 0）</text></svg>')
    low, high = min(values), max(values)
    if high == low:
        high = low + 1.
    epochs = sorted({epoch for _, _, points in series for epoch, _ in points})
    left, right, top, bottom = 80., 860., 30., 260.

    def x(epoch: int) -> float:
        if len(epochs) == 1:
            return (left + right) / 2
        return left + (right - left) * (epoch - epochs[0]) / (epochs[-1] - epochs[0])

    def y(value: float) -> float:
        return bottom - (bottom - top) * (value - low) / (high - low)

    parts: list[str] = [f'<text x="8" y="{top + 4}" font-size="12">BCE {high:.4f}</text>',
                        f'<text x="8" y="{bottom}" font-size="12">BCE {low:.4f}</text>']
    for epoch in epochs:
        parts.append(f'<text x="{x(epoch):.1f}" y="{bottom + 22:.1f}" font-size="12">'
                     f'epoch {epoch}</text>')
    legend: list[str] = []
    for mode, field, points in series:
        if not points:
            continue
        color = _COLOR[(mode, field)]
        coords = ' '.join(f'{x(epoch):.1f},{y(value):.1f}' for epoch, value in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                     f'points="{coords}"/>')
        for epoch, value in points:
            parts.append(f'<circle cx="{x(epoch):.1f}" cy="{y(value):.1f}" r="3" '
                         f'fill="{color}"><title>{_text(mode)} epoch {epoch} '
                         f'{_text(field)}={value:.6g}</title></circle>')
        label = f'{mode} {field}'
        legend.append(f'<text x="640" y="{30 + 20 * len(legend)}" font-size="12" '
                      f'fill="{color}">{_text(label)}</text>')
    parts.extend(legend)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300" '
            'role="img" aria-label="保存的逐epoch学习曲线">'
            '<rect width="900" height="300" fill="#fbfbfb"/>'
            '<line x1="80" y1="260" x2="860" y2="260" stroke="#888"/>'
            + ''.join(parts) + '</svg>')


def _split_rows(summary: Mapping[str, object]) -> list[str]:
    rows: list[str] = []
    for split in _SPLITS:
        for mode in _ARMS:
            metric = _metric(summary, mode, split)
            rows.append(f'{split.upper()} / {_text(_ARM_LABEL[mode])}：'
                        f'rows={_text(metric.get("rows", "未保存"))}；'
                        f'绝对bit ACC={_value(metric, "acc")}；BCE={_value(metric, "bce")}；'
                        f'TRAIN基线 ACC={_value(metric, "baseline_acc")}'
                        f' / BCE={_value(metric, "baseline_bce")}')
    return rows


def _diagnostic_rows(summary: Mapping[str, object]) -> list[str]:
    constant = _map(_map(summary.get('diagnostics')).get('reference_constant'))
    splits = _map(constant.get('splits'))
    rows = [f'TRAIN 残差常量：rows={_text(constant.get("train_residual_rows", "未保存"))}；'
            f'ones={_text(constant.get("train_residual_ones", "未保存"))}；'
            f'多数={_text(constant.get("train_residual_constant", "未保存"))}；'
            f'Jeffreys 残差概率={_text(constant.get("train_residual_probability", "未保存"))}']
    for split in _SPLITS:
        values = _map(splits.get(split))
        flip = _map(values.get('fixed_flip'))
        probability = _map(values.get('residual_probability'))
        if not values:
            rows.append(f'{split.upper()}：诊断未保存（不补 0）')
            continue
        rows.append(f'{split.upper()}：fixed_flip 绝对bit ACC={_value(flip, "acc")}'
                    f' / BCE={_value(flip, "bce")}；'
                    f'residual_probability 绝对bit ACC={_value(probability, "acc")}'
                    f' / BCE={_value(probability, "bce")}')
    return rows


def _selection_rows(summary: Mapping[str, object]) -> list[str]:
    rows: list[str] = []
    for mode in _ARMS:
        selected = _map(_arm(summary, mode).get('selected'))
        if not selected:
            rows.append(f'{mode}：selected 未保存（不补 0）')
            continue
        rows.append(f'{mode}：epoch={_text(selected.get("epoch"))}；'
                    f'updates={_text(selected.get("updates"))}；'
                    f'DEV BCE={_text(selected.get("selected_dev_bce"))}；'
                    f'SHA256={_text(selected.get("sha256", "未保存"))}')
    pairing = _map(summary.get('pairing'))
    identities = _map(pairing.get('initial_state_sha256'))
    rows.append('两臂初值 SHA256 相同：'
                + _text(identities.get('direct') == identities.get('residual')
                        if identities else '未保存'))
    rows.append(f'shuffle_seed={_text(pairing.get("shuffle_seed", "未保存"))}')
    panels = _map(pairing.get('panels'))
    for split in _SPLITS:
        identity = _map(panels.get(split))
        rows.append(f'{split.upper()} 面板：rows={_text(identity.get("rows", "未保存"))}；'
                    f'x={_text(identity.get("x_sha256", "未保存"))}；'
                    f'y={_text(identity.get("y_sha256", "未保存"))}；'
                    f'h={_text(identity.get("h_sha256", "未保存"))}')
    return rows


def render_residual_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Exactly three files for the original RunStore; never writes or reads here."""
    summary = _map(saved.get('summary'))
    title = '残差配对 MLP 训练报告'
    status = saved.get('status', 'UNKNOWN')
    if status == 'RUNNING':
        status = 'UNKNOWN（RUNNING，无最终记录）'
    synthetic = summary.get('synthetic') is True or summary.get('backend_identity') == 'demo'
    identity = ('合成工程演示，不是真实密码科学结果。' if synthetic
                else '后端由调用方声明，密码实现及 KAT 未核验；不认证科学有效性。')
    warning = ('限制：scientific_validation=false；执行成功不等于残差方案优于 direct，'
               '更不是密码攻击或新颖性证明。配对精确检验只回答固定key、固定训练模型下的新明文；'
               'DEV 筛选 p 只作预定探索，不是有效确认 p 值或多重比较校正。'
               '未做真实密码/KAT/GPU/生产RSS/跨key验证；低轮结果不冒充6轮进展。'
               '资源检查为协作式；本报告只展示保存结果，不解码模型/数据或重算指标。')
    top_note = ('顶层 summary.metrics/learning_curve/selected 是旧公共空占位，'
                '不作为结果展示；本报告只消费已保存的 arms/paired/diagnostics。')
    diagnostic_note = ('诊断 fixed_flip：按 TRAIN 残差多数选出的硬翻转，BCE 使用与之配套的'
                       '固定单位幅度 logit（概率 sigmoid(+1)≈0.7311 / '
                       'sigmoid(-1)≈0.2689），它是诊断而不是拟合概率；'
                       'residual_probability 使用 TRAIN Jeffreys 概率 '
                       '(ones+0.5)/(rows+1) 对应 logit。二者概率定义不同，均在 TRAIN 冻结，'
                       '不在 DEV/TEST 重新拟合，也不是新增 fit。')
    curve_note = ('曲线只连接已保存的逐 epoch train_bce 与 DEV mean_bce，不重新前向或重算；'
                  '两臂各自的选模机会全部展示。')
    details = {'recipe': saved.get('recipe', {}), '实际计数': saved.get('counts', {}),
               'paired': summary.get('paired', {}),
               'cost/meter': summary.get('meter', {}), 'refs': summary.get('refs', {}),
               'source': saved.get('source', {}), 'policy': saved.get('policy', {}),
               '原因': summary.get('reason', '')}
    paired_lines = ''.join(
        f'<p>{_text(_paired_text(split, _map(_map(summary.get("paired")).get(split))))}</p>'
        for split in _SPLITS)
    heading = f'{title}：{_text(saved.get("run_id", "未知"))}'
    body = (f'<h1>{heading}</h1><p>执行状态：{_text(status)}；'
            f'backend={_text(summary.get("backend_identity", "UNVERIFIED"))}</p>'
            f'<p>{identity}</p><p>{warning}</p>'
            f'<h2>方法、公开参考与TEST时序</h2>{_method_svg()}'
            f'<h2>已保存的绝对bit结果</h2><p>{top_note}</p>'
            + ''.join(f'<p>{row}</p>' for row in _split_rows(summary))
            + '<h2>配对差与精确条件检验</h2>' + paired_lines
            + f'<h2>已保存结果图</h2>{_paired_svg(summary)}'
            f'<h2>全部学习曲线</h2><p>{curve_note}</p>{_curve_svg(summary)}'
            f'<h2>固定翻转与残差概率诊断</h2><p>{diagnostic_note}</p>'
            + ''.join(f'<p>{row}</p>' for row in _diagnostic_rows(summary))
            + '<h2>选模与配对身份</h2>'
            + ''.join(f'<p>{row}</p>' for row in _selection_rows(summary))
            + f'<h2>来源、计数与限制</h2><pre>{_json(details)}</pre>'
            '<p>source.json 绑定已冻结 run SHA；不提供模型/数据下载。</p>')
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
