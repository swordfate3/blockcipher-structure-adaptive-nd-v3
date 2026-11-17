"""Pure saved-only HTML report for the fixed differential-relations experiment.

The input is the mapping returned by ``read_saved``. This module never reads or
writes files, never imports NumPy/Torch/training/data/cipher modules, never
runs a model, creates labels or recomputes scientific metrics. Every displayed
number comes from the saved mapping; missing or undefined values stay explicit.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from typing import cast

FIT_IDS = ('F1', 'F2', 'F3', 'F4', 'F5')
SPLITS = ('train', 'dev', 'test')
_MISSING = object()
_COLORS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf')


def _map(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _seq(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(cast(list[object] | tuple[object, ...], value))
    return []


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _json(value: object) -> str:
    try:
        return escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
                      quote=True)
    except (TypeError, ValueError):
        return _text(value)


def _plain_show(value: object = _MISSING) -> str:
    if value is _MISSING:
        return '未保存'
    if value is None:
        return 'undefined'
    if isinstance(value, (Mapping, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _show(value: object = _MISSING) -> str:
    return _text(_plain_show(value))


def _relation_value(relation: Mapping[str, object], name: str) -> object:
    aliases = {'recall_0': ('recall_0', 'recall0'),
               'recall_1': ('recall_1', 'recall1')}
    for key in aliases.get(name, (name,)):
        if key in relation:
            return relation[key]
    return _MISSING


def _finite(value: object) -> float | None:
    if type(value) in (int, float):
        number = float(cast(float, value))
        if math.isfinite(number):
            return number
    return None


def _fits(summary: Mapping[str, object]) -> list[tuple[str, Mapping[str, object]]]:
    rows: list[tuple[str, Mapping[str, object]]] = []
    seen: set[str] = set()
    for index, item in enumerate(_seq(summary.get('fits'))):
        if not isinstance(item, Mapping):
            continue
        fit = _map(item)
        saved_id = fit.get('fit_id')
        fit_id = saved_id if type(saved_id) is str and saved_id else f'FIT{index + 1}'
        if fit_id in seen:
            continue
        seen.add(fit_id)
        rows.append((fit_id, fit))
    for fit_id in FIT_IDS:
        if fit_id not in seen:
            rows.append((fit_id, {}))
    return rows


def _fit(
    summary: Mapping[str, object], wanted: str,
) -> Mapping[str, object]:
    for fit_id, fit in _fits(summary):
        if fit_id == wanted:
            return fit
    return {}


def _relations(value: object) -> list[Mapping[str, object]]:
    metrics = _map(value)
    raw = metrics.get('relations')
    return [_map(item) for item in _seq(raw) if isinstance(item, Mapping)]


def _relation_list(value: object) -> list[Mapping[str, object]]:
    """Only explicit TEST-shaped saved structures, never dev/train fallback."""
    if isinstance(value, (list, tuple)):
        return [_map(item) for item in _seq(value) if isinstance(item, Mapping)]
    metrics = _map(value)
    direct = _relations(metrics)
    if direct:
        return direct
    return _relations(metrics.get('test'))


def _saved_accuracy_rows(
    summary: Mapping[str, object],
) -> list[Mapping[str, object]] | None:
    raw = summary.get('accuracy_comparisons', _MISSING)
    if raw is _MISSING or raw is None:
        return None
    items: list[object] | None = None
    if isinstance(raw, (list, tuple)):
        items = _seq(raw)
    elif isinstance(raw, Mapping):
        for key in ('items', 'comparisons', 'rows', 'results'):
            value = _map(raw).get(key)
            if isinstance(value, (list, tuple)):
                items = _seq(value)
                break
    if items is None:
        return None
    mapped = [_map(item) for item in items if isinstance(item, Mapping)]
    if items and not mapped:
        return None
    return mapped


def _saved_accuracy_improvement(
    summary: Mapping[str, object], baseline: str, index: int,
) -> object:
    rows = _saved_accuracy_rows(summary)
    if rows is None:
        return _MISSING
    names = {f'R{index + 1}', str(index + 1)}
    for row in rows:
        if (str(row.get('baseline')) == baseline
                and str(row.get('relation')) in names
                and 'accuracy_improvement' in row):
            return row['accuracy_improvement']
    return _MISSING


def _reason_text(
    relation: Mapping[str, object], columns: tuple[str, ...],
) -> str:
    has_undefined = any(_relation_value(relation, name) is None for name in columns)
    raw = relation.get('undefined_reasons', _MISSING)
    if raw is _MISSING or raw is None:
        return _text('原因未保存（指标 undefined）') if has_undefined else '未保存'
    if isinstance(raw, (list, tuple)):
        if not raw:
            if has_undefined:
                return _text('空原因列表（指标 undefined）')
            return '[]'
        if all(isinstance(item, str) for item in raw):
            return _text('、'.join(cast(list[str], raw)))
        return _show(raw)
    return _show(raw)


def _split_label(fit: Mapping[str, object], recipe: Mapping[str, object]) -> str:
    for key in ('split', 'train_split', 'dataset'):
        value = fit.get(key)
        if type(value) is str and value:
            return value
    rows = _finite(fit.get('train_rows'))
    small = _finite(recipe.get('train_small'))
    large = _finite(recipe.get('train_large'))
    if rows is not None and small is not None and rows == small:
        return 'small'
    if rows is not None and large is not None and rows == large:
        return 'large'
    return '未保存'


def _target_label(fit: Mapping[str, object], fit_id: str, split: str) -> str:
    version = fit.get('target_version')
    if version == 'row-permutation-v1':
        # F5 TRAIN/DEV use the permuted target; TEST is frozen/real by protocol.
        if split in ('train', 'dev'):
            return '置乱标签（row-permutation-v1）'
        return '真实标签'
    if version == 'real-v1':
        return '真实标签（real-v1）'
    if type(version) is str and version:
        return version
    if fit_id == 'F5':
        return '置乱标签' if split in ('train', 'dev') else '真实标签'
    return '真实标签'


def _completion_text(summary: Mapping[str, object], status: object) -> str:
    completion = summary.get('completion')
    saved = [fit_id for fit_id, fit in _fits(summary) if fit]
    missing = [fit_id for fit_id in FIT_IDS if fit_id not in saved]
    saved_text = '、'.join(saved) if saved else '无'
    missing_text = '、'.join(f'{fit_id} 未保存' for fit_id in missing)
    if completion == 'COMPLETE':
        text = f'COMPLETE；已保存 fit：{saved_text}。'
        if missing:
            text += f' 但缺少 {missing_text}，不补0、不虚构完成。'
        return text
    if completion == 'PARTIAL':
        return (f'PARTIAL（未完成）；已保存 fit：{saved_text}；'
                f'缺失：{missing_text}。缺少 F3 或 TEST 时没有主结论，不以0填充。')
    if completion == 'ALL_TRAIN_CONSTANT':
        return ('ALL_TRAIN_CONSTANT；TRAIN 四关系全为常量，按设计停止神经拟合，'
                '未生成 TEST，因此没有模型或基线预测的主结果。')
    return (f'completion={_plain_show(completion)}；status={_plain_show(status)}；'
            f'已保存 fit：{saved_text}。仅展示已保存内容，不推断未记录阶段。')


def _comparisons(raw: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    inner = raw.get('comparisons')
    source = _map(inner) if inner is not None else raw
    result: dict[str, Mapping[str, object]] = {}
    for key, value in source.items():
        item = _map(value)
        if key in ('constant', 'stump') or any(
                name in item for name in ('mean_difference', 'baseline_bce', 'ci_low')):
            result[str(key)] = item
    return result


def _main_bce_text(summary: Mapping[str, object]) -> str:
    completion = summary.get('completion')
    prefix = ('' if completion == 'COMPLETE'
              else f'未完成（completion={_plain_show(completion)}）：')
    raw_value = summary.get('main_comparisons', _MISSING)
    if raw_value is _MISSING or raw_value is None:
        return (f'{prefix}未保存 main_comparisons，不能判断主BCE门槛；'
                '不重算、不以0或推测补全。')
    if not isinstance(raw_value, Mapping):
        return (f'{prefix}main_comparisons 结构未识别，不能判断主BCE门槛；'
                f'保存内容={_plain_show(raw_value)}。')
    raw = _map(raw_value)
    signal = raw.get('main_signal')
    complete = completion == 'COMPLETE'
    f3 = _fit(summary, 'F3')
    f3_test = _relations(_map(_map(f3.get('final_metrics')).get('test'))) if f3 else []
    pieces = [f'model_bce={_plain_show(raw.get("model_bce", _MISSING))}']
    for name, item in _comparisons(raw).items():
        pieces.append(
            f'{name}: mean_difference={_plain_show(item.get("mean_difference"))}, '
            f'ci_low={_plain_show(item.get("ci_low"))}, '
            f'ci_high={_plain_show(item.get("ci_high"))}, '
            f'relative_improvement={_plain_show(item.get("relative_improvement"))}, '
            f'uncertain={_plain_show(item.get("uncertain", _MISSING))}')
    detail = '；'.join(pieces)
    if signal is True and complete and f3 and f3_test:
        verdict = ('main_signal=True（保存值）；保存记录显示达到其保存的信号门槛，'
                   '但本报告不重算CI/相对改善，也不外推为跨key或密码学结论。')
    elif signal is True:
        blockers: list[str] = []
        if not complete:
            blockers.append('周期未完成')
        if not f3:
            blockers.append('F3 未保存')
        elif not f3_test:
            blockers.append('F3 TEST 逐关系指标未保存')
        verdict = ('main_signal=True（保存值），但' + '、'.join(blockers)
                   + '；不能展示为主signal成功，也不能作为主结论。')
    elif signal is False:
        verdict = 'main_signal=False（保存值）；保存记录未显示达到信号门槛，不判为通过。'
    else:
        verdict = 'main_signal=未保存，不能判断主BCE门槛。'
    return f'{prefix}{verdict} 保存字段：{detail}。'


def _baseline_lookup(
    summary: Mapping[str, object], split: str, kind: str,
) -> object:
    group = _map(_map(summary.get('baselines')).get(split))
    if kind in group:
        return group[kind]
    test = _map(group.get('test'))
    if kind in test:
        return test[kind]
    return _MISSING


def _baseline_relations(
    summary: Mapping[str, object], split: str, kind: str,
) -> list[Mapping[str, object]] | None:
    value = _baseline_lookup(summary, split, kind)
    if value is _MISSING:
        return None
    relations = _relation_list(value)
    return relations or None


def _relation_diff_text(summary: Mapping[str, object]) -> str:
    f3 = _fit(summary, 'F3')
    if not f3:
        return 'F3 未保存，无法报告逐关系与 large 基线的差值；不补0。'
    test = _map(_map(f3.get('final_metrics')).get('test'))
    model = _relations(test)
    if not model:
        return 'F3 TEST 逐关系指标未保存，无法报告与 large 基线的差值；不补0。'
    pieces: list[str] = []
    for kind in ('constant', 'stump'):
        baseline = _baseline_relations(summary, 'large', kind)
        if not baseline:
            pieces.append(f'large/{kind} 基线逐关系指标未保存，差值未保存')
            continue
        lines: list[str] = []
        for index in range(4):
            relation = model[index] if index < len(model) else {}
            base = baseline[index] if index < len(baseline) else {}
            saved_improvement = _saved_accuracy_improvement(summary, kind, index)
            lines.append(
                f'R{index + 1}：模型ACC={_plain_show(relation.get("accuracy"))}、'
                f'基线ACC={_plain_show(base.get("accuracy"))}、'
                f'ACC差(saved)={_plain_show(saved_improvement)}、'
                f'模型BCE={_plain_show(relation.get("bce"))}、'
                f'基线BCE={_plain_show(base.get("bce"))}、BCE差=未保存')
        pieces.append(f'相对 {kind}：' + '；'.join(lines))
    if not pieces:
        return '未保存 large 基线逐关系指标，差值未保存；不以0填充。'
    return ('F3 TEST（saved）并列保存的原始指标；ACC差只读保存的 '
            'accuracy_improvement，BCE 差无保存 comparison 时标未保存：'
            + '；'.join(pieces) + '。')


def _scale_text(summary: Mapping[str, object]) -> str:
    labels = {'F1': 'F1 small', 'F2': 'F2 small', 'F3': 'F3 large（主候选）',
              'F4': 'F4 large/init B', 'F5': 'F5 large/null'}
    parts: list[str] = []
    for fit_id in FIT_IDS:
        fit = _fit(summary, fit_id)
        if not fit:
            parts.append(f'{labels[fit_id]}：未保存')
            continue
        final = _map(fit.get('final_metrics'))
        dev = _map(final.get('dev')).get('mean_bce', _MISSING)
        test = _map(final.get('test')).get('mean_bce', _MISSING)
        parts.append(f'{labels[fit_id]}：DEV={_plain_show(dev)}、'
                     f'TEST={_plain_show(test)}')
    return ('仅并排保存的 DEV/TEST mean_bce：' + '；'.join(parts)
            + '。不据此重算、检验或宣称显著的数据规模效应。')


def _method_svg(recipe: Mapping[str, object]) -> str:
    cipher = _text(recipe.get('cipher', '未保存'))
    rounds = _text(recipe.get('rounds', '未保存'))
    alpha = _text(recipe.get('alpha', '未保存'))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 205" '
        'role="img" aria-label="方法信息边界示意图">'
        '<rect x="1" y="1" width="978" height="203" fill="#eef3f8" stroke="#9fb3c8"/>'
        '<text x="18" y="28" font-size="16" font-weight="bold">'
        '标签生成侧（密文不进入模型）</text>'
        f'<text x="18" y="54" font-size="14">C = E_K(P)，C′ = E_K(P XOR α)，'
        f'ΔC = C XOR C′；{cipher} rounds={rounds}，α={alpha}</text>'
        '<text x="18" y="78" font-size="14">'
        '四掩码 popcount(β ∩ ΔC) mod 2 → R1..R4 布尔标签；真实 key 只留在标签侧。</text>'
        '<text x="18" y="122" font-size="16" font-weight="bold">模型可见侧</text>'
        '<text x="18" y="148" font-size="14">'
        '明文对 P、P XOR α（公开输入）→ shared-M 明文对网络 → 4 个关系概率 p(R1..R4)</text>'
        '<text x="18" y="174" font-size="14">'
        '密文、ΔC、key、逐样本标签均不作为模型输入；本图只描述已保存方法边界。</text>'
        '</svg>'
    )


def _curve_svg(summary: Mapping[str, object]) -> str:
    series: list[tuple[str, list[tuple[float, float]]]] = []
    for fit_id, fit in _fits(summary):
        points: list[tuple[float, float]] = []
        for item in _seq(fit.get('dev_curve')):
            point = _map(item)
            step = _finite(point.get('step'))
            bce = _finite(point.get('bce'))
            if step is not None and bce is not None:
                points.append((step, bce))
        series.append((fit_id, points))
    valid = [(fit_id, points) for fit_id, points in series if points]
    left, top, width, height = 62.0, 24.0, 770.0, 300.0
    if not valid:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 380" '
            'role="img" aria-label="五fit DEV BCE曲线">'
            '<rect x="1" y="1" width="898" height="378" fill="#fbfcfd" stroke="#9fb3c8"/>'
            '<text x="24" y="44" font-size="16">无保存 DEV 曲线数据；不伪造曲线或补点。</text>'
            '</svg>'
        )
    xs = [x for _, points in valid for x, _ in points]
    ys = [y for _, points in valid for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    if ymin == ymax:
        ymin, ymax = ymin - 0.05, ymax + 0.05
    xpad = (xmax - xmin) * 0.04
    ypad = (ymax - ymin) * 0.08
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * width

    def sy(value: float) -> float:
        return top + height - (value - ymin) / (ymax - ymin) * height

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 380" '
        'role="img" aria-label="五fit DEV BCE曲线">',
        '<rect x="1" y="1" width="898" height="378" fill="#fbfcfd" stroke="#9fb3c8"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" '
        f'y2="{top + height}" stroke="#445566"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#445566"/>',
        f'<text x="{left + width / 2}" y="365" font-size="13" text-anchor="middle">'
        'step（saved dev_curve）</text>',
        f'<text x="16" y="{top + 8}" font-size="13">DEV BCE</text>',
        f'<text data-axis="ymax" x="{left - 6}" y="{top + 4}" font-size="11" '
        f'text-anchor="end">{_text(f"{ymax:.4g}")}</text>',
        f'<text data-axis="ymin" x="{left - 6}" y="{top + height + 4}" font-size="11" '
        f'text-anchor="end">{_text(f"{ymin:.4g}")}</text>',
        f'<text x="{left}" y="{top + height + 17}" font-size="11">'
        f'{_text(f"{xmin:.4g}")}</text>',
        f'<text x="{left + width}" y="{top + height + 17}" font-size="11" '
        f'text-anchor="end">{_text(f"{xmax:.4g}")}</text>',
    ]
    for index, (fit_id, points) in enumerate(series):
        color = _COLORS[index % len(_COLORS)]
        coords = ' '.join(f'{sx(x):.2f},{sy(y):.2f}' for x, y in points)
        parts.append(f'<g data-fit="{_text(fit_id)}" data-points="{len(points)}">')
        if len(points) >= 2:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                         f'points="{coords}"/>')
        if points:
            for x, y in points:
                parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.5" '
                             f'fill="{color}"><title>step={x:.6g}, '
                             f'dev_bce={y:.6g}</title></circle>')
        else:
            parts.append('<text x="80" y="60" font-size="13">无保存点</text>')
        legend_y = 22 + index * 18
        parts.append(f'<rect x="630" y="{legend_y - 10}" width="10" height="10" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="646" y="{legend_y}" font-size="12">'
                     f'{_text(fit_id)} ({len(points)} 点)</text>')
        parts.append('</g>')
    parts.append('</svg>')
    return ''.join(parts)


def _comparison_svg(
    summary: Mapping[str, object],
) -> tuple[str, str]:
    f3 = _fit(summary, 'F3')
    model = _relations(_map(_map(f3.get('final_metrics')).get('test')))
    if not model or not any(_finite(item.get('accuracy')) is not None for item in model):
        return '', ('F3 TEST 逐关系指标未保存，不能判断逐关系对比；'
                    '不绘制图形、不以0代替模型指标。')
    baselines: list[tuple[str, list[Mapping[str, object]], str]] = []
    for kind, color in (('constant', '#777777'), ('stump', '#e08a1e')):
        relations = _baseline_relations(summary, 'large', kind)
        if relations and any(_finite(item.get('accuracy')) is not None
                             for item in relations):
            baselines.append((kind, relations, color))
    if not baselines:
        return '', ('基线逐关系 accuracy 未保存：未完成对比，不能判断逐关系基线差；'
                    '不绘制图形、不以0代替基线。')
    left, top, width, height = 64.0, 34.0, 700.0, 250.0
    group_width = width / 4
    all_series = [('F3', model, '#1f77b4')] + baselines
    bar_width = min(34.0, (group_width - 14.0) / len(all_series))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 340" '
        'role="img" aria-label="四关系TEST准确率与基线对比">',
        '<rect x="1" y="1" width="898" height="338" fill="#fbfcfd" stroke="#9fb3c8"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" '
        f'y2="{top + height}" stroke="#445566"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#445566"/>',
        f'<text x="16" y="{top + 8}" font-size="12">TEST ACC</text>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + height - tick * height
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" '
                     'stroke="#dde3ea"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.2f}" font-size="11" '
                     f'text-anchor="end">{tick:.2f}</text>')
    for index in range(4):
        center = left + group_width * (index + 0.5)
        parts.append(f'<text x="{center:.2f}" y="{top + height + 22}" font-size="13" '
                     f'text-anchor="middle">R{index + 1}</text>')
        start = center - (bar_width * len(all_series)) / 2
        for series_index, (label, relations, color) in enumerate(all_series):
            relation = relations[index] if index < len(relations) else {}
            accuracy = _finite(relation.get('accuracy'))
            if accuracy is None or not 0.0 <= accuracy <= 1.0:
                continue
            x = start + series_index * bar_width
            y = top + height - accuracy * height
            parts.append(f'<rect data-series="{_text(label)}" '
                         f'data-relation="R{index + 1}" x="{x:.2f}" y="{y:.2f}" '
                         f'width="{bar_width - 3:.2f}" height="{accuracy * height:.2f}" '
                         f'fill="{color}"><title>{_text(label)} R{index + 1} '
                         f'ACC={accuracy:.6g}</title></rect>')
    legend_x = left
    for label, _series, color in all_series:
        parts.append(f'<rect x="{legend_x}" y="{top + height + 32}" width="11" '
                     f'height="11" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 16}" y="{top + height + 42}" '
                     f'font-size="12">{_text(label)}</text>')
        legend_x += 120
    parts.append('</svg>')
    note = ('仅绘制明确保存的 accuracy；模型为 F3，基线为 large。'
            '未保存的柱不绘制，不以0代替；本图不重新计算准确率或显著性。')
    return ''.join(parts), note


def _relation_table(
    relations: list[Mapping[str, object]], *, target: str | None = None,
) -> str:
    columns = ('rows', 'ones', 'bce', 'accuracy', 'balanced_accuracy',
               'recall_0', 'recall_1', 'mcc')
    head = '<tr><th>关系</th>' + ('<th>目标</th>' if target is not None else '')
    head += ''.join(f'<th>{_text(name)}</th>' for name in columns)
    head += '<th>undefined_reasons</th></tr>'
    rows: list[str] = []
    for index in range(4):
        relation = relations[index] if index < len(relations) else {}
        cells = ''.join(f'<td>{_show(_relation_value(relation, name))}</td>'
                        for name in columns)
        reason = _reason_text(relation, columns)
        rows.append(f'<tr><td>R{index + 1}</td>'
                    + (f'<td>{_text(target)}</td>' if target is not None else '')
                    + cells + f'<td>{reason}</td></tr>')
    return ('<table><thead>' + head + '</thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def _summary_table(summary: Mapping[str, object], recipe: Mapping[str, object]) -> str:
    header = ('<tr><th>fit</th><th>数据</th><th>train_rows</th><th>lr</th>'
              '<th>init_seed</th><th>updates</th><th>selected_step</th>'
              '<th>DEV mean_bce</th><th>TEST mean_bce</th><th>TEST 逐关系 ACC</th>'
              '<th>TRAIN/DEV 目标</th></tr>')
    rows: list[str] = []
    for fit_id, fit in _fits(summary):
        final = _map(fit.get('final_metrics'))
        dev = _map(final.get('dev'))
        test = _map(final.get('test'))
        test_relations = _relations(test)
        acc_text = ('、'.join(f'R{index + 1}={_show(item.get("accuracy"))}'
                             for index, item in enumerate(test_relations))
                    if test_relations else '未保存')
        split = _split_label(fit, recipe) if fit else '未保存'
        target = _target_label(fit, fit_id, 'train') if fit else '未保存'
        css = (' class="main-candidate" style="font-weight:bold;background:#fff3cd"'
               if fit_id == 'F3' else '')
        rows.append(
            f'<tr{css} data-fit="{_text(fit_id)}"><td>{_text(fit_id)}</td>'
            f'<td>{_text(split)}</td><td>{_show(fit.get("train_rows"))}</td>'
            f'<td>{_show(fit.get("learning_rate"))}</td>'
            f'<td>{_show(fit.get("init_seed"))}</td>'
            f'<td>{_show(fit.get("updates"))}</td>'
            f'<td>{_show(fit.get("selected_step"))}</td>'
            f'<td>{_show(dev.get("mean_bce", _MISSING))}</td>'
            f'<td>{_show(test.get("mean_bce", _MISSING))}</td>'
            f'<td>{acc_text}</td><td>{_text(target)}</td></tr>')
    return ('<table><thead>' + header + '</thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def _per_fit_relations(summary: Mapping[str, object]) -> str:
    parts: list[str] = []
    for fit_id, fit in _fits(summary):
        if not fit:
            parts.append(f'<h3>{_text(fit_id)}：未保存</h3>')
            continue
        final = _map(fit.get('final_metrics'))
        for split in SPLITS:
            relations = _relations(final.get(split))
            target = _target_label(fit, fit_id, split)
            parts.append(f'<h3>{_text(fit_id)}/{_text(split.upper())}'
                         f'（目标：{_text(target)}）</h3>')
            if relations:
                parts.append(_relation_table(relations, target=target))
            else:
                parts.append('<p>该划分逐关系指标未保存；不补0。</p>')
    return ''.join(parts)


def _baseline_section(summary: Mapping[str, object]) -> str:
    raw = summary.get('baselines', _MISSING)
    if raw is _MISSING or raw is None:
        return '<p>TRAIN-only 基线未保存；不绘制或填充基线结果。</p>'
    if not isinstance(raw, Mapping):
        return ('<p>TRAIN-only 基线结构未识别；不推断其科学含义。</p>'
                f'<pre>{_json(raw)}</pre>')
    baselines = _map(raw)
    parts: list[str] = []
    for split in ('small', 'large'):
        if split not in baselines:
            parts.append(f'<h3>{split} 基线</h3><p>未保存。</p>')
            continue
        group = _map(baselines.get(split))
        parts.append(f'<h3>{split} 基线（train_rows={_show(group.get("train_rows"))}）</h3>')
        for kind in ('constant', 'stump'):
            relations = _baseline_relations(summary, split, kind)
            if relations:
                parts.append(f'<h4>{kind}</h4>' + _relation_table(relations))
            else:
                value = _baseline_lookup(summary, split, kind)
                shown = '未保存' if value is _MISSING else _json(value)
                parts.append(f'<h4>{kind}</h4><p>逐关系结构未保存；保存内容：</p>'
                             f'<pre>{shown}</pre>')
        for key, value in group.items():
            if key not in ('train_rows', 'constant', 'stump', 'test'):
                parts.append(f'<h4>其他保存字段：{_text(key)}</h4><pre>{_json(value)}</pre>')
    for key, value in baselines.items():
        if key not in ('small', 'large'):
            parts.append(f'<h4>其他保存基线：{_text(key)}</h4><pre>{_json(value)}</pre>')
    return ''.join(parts)


def _main_comparison_section(summary: Mapping[str, object]) -> str:
    raw = summary.get('main_comparisons', _MISSING)
    if raw is _MISSING or raw is None:
        return '<p>主比较与CI未保存；不重算、不补字段。</p>'
    if not isinstance(raw, Mapping):
        return ('<p>主比较与CI结构未识别；不推断其科学含义，原样转义展示。</p>'
                f'<pre>{_json(raw)}</pre>')
    value = _map(raw)
    known = ('model_bce', 'comparisons', 'main_signal')
    columns = ('baseline_bce', 'mean_difference', 'relative_improvement',
               'ci_low', 'ci_high', 'confidence', 'sample_sd',
               'max_abs_difference', 'first_half_mean', 'second_half_mean',
               'max_centered_square_fraction', 'uncertain', 'reasons')
    head = '<tr><th>比较</th>' + ''.join(f'<th>{_text(name)}</th>' for name in columns) + '</tr>'
    rows: list[str] = []
    for name, item in _comparisons(value).items():
        rows.append(f'<tr><td>{_text(name)}</td>'
                    + ''.join(f'<td>{_show(item.get(key))}</td>' for key in columns)
                    + '</tr>')
    table = ('<table><thead>' + head + '</thead><tbody>' + ''.join(rows) + '</tbody></table>')
    scalar = ''.join(f'<li>{_text(key)} = {_show(value.get(key))}</li>'
                     for key in known if key != 'comparisons' and key in value)
    extra = {key: item for key, item in value.items() if key not in known}
    parts = [f'<ul>{scalar}</ul>', table]
    if extra:
        parts.append(f'<p>其他保存主比较字段（原样转义）：</p><pre>{_json(extra)}</pre>')
    parts.append(f'<p>完整保存结构：</p><pre>{_json(value)}</pre>')
    return ''.join(parts)


def _accuracy_comparison_section(summary: Mapping[str, object]) -> str:
    raw = summary.get('accuracy_comparisons', _MISSING)
    if raw is _MISSING or raw is None:
        return '<p>逐关系准确率次级比较未保存；不计算 p、不补 8 项。</p>'
    rows = _saved_accuracy_rows(summary)
    if rows is None:
        return ('<p>逐关系准确率次级比较结构未识别；不推断其科学含义，'
                '原样转义展示。</p>'
                f'<pre>{_json(raw)}</pre>')
    columns = ('baseline', 'relation', 'rows', 'model_only_correct',
               'baseline_only_correct', 'accuracy_improvement', 'p',
               'adjusted_p', 'alpha', 'accuracy_signal')
    head = '<tr>' + ''.join(f'<th>{_text(name)}</th>' for name in columns) + '</tr>'
    body: list[str] = []
    for row in rows:
        body.append('<tr>' + ''.join(
            f'<td>{_show(row.get(name))}</td>' for name in columns) + '</tr>')
    count_note = (f'保存条目数={len(rows)}（预期固定8项，不补不删）；'
                  'p、adjusted_p、accuracy_signal 均为保存值。')
    return ('<p>' + _text(count_note) + '本表为独立次级结果，'
            '不与主BCE表混同；报告不计算 p 或 Holm 校正。</p>'
            '<table><thead>' + head + '</thead><tbody>'
            + ''.join(body) + '</tbody></table>')


def _recipe_section(recipe: Mapping[str, object]) -> str:
    if not recipe:
        return '<p>recipe 未保存。</p>'
    rows: list[str] = []
    extras: list[tuple[str, object]] = []
    for key, value in recipe.items():
        if isinstance(value, (Mapping, list, tuple)):
            extras.append((str(key), value))
        else:
            rows.append(f'<tr><td>{_text(key)}</td><td>{_show(value)}</td></tr>')
    table = ('<table><thead><tr><th>recipe 字段</th><th>保存值</th></tr></thead>'
             '<tbody>' + ''.join(rows) + '</tbody></table>')
    parts = [table]
    for key, value in extras:
        label = '插件 source（saved）' if key == 'plugin_source' else key
        parts.append(f'<h4>{_text(label)}</h4><pre>{_json(value)}</pre>')
    return ''.join(parts)


def _unknown_summary(summary: Mapping[str, object]) -> str:
    known = {'method', 'completion', 'reason', 'fits', 'baselines', 'main_comparisons',
             'accuracy_comparisons'}
    extra = {key: value for key, value in summary.items() if key not in known}
    if not extra:
        return ''
    return ('<h2>其他 summary 保存结构</h2><p>以下字段按原样转义展示，'
            '报告不对其做科学解释。</p><pre>' + _json(extra) + '</pre>')


def render_report(saved: Mapping[str, object]) -> bytes:
    """Render one self-contained UTF-8 HTML report from saved values only.

    The function is pure: no filesystem, model, label or metric access. Missing
    fields stay visibly missing/undefined; no zeros or new scientific results
    are synthesized. All dynamic text is HTML-escaped.
    """
    summary = _map(saved.get('summary'))
    recipe = _map(saved.get('recipe'))
    run_id = saved.get('run_id', _MISSING)
    status = saved.get('status', _MISSING)
    method = summary.get('method', recipe.get('method', _MISSING))
    completion = summary.get('completion')
    reason = summary.get('reason', _MISSING)
    counts = saved.get('counts')
    run_ref = saved.get('run_ref', _MISSING)
    run_sha = saved.get('run_sha256', _MISSING)

    f5_note = ('F5：TRAIN/DEV 目标为整行置乱标签，TEST 目标为真实标签；'
               '两者目标不同，数值差距不是泛化差距，也不把置乱 DEV 与真实 DEV 作同目标排名。')
    curve_svg = _curve_svg(summary)
    comparison_svg, comparison_note = _comparison_svg(summary)
    title = f'多位差分关系 saved-only 报告：{_text(run_id)}'
    body: list[str] = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{title}</title></head><body>',
        '<header>',
        f'<h1>{title}</h1>',
        f'<p>run_id={_show(run_id)}；status={_show(status)}；'
        f'method={_show(method)}；completion={_show(completion)}。</p>',
        f'<p><strong>执行完成范围：</strong>{_text(_completion_text(summary, status))}</p>',
        f'<p><strong>主BCE判断：</strong>{_text(_main_bce_text(summary))}</p>',
        f'<p><strong>逐关系基线差：</strong>{_text(_relation_diff_text(summary))}</p>',
        f'<p><strong>规模/初始化/置乱对照（仅saved）：</strong>'
        f'{_text(_scale_text(summary))}</p>',
        '<p><strong>单key限制：</strong>整次实验固定一个未知私有key；'
        '结论只描述该key与本次固定面板，不是跨key平均、独立复现或密码学证明；'
        '两个初始化seed不等于两个独立密钥。</p>',
        '</header>',
        '<main>',
        '<p>本报告只读 run.json 保存值；不读取数据/模型、不重新计算科学指标、'
        '不生成标签、不执行模型前向。</p>',
        '<h2>方法与信息边界</h2>',
        _method_svg(recipe),
        '<p>模型只接收明文对；密文只用于标签侧。四个掩码是公开任务参数，'
        '但逐样本标签不是模型输入。F3 为预定主候选；F4 只作初始化稳健性描述，'
        'F1/F2/F5 不替换 F3。</p>',
        f'<p>{_text(f5_note)}</p>',
        '<h2>五fit DEV BCE 曲线（saved dev_curve）</h2>',
        curve_svg,
        '<p>曲线只连接保存的 step/bce 点；空曲线、缺失 fit 或非有限点不补点、不插值。</p>',
        '<h2>四关系 TEST ACC 与 large 基线对比</h2>',
        comparison_svg,
        f'<p>{_text(comparison_note)}</p>',
        '<h2>五fit汇总（F3 主候选突出）</h2>',
        _summary_table(summary, recipe),
        f'<p>{_text(f5_note)}</p>',
        '<h2>逐fit逐关系指标（saved）</h2>',
        _per_fit_relations(summary),
        '<h2>TRAIN-only 基线（saved）</h2>',
        _baseline_section(summary),
        '<h2>主比较与CI（saved，不重算）</h2>',
        _main_comparison_section(summary),
        '<h2>逐关系准确率次级比较（McNemar/Holm, saved）</h2>',
        _accuracy_comparison_section(summary),
        '<h2>参数、计数与来源（saved）</h2>',
        '<h3>recipe</h3>',
        _recipe_section(recipe),
        '<h3>counts</h3>',
        f'<pre>{_show(counts)}</pre>',
        f'<p>reason：{_show(reason)}</p>',
        f'<p>run_ref：{_show(run_ref)}；run_sha256：{_show(run_sha)}</p>',
        _unknown_summary(summary),
        '</main>',
        '<footer><p>限制：本报告不认证密码实现、单key泛化、显著性或跨key复现；'
        '缺少字段保持未保存/undefined，不当作0或成功。</p></footer>',
        '</body></html>',
    ]
    return '\n'.join(body).encode('utf-8')
