"""Saved-only Fourier search report; no data/model/label access or new metrics.

Input is the validated mapping from artifacts.read_saved(). Every displayed
number comes from the saved summary/recipe/counts; an unreleased or unvisited
split is labelled as such and is never filled with 0 or 50%. This module has no
filesystem, cipher, or NumPy dependency and never re-fits, re-scores or queries.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from html import escape
from typing import cast

_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SIDES = ("real", "null")
_SPLITS = ("dev", "test")


def _map(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _markdown(value: object) -> str:
    # The same text is safe in both renderers: Markdown punctuation is encoded.
    return "".join(
        f"&#{ord(character)};" if character in "\\`*!|~\n\r" else character
        for character in _text(value)
    )


def _number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(cast("int | float", value))
    return number if math.isfinite(number) else None


def _value(value: object, digits: str = "g") -> str:
    number = _number(value)
    return "未保存（不补 0）" if number is None else format(number, digits)


def _count(value: object) -> str:
    return str(value) if type(value) is int and value >= 0 else "未保存（不补 0）"


def _signed(value: object) -> str:
    # Signed saved sums may legitimately be negative; do not reuse the nonnegative
    # counter formatter or hide them as missing.
    return str(value) if type(value) is int else "未保存（不补 0）"


def _status(value: object) -> str:
    return str(value) if isinstance(value, str) and value else "UNKNOWN"


def _method_svg() -> str:
    flow = ("64 层 paired 采样（每层固定 pairs_per_layer 对，pair 内共享低后缀）→ "
            "逐层 bucket 质量 score>tau²/2 保留 ≤beam_width（超限按 score 降、mask 升截断）→ "
            "候选 mask → 独立 coef 面板求样本均值系数（abs(coef)≥tau/2，intercept=mean）→ "
            "保存 JSON 模型并 owner-read/SHA/严格 codec 加载 → DEV 资源门 → "
            "仅有 real 通过才释放并评分一次 fixed TEST")
    note = ("纯频谱搜索，不训练神经网络、不调用 optimizer、不统计 NN forward；"
            "null 是固定公开伪随机函数的独立标签对照，只描述、不当零分布 p 值。")
    closing = "本图只说明保存的方法接线；所有结果数字来自原始 summary，不重算。"
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 130" '
            'role="img" aria-label="Fourier 搜索方法流程图">'
            '<rect width="1400" height="130" fill="#eef3f8"/>'
            f'<text x="12" y="36" font-size="14">{_text(flow)}</text>'
            f'<text x="12" y="70" font-size="13">{_text(note)}</text>'
            f'<text x="12" y="104" font-size="13">{_text(closing)}</text></svg>')


def _result_svg(summary: Mapping[str, object]) -> str:
    bars: list[str] = []
    for side_index, side in enumerate(_SIDES):
        state = _map(summary.get(side))
        for index, split in enumerate(_SPLITS):
            metric = _map(state.get(split))
            y = 20 + (side_index * 2 + index) * 44
            label = f"{side}/{split.upper()}"
            accuracy = _number(metric.get("accuracy"))
            baseline = _number(metric.get("baseline_accuracy"))
            bars.append(f'<text x="8" y="{y + 15}" font-size="13">{_text(label)}</text>')
            if accuracy is not None and 0 <= accuracy <= 1:
                bars.append(f'<rect x="140" y="{y}" width="{accuracy * 500:.2f}" '
                            'height="18" fill="#276785" opacity="0.85"/>')
                bars.append(f'<text x="652" y="{y + 14}" font-size="13">'
                            f'ACC={accuracy:.6f}</text>')
            else:
                bars.append(f'<text x="140" y="{y + 14}" font-size="13">'
                            '未访问/未保存（不补 0）</text>')
            if baseline is not None and 0 <= baseline <= 1:
                bars.append(f'<line x1="{140 + baseline * 500:.2f}" y1="{y - 2}" '
                            f'x2="{140 + baseline * 500:.2f}" y2="{y + 20}" '
                            'stroke="#a33" stroke-width="2"/>')
                bars.append(f'<text x="140" y="{y + 34}" font-size="11">'
                            f'常量基线 {baseline:.6f}（红竖线）</text>')
    body = "".join(bars)
    if not body:
        body = ('<text x="8" y="30">无已保存 DEV/TEST 指标，不绘制虚假成绩'
                '（未访问不补 0 或半数）</text>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 240" role="img" '
            'aria-label="已保存的 DEV/TEST 猜对比例">'
            '<rect width="1000" height="240" fill="#fbfbfb"/>'
            '<line x1="140" y1="10" x2="140" y2="230" stroke="#888"/>'
            + body + '</svg>')


def _layer_lines(summary: Mapping[str, object], side: str) -> list[str]:
    state = _map(summary.get(side))
    raw = state.get("layers")
    layers = raw if isinstance(raw, list) else []
    if not layers:
        return [f"{side}：无已保存搜索层（search_status="
                f"{_status(summary.get('search_status'))}；空 beam 立即停止后续新查询，"
                "未检出不等于不存在重谱）"]
    tested = sum(value for value in (_integer(item, "tested") for item in layers)
                 if value is not None)
    truncated = sum(value for value in (_integer(item, "truncated") for item in layers)
                    if value is not None)
    lines = [f"{side}：已保存 {len(layers)} 层；tested 合计={tested}；"
             f"truncated 合计={truncated}（beam 截断是覆盖不完整，不是不存在重谱）"]
    for index, raw_layer in enumerate(layers, start=1):
        layer = _map(raw_layer)
        masks = layer.get("masks")
        kept = ",".join(f"0x{int(mask):016x}" for mask in masks
                        if type(mask) is int) if isinstance(masks, (list, tuple)) else ""
        lines.append(
            f"第 {index} 层：depth={_count(layer.get('depth'))}；"
            f"tested={_count(layer.get('tested'))}；"
            f"survivors={_count(layer.get('survivors'))}；"
            f"kept={_count(layer.get('kept'))}；"
            f"truncated={_count(layer.get('truncated'))}；"
            f"score_range=[{_value(layer.get('score_min'))}, "
            f"{_value(layer.get('score_max'))}]；"
            f"sample_bucket_products={_count(layer.get('sample_bucket_products'))}；"
            f"pair_collisions={_count(layer.get('pair_collisions'))}；"
            f"kept_masks={kept or '无'}")
    return lines


def _integer(entry: object, key: str) -> int | None:
    value = _map(entry).get(key)
    return value if type(value) is int and value >= 0 else None


def _model_lines(summary: Mapping[str, object], side: str) -> list[str]:
    state = _map(summary.get(side))
    terms = state.get("terms")
    term_list = terms if isinstance(terms, list) else []
    status = _status(state.get("coef_status", summary.get("coef_status")))
    if not term_list:
        return [f"{side}：coef_status={status}；无已保存模型/非零项，"
                "因此不生成 DEV/TEST，不用 0 或半数填充"]
    lines = [f"{side}：model_ref={_status(state.get('model_ref'))}；"
             f"model_sha256={_status(state.get('model_sha256'))}；"
             f"intercept={_value(state.get('intercept'))}；"
             f"baseline={_count(state.get('baseline'))}；"
             f"count={_count(state.get('count'))}；"
             f"score=intercept+Σcoef·chi，score<0 预测 1"]
    for raw_term in term_list:
        term = _map(raw_term)
        mask = term.get("mask")
        mask_text = f"0x{int(mask):016x}" if type(mask) is int else "未保存"
        lines.append(f"{side} 项：mask={mask_text}；"
                     f"signed_sum={_signed(term.get('signed_sum'))}；"
                     f"coefficient={_value(term.get('coefficient'))}")
    return lines


def _split_lines(summary: Mapping[str, object], side: str) -> list[str]:
    state = _map(summary.get(side))
    lines: list[str] = []
    for split in _SPLITS:
        metric = _map(state.get(split))
        label = f"{side}/{split.upper()}"
        if not metric:
            lines.append(
                f"{label}：未访问/未启动（status="
                f"{_status(summary.get(f'{split}_status'))}；未访问不等于阴性，"
                "不补 0，也不填半数默认值）")
            continue
        lines.append(
            f"{label}：count={_count(metric.get('count'))}；"
            f"correct={_count(metric.get('correct'))}；"
            f"accuracy={_value(metric.get('accuracy'))}；"
            f"baseline={_count(metric.get('baseline'))}；"
            f"baseline_correct={_count(metric.get('baseline_correct'))}；"
            f"baseline_accuracy={_value(metric.get('baseline_accuracy'))}；"
            f"delta={_value(metric.get('delta'))}")
        if split == "test":
            lines.append(
                f"{label} 预注册统计：d_sum={_signed(metric.get('d_sum'))}；"
                f"d_histogram={_json(metric.get('d_histogram', {}))}；"
                f"lower={_value(metric.get('lower'))}；"
                f"alpha={_value(metric.get('alpha'))}；"
                f"positive_exploratory={metric.get('positive_exploratory')}")
    return lines


def _cost_lines(saved: Mapping[str, object]) -> list[str]:
    counts = _map(saved.get("counts"))
    lines: list[str] = []
    for unit in ("real_blocks", "null_words", "optimizer_updates", "forward_rows"):
        lines.append(f"{unit}：attempted={_count(counts.get(f'attempted_{unit}'))}；"
                     f"completed={_count(counts.get(f'completed_{unit}'))}")
    for side in _SIDES:
        for unit in ("sample_bucket_products", "coefficient_rows", "scoring_rows"):
            lines.append(
                f"{side}_{unit}：attempted="
                f"{_count(counts.get(f'{side}_{unit}_attempted'))}；completed="
                f"{_count(counts.get(f'{side}_{unit}_completed'))}")
    lines.append(f"wall_elapsed_seconds={_value(counts.get('wall_elapsed_seconds'))}；"
                 f"cpu_elapsed_seconds={_value(counts.get('cpu_elapsed_seconds'))}；"
                 f"rss_bytes={_count(counts.get('rss_bytes'))}")
    return lines


def _source_line(saved: Mapping[str, object], summary: Mapping[str, object]) -> str:
    claimed = summary.get("source_sha256")
    source = _map(saved.get("source"))
    run_sha = saved.get("run_sha256")
    if not isinstance(claimed, str) or _SHA.fullmatch(claimed) is None:
        return ("保存的 source_sha256 缺失或格式无效：不推断、不补造来源身份，"
                "也不把该运行当成身份已核验。")
    if not source:
        return ("summary 有 source_sha256，但保存记录缺少 source 映射："
                "来源身份未核验，不补造。")
    return (f"summary source_sha256={claimed}；source 映射 {len(source)} 个文件；"
            f"saved run SHA256={_status(run_sha)}。")


def render_spectral_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Return exactly report.md/index.html/source.json; never write or read here."""
    summary = _map(saved.get("summary"))
    recipe = _map(saved.get("recipe"))
    status = saved.get("status", "UNKNOWN")
    if status == "RUNNING":
        status = "UNKNOWN（RUNNING，缺少最终记录）"
    backend = _status(summary.get("backend_identity", "UNVERIFIED"))
    synthetic = summary.get("synthetic") is True or backend == "demo"
    if synthetic:
        identity = ("合成工程演示：人工 oracle/公共字节与 tiny synthetic 配方，"
                    "不是 SIMON，也不是真实密码科学结果或科学验证。")
    elif backend == "simon":
        identity = ("后端身份由装配层声明为 simon；本报告不认证密码实现、KAT "
                    "或科学有效性。")
    else:
        identity = "后端身份未核验：只是工程记录，不得标为真实科学验证成功。"
    scientific = summary.get("scientific_validation")
    scientific_text = ("True" if scientific is True else "False" if scientific is False
                       else "未保存（不补造）")
    warning = ("限制：scientific_validation 只按保存值原样展示，报告不重新裁定；"
               "执行完成不等于发现有效预测。"
               "单 key、一次尝试、固定 beam，未检出不等于不存在重谱；null 仅描述、"
               "不估零分布 p 值；分数是 bucket 质量估计，不是单个系数；"
               "资源检查为协作式。本报告只展示保存值，不解码模型/数据、不重新生成标签、"
               "不前向、不重算指标。")
    tau = _number(summary.get("tau"))
    beam = summary.get("beam_width")
    tau_text = ("未保存" if tau is None else f"{tau:g}")
    threshold = ("未保存" if tau is None else f"{tau * tau / 2:g}")
    search_lines = [f"方法：{_status(summary.get('method'))}；outcome="
                    f"{_status(summary.get('outcome'))}；"
                    f"search_status={_status(summary.get('search_status'))}；"
                    f"coef_status={_status(summary.get('coef_status'))}；"
                    f"dev_status={_status(summary.get('dev_status'))}；"
                    f"test_status={_status(summary.get('test_status'))}；"
                    f"null_status={_status(summary.get('null_status'))}",
                    f"tau={tau_text}；beam_width={_count(beam)}；保留条件 "
                    f"score 严格大于 tau²/2 = {threshold}；"
                    f"sampling={_status(summary.get('sampling'))}；"
                    f"split_exact_intersection="
                    f"{_status(summary.get('split_exact_intersection'))}"]
    layer_lines = [line for side in _SIDES for line in _layer_lines(summary, side)]
    model_lines = [line for side in _SIDES for line in _model_lines(summary, side)]
    split_lines = [line for side in _SIDES for line in _split_lines(summary, side)]
    cost_lines = _cost_lines(saved)
    limits = summary.get("limitations")
    limits_text = ("；".join(str(item) for item in limits)
                   if isinstance(limits, list) else "未保存")
    boundary = ("结论边界：这是固定单 key 的探索性有界频谱搜索，不是跨 key 确认、"
                "全部明文域或论文复现；beam 截断/未检出/未访问都不证明不存在重谱或不可学。"
                f"方法限制标记={limits_text}。")
    reason = summary.get("reason", "")
    interrupted = summary.get("interrupted_phase", "")
    reason_text = _status(reason) if reason else "无"
    interrupted_text = _status(interrupted) if interrupted else "无"
    outcome_line = (f"保存终态：{_status(status)}；公开原因：{reason_text}；"
                    f"中断阶段：{interrupted_text}")
    source_line = _source_line(saved, summary)
    heading = f"六轮目标位 Fourier 搜索报告：{saved.get('run_id', '未知')}"
    paragraphs = [
        f"# {_markdown(heading)}",
        _markdown(f"执行状态：{_status(status)}；backend={backend}。"),
        _markdown(f"保存的 scientific_validation={scientific_text}（原样展示，不重新裁定）。"),
        _markdown(identity),
        _markdown(f"警告：{warning}"),
        _markdown(outcome_line),
        "## 方法与阈值",
        _method_svg(),
        _markdown("；".join(search_lines)),
        f"配方（保存值）：{_markdown(_json(recipe))}",
        "## 已保存搜索层与 beam 截断",
        *(_markdown(line) for line in layer_lines),
        "## 已保存模型项与系数",
        *(_markdown(line) for line in model_lines),
        "## 已保存 DEV/TEST 统计",
        *(_markdown(line) for line in split_lines),
        "## 查询、评分与资源成本",
        *(_markdown(line) for line in cost_lines),
        "## 来源身份与结论边界",
        _markdown(source_line),
        _markdown(boundary),
        "## 已保存结果图",
        _result_svg(summary),
    ]
    html_parts: list[str] = [f"<h1>{_text(heading)}</h1>"]
    for paragraph in paragraphs[1:]:
        if paragraph.startswith("## "):
            html_parts.append(f"<h2>{_text(paragraph[3:])}</h2>")
        elif paragraph.startswith("<svg"):
            html_parts.append(paragraph)
        else:
            html_parts.append(f"<p>{paragraph}</p>")
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>六轮目标位 Fourier 搜索报告</title><style>"
            "body{max-width:1200px;margin:auto;padding:1em;font-family:sans-serif;"
            "line-height:1.7}svg{width:100%;border:1px solid #ddd}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><body>"
            + "".join(html_parts) + "</body></html>")
    source = {"schema": 1, "run_id": saved["run_id"], "run_ref": saved["run_ref"],
              "run_sha256": saved["run_sha256"]}
    return {"report.md": ("\n\n".join(paragraphs) + "\n").encode("utf-8"),
            "index.html": html.encode("utf-8"),
            "source.json": (_json(source) + "\n").encode("utf-8")}
