"""Pure saved-result rendering. No I/O, clocks, oracle, models or metric math.

Input is the validated mapping from artifacts.read_saved(), not a live store.
Only the saved summary supplies dimensions/outcomes/counts/times. Payloads supply
at most four display representatives; their length is never used as a metric.
The original finalized owner writes the returned files under its single budget.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _markdown(value: object) -> str:
    # Escape Markdown punctuation as entities as well as HTML: user text is text,
    # never an injected link/image/code block when a Markdown viewer renders it.
    return "".join(f"&#{ord(character)};" if character in "\\`*_{}[]()!#+-.|~\n\r"
                   else character for character in _text(value))


def _dimension(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 2080 else None


def render_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Return report.md, index.html, source.json; never mutate the saved mapping.

    Source SHA is the *saved run.json* SHA provided by read_saved. It is not a
    fresh source-code hash, a recomputation or a link exposing private data.
    SUCCEEDED describes execution only. Backend identity is explicit caller
    metadata; absent/unknown identity stays unverified, never assumed real.
    """
    summary = _mapping(saved.get("summary"))
    backend = summary.get("backend_identity", "UNVERIFIED")
    if backend in ("demo", "engineering"):
        identity = "合成工程演示：人工数据，不是真实密码科学结果。"
    elif backend == "simon":
        identity = "真实后端身份由装配层记录；本报告不认证密码实现或科学有效性。"
    else:
        identity = "后端身份未核验：仅工程执行记录，不得标为真实科学验证成功。"
    status = saved.get("status", "UNKNOWN")
    if status == "RUNNING":
        status = "UNKNOWN（RUNNING，缺少科学终态）"
    meter = _mapping(summary.get("meter"))
    warning = ("这是经验关系检验，不是证明，也不提供显著性或 p 值。小样本必有经验核；"
               "确认失败判整个冻结空间失败，不挑选幸存者。空空间为 NOT_APPLICABLE。"
               "无真实密码工程自检只验证接线，不能代替未来科学正负控制。")
    timing = (f"耗时：墙钟 {meter.get('wall_elapsed_seconds', '未保存')} 秒；"
              f"CPU {meter.get('cpu_elapsed_seconds', '未保存')} 秒。"
              "限制：资源检查为协作式而非硬隔离；保存值不代表跨 key 证明。")
    counts = _json(saved.get("counts", {}))
    settings = _json({"recipe": saved.get("recipe", {}), "policy": saved.get("policy", {})})
    reason = summary.get("reason", "无公开终止原因")
    paragraphs = [f"# 积分经验运行报告：{_markdown(saved.get('run_id', '未知'))}",
                  f"结论：执行状态 {_markdown(status)}。{identity}", warning,
                  "方法：D 发现 → S 筛选 → 完整规范基冻结 → C 核验全部；"
                  "随后独立 null D/S/C。非空 null 全通过即 ALARM，停止新增 cell。",
                  f"数据量（保存的实际计数，不重新采样或重算指标）：\n<pre>{_text(counts)}</pre>",
                  _markdown(timing), f"公开原因：{_markdown(reason)}",
                  f"设置与预算：\n<pre>{_text(settings)}</pre>"]
    method_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 90" '
        'role="img" aria-label="固定经验流程图">'
        '<rect x="1" y="1" width="758" height="88" fill="#eef3f8"/>'
        '<text x="15" y="32" font-size="17">real：D → K_D → S → K_DS 完整基冻结 → C 全核验</text>'
        '<text x="15" y="66" font-size="17">'
        'null：独立 D/S/冻结/C；非空全过 → ALARM → 停止新 cell</text></svg>')
    chart: list[str] = []
    cell_html: list[str] = []
    payloads = _mapping(saved.get("payloads"))
    representative_count = 0
    raw_cells = summary.get("cells", [])
    cell_values = raw_cells if isinstance(raw_cells, (list, tuple)) else ()
    for index, raw_cell in enumerate(cell_values):
        cell = _mapping(raw_cell)
        heading = (f"Cell {index + 1}：{_json(cell.get('cell', {}))}；"
                   f"结论 {cell.get('outcome', 'INCOMPLETE')}")
        paragraphs.append(f"## {_markdown(heading)}")
        cell_html.append(f"<h2>{_text(heading)}</h2>")
        for side in ("real", "null"):
            state = _mapping(cell.get(side))
            d = _dimension(state.get("discovery_dimension"))
            ds = _dimension(state.get("frozen_dimension"))
            confirmation = state.get("confirmation", "PENDING")
            line = (f"{side}：发现维数 {d if d is not None else '未保存'}；"
                    f"冻结维数 {ds if ds is not None else '未保存'}；确认 {confirmation}。"
                    "完整基保留在原始载荷，报告全局至多展示 4 个代表。")
            paragraphs.append(_markdown(line))
            cell_html.append(f"<p>{_text(line)}</p>")
            for label, value in (("D", d), ("DS", ds)):
                if value is not None:
                    y = 30 + len(chart) * 28
                    # Geometric rendering only: dimension comes solely from saved summary.
                    chart.append(f'<g data-dimension="{value}"><text x="8" y="{y}">'
                                 f'{index + 1}/{side}/{label}: {value}</text>'
                                 f'<rect x="210" y="{y - 15}" width="{value / 4:.2f}" '
                                 'height="16" fill="#276785"/></g>')
            payload = _mapping(payloads.get(str(state.get("basis_ref", ""))))
            basis = payload.get("frozen_basis", ())
            if isinstance(basis, (tuple, list)):
                for vector in basis[:max(0, 4 - representative_count)]:
                    if type(vector) is int and 0 <= vector < (1 << 2080):
                        representative_count += 1
                        shown = f"代表 {representative_count}：{hex(vector)}"
                        paragraphs.append(shown)
                        cell_html.append(f'<p class="representative">{shown}</p>')
    result_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 760 {max(65, 35 + len(chart) * 28)}" '
        'role="img" aria-label="已保存的经验空间维数（非科学成绩）">'
        + ("".join(chart) or '<text x="8" y="30">尚无已保存维数，不绘制虚假成绩</text>')
        + '</svg>')
    paragraphs.extend(["## 方法流程图", method_svg, "## 已保存结果图（经验维数）", result_svg])
    html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>积分经验运行报告</title><style>'
        'body{max-width:1000px;margin:2em auto;padding:1em;font-family:sans-serif;line-height:1.7}'
        'svg{width:100%;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        '.representative{overflow-wrap:anywhere}.warning{background:#fff1cf;padding:1em}'
        '</style><body>'
        f'<h1>积分经验运行报告：{_text(saved.get("run_id", "未知"))}</h1>'
        f'<p>结论：执行状态 {_text(status)}。{identity}</p>'
        f'<p class="warning">{warning}</p><h2>方法与流程</h2>{method_svg}'
        f'<h2>数据量</h2><pre>{_text(counts)}</pre>'
        f'<p>{_text(timing)}</p><p>公开原因：{_text(reason)}</p>'
        f'<details><summary>设置与预算</summary><pre>{_text(settings)}</pre></details>'
        f'<h2>已保存结果图（经验维数）</h2>{result_svg}' + ''.join(cell_html)
        + '<p>来源：source.json 记录冻结 run SHA；本页不提供原始数据或模型下载。</p></body></html>')
    source = {"schema": 1, "run_id": saved["run_id"], "run_ref": saved["run_ref"],
              "run_sha256": saved["run_sha256"]}
    return {"report.md": ("\n\n".join(paragraphs) + "\n").encode("utf-8"),
            "index.html": html.encode("utf-8"),
            "source.json": (_json(source) + "\n").encode("utf-8")}
