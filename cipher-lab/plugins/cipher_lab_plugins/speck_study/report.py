"""Saved-only study report: stdlib mapping -> exact three files, no I/O/science."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any

LIMITS = (
    "single-key 单密钥探索，不外推跨key；有限样本与噪声不证明不可预测。"
    "采样有放回、独立域不保证unique/unseen；覆盖/交集 UNKNOWN。"
    "主score<0输出1，tie0；官方tie1仅固定描述，不按结果翻极性。"
    "实际GPU/RSS为采样监测，非OS硬隔离/瞬时peak保证。"
    "partial/停止/UNKNOWN不是阴性；未访问TEST不填0或50%。"
)


def render_report(saved: Mapping[str, Any]) -> dict[str, bytes]:
    summary = saved.get("summary", {})
    source = {"schema": 1, **{k: saved[k] for k in ("run_id", "run_ref", "run_sha256")}}
    rows = []
    for name in (
        "r1",
        "r_gate",
        "r_model",
        "r_agreement",
        "r_classification",
        "b_model",
        "dev",
        "test",
    ):
        value = summary.get(name)
        rows.append(
            f"{name}: "
            + (
                "未访问 / 未完成"
                if value is None
                else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
        )
    body = "\n".join(rows)
    counts = json.dumps(
        saved.get("counts", {}), ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    text = (
        f"# SPECK Fourier 保存结果 — {saved['run_id']}\n\n"
        "流程：R0 → R1 teacher/Eq4门 → R2全桶GL → R3系数/一致性/公开面板 → "
        "Rgate → teacher close/reap → B新key → Bsearch → Bcoef → "
        "严格持久读取 → DEV门 → 条件TEST\n\n"
        f"状态：{saved.get('status', 'UNKNOWN')} / {summary.get('reason', 'UNKNOWN')}\n"
        f"reproduction_matched: {summary.get('reproduction_matched', False)}\n"
        f"prediction_positive: {summary.get('prediction_positive', False)}\n"
        f"scientific_validation: {summary.get('scientific_validation', False)}\n\n"
        f"{body}\n\n实际工作账（attempt不退款，teacher为forward）：\n{counts}\n\n{LIMITS}\n"
    )
    bars: list[str] = []
    # Charts use ONLY saved metrics. No binary/model decode or label generation.
    for name in ("r_agreement", "r_classification", "dev", "test"):
        metric = summary.get(name)
        if isinstance(metric, Mapping) and type(metric.get("accuracy")) in (int, float):
            value = metric["accuracy"]
            if 0 <= value <= 1:
                y = 30 + len(bars) * 40
                bars.append(
                    f'<text x="0" y="{y}">{html.escape(name)} {value:.7f}</text>'
                    f'<rect x="230" y="{y - 15}" width="{value * 300:.3f}" '
                    'height="20" fill="teal"/>'
                )
    svg = (
        (
            '<svg viewBox="0 0 560 210" role="img" aria-label="实际保存准确率">'
            + "".join(bars)
            + "</svg>"
        )
        if bars
        else "<p>暂无已保存完整准确率，不补绘结果。</p>"
    )
    diagram = (
        "<p>R0 → R1 → R2 → R3 → Rgate → teacher回收 → Bsearch → Bcoef → "
        "durable → DEV → 条件TEST</p>"
    )
    page = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>SPECK 保存结果</title>'
        f"<body>{diagram}{svg}<pre>{html.escape(text)}</pre></body></html>"
    )
    return {
        "report.md": text.encode(),
        "index.html": page.encode(),
        "source.json": json.dumps(source, sort_keys=True, ensure_ascii=True).encode(),
    }
