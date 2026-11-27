"""Render the masked-card result from saved aggregates."""
from __future__ import annotations

import html
from typing import Any, cast

from ..present_card_fresh.report import report_files as render_pair_report


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    files = render_pair_report(saved)
    record = cast('dict[str, Any]', saved)
    aux = record['summary'].get('fit', {}).get('auxiliary_train', {})
    title = 'PRESENT-80 七轮结构卡与训练期遮挡补全'
    body = files['report.md'].decode().replace('PRESENT-80 七轮：公开结构卡，纯单对分类', title, 1)
    objective = ('训练使用完整视图分类 BCE 加权叠加遮挡补全 BCE。'
                 'DEV 与 TEST 使用完整视图。')
    stats = (f"补全 TRAIN BCE {aux.get('bce', 'UNKNOWN')}，"
             f"监督位数 {aux.get('targets', 'UNKNOWN')}。")
    header, _, details = body.partition('\n\n')
    markdown = f'{header}\n\n{objective}\n{stats}\n\n{details}'
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            f'<title>{html.escape(title)}</title><pre>{html.escape(markdown)}</pre></html>')
    return {**files, 'report.md': markdown.encode(), 'index.html': page.encode()}
