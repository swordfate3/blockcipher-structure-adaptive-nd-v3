"""Render saved results for the three-arm feature fusion experiment."""
from __future__ import annotations

import html
import json
from typing import Any, cast

from . import config

_LABELS = {config.BASE: '原结构卡', config.SHUFFLED: '打乱接线融合', config.P_LAYER: 'P 层接线融合'}


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary = record['summary']
    lines = [
        '# PRESENT-80 七轮中间特征融合', '', f"运行 `{record['run_id']}`。", '',
        '共享 TRAIN、DEV → 三组从相同骨干初值训练 → 原序 DEV BCE 选模',
        '→ 严格回读三组最佳权重 → 在一份共享 TEST 上打分。', '',
        '| 训练组 | 判对数 | TEST 准确率 | TEST BCE | 参数量 | 最佳轮 / 完成轮 | 拟合秒数 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for slot in config.SLOTS:
        arm, test = summary['arms'][slot], summary['test'][slot]
        lines.append(f"| {_LABELS[slot]} | {test['correct_pairs']:,}/{test['pairs']:,} | "
                     f"{test['accuracy']:.4%} | {test['bce']:.8f} | {arm['parameters']:,} | "
                     f"{arm['selected']['epoch']}/{arm['epochs_completed']} | "
                     f"{arm['fit_elapsed_seconds']:.2f} |")
    lines.extend(['', '| 配对比较 | 准确率差（百分点） | 仅前者正确 | 仅后者正确 |',
                  '|---|---:|---:|---:|'])
    for first, second in config.CONTRASTS:
        paired = summary['paired'][f'{first}_vs_{second}']
        lines.append(f"| {_LABELS[first]}减{_LABELS[second]} | "
                     f"{paired['percentage_points']:+.4f} | {paired['joint_only_correct']:,} | "
                     f"{paired['control_only_correct']:,} |")
    lines.extend(['', '两组融合各增加 9,216 个参数，接线保留相同的置换环结构。',
                  '结果覆盖本次一把合成密钥、固定初始化及七轮单对分类。'])
    markdown = '\n'.join(lines) + '\n'
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            f'<title>PRESENT-80 中间特征融合</title><pre>{html.escape(markdown)}</pre></html>')
    source = {key: record[key] for key in ('run_ref', 'run_sha256', 'run_id')}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps({'schema': 1, **source}).encode()}
