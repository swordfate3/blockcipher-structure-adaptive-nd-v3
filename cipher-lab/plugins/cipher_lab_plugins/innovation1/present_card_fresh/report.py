"""Render saved pair aggregates and evidence, never open science artifacts."""
from __future__ import annotations

import html
import json
from typing import Any, cast


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary, recipe = record['summary'], record['recipe']
    title = 'PRESENT-80 七轮：公开结构卡，纯单对分类'
    text = [
        f'# {title}', '',
        f"运行：{record['run_id']}；保存科学终态：{record['status']}；"
        f"阶段：{summary.get('phase', 'UNKNOWN')}；原因：{summary.get('reason', 'UNKNOWN')}。",
        'native exit: UNKNOWN；SSH exit: UNKNOWN。须独立取证，不由保存终态推定。', '',
        '公开双路径门槛（4 单位）→ 一次合成 key → TRAIN/DEV → 一次 fresh fit'
        ' → 磁盘 SHA 与严格回读 → 一次 TEST。',
        '明文正类固定差分 0xD00000，负类端点独立；两类都完整加密至 C7。'
        '单对读出；四槽仅用于形状，不聚合。平衡参考 50%。',
        '仅观察一次合成 key、固定 seed 与缩减轮数；无真实目标、密钥恢复、历史续训或同期基线。',
    ]
    arm = summary.get('fit', {})
    best = arm.get('selected', {})
    text.extend(['', f"完成 epoch：{arm.get('epochs_completed', 'UNKNOWN')}；"
                 f"停止原因：{arm.get('stop_reason', 'UNKNOWN')}；"
                 f"最佳 epoch：{best.get('epoch', 'UNKNOWN')}；"
                 f"最佳 step：{best.get('step', 'UNKNOWN')}；"
                 f"DEV 分类 BCE：{best.get('bce', 'UNKNOWN')}；"
                 f"严格回读：{arm.get('best_reloaded', 'UNKNOWN')}。"])
    test = summary['test'] if record['status'] == 'SUCCEEDED' else None
    if test is None:
        text.append('TEST 未提交。')
    else:
        text.extend(['', '## TEST（单对）', '',
                     f"{test['correct_pairs']}/{test['pairs']}；ACC {test['accuracy']:.4%}；"
                     f"BCE {test['bce']:.8f}。"])
    text.extend(['', '## 保存证据', '', '```json', json.dumps({
        'recipe': recipe, 'policy': record.get('policy'), 'counts': record.get('counts'),
        'kat': summary.get('kat'), 'device': summary.get('device'),
        'runtime': summary.get('runtime'), 'cuda_peaks': summary.get('cuda_peaks'),
        'panel_hashes': summary.get('panel_hashes'), 'best': best,
        'initial_sha256': arm.get('initial_sha256'), 'geometry': arm.get('geometry'),
        'test': test, 'artifacts': record.get('artifacts'),
    }, ensure_ascii=False, sort_keys=True, indent=2), '```', ''])
    markdown = '\n'.join(text)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            f'<title>{html.escape(title)}</title><pre>{html.escape(markdown)}</pre></html>')
    source = {'schema': 1, 'run_id': record['run_id'], 'run_ref': record['run_ref'],
              'run_sha256': record['run_sha256']}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(source, sort_keys=True).encode()}
