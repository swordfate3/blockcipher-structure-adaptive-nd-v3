"""Render only the saved comparison; no numerical dependencies."""
from __future__ import annotations

import html
import json
from typing import Any, cast


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary, recipe = record['summary'], record['recipe']
    title = 'PRESENT-80 总7轮：单对分类与训练时遮挡辅助'
    rows = [f'# {title}', '',
            f"运行：{record['run_id']}；保存终态：{record['status']}。",
            'native exit 与 SSH exit 需独立取证，未由本记录推断。', '',
            '同一新合成 key → 共享 TRAIN / DEV → A/B 两个独立 fit → '
            '两个 best 严格从磁盘恢复 → 共享 TEST → 逐对配对比较。',
            'A/B 都只用完整 C7 分类；B 仅在训练时增加遮挡视图和补全 BCE。'
            '两臂分类初值相同，计算量不匹配。每对一个判定，平衡随机参考为 50%。'
            '历史四对组分数不可直接比较。',
            '结果限于单 key、固定 seed 和缩减轮合成面板；不推断显著性、全轮或真实系统。',
            f"固定配方：TRAIN {recipe['train_rows']:,} 对、{recipe['train_ciphertexts']:,} 条密文；"
            f"DEV {recipe['dev_rows']:,} 对，TEST {recipe['test_rows']:,} 对。"]
    if record['status'] == 'SUCCEEDED':
        test, fits = summary['test'], summary['fits']
        rows.extend(['', '|臂|完成 epoch|最佳 epoch|DEV BCE|TEST 正确对|ACC|TEST BCE|',
                     '|---|---:|---:|---:|---:|---:|---:|'])
        for slot in recipe['slots']:
            arm, score = fits[slot], test['by_slot'][slot]
            best = arm['selected']
            rows.append(f"|{slot}|{arm['epochs_completed']}|{best['epoch']}|{best['bce']:.8f}|"
                        f"{score['correct_pairs']}/{score['pairs']}|{score['accuracy']:.4%}|"
                        f"{score['bce']:.8f}|")
        rows.append(f"\nB−A：{test['b_vs_a']['percentage_points']:+.4f} pp。"
                    '配对计数中的 joint 指 B，control 指 A。')
        rows.extend(['',
                     '|臂|更新|训练分类对前向|DEV 分类对前向|TEST 分类对前向|遮挡对前向|目标位|',
                     '|---|---:|---:|---:|---:|---:|---:|'])
        for slot in recipe['slots']:
            work = fits[slot]['work']
            rows.append(f"|{slot}|{work['optimizer_updates']}|{work['train_class_rows']}|"
                        f"{work['dev_class_rows']}|{test['forward_rows_by_slot'][slot]}|"
                        f"{work['masked_forward_rows']}|{work['aux_target_bits']}|")
    else:
        rows.append('TEST 未提交。')
    evidence = {'recipe': recipe, 'counts': record.get('counts'), 'policy': record.get('policy'),
                'summary': summary, 'artifacts': record.get('artifacts')}
    rows.extend(['', '## 保存证据', '', '```json',
                 json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2), '```', ''])
    markdown = '\n'.join(rows)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            f'<title>{title}</title><pre>{html.escape(markdown)}</pre></html>')
    source = {'schema': 1, 'run_id': record['run_id'], 'run_ref': record['run_ref'],
              'run_sha256': record['run_sha256']}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(source, sort_keys=True).encode()}
