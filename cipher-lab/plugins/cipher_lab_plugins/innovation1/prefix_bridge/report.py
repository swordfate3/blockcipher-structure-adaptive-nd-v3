"""Presentation from the saved RunStore record only; no panel/model imports."""
from __future__ import annotations

import html
import json
from typing import Any, cast


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary, recipe = record['summary'], record['recipe']
    text = [
        '# PRESENT-80：理想二轮状态辅助分类', '',
        f"运行：{record['run_id']}；保存科学终态：{record['status']}；"
        f"阶段：{summary.get('phase', 'UNKNOWN')}；原因：{summary.get('reason', 'UNKNOWN')}。",
        'native exit：UNKNOWN；SSH exit：UNKNOWN。两者分别由启动器和控制端取证，'
        '不能从科学终态补零。', '',
        '同一新 key → 共享 TRAIN / DEV → 13 个独立 fit / DEV 选模 → '
        '13 份磁盘 best 严格恢复 → 共享 TEST → 组级配对比较。',
        '一组是四对；ACC = 正确组数 / TEST 组数，不是位或单对准确率，平衡随机参考为 50%。',
        'S2-only 只训练及评分一次，六轮共用；r2 的 joint 重复同一状态。',
        '理想 S2 oracle 是额外输入，可能含秘密相关关系，不是实际可用的预测器。'
        '单路屏蔽后有效连接更少；同构不等于相同有效容量。',
        '单 key、单面板的描述性结果不证明轮数扩展、现实攻击或 joint 对任一路的因果依赖。',
    ]
    if 's2_boundary' in recipe:
        text.append(f"S2 定义：{recipe['s2_boundary']}。")
    fits = summary.get('fits', {})
    if fits:
        text.extend(['', '|fit|完成 epoch|选中 epoch|选中 DEV BCE|停止原因|参数数|严格恢复|',
                     '|---|---:|---:|---:|---|---:|---|'])
        for slot in recipe['slots']:
            arm = fits.get(slot, {})
            best = arm.get('selected', {})
            text.append(f"|{slot}|{arm.get('epochs_completed', 'UNKNOWN')}|"
                        f"{best.get('epoch', 'UNKNOWN')}|{best.get('bce', 'UNKNOWN')}|"
                        f"{arm.get('stop_reason', 'UNKNOWN')}|{arm.get('parameters', 'UNKNOWN')}|"
                        f"{arm.get('best_reloaded', 'UNKNOWN')}|")
    if record['status'] == 'SUCCEEDED':
        test = summary['test']
        by_slot = test['by_slot']

        def cell(key: str) -> str:
            s = by_slot[key]
            return f"{s['correct_groups']}/{s['groups']}; {s['accuracy']:.4%}; {s['bce']:.8f}"

        text.extend(['', '## TEST（正确组/总组；ACC；BCE）', '',
                     '|轮|Cr-only|共享 S2-only|joint|joint−Cr (pp)|joint−S2 (pp)|',
                     '|---:|---|---|---|---:|---:|'])
        for row in test['rounds']:
            text.append(f"|{row['round']}|{cell(row['cipher_slot'])}|{cell(row['s2_slot'])}|"
                        f"{cell(row['joint_slot'])}|{row['joint_vs_cipher']['percentage_points']:+.4f}|"
                        f"{row['joint_vs_s2']['percentage_points']:+.4f}|")
    else:
        text.append('TEST 未提交。')
    text.extend(['', '## 保存证据', '', '```json', json.dumps({
        'recipe': recipe, 'counts': record.get('counts'), 'policy': record.get('policy'),
        'kat': summary.get('kat'), 'panel_hashes': summary.get('panel_hashes'),
        'trajectories': summary.get('trajectories'),
        'roundstateoutputs': summary.get('roundstateoutputs'),
        'test': summary.get('test') if record['status'] == 'SUCCEEDED' else None,
        'artifacts': record.get('artifacts'),
    }, ensure_ascii=False, sort_keys=True, indent=2), '```', ''])
    markdown = '\n'.join(text)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>PRESENT-80 理想二轮状态辅助分类</title><pre>'
            + html.escape(markdown) + '</pre></html>')
    source = {'schema': 1, 'run_id': record['run_id'], 'run_ref': record['run_ref'],
              'run_sha256': record['run_sha256']}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(source, sort_keys=True).encode()}
