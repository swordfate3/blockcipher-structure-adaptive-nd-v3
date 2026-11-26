"""Only render the saved record. Do not reopen labels, checkpoints or ciphers."""
from __future__ import annotations

import html
import json
from typing import Any, cast


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    record = cast('dict[str, Any]', saved)
    summary, recipe = record['summary'], record['recipe']
    text = [
        '# PRESENT-80：S2 选态、总七轮 C7-only 分类', '',
        f"运行：{record['run_id']}；保存科学终态：{record['status']}；"
        f"阶段：{summary.get('phase', 'UNKNOWN')}；原因：{summary.get('reason', 'UNKNOWN')}。",
        'native exit：UNKNOWN；SSH exit：UNKNOWN。须独立取证，不能由保存终态推定。', '',
        '在含 K3 白化的完整 S2 选态；正类以固定异或差配对，负类独立取样。',
        '两类均用同一合成 key 做 S2→P；从 P 完整前向七轮，同一轨迹复核 S2、仅 C7 入模。',
        'TRAIN/DEV → 一次 fresh Shared-M fit → 磁盘 SHA 与严格回读 → 一次 TEST。',
        '本实验假定理想选态能力，不是实际预测器，不做密钥恢复，也不针对真实目标。',
        '单 key 与缩减轮数不支持跨 key 或现实攻击结论。旧 52.3071% 来自不同协议，不是同期对照。',
        '每组四对；正确组/总组，平衡参考 50%。公开 gate 的 5 单位计入 real_blocks，'
        '不是新官方 KAT；下列科学轨迹计数排除 gate。',
    ]
    arm = summary.get('fit', {})
    if arm:
        best = arm.get('selected', {})
        text.extend(['', f"完成 epoch：{arm.get('epochs_completed', 'UNKNOWN')}；"
                     f"停止原因：{arm.get('stop_reason', 'UNKNOWN')}；"
                     f"最佳 epoch：{best.get('epoch', 'UNKNOWN')}；"
                     f"DEV BCE：{best.get('bce', 'UNKNOWN')}；"
                     f"严格回读：{arm.get('best_reloaded', 'UNKNOWN')}。"])
    test = summary['test'] if record['status'] == 'SUCCEEDED' else None
    if test is None:
        text.append('TEST 未提交。')
    else:
        text.extend(['', '## TEST（组级）', '',
                     f"{test['correct_groups']}/{test['groups']}；ACC {test['accuracy']:.4%}；"
                     f"BCE {test['bce']:.8f}。"])
    text.extend(['', '## 保存证据', '', '```json', json.dumps({
        'recipe': recipe, 'policy': record.get('policy'), 'counts': record.get('counts'),
        'kat': summary.get('kat'), 'device': summary.get('device'),
        'runtime': summary.get('runtime'), 'cuda_peaks': summary.get('cuda_peaks'),
        'panel_hashes': summary.get('panel_hashes'), 'science': summary.get('science'),
        'test': test, 'artifacts': record.get('artifacts'),
    }, ensure_ascii=False, sort_keys=True, indent=2), '```', ''])
    markdown = '\n'.join(text)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<title>PRESENT-80 S2 选态、总七轮 C7-only 分类</title><pre>'
            + html.escape(markdown) + '</pre></html>')
    source = {'schema': 1, 'run_id': record['run_id'], 'run_ref': record['run_ref'],
              'run_sha256': record['run_sha256']}
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(source, sort_keys=True).encode()}
