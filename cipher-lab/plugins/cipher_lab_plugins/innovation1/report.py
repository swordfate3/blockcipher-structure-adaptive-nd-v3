"""Pure saved-only report: no model, metrics, numpy, data generation or file I/O."""
from __future__ import annotations

import html
import json
from typing import Any


def render(saved: dict[str, Any]) -> dict[str, bytes]:
    summary = saved['summary']
    rows = []
    bars = []
    comparisons = summary.get('compare', {}) if saved['status'] == 'SUCCEEDED' else {}
    for index, (task, comparison) in enumerate(comparisons.items()):
        m, d = comparison['shared'], comparison['dedicated']
        difference = 100 * comparison['accuracy_difference']
        interval = [100 * value for value in comparison['interval']]
        rows.append(f"| {task} | {100*m['accuracy']:.4f} | {100*d['accuracy']:.4f} | "
                    f"{difference:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] | "
                    f"{m['bce']:.6f} | {d['bce']:.6f} |")
        for arm, scores, color in ((0, m, '#2463eb'), (1, d, '#d97706')):
            width = max(0., min(100., 100 * scores['accuracy'])) * 3
            y = 30 + index * 60 + arm * 22
            bars.append(f'<text x="0" y="{y+13}">{html.escape(task)} '
                        f'{"M" if arm == 0 else "D"}</text>'
                        f'<rect x="110" y="{y}" width="{width:.3f}" height="16" '
                        f'fill="{color}"/>')
    method = ('在线批一次生成 → 同tensor M更新 → D更新 → epoch DEV选模 → '
              '5个best磁盘严格重载 → 晚构造共同TEST → 保存比较 → 纯saved报告')
    limitations = ('单初始化；区间仅反映固定模型/固定类配额下组级抽样，不含初始化波动；'
                   '包含0不表示等效，不推及每个key/最差key。共同新配方不是旧结果复现。'
                   'M总更新为每D的4倍；原生初始化/BN及选模策略不同；参数少不等于更快。'
                   '采样资源门禁不是OS硬隔离，GPU/Windows现场能力尚未验证。'
                   '保存计数仅截至科学封存；命令输出另给出报告阶段用量与请求退出码，'
                   '不能由保存科学终态推断命令成功。')
    evidence = json.dumps(dict(status=saved['status'], summary=summary, counts=saved['counts'],
                               source=saved['source'], artifacts=saved['artifacts']),
                          ensure_ascii=False, indent=2)
    markdown = ('# 创新1共享/专用同批对照\n\n'
                f"状态：{saved['status']}；"
                f"backend={summary.get('backend_identity', 'UNVERIFIED')}；"
                f"synthetic={summary.get('synthetic')}；scientific_validation=False\n\n"
                'demo为人工工程闭环，非训练/科学结果。\n\n## 方法\n\n' + method + '\n\n'
                '|任务|M ACC %|D ACC %|M−D pp|98.75%区间 pp|M BCE|D BCE|\n'
                '|---|---:|---:|---:|---|---:|---:|\n' + '\n'.join(rows) +
                '\n\n## 限制\n\n' + limitations + '\n\n## 保存证据\n\n```json\n' +
                evidence + '\n```\n')
    flow = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 80" '
            'role="img" aria-label="方法流程"><rect width="900" height="80" '
            'fill="#eef2ff"/><text x="10" y="30">'
            '同批 → M/D更新 → DEV → 5best磁盘重载 → TEST → saved比较 → report'
            '</text><text x="10" y="60">TEST不参与训练/选模；无自动重试/续训</text></svg>')
    chart = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300" '
             'role="img" aria-label="已保存ACC，不重算指标">' + ''.join(bars) + '</svg>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>创新1保存报告</title>'
            '<h1>创新1保存报告（非科学认证）</h1>' + flow + chart + '<pre>' +
            html.escape(markdown) + '</pre></html>')
    source = dict(schema=1, run_id=saved['run_id'], run_ref=saved['run_ref'],
                  run_sha256=saved['run_sha256'])
    return {'report.md': markdown.encode(), 'index.html': page.encode(),
            'source.json': json.dumps(source, sort_keys=True).encode()}
