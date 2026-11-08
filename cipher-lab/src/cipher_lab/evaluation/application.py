"""Explicit saved-file evaluation and pure report rendering; no training/collection.

Source references expose only SHA/size and fixed logical names, never user paths.
Checkpoint schema currently carries no backend identity; provenance stays UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import cast

from cipher_lab.artifacts import ArtifactError, RunStore, source_identity
from cipher_lab.execution import Meter, Stopped

MAX_BYTES = 256 * 1024 * 1024


def read_source(path: Path, expected_sha: str) -> bytes:
    """Stat then bounded read then SHA BEFORE invoking any binary decoder."""
    if re.fullmatch('[0-9a-f]{64}', expected_sha) is None or path.is_symlink():
        raise ValueError('INVALID_SOURCE')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
        raise ValueError('INVALID_SOURCE')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or opened.st_size != info.st_size
                or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
            raise ValueError('INVALID_SOURCE')
        content = stream.read(MAX_BYTES + 1)
    if (len(content) != info.st_size or len(content) > MAX_BYTES
            or hashlib.sha256(content).hexdigest() != expected_sha):
        raise ValueError('INVALID_SOURCE')
    return content


def run_evaluation(checkpoint: Path, checkpoint_sha: str, dataset: Path, dataset_sha: str,
                   meter: Meter, store: RunStore) -> dict[str, object]:
    """New run only. Four unit budget: only shared evaluator reserves forward rows."""
    summary: dict[str, object] = {
        'method': 'explicit-evaluation-v1', 'backend_identity': 'UNVERIFIED',
        'synthetic': None, 'scientific_validation': False,
        'dataset_role': 'explicit-saved-panel', 'reason': '', 'metrics': {},
        'source_refs': {}, 'meter': meter.snapshot(),
    }
    status, reason = 'SUCCEEDED', ''
    try:
        frozen = store.source_snapshot()
        meter.check()
        if source_identity() != frozen:
            raise ValueError('SOURCE_CHANGED')
        store.write_snapshot({'summary': summary})
        # Local imports are inside CLI's fixed setup/method boundary, never check.
        from cipher_lab.data.bundles import decode_bundle
        from cipher_lab.evaluation.checkpoints import load_checkpoint
        from cipher_lab.evaluation.metrics import EvaluationPanels, evaluate_batches

        model_bytes = read_source(checkpoint, checkpoint_sha)
        data_bytes = read_source(dataset, dataset_sha)
        summary['source_refs'] = {
            'checkpoint': {'path': 'explicit/checkpoint', 'sha256': checkpoint_sha,
                           'bytes': len(model_bytes)},
            'dataset': {'path': 'explicit/dataset', 'sha256': dataset_sha,
                        'bytes': len(data_bytes)}}
        store.write_snapshot({'summary': summary})
        loaded = load_checkpoint(model_bytes, checkpoint_sha)
        bundle = decode_bundle(data_bytes)
        spec = loaded.public_model_spec
        if bundle.method != spec.method or bundle.task_declarations != loaded.task_declarations:
            raise ValueError('DECLARATION_MISMATCH')
        summary.update(model_method=spec.method, model_spec=spec.model.to_dict(),
                       task_declarations=[t.to_dict() for t in loaded.task_declarations],
                       recipe_sha256=loaded.recipe_sha)
        meter.check()
        # Match the panel's own row count for tiny bundles; bounded batches otherwise.
        batch_size = min(256, min(len(x) for x, _ in bundle.panels.values()))
        summary['metrics'] = evaluate_batches(loaded.model,
            EvaluationPanels.from_bundle(bundle, loaded.baseline), spec, meter, batch_size)
        meter.check()
        if source_identity() != frozen:
            raise ValueError('SOURCE_CHANGED')
    except ArtifactError:
        raise  # never retry a storage-sealed owner
    except Stopped as error:
        status, reason = 'STOPPED', error.reason
    except Exception:
        status, reason = 'FAILED', 'EVALUATION_FAILED'
    summary['reason'], summary['meter'] = reason, meter.snapshot()
    result: dict[str, object] = {'status': status, 'summary': summary}
    store.finalize(result)
    return result


def render_evaluation_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Only saved public aggregates; this module has no eager tensor imports."""
    summary = cast(Mapping[str, object], saved.get('summary', {}))
    metrics = cast(Mapping[str, object], summary.get('metrics', {}))
    tasks = cast(Mapping[str, Mapping[str, object]], metrics.get('tasks', {}))
    bars: list[str] = []
    for name, values in tasks.items():
        acc = values.get('acc')
        if type(acc) in (float, int) and 0 <= cast(float, acc) <= 1:
            y = 30 + len(bars) * 30
            bars.append(f'<text x="8" y="{y}">{escape(name)} ACC={acc}</text>'
                        f'<rect x="400" y="{y-16}" width="{float(cast(float, acc))*350:.2f}" '
                        'height="18" fill="#276785"/>')
    flow = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 70" '
            'aria-label="显式评价流程"><text x="8" y="30">'
            '显式文件 → stat/有界读取/SHA → 严格 CPU 加载 → 声明精确匹配 → forward → ACC/BCE'
            '</text></svg>')
    figure = ('<svg xmlns="http://www.w3.org/2000/svg" '
              f'viewBox="0 0 850 {max(70, 45+30*len(bars))}" aria-label="已保存 ACC">'
              + (''.join(bars) or '<text x="8" y="30">无已保存指标</text>') + '</svg>')
    details = json.dumps(dict(summary), ensure_ascii=False, sort_keys=True, indent=2)
    body = (f'<h1>显式保存面板评价：{escape(str(saved["run_id"]))}</h1>'
            f'<p>状态：{escape(str(saved.get("status", "UNKNOWN")))}</p>'
            '<p>explicit-saved-panel：不推断为原 TEST；同任务其他已保存面板允许新评价。</p>'
            '<p>backend=UNVERIFIED；synthetic=unknown。checkpoint 未保存后端身份，'
            '不能认证真实或 demo 来源；scientific_validation=false。</p>'
            '<p>不训练、不采集、不生成标签；TRAIN 基线来自 checkpoint。'
            '仅 forward_rows 扣账。未验证真实密码/KAT/生产 RSS/Windows。</p>'
            f'<h2>方法流程</h2>{flow}<h2>已保存结果图</h2>{figure}'
            f'<h2>来源、ACC/BCE 与预算</h2><pre>{escape(details)}</pre>')
    source = {k: saved[k] for k in ('run_id', 'run_ref', 'run_sha256')}
    source['schema'] = 1
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<title>显式评价报告</title><body>' + body + '</body></html>')
    return {'report.md': ('# 显式评价\n\n' + body + '\n').encode(),
            'index.html': html.encode(),
            'source.json': (json.dumps(source, ensure_ascii=False, indent=2) + '\n').encode()}
