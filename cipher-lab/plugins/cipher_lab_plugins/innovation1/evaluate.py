"""Explicit saved-input evaluation; no training/cipher/private generator dependency."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cipher_lab.artifacts import RunStore, read_binary, read_saved

from .checkpoint import MAX_CHECKPOINT, SLOTS, canonical, identity
from .config import _decode, validate
from .dataset import MAX_PANEL, decode_panel
from .scoring import bce, score_panel


def evaluate(root: Path, run_id: str, run_sha256: str, checkpoint_ref: str,
             checkpoint_sha256: str, dataset_ref: str, dataset_sha256: str,
             store: RunStore, resources: Any, *, backend: str = 'UNVERIFIED',
             device: str = 'cuda:0', max_forward_rows: int | None = None,
             batch_groups: int | None = None) -> dict[str, Any]:
    from .config import Recipe
    saved: Any = read_saved(root, run_id)
    if saved['run_sha256'] != run_sha256 or saved['status'] != 'SUCCEEDED':
        raise ValueError('RUN_IDENTITY')
    if (store.current_source_identity() != store.source_snapshot()
            or canonical(store.source_snapshot()) != canonical(saved['source'])):
        raise ValueError('SOURCE_IDENTITY')
    summary: Any = saved['summary']
    recipe = _decode(saved['recipe'], Recipe(), 'recipe')
    validate(recipe)
    if (summary.get('backend_identity') != backend or backend not in ('demo', 'ciphers')
            or type(summary.get('synthetic')) is not bool
            or summary['synthetic'] != (backend == 'demo')):
        raise ValueError('BACKEND_IDENTITY')
    best = summary['best']
    choices = [s for s in SLOTS if best[s]['ref'] == checkpoint_ref]
    if len(choices) != 1:
        raise ValueError('CHECKPOINT_REFERENCE')
    slot = choices[0]
    entry = best[slot]
    if entry['sha256'] != checkpoint_sha256:
        raise ValueError('CHECKPOINT_SHA')
    meta = entry['metadata']
    expected = identity(recipe, saved['source'], backend, slot,
                        meta['epoch'], meta['score'])
    if canonical(expected) != canonical(meta):
        raise ValueError('CHECKPOINT_IDENTITY')
    panels = [(t, split) for t in recipe.tasks for split in ('dev', 'test')
              if dataset_ref == f'dataset/{split}-{t.cipher}.npz']
    if len(panels) != 1:
        raise ValueError('DATASET_REFERENCE')
    task, split = panels[0]
    if slot != 'M-best' and slot != f'D-{task.cipher}-best':
        raise ValueError('TASK_IDENTITY')
    size = recipe.training.dev_groups if split == 'dev' else recipe.training.test_groups
    budget = size if max_forward_rows is None else max_forward_rows
    chunk = recipe.training.eval_batch_groups if batch_groups is None else batch_groups
    if (type(budget) is not int or budget < size or type(chunk) is not int
            or not 1 <= chunk <= recipe.training.eval_batch_groups):
        raise ValueError('EVALUATION_BUDGET')
    resources.check()
    raw = read_binary(root, run_id, checkpoint_ref, checkpoint_sha256, MAX_CHECKPOINT)
    panel = decode_panel(read_binary(root, run_id, dataset_ref, dataset_sha256, MAX_PANEL),
                         recipe, task, split, saved['source'], backend)
    runtime: Any
    if backend == 'demo':
        from .manual_runtime import ManualRuntime
        runtime = ManualRuntime(recipe)
        runtime.release()
    else:
        from .models import TorchRuntime
        runtime = TorchRuntime(recipe, device, inference_only=True)
    runtime.restore(slot, raw, expected)
    logits = score_panel(panel, runtime, (slot,),
                         chunk, resources)[slot]
    import numpy as np
    value = dict(bce=bce(panel.labels, logits),
                 accuracy=float(np.mean((logits >= 0) == panel.labels)))
    store.write_snapshot({'recipe': {'method': 'innovation1-explicit-evaluation-v1'}})
    store.write_payload('evaluation-predictions', dict(labels=panel.labels.tolist(),
                                                      logits=logits.tolist()))
    result = dict(status='SUCCEEDED', counts=resources.snapshot(), summary=dict(
        method='innovation1-explicit-evaluation-v1', backend_identity=backend,
        synthetic=backend == 'demo', scientific_validation=False, scores=value,
        input_run_id=run_id, input_run_sha256=run_sha256, checkpoint_ref=checkpoint_ref,
        checkpoint_sha256=checkpoint_sha256, dataset_ref=dataset_ref,
        dataset_sha256=dataset_sha256, task=vars(task), slot=slot,
        max_forward_rows=budget, batch_groups=chunk))
    resources.check()
    if store.current_source_identity() != store.source_snapshot():
        raise ValueError('SOURCE_CHANGED')
    store.finalize(result)
    return result
