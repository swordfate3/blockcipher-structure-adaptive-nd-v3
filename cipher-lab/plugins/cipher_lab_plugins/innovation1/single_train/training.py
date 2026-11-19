"""One online two-arm loop with a disk-only two-best barrier before TEST."""
from __future__ import annotations

import math
from collections.abc import Callable
from hashlib import sha256
from typing import Any, Protocol

import numpy as np

from cipher_lab.artifacts import ArtifactError, RunStore

from ..contracts import MAX_CHECKPOINT
from ..resources import Resources, ResourceStop
from ..scoring import bce, predict, tensor_digest
from .config import SLOTS, TASK, SingleRecipe, estimate, learning_rate
from .data import PairBatch, encode_panel
from .weights import identity


class PanelEncoder(Protocol):
    def __call__(self, batch: PairBatch, recipe: SingleRecipe, source: dict[str, str],
                 backend: str, *, logits: dict[str, Any] | None = None,
                 weights: dict[str, Any] | None = None) -> bytes: ...


CheckpointIdentity = Callable[[SingleRecipe, dict[str, str], str, str, int, float], dict[str, Any]]
TestResult = Callable[[PairBatch, dict[str, dict[str, float]], dict[str, Any]], dict[str, Any]]


def _legacy_result(panel: PairBatch, scores: dict[str, dict[str, float]],
                   logits: dict[str, Any]) -> dict[str, Any]:
    return {'m_minus_d': {metric: scores[SLOTS[0]][metric] - scores[SLOTS[1]][metric]
                          for metric in ('accuracy', 'bce')}}


def train(recipe: SingleRecipe, provider: Any, runtime: Any, store: RunStore,
          resources: Resources, summary: dict[str, Any], backend: str) -> None:
    train_pair(recipe, provider, runtime, store, resources, summary, backend, slots=SLOTS,
               panel_bytes=encode_panel, checkpoint_identity=identity, test_result=_legacy_result)


def train_pair(recipe: SingleRecipe, provider: Any, runtime: Any, store: RunStore,
               resources: Resources, summary: dict[str, Any], backend: str, *,
               slots: tuple[str, str], panel_bytes: PanelEncoder,
               checkpoint_identity: CheckpointIdentity, test_result: TestResult) -> None:
    """Compute and snapshot only; application owns the single terminal transition."""
    tr = recipe.training
    source = store.source_snapshot()
    best: dict[str, Any] = summary['best']
    pending: list[dict[str, Any]] = summary['pending']
    refs: list[str] = summary['batch_refs']
    check = resources.check

    def check_source() -> None:
        check()
        try:
            current = store.current_source_identity()
        except ArtifactError:
            raise ResourceStop('SOURCE_CHANGED') from None
        if current != source:
            raise ResourceStop('SOURCE_CHANGED')
        check()

    def snapshot() -> None:
        check()
        store.write_snapshot(dict(summary=summary, counts=resources.snapshot()))
        check()

    def flush() -> None:
        if pending:
            snapshot()  # a sealing storage fault must not lose committed update evidence
            refs.append(store.write_payload(f'batches-{len(refs):04d}', pending))
            pending.clear()
            snapshot()

    def score(panel: PairBatch) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        chunks: dict[str, list[Any]] = {s: [] for s in slots}
        for start in range(0, panel.groups, tr.eval_batch_groups):
            check()
            loc = summary['location']
            loc['start_group'] = start
            batch = panel.slice(start, start + tr.eval_batch_groups)
            x, y = runtime.prepare(batch)
            digest = tensor_digest(x, y)
            for slot in slots:
                loc['slot'] = slot
                chunks[slot].append(predict(runtime, slot, TASK, x, y, resources))
                if digest != tensor_digest(x, y):
                    raise ValueError('BATCH_MUTATED')
        logits = {slot: np.concatenate(chunks[slot]) for slot in slots}
        scores = {slot: dict(accuracy=float(np.mean((z >= 0) == panel.labels)),
                             bce=bce(panel.labels, z)) for slot, z in logits.items()}
        return scores, logits

    check_source()
    # Only the committed pending rows (at most 256) need inline failure evidence.
    # Earlier chunks are already registered payloads, not copied into the seal.
    store._failure_evidence_key = 'pending'
    summary['models'] = runtime.metadata()
    summary['location'] = dict(phase='dev-generate')
    dev = provider.panel('dev')
    check()
    raw = panel_bytes(dev, recipe, source, backend)
    snapshot()  # seal from the collected counts if the dataset write fails
    ref = store.write_dataset('dev-SKINNY', raw)
    check()
    summary['dev_panel'] = dict(ref=ref, sha256=sha256(raw).hexdigest(), digest=dev.digest)
    del raw
    snapshot()
    for epoch in range(1, tr.epochs + 1):
        check_source()
        losses = dict.fromkeys(slots, 0.)
        for offset in range(tr.batches_per_epoch):
            cycle = (epoch - 1) * tr.batches_per_epoch + offset
            summary['location'] = dict(phase='train', epoch=epoch, cycle=cycle)
            check()
            batch = provider.train(cycle)
            x, y = runtime.prepare(batch)
            digest = tensor_digest(x, y)
            lr = learning_rate(recipe, cycle)
            row: dict[str, Any] = dict(epoch=epoch, cycle=cycle, lr=lr, groups=batch.groups,
                                      ciphertext_labels_sha256=batch.digest, input_sha256=digest,
                                      loss={}, updates_completed=[])
            pending.append(row)
            for slot in slots:
                summary['location']['slot'] = slot
                resources.reserve('optimizer_updates', 1)
                summary['updates'][slot]['attempted'] += 1
                loss = runtime.update(slot, TASK, x, y, lr, resources)
                resources.complete('optimizer_updates', 1)
                summary['updates'][slot]['completed'] += 1
                row['updates_completed'].append(slot)
                if len(row['updates_completed']) == len(slots):
                    summary['paired_completed'] += 1
                if math.isfinite(loss):
                    row['loss'][slot] = loss
                if cycle == 0 and slot == slots[0]:
                    snapshot()  # publish the first update before starting the second arm
                check()
                if not math.isfinite(loss) or tensor_digest(x, y) != digest:
                    raise ValueError('UPDATE_INVALID')
                losses[slot] += loss
            del batch, x, y
            if cycle == 0:
                snapshot()  # first full batch is visible, too
            if len(pending) >= 256:
                # 256 completed batches (131072 groups in the exact recipe).
                check_source()
                flush()
        flush()
        summary['location'] = dict(epoch=epoch, phase='dev-score')
        scores, _ = score(dev)
        summary['curves'].append(dict(epoch=epoch, dev=scores, train_bce={
            s: v / tr.batches_per_epoch for s, v in losses.items()}))
        snapshot()  # full DEV and counts survive before any best write
        for slot in slots:
            val = scores[slot]['bce']
            if slot in best and val >= best[slot]['bce']:
                continue
            summary['location'] = dict(epoch=epoch, phase='best-save', slot=slot)
            check_source()
            meta = checkpoint_identity(recipe, source, backend, slot, epoch, val)
            raw = runtime.checkpoint(slot, meta)
            check()
            ref = store.write_model(slot, raw, replace=slot in best)
            best[slot] = dict(ref=ref, sha256=sha256(raw).hexdigest(), epoch=epoch, bce=val,
                              dev_digest=dev.digest)
            del raw
            snapshot()
    summary['location'] = dict(phase='persisted-best-reload')
    check_source()
    if set(best) != set(slots):
        raise ValueError('BEST_INCOMPLETE')
    runtime.release()
    check()
    for slot in slots:
        summary['location']['slot'] = slot
        selected = best[slot]
        if selected['ref'] != f'model/{slot}.pt':
            raise ValueError('BEST_REF')
        # Canonical run.json registration, exact current owner path, bounded size and SHA.
        raw = store.read_model(selected['ref'], selected['sha256'], MAX_CHECKPOINT)
        check()
        runtime.load_best(slot, raw, recipe, source, selected['epoch'], selected['bce'], check)
        del raw
        summary['persisted_reloads'][slot] = dict(selected)
        snapshot()
    summary['location'] = dict(phase='test-generate')
    check_source()
    test = provider.panel('test')
    summary['location'] = dict(phase='test-score')
    scores, logits = score(test)
    summary['location'] = dict(phase='test-save')
    raw = panel_bytes(test, recipe, source, backend, logits=logits,
                      weights=summary['persisted_reloads'])
    snapshot()  # include generation and both scoring passes, not TEST results
    ref = store.write_dataset('test-SKINNY', raw)
    check()
    summary['test_panel'] = dict(ref=ref, sha256=sha256(raw).hexdigest(), digest=test.digest)
    summary.update(test_scores=scores, **test_result(test, scores, logits))
    snapshot()
    check_source()
    counts = resources.snapshot()
    totals = estimate(recipe)
    if any(counts[f'{kind}_{unit}'] != totals[unit]
           for kind in ('attempted', 'completed')
           for unit in ('real_blocks', 'optimizer_updates', 'forward_rows', 'null_words')):
        raise ValueError('COUNT_MISMATCH')
