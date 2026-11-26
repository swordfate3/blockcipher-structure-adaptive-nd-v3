"""One loop with a disk-only current best checkpoint barrier before TEST."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Protocol

import numpy as np

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.training.early_stopping import EarlyStopping, Settings

from ..config import Task, canonical
from ..contracts import MAX_CHECKPOINT
from ..resources import Resources, ResourceStop
from ..scoring import bce, mse, predict, tensor_digest
from .config import TASK, SingleRecipe, estimate, learning_rate
from .data import PairBatch


class PanelEncoder(Protocol):
    def __call__(self, batch: PairBatch, recipe: SingleRecipe, source: dict[str, str],
                 backend: str, /, *, logits: dict[str, Any] | None = None,
                 weights: dict[str, Any] | None = None) -> bytes: ...


CheckpointIdentity = Callable[[SingleRecipe, dict[str, str], str, str, int, float], dict[str, Any]]
TestResult = Callable[[PairBatch, dict[str, dict[str, float]], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class SavedTest:
    """Opt in to verified archive TEST and terminal-only results."""

    open_panel: Callable[[dict[str, Any]], PairBatch]
    observe_routes: Callable[[str], dict[str, float]]
    estimate: Callable[[SingleRecipe], dict[str, int]]
    batch_row_bytes: int


def train_pair(recipe: SingleRecipe, provider: Any, runtime: Any, store: RunStore,
               resources: Resources, summary: dict[str, Any], backend: str, *,
               slots: tuple[str] | tuple[str, str], panel_bytes: PanelEncoder,
               checkpoint_identity: CheckpointIdentity, test_result: TestResult,
               early_stopping: Settings | None = None, cosine_epochs: int | None = None,
               saved_test: SavedTest | None = None, task: Task = TASK) -> dict[str, Any] | None:
    """Compute and snapshot only; application owns the single terminal transition."""
    if (type(slots) is not tuple or len(slots) not in (1, 2) or len(slots) != len(set(slots))
            or (len(slots) == 1 and (saved_test is None or early_stopping is not None
                                    or cosine_epochs is not None))):
        raise ValueError('TRAIN_SLOTS')
    completed_key = 'completed_batches' if len(slots) == 1 else 'paired_completed'
    tr = recipe.training
    stopper = (None if early_stopping is None else
               EarlyStopping(slots, early_stopping, epoch_ceiling=tr.epochs))
    loss_key = recipe.loss_key
    counted_recipe = recipe
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
        value = summary
        if saved_test is not None:
            # Bounded evidence copied by the store on a storage seal. Registered
            # ledgers are already in artifacts; never duplicate their full rows here.
            progress: dict[str, Any] = {key: summary[key] for key in (
                'location', 'updates', completed_key, 'pending', 'best', 'persisted_reloads',
                'reload_routes', 'training_termination') if key in summary}
            progress['last_dev'] = summary['curves'][-1] if summary['curves'] else None
            value = dict(summary, progress=progress)
        store.write_snapshot(dict(summary=value, counts=resources.snapshot()))
        check()

    def flush() -> None:
        if pending:
            if saved_test is not None and any(
                    len(canonical(row)) + 1 > saved_test.batch_row_bytes for row in pending):
                raise ValueError('BATCH_ROW_SIZE')
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
            batch = panel.slice(start, min(start + tr.eval_batch_groups, panel.groups))
            x, y = runtime.prepare(batch)
            digest = tensor_digest(x, y)
            for slot in slots:
                loc['slot'] = slot
                chunks[slot].append(predict(runtime, slot, task, x, y, resources))
                if digest != tensor_digest(x, y):
                    raise ValueError('BATCH_MUTATED')
        logits = {slot: np.concatenate(chunks[slot]) for slot in slots}
        scores = {slot: dict(accuracy=float(np.mean((z >= 0) == panel.labels)),
                             bce=bce(panel.labels, z)) for slot, z in logits.items()}
        if recipe.fixed_pool and saved_test is None:
            for slot, z in logits.items():
                scores[slot]['mse'] = mse(panel.labels, z)
        return scores, logits

    def save_split(panel: PairBatch) -> None:
        check()
        raw = panel_bytes(panel, recipe, source, backend)
        snapshot()  # seal from the collected counts if the dataset write fails
        ref = store.write_dataset(f'{panel.split}-{task.cipher}', raw)
        check()
        summary[f'{panel.split}_panel'] = dict(ref=ref, sha256=sha256(raw).hexdigest(),
                                              digest=panel.digest)
        del raw
        snapshot()

    check_source()
    # Only the committed pending rows (at most 256) need inline failure evidence.
    # Earlier chunks are already registered payloads, not copied into the seal.
    if hasattr(store, '_failure_evidence_key'):
        setattr(store, '_failure_evidence_key', 'pending' if saved_test is None else 'progress')
    summary['models'] = runtime.metadata()
    if recipe.fixed_pool:
        summary['location'] = dict(phase='pool-prepare')
        summary['pool_status'] = 'PREPARING'
        snapshot()
        train = provider.prepare_pool()
        expected_blocks = (0 if recipe.reuse_pool else
                           2 * (tr.batches_per_epoch * tr.batch_groups + tr.dev_groups))
        counts = resources.snapshot()
        if any(counts[f'{kind}_real_blocks'] != expected_blocks
               for kind in ('attempted', 'completed')):
            raise ValueError('POOL_COUNT_MISMATCH')
        check_source()
        if not recipe.reuse_pool:
            save_split(train)
        del train
        summary['location'] = dict(phase='pool-dev-bind' if recipe.reuse_pool else 'pool-dev-save')
    else:
        summary['location'] = dict(phase='dev-generate')
    dev = provider.panel('dev')
    if recipe.reuse_pool:
        counts = resources.snapshot()
        if any(counts[f'{kind}_real_blocks'] != 0 for kind in ('attempted', 'completed')):
            raise ValueError('POOL_COUNT_MISMATCH')
        summary.update(provider.pool_provenance())
    else:
        save_split(dev)
    if recipe.fixed_pool:
        summary['pool_status'] = 'READY'
        snapshot()
        provider.activate()
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
            lr = learning_rate(recipe, cycle, cosine_epochs=cosine_epochs)
            row: dict[str, Any] = dict(epoch=epoch, cycle=cycle, lr=lr, groups=batch.groups,
                                      ciphertext_labels_sha256=batch.digest, input_sha256=digest,
                                      loss={}, updates_completed=[])
            pending.append(row)
            for slot in slots:
                summary['location']['slot'] = slot
                resources.reserve('optimizer_updates', 1)
                summary['updates'][slot]['attempted'] += 1
                loss = runtime.update(slot, task, x, y, lr, resources)
                resources.complete('optimizer_updates', 1)
                summary['updates'][slot]['completed'] += 1
                row['updates_completed'].append(slot)
                if len(row['updates_completed']) == len(slots):
                    summary[completed_key] += 1
                if math.isfinite(loss):
                    row['loss'][slot] = loss
                if cycle == 0 and slot == slots[0]:
                    snapshot()  # publish the first update immediately
                check()
                if not math.isfinite(loss) or tensor_digest(x, y) != digest:
                    raise ValueError('UPDATE_INVALID')
                losses[slot] += loss
            del batch, x, y
            if cycle == 0:
                snapshot()  # first full batch is visible, too
            if len(pending) >= 256:
                check_source()
                flush()
        flush()
        summary['location'] = dict(epoch=epoch, phase='dev-score')
        scores, _ = score(dev)
        summary['curves'].append({'epoch': epoch, 'dev': scores, f'train_{loss_key}': {
            s: v / tr.batches_per_epoch for s, v in losses.items()}})
        if saved_test is not None:
            summary['last_dev'] = dict(epoch=epoch, dev=scores)
        snapshot()  # full DEV and counts survive before any best write
        if saved_test is not None:
            summary['curves'][-1]['routes'] = {
                slot: saved_test.observe_routes(slot) for slot in slots}
            snapshot()
        for slot in slots:
            val = scores[slot][loss_key]
            if slot in best and val >= best[slot][loss_key]:
                continue
            summary['location'] = dict(epoch=epoch, phase='best-save', slot=slot)
            check_source()
            meta = checkpoint_identity(recipe, source, backend, slot, epoch, val)
            raw = runtime.checkpoint(slot, meta)
            check()
            ref = store.write_model(slot, raw, replace=slot in best)
            best[slot] = dict(ref=ref, sha256=sha256(raw).hexdigest(), epoch=epoch,
                              dev_digest=summary['dev_panel']['digest'], **{loss_key: val})
            if saved_test is not None:
                best[slot]['dev'] = dict(scores[slot])
            del raw
            snapshot()
        if stopper is not None:
            reason = stopper.observe(epoch, {slot: scores[slot][loss_key] for slot in slots})
            if reason is not None:
                summary['training_termination'] = dict(reason=reason, completed_epochs=epoch)
                snapshot()  # Normal training end is evidence, not a run success seal.
                if reason == 'EARLY_STOPPING':
                    counted_recipe = replace(recipe, training=replace(tr, epochs=epoch))
                break
    if stopper is None and saved_test is not None:
        summary['training_termination'] = dict(reason='EPOCH_CEILING', completed_epochs=tr.epochs)
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
        runtime.load_best(slot, raw, recipe, source, selected['epoch'], selected[loss_key], check)
        del raw
        summary['persisted_reloads'][slot] = dict(selected)
        snapshot()
        if saved_test is not None:
            summary['reload_routes'][slot] = saved_test.observe_routes(slot)
            snapshot()
    summary['location'] = dict(phase='test-generate' if saved_test is None else 'test-reuse')
    check_source()
    test = (provider.panel('test') if saved_test is None
            else saved_test.open_panel(summary['persisted_reloads']))
    summary['location'] = dict(phase='test-score')
    scores, logits = score(test)
    summary['location'] = dict(phase='test-save')
    raw = panel_bytes(test, recipe, source, backend, logits=logits,
                      weights=summary['persisted_reloads'])
    snapshot()  # include both scoring passes, not TEST results
    ref = store.write_dataset(f'test-{task.cipher}', raw)
    check()
    panel_info = dict(ref=ref, sha256=sha256(raw).hexdigest(), digest=test.digest)
    if saved_test is None:
        summary['test_panel'] = panel_info
        summary.update(test_scores=scores, **test_result(test, scores, logits))
        snapshot()
    check_source()
    counts = resources.snapshot()
    totals = (estimate if saved_test is None else saved_test.estimate)(counted_recipe)
    if (any(counts[f'{kind}_{unit}'] != totals[unit]
            for kind in ('attempted', 'completed')
            for unit in ('real_blocks', 'optimizer_updates', 'forward_rows', 'null_words'))
            or ((stopper is not None or saved_test is not None)
                and (summary[completed_key] != totals['cycles']
                     or any(summary['updates'][slot][kind] != totals['updates_each']
                            for slot in slots for kind in ('attempted', 'completed'))))):
        raise ValueError('COUNT_MISMATCH')
    if saved_test is None:
        return None
    return dict(test_panel=panel_info, test_scores=scores, **test_result(test, scores, logits))
