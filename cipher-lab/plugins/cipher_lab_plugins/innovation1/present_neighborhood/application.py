"""Assemble PRESENT-specific adapters, then run the existing training loop."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Stopped

from ..four_single.replay import TestSourceError
from ..layout_probe.reuse import PoolSourceError
from ..resources import Resources, ResourceStop
from ..single_train.config import BACKEND, SingleRecipe
from ..single_train.data import PairBatch
from ..single_train.training import SavedTest, train_pair
from .config import ROW_BYTES, TASK, NeighborhoodRecipe, estimate, official
from .data import NeighborhoodPools, encode_test
from .runtime import NeighborhoodRuntime


def _assemble(recipe: NeighborhoodRecipe, check: Callable[[], None]
              ) -> tuple[NeighborhoodPools, NeighborhoodRuntime]:
    pools = NeighborhoodPools(recipe, check)
    return pools, NeighborhoodRuntime(recipe, check, pools.pool_provenance)


def execute(recipe: NeighborhoodRecipe, store: RunStore, resources: Resources, *,
            launch: dict[str, str], backend: str = BACKEND,
            assemble: Callable[[NeighborhoodRecipe, Callable[[], None]],
                               tuple[NeighborhoodPools, NeighborhoodRuntime]] | None = None
            ) -> None:
    if recipe.run_id != store._run_id:
        raise ValueError('RUN_IDENTITY')
    if backend == BACKEND and assemble is None:
        official(recipe, store._run_id)
        assemble = _assemble
    elif backend != 'artificial' or assemble is None:
        raise ValueError('BACKEND_MISMATCH')
    slots = recipe.slots
    summary: dict[str, Any] = dict(method=recipe.method, backend_identity=backend,
        synthetic=backend != BACKEND, scientific_validation=False, launch=launch,
        updates={s: dict(attempted=0, completed=0) for s in slots},
        best={}, curves=[], pending=[], batch_refs=[], persisted_reloads={}, reload_routes={},
        location=dict(phase='assembly'))
    summary['completed_batches' if len(slots) == 1 else 'paired_completed'] = 0
    status = 'FAILED'
    result: dict[str, Any] | None = None
    try:
        resources.check()
        policy = dict(vars(recipe.resources), request=dict(launch),
                      deadline_utc=resources.meter._policy.deadline_utc)
        store.write_snapshot(dict(recipe=recipe.to_dict(), policy=policy,
                                  summary=summary, counts=resources.snapshot()))
        pools, runtime = assemble(recipe, resources.check)
        resources.check()

        def panel_bytes(batch: PairBatch, rec: SingleRecipe, source: dict[str, str],
                        backend: str, *, logits: dict[str, Any] | None = None,
                        weights: dict[str, Any] | None = None) -> bytes:
            if rec is not recipe or logits is None or weights is None or pools.test_input is None:
                raise ValueError('TEST_PANEL_IDENTITY')
            return encode_test(batch, recipe, source, backend, logits=logits,
                               weights=weights, reused_input=pools.test_input)

        def test_result(batch: PairBatch, scores: dict[str, dict[str, float]],
                        logits: dict[str, Any]) -> dict[str, Any]:
            if len(slots) == 1:
                return dict(test_input=pools.test_input)
            a, b = (scores[slot] for slot in slots)
            return dict(test_input=pools.test_input, paired_test=dict(rows=batch.groups,
                direction='B-minus-A', delta_acc_pp=100 * (b['accuracy'] - a['accuracy']),
                delta_bce=b['bce'] - a['bce']))

        def totals(rec: SingleRecipe) -> dict[str, int]:
            if rec is not recipe:
                raise ValueError('NEIGHBORHOOD_RECIPE')
            return estimate(recipe)

        result = train_pair(recipe, pools, runtime, store, resources, summary, backend, slots=slots,
                   task=TASK, panel_bytes=panel_bytes, checkpoint_identity=runtime.identity,
                   test_result=test_result,
                   saved_test=SavedTest(pools.load_test, runtime.routes, totals, ROW_BYTES))
        resources.check()
        if result is None:
            raise ValueError('TEST_EVIDENCE_MISSING')
        summary['reason'] = 'COMPLETE'
        status = 'SUCCEEDED'
    except ArtifactError:
        if store._sealed:
            return
        summary['reason'] = 'ARTIFACT_FAILED'
    except TestSourceError:
        summary['reason'] = 'TEST_SOURCE_INVALID'
    except PoolSourceError:
        summary['reason'] = 'POOL_SOURCE_INVALID'
    except (ResourceStop, Stopped) as error:
        status = 'STOPPED'
        summary['reason'] = error.reason
        if isinstance(error, ResourceStop):
            summary['detail_code'] = error.detail_code
    except KeyboardInterrupt:
        status = 'STOPPED'
        summary['reason'] = 'INTERRUPTED'
    except Exception:
        summary['reason'] = 'EXECUTION_FAILED'
    try:
        terminal = dict(summary, **result) if status == 'SUCCEEDED' and result else summary
        store.finalize(dict(status=status, summary=terminal, counts=resources.snapshot()))
    except ArtifactError:
        return
