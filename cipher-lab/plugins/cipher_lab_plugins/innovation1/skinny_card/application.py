"""Bind the archive, checkpoint identity and observations to the shared pair loop."""
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
from .config import ROW_BYTES, SLOTS, CardRecipe, estimate, official
from .data import CardPools, encode_test
from .runtime import CardRuntime


def _assemble(recipe: CardRecipe, check: Callable[[], None]) -> tuple[CardPools, CardRuntime]:
    pools = CardPools(recipe, check)
    return pools, CardRuntime(recipe, check, pools.pool_provenance)


def execute(recipe: CardRecipe, store: RunStore, resources: Resources, *,
            launch: dict[str, str], backend: str = BACKEND,
            assemble: Callable[[CardRecipe, Callable[[], None]],
                               tuple[CardPools, CardRuntime]] | None = None) -> None:
    if recipe.run_id != store._run_id:
        raise ValueError('RUN_IDENTITY')
    if backend == BACKEND and assemble is None:
        official(recipe, store._run_id)
        assemble = _assemble
    elif backend != 'artificial' or assemble is None:
        raise ValueError('BACKEND_MISMATCH')
    summary: dict[str, Any] = dict(method=recipe.method, backend_identity=backend,
        synthetic=backend != BACKEND, scientific_validation=False, launch=launch,
        updates={s: dict(attempted=0, completed=0) for s in SLOTS}, paired_completed=0,
        best={}, curves=[], pending=[], batch_refs=[], persisted_reloads={}, reload_routes={},
        location=dict(phase='assembly'))
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
            b, c = (scores[slot] for slot in SLOTS)
            return dict(test_input=pools.test_input, paired_test=dict(rows=batch.groups,
                direction='C-minus-B', delta_acc_pp=100 * (c['accuracy'] - b['accuracy']),
                delta_bce=c['bce'] - b['bce']))

        def totals(rec: SingleRecipe) -> dict[str, int]:
            if not isinstance(rec, CardRecipe):
                raise ValueError('CARD_RECIPE')
            return estimate(rec)

        result = train_pair(recipe, pools, runtime, store, resources, summary, backend, slots=SLOTS,
                   panel_bytes=panel_bytes, checkpoint_identity=runtime.identity,
                   test_result=test_result, early_stopping=recipe.early_stopping,
                   cosine_epochs=recipe.cosine_epochs,
                   saved_test=SavedTest(pools.load_test, runtime.routes, totals, ROW_BYTES))
        resources.check()
        if result is None:
            raise ValueError('TEST_EVIDENCE_MISSING')
        summary['reason'] = 'COMPLETE'
        status = 'SUCCEEDED'
    except ArtifactError:
        if store._sealed:
            return  # the owner already attempted its storage failure seal
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
