"""Lazy production assembly and layout-specific bindings for the shared loop."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Stopped

from ..resources import Resources, ResourceStop
from ..single_train.config import BACKEND
from .config import SLOTS, LayoutRecipe, official

if TYPE_CHECKING:
    from ..single_train.config import SingleRecipe
    from ..single_train.data import PairBatch


def assemble(recipe: LayoutRecipe, resources: Resources) -> tuple[Any, Any]:
    from secrets import token_bytes

    from cipher_lab.ciphers import skinny

    from .data import FixedPoolProvider, Provider
    from .runtime import MSERuntime, Runtime

    resources.check()
    if recipe.reuse_pool:
        from .reuse import ReusedPoolProvider

        return ReusedPoolProvider(recipe, skinny.encrypt_many, token_bytes, resources), Runtime(
            resources.check)
    runtime_type, provider_type = ((MSERuntime, FixedPoolProvider) if recipe.fixed_pool
                                   else (Runtime, Provider))
    runtime = runtime_type(resources.check)
    return provider_type(recipe, skinny.encrypt_many, token_bytes, resources), runtime


def _train(recipe: SingleRecipe, provider: Any, runtime: Any, store: RunStore,
           resources: Resources, summary: dict[str, Any], backend: str) -> None:
    from ..single_train.training import train_pair
    from .data import encode_panel
    from .metrics import paired_statistics
    from .weights import identity

    if type(recipe) is not LayoutRecipe:
        raise ValueError('LAYOUT_RECIPE')

    def test_result(panel: PairBatch, scores: dict[str, dict[str, float]],
                    logits: dict[str, Any]) -> dict[str, Any]:
        paired = paired_statistics(panel.labels, logits[SLOTS[0]], logits[SLOTS[1]])
        baseline = math.log(2.)
        return dict(paired_test=paired, constant_bce=baseline,
                    bce_minus_constant={slot: scores[slot]['bce'] - baseline for slot in SLOTS})

    def panel_bytes(batch: PairBatch, rec: SingleRecipe, source: dict[str, str],
                    backend: str, *, logits: dict[str, Any] | None = None,
                    weights: dict[str, Any] | None = None) -> bytes:
        # Copy only the small verified provenance, not the old dataset bytes.
        provenance = {k: summary[k] for k in ('pool_source', 'train_panel', 'dev_panel')
                      } if recipe.reuse_pool else None
        return encode_panel(batch, rec, source, backend, logits=logits, weights=weights,
                            reuse_provenance=provenance)

    train_pair(recipe, provider, runtime, store, resources, summary, backend, slots=SLOTS,
               panel_bytes=panel_bytes, checkpoint_identity=identity, test_result=test_result,
               early_stopping=recipe.early_stopping, cosine_epochs=recipe.cosine_epochs)


def _execute(recipe: LayoutRecipe, store: RunStore, resources: Resources, backend: str,
             assemble_pair: Callable[[LayoutRecipe, Resources], tuple[Any, Any]] | None = None, *,
             launch: dict[str, str] | None = None) -> None:
    # Only the guarded synthetic loop tests use the artificial identity.
    if recipe.run_id != store._run_id:
        raise ValueError('RUN_IDENTITY')
    if backend == BACKEND and assemble_pair is None:
        official(recipe, store._run_id)
        assemble_pair = assemble
    elif backend != 'artificial' or assemble_pair is None:
        raise ValueError('BACKEND_MISMATCH')
    summary: dict[str, Any] = dict(method=recipe.method, backend_identity=backend,
        synthetic=backend != BACKEND, scientific_validation=False,
        updates={s: dict(attempted=0, completed=0) for s in SLOTS},
        paired_completed=0, best={}, curves=[], pending=[], batch_refs=[], persisted_reloads={},
        location=dict(phase='assembly'))
    if launch is not None:
        summary['launch'] = launch
    from .reuse import PoolSourceError

    status = 'FAILED'
    try:
        resources.check()
        policy = dict(vars(recipe.resources))
        if launch is not None:
            policy['request'] = dict(launch)  # never overwrite the saved request identity
        store.write_snapshot(dict(recipe=recipe.to_dict(), policy=policy,
                                  summary=summary, counts=resources.snapshot()))
        resources.check()
        provider, runtime = assemble_pair(recipe, resources)
        resources.check()
        _train(recipe, provider, runtime, store, resources, summary, backend)
        resources.check()
        summary['reason'] = 'COMPLETE'
        status = 'SUCCEEDED'
    except ArtifactError:
        return  # storage owns its failure transition; no second seal
    except PoolSourceError:
        summary['reason'] = 'POOL_SOURCE_INVALID'
    except (ResourceStop, Stopped) as error:
        status = 'STOPPED'
        summary['reason'] = error.reason
    except KeyboardInterrupt:
        status = 'STOPPED'
        summary['reason'] = 'INTERRUPTED'
    except Exception:
        summary['reason'] = 'EXECUTION_FAILED'
    try:
        store.finalize(dict(status=status, summary=summary, counts=resources.snapshot()))
    except ArtifactError:
        return
