"""One device and resource lifecycle; assemble only the selected trainer."""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Clocks, Meter, Stopped

from . import resources
from .config import GPU_UUID, Recipe, require_official, validate_run_id
from .layout_probe.config import LayoutRecipe, official
from .resources import Resources, ResourceStop
from .single_train.config import BACKEND, DEVICE, make_policy


def make_resources(recipe: Recipe | LayoutRecipe) -> Resources:
    clocks = Clocks()
    policy = (make_policy(recipe, clocks) if isinstance(recipe, LayoutRecipe)
              else resources.policy(recipe, clocks))
    return Resources(Meter(policy, clocks, resources.production_rss_bytes))


@contextmanager
def _device_stage(detail: str) -> Iterator[None]:
    try:
        yield
    except (ResourceStop, Stopped):
        raise
    except Exception:
        raise ResourceStop(detail) from None


def prepare_device(recipe: Recipe | LayoutRecipe, gpu_uuid: str) -> None:
    if ('torch' in sys.modules or sys.platform not in ('linux', 'win32')
            or not GPU_UUID.fullmatch(gpu_uuid)):
        raise ResourceStop('DEVICE_ENTRY_REJECTED')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    with _device_stage('TORCH_IMPORT_FAILED'):
        import torch
    with _device_stage('CUDA_COUNT_FAILED'):
        if torch.cuda.device_count() != 1:
            raise ResourceStop('CUDA_COUNT_MISMATCH')
    limits = recipe.resources
    with _device_stage('TORCH_CONFIGURE_FAILED'):
        torch.set_num_threads(limits.intra_threads)
        torch.set_num_interop_threads(limits.inter_threads)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    with _device_stage('CUDA_SELECT_FAILED'):
        torch.cuda.set_device(0)
    with _device_stage('DEVICE_PROPERTIES_FAILED'):
        total = torch.cuda.get_device_properties(0).total_memory
        if type(total) is not int or total <= 0:
            raise ValueError
        if total < limits.gpu_bytes:
            raise ResourceStop('GPU_MEMORY_LIMIT')
    with _device_stage('MEMORY_FRACTION_FAILED'):
        torch.cuda.set_per_process_memory_fraction(limits.gpu_bytes / total, 0)


def _production_provider(recipe: Recipe, res: Resources) -> Any:
    from secrets import token_bytes

    from cipher_lab.ciphers import present, simeck, simon, skinny

    from .data import Provider

    return Provider(recipe, dict(SIMECK=simeck.encrypt_many, PRESENT=present.encrypt_many,
                                 SIMON=simon.encrypt_many, SKINNY=skinny.encrypt_many),
                    token_bytes, res)


def _assemble_four(recipe: Recipe, res: Resources) -> tuple[Any, Any]:
    from .models import TorchRuntime

    runtime = TorchRuntime(recipe, DEVICE)
    res.check()
    return _production_provider(recipe, res), runtime


def execute(recipe: Recipe | LayoutRecipe, store: RunStore, res: Resources, *,
            gpu_uuid: str, run_id: str) -> None:
    """No production overrides: a resolved recipe must match the frozen base."""
    validate_run_id(run_id)
    if store._run_id != run_id or not GPU_UUID.fullmatch(gpu_uuid):
        raise ValueError('RUN_IDENTITY')
    if isinstance(recipe, LayoutRecipe):
        official(recipe, run_id)
    else:
        require_official(recipe)

    launch = dict(gpu_uuid=gpu_uuid, device=DEVICE, gpu_memory_scope='process_allocator')
    # request is the stable saved request identity; a later storage seal keeps policy
    # even when the reduced failure summary drops launch.
    policy = dict(vars(recipe.resources), request=dict(launch))
    summary = dict(method=recipe.method, backend_identity=BACKEND, synthetic=False,
                   scientific_validation=False, launch=launch, location=dict(phase='device-setup'))
    status = 'FAILED'
    try:
        store.write_snapshot(dict(recipe=recipe.to_dict(), policy=policy,
                                  summary=summary, counts=res.snapshot()))
        res.check()
        prepare_device(recipe, gpu_uuid)
        res.check()
        summary['location'] = dict(phase='assembly')
        if isinstance(recipe, LayoutRecipe):
            from .layout_probe.application import _execute
        else:
            from .training import run

            provider, runtime = _assemble_four(recipe, res)
            res.check()
    except ArtifactError:
        return
    except (ResourceStop, Stopped) as error:
        status = 'STOPPED'
        summary['reason'] = error.reason
        if isinstance(error, ResourceStop):
            summary['detail_code'] = error.detail_code
    except KeyboardInterrupt:
        status = 'STOPPED'
        summary['reason'] = 'INTERRUPTED'
    except Exception:
        summary['reason'] = 'SETUP_FAILED'
    else:
        if isinstance(recipe, LayoutRecipe):
            _execute(recipe, store, res, BACKEND, launch=launch)
        else:
            run(recipe, provider, runtime, store, res, backend=BACKEND, launch=launch)
        return
    try:
        store.finalize(dict(status=status, summary=summary, counts=res.snapshot()))
    except ArtifactError:
        return
