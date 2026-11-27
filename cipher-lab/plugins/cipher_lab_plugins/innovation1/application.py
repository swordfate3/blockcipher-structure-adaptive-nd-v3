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
from .four_single.config import FourSingleRecipe, ReplayRecipe, validate_replay_root
from .four_single.config import make_policy as make_four_single_policy
from .four_single.config import official as official_four_single
from .four_single.interleave_config import InterleaveRecipe
from .four_single.interleave_config import make_policy as make_interleave_policy
from .four_single.interleave_config import validate_launch as validate_interleave_launch
from .layout_probe.config import LayoutRecipe, official
from .present_neighborhood.config import GPU_UUID as NEIGHBORHOOD_GPU_UUID
from .present_neighborhood.config import NeighborhoodRecipe, PresentCardRecipe
from .present_neighborhood.config import make_policy as make_neighborhood_policy
from .present_neighborhood.config import official as official_neighborhood
from .present_neighborhood.config import validate_root as validate_neighborhood_root
from .resources import Resources, ResourceStop
from .serial_bc.config import (
    DEADLINE_ENV,
    INDEPENDENT_ROOT,
    VOLUME_GPU_UUID,
    VOLUME_ROOT,
    IndependentRecipe,
    SerialRecipe,
    VolumeRecipe,
)
from .serial_bc.config import make_policy as make_serial_policy
from .serial_bc.config import official as official_serial
from .serial_bc.config import validate_root as validate_serial_root
from .single_train.config import BACKEND, DEVICE, make_policy
from .skinny_card.config import GPU_UUID as CARD_GPU_UUID
from .skinny_card.config import CardRecipe
from .skinny_card.config import make_policy as make_card_policy
from .skinny_card.config import official as official_card
from .skinny_card.config import validate_root as validate_card_root


def make_resources(recipe: Recipe | LayoutRecipe | FourSingleRecipe) -> Resources:
    clocks = Clocks()
    policy = (make_card_policy(recipe, clocks, os.environ.get(DEADLINE_ENV))
              if isinstance(recipe, CardRecipe)
              else make_neighborhood_policy(recipe, clocks, os.environ.get(DEADLINE_ENV))
              if isinstance(recipe, NeighborhoodRecipe)
              else make_interleave_policy(recipe, clocks, os.environ.get(DEADLINE_ENV))
              if isinstance(recipe, InterleaveRecipe)
              else make_serial_policy(recipe, clocks, os.environ.get(DEADLINE_ENV))
              if isinstance(recipe, SerialRecipe)
              else make_policy(recipe, clocks) if isinstance(recipe, LayoutRecipe)
              else make_four_single_policy(recipe, clocks) if isinstance(recipe, FourSingleRecipe)
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


def prepare_device(recipe: Recipe | LayoutRecipe | FourSingleRecipe, gpu_uuid: str) -> None:
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


def execute(recipe: Recipe | LayoutRecipe | FourSingleRecipe, store: RunStore, res: Resources, *,
            gpu_uuid: str, run_id: str) -> None:
    """No production overrides: a resolved recipe must match the frozen base."""
    validate_run_id(run_id)
    if store._run_id != run_id or not GPU_UUID.fullmatch(gpu_uuid):
        raise ValueError('RUN_IDENTITY')
    if isinstance(recipe, LayoutRecipe):
        official(recipe, run_id)
    elif isinstance(recipe, SerialRecipe):
        official_serial(recipe, run_id)
        if isinstance(recipe, VolumeRecipe):
            validate_serial_root(store._root, expected=VOLUME_ROOT)
            if gpu_uuid.casefold() != VOLUME_GPU_UUID.casefold():
                raise ValueError('VOLUME_GPU_UUID')
        elif isinstance(recipe, IndependentRecipe):
            validate_serial_root(store._root, expected=INDEPENDENT_ROOT)
        else:
            validate_serial_root(store._root)
    elif isinstance(recipe, NeighborhoodRecipe):
        official_neighborhood(recipe, run_id)
        if isinstance(recipe, PresentCardRecipe):
            validate_neighborhood_root(store._root, recipe=recipe)
        else:
            validate_neighborhood_root(store._root)
        if gpu_uuid.casefold() != NEIGHBORHOOD_GPU_UUID.casefold():
            raise ValueError('NEIGHBORHOOD_GPU_UUID')
    elif isinstance(recipe, CardRecipe):
        official_card(recipe, run_id)
        validate_card_root(store._root)
        if gpu_uuid.casefold() != CARD_GPU_UUID.casefold():
            raise ValueError('CARD_GPU_UUID')
    elif isinstance(recipe, FourSingleRecipe):
        official_four_single(recipe, run_id)
        if isinstance(recipe, InterleaveRecipe):
            validate_interleave_launch(store._root, gpu_uuid)
        elif isinstance(recipe, ReplayRecipe):
            validate_replay_root(store._root)
    else:
        require_official(recipe)

    launch = dict(gpu_uuid=gpu_uuid, device=DEVICE, gpu_memory_scope='process_allocator')
    # request is the stable saved request identity; a later storage seal keeps policy
    # even when the reduced failure summary drops launch.
    policy = dict(vars(recipe.resources), request=dict(launch))
    if isinstance(recipe, (SerialRecipe, CardRecipe, NeighborhoodRecipe, InterleaveRecipe)):
        policy['deadline_utc'] = res.meter._policy.deadline_utc
    summary = dict(method=recipe.method, backend_identity=BACKEND, synthetic=False,
                   scientific_validation=False, launch=launch, location=dict(phase='device-setup'))
    if isinstance(recipe, VolumeRecipe):
        summary['dev_observations'] = []
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
        elif isinstance(recipe, SerialRecipe):
            from .serial_bc.training import execute as execute_serial
        elif isinstance(recipe, NeighborhoodRecipe):
            from .present_neighborhood.application import execute as execute_neighborhood
        elif isinstance(recipe, CardRecipe):
            from .skinny_card.application import execute as execute_card
        elif isinstance(recipe, FourSingleRecipe):
            from .four_single.training import execute as execute_four_single
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
        elif isinstance(recipe, SerialRecipe):
            execute_serial(recipe, store, res, launch=launch)
        elif isinstance(recipe, NeighborhoodRecipe):
            execute_neighborhood(recipe, store, res, launch=launch)
        elif isinstance(recipe, CardRecipe):
            execute_card(recipe, store, res, launch=launch)
        elif isinstance(recipe, FourSingleRecipe):
            execute_four_single(recipe, store, res, launch=launch)
        else:
            run(recipe, provider, runtime, store, res, backend=BACKEND, launch=launch)
        return
    try:
        store.finalize(dict(status=status, summary=summary, counts=res.snapshot()))
    except ArtifactError:
        return
