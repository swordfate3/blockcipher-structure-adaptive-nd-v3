"""Lazy runtime assembly. This is the only private RNG / real-cipher wiring point."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, cast

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.execution import Clocks, Meter, Stopped

from .config import Recipe
from .resources import (
    GpuGate,
    Resources,
    ResourceStop,
    gpu_sample,
    manual_resources,
    pci,
    policy,
    production_rss_bytes,
)


@dataclass(frozen=True)
class Execution:
    """Scientific result plus retained clocks/RSS lifecycle for command closeout.

    Not a resume token or a new owner. Only CLI uses resources after science is
    sealed; it must never finalize again when report-phase checks stop.
    """
    result: dict[str, Any]
    resources: Resources | None


def production_resources(recipe: Recipe, device: str, gpu_uuid: str,
                         gpu_pci: str) -> Resources:
    if (device != 'cuda:0' or sys.platform not in ('linux', 'win32')
            or 'torch' in sys.modules or not gpu_uuid or not gpu_pci):
        # No re-binding an already imported CUDA runtime or silent CPU fallback.
        raise ResourceStop()
    gate = GpuGate(gpu_uuid, gpu_pci, recipe.resources.gpu_bytes,
                   lambda: gpu_sample(gpu_uuid))
    resources = Resources(Meter(policy(recipe), Clocks(), production_rss_bytes), gate)
    resources.check()  # actual telemetry required before numerical imports/constructors
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    import torch

    from .device import actual_attestor
    try:
        if torch.cuda.device_count() != 1:
            raise ResourceStop()
        torch.set_num_threads(recipe.resources.intra_threads)
        torch.set_num_interop_threads(recipe.resources.inter_threads)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)
        cast(Any, torch.cuda).init()
        evidence = actual_attestor(torch)
        if (evidence['gpu_uuid'] != gpu_uuid or pci(evidence['pci_bus_id']) != pci(gpu_pci)
                or evidence['logical_device'] != 0 or evidence['pid'] != os.getpid()
                or evidence['evidence'] != 'cuda-current-context'):
            raise ResourceStop()
        total = torch.cuda.get_device_properties(0).total_memory
        if total < recipe.resources.gpu_bytes:
            raise ResourceStop('GPU_LIMIT')
        torch.cuda.set_per_process_memory_fraction(recipe.resources.gpu_bytes / total, 0)
        resources.check()
        return resources
    except Exception:
        raise ResourceStop() from None


def _production_provider(recipe: Recipe, resources: Resources) -> Any:
    from secrets import token_bytes

    from cipher_lab.ciphers import present, simeck, simon, skinny

    from .data import Provider
    return Provider(recipe, dict(SIMECK=simeck.encrypt_many, PRESENT=present.encrypt_many,
                                 SIMON=simon.encrypt_many, SKINNY=skinny.encrypt_many),
                    token_bytes, resources)


def _setup_failure(store: RunStore, recipe: Recipe | None, backend: str,
                   error: Exception, resources: Resources | None) -> dict[str, Any]:
    stopped = isinstance(error, (ResourceStop, Stopped))
    summary = dict(method=recipe.method if recipe else 'innovation1-explicit-evaluation-v1',
                   backend_identity=backend, synthetic=backend == 'demo',
                   scientific_validation=False,
                   reason=(error.reason if isinstance(error, (Stopped, ResourceStop))
                           else 'SETUP_FAILED'))
    result = dict(status='STOPPED' if stopped else 'FAILED', summary=summary,
                  counts={} if resources is None else resources.snapshot())
    try:
        store.finalize(result)
    except ArtifactError:
        pass
    return result


def execute(recipe: Recipe, store: RunStore, *, backend: str, device: str,
            gpu_uuid: str = '', gpu_pci: str = '') -> Execution:
    resources: Resources | None = None
    models: Any
    provider: Any
    try:
        store.write_snapshot({'recipe': recipe.to_dict(), 'summary': dict(
            method=recipe.method, backend_identity=backend, synthetic=backend == 'demo',
            scientific_validation=False)})
        if backend == 'demo' and device == 'cpu':
            from .synthetic import ManualProvider, ManualRuntime
            resources = manual_resources(recipe)
            provider, models = ManualProvider(recipe, resources), ManualRuntime(recipe)
        elif backend == 'ciphers':
            resources = production_resources(recipe, device, gpu_uuid, gpu_pci)
            from .models import TorchRuntime
            models = TorchRuntime(recipe, device)
            resources.check()
            provider = _production_provider(recipe, resources)
        else:
            raise ValueError('BACKEND_DEVICE')
    except ArtifactError:
        return Execution(dict(status='FAILED', reason='ARTIFACT_FAILED'), resources)
    except Exception as error:
        return Execution(_setup_failure(store, recipe, backend, error, resources), resources)
    from .training import run
    return Execution(run(recipe, provider, models, store, resources, backend=backend), resources)


def execute_evaluation(args: Any, store: RunStore) -> Execution:
    resources: Resources | None = None
    try:
        from .config import _decode, validate
        saved = read_saved(args.run_root, args.run_id)
        if saved['run_sha256'] != args.run_sha256:
            raise ValueError('RUN_SHA')
        recipe = _decode(saved['recipe'], Recipe(), 'recipe')
        validate(recipe)
        resources = (manual_resources(recipe) if args.backend == 'demo' else
                     production_resources(recipe, args.device, args.gpu_uuid, args.gpu_pci))
        from .evaluate import evaluate
        result = evaluate(args.run_root, args.run_id, args.run_sha256,
                        args.checkpoint_ref, args.checkpoint_sha256, args.dataset_ref,
                        args.dataset_sha256, store, resources, backend=args.backend,
                        device=args.device, max_forward_rows=args.max_forward_rows,
                        batch_groups=args.batch_groups)
        return Execution(result, resources)
    except ArtifactError:
        # Input read errors have not sealed output owner; keep UNKNOWN rather than
        # second-finalizing an output error. Caller reports saved truth only.
        return Execution(dict(status='FAILED', reason='ARTIFACT_FAILED'), resources)
    except Exception as error:
        return Execution(_setup_failure(store, None, args.backend, error, resources), resources)
