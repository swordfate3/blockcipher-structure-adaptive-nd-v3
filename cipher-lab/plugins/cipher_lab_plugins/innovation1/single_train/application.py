"""Wire only the approved GPU, fresh secret source, SKINNY oracle and two arms."""
from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Stopped

from ..resources import DETAIL_CODES, Resources, ResourceStop
from .config import ARCHITECTURE, BACKEND, DEVICE, FIXED, GPU_PCI, GPU_UUID, METHOD, SLOTS

if TYPE_CHECKING:
    from .config import SingleRecipe


def summary_for(backend: str, *, method: str = METHOD,
                slots: tuple[str, str] = SLOTS) -> dict[str, Any]:
    return dict(method=method, backend_identity=backend, synthetic=backend != BACKEND,
                scientific_validation=False, updates={s: dict(attempted=0, completed=0)
                for s in slots}, paired_completed=0, best={}, curves=[], pending=[], batch_refs=[],
                persisted_reloads={}, location=dict(phase='setup'))


_DEVICE_STAGES = frozenset({
    'entry-check', 'gate-setup', 'resource-setup', 'initial-resource-check', 'torch-import',
    'torch-configure', 'cuda-select', 'cuda-init', 'context-establish', 'context-attest',
    'identity-check', 'device-properties', 'memory-fraction', 'final-resource-check', 'returned',
})


def _seconds(value: Any) -> float | None:
    if type(value) in (float, int) and -2**63 < value < 2**63 and math.isfinite(value):
        return float(value)
    return None


def _public_fields(row: Any, *, text: dict[str, str], integers: tuple[str, ...]
                   ) -> dict[str, Any]:
    if type(row) is not dict:
        return {}
    public: dict[str, Any] = {}
    for field, pattern in text.items():
        value = row.get(field)
        if type(value) is str and len(value) <= 64 and re.fullmatch(pattern, value):
            public[field] = value
    for field in integers:
        value = row.get(field)
        if type(value) is int and 0 <= value < 2**63:
            public[field] = value
    return public


def admit_device(resources: Resources, evidence: dict[str, Any]) -> None:
    from ..application import production_resources

    stages: dict[str, Any] = {}
    evidence['stages'] = stages

    def observe(event: dict[str, Any]) -> None:
        stage, phase = event['stage'], event['event']
        last = _seconds(event['gate_last'])
        age = None
        try:
            resources.meter.check()
        finally:
            # Proof age at the existing observer gate check; no new GPU probe.
            age = None if last is None else time.monotonic() - last
            if stage in _DEVICE_STAGES and phase in ('begin', 'success', 'stop'):
                evidence['stage'] = stage
                sample = _public_fields(event.get('gate_latest'), text={
                    'uuid': r'GPU-[0-9a-f-]{36}',
                    'pci': r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',
                }, integers=('bytes',))
                if sample:
                    sample['sampled_at'] = last
                entry = dict(event=phase, elapsed_seconds=_seconds(event.get('elapsed_seconds')),
                             duration_seconds=_seconds(event.get('duration_seconds')),
                             last_sample=sample or None, proof_age_seconds_at_check=_seconds(age))
                detail = event.get('detail_code')
                if type(detail) is str and detail in DETAIL_CODES:
                    entry['detail_code'] = detail
                    evidence['detail_code'] = detail
                stages[stage] = entry
                attestation = _public_fields(event.get('attestation'), text={
                    'gpu_uuid': r'GPU-[0-9a-f-]{36}',
                    'pci_bus_id': r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',
                    'evidence': 'cuda-current-context', 'luid': r'[0-9a-f]{16}',
                }, integers=('pid', 'logical_device', 'device_node_mask'))
                if attestation:
                    evidence['attestation'] = attestation
        # The legacy admission still owns this gate. Do not renew/switch it.
        if phase != 'stop' and age is not None and age > 5.0:
            raise ResourceStop(detail_code='GPU_PROOF_STALE')

    resources.check()
    wired = production_resources(ARCHITECTURE, DEVICE, GPU_UUID, GPU_PCI, observer=observe)
    if wired.gpu is None:
        raise ResourceStop(detail_code='GPU_SAMPLE_INVALID')
    resources.gpu = wired.gpu  # never carry the legacy run's Meter into the new owner
    resources.check()


def _production(resources: Resources, evidence: dict[str, Any]) -> tuple[Any, Any]:
    admit_device(resources, evidence)
    from secrets import token_bytes

    from cipher_lab.ciphers import skinny

    from .data import Provider
    from .models import Runtime

    resources.check()
    runtime = Runtime(resources.check)
    provider = Provider(FIXED, skinny.encrypt_many, token_bytes, resources)
    return provider, runtime


Assemble = Callable[[Resources, dict[str, Any]], tuple[Any, Any]]
Train = Callable[["SingleRecipe", Any, Any, RunStore, Resources, dict[str, Any], str], None]


def _execute(recipe: SingleRecipe, store: RunStore, resources: Resources, backend: str,
             assemble: Assemble) -> None:
    def train(*args: Any) -> None:
        from .training import train as legacy_train

        legacy_train(*args)

    execute_pair(recipe, store, resources, backend, assemble, train, summary_for(backend))


def execute_pair(recipe: SingleRecipe, store: RunStore, resources: Resources, backend: str,
                 assemble: Assemble, train: Train, summary: dict[str, Any]) -> None:
    setup: dict[str, Any] = {}
    summary['device_setup'] = setup
    status = 'FAILED'
    try:
        resources.check()
        store.write_snapshot(dict(recipe=recipe.to_dict(), policy=dict(vars(recipe.resources)),
                                  summary=summary, counts=resources.snapshot()))
        resources.check()
        provider, runtime = assemble(resources, setup)
        resources.check()
        if setup:
            # Preserve bounded setup evidence before long DEV collection.
            store.write_snapshot(dict(summary=summary, counts=resources.snapshot()))
            resources.check()
        resources.check()
        train(recipe, provider, runtime, store, resources, summary, backend)
        resources.check()
        summary['reason'] = 'COMPLETE'
        status = 'SUCCEEDED'
    except ArtifactError:
        # RunStore owns its own I/O failure seal. Never attempt a second seal.
        return
    except (Stopped, ResourceStop) as error:
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
        store.finalize(dict(status=status, summary=summary, counts=resources.snapshot()))
    except ArtifactError:
        return


def execute(store: RunStore, resources: Resources) -> None:
    """The production command cannot inject a recipe, runtime, oracle or alternate backend."""
    _execute(FIXED, store, resources, BACKEND, _production)
