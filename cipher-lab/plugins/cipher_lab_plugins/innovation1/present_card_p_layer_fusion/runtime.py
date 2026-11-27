"""Arm-local swapping and bound checkpoints for the public fusion control."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from cipher_lab.execution import Meter

from ..checkpoint import canonical
from ..config import Task
from ..prefix_bridge.data import Floats, Labels
from ..prefix_bridge.model import initial_sha256
from ..present_card_fresh import config as pure_config
from ..present_card_fresh import model as pure_model
from ..present_card_fresh import runtime as pure_runtime
from ..present_card_fresh import weights as pure_weights
from ..present_card_swap import runtime as swap_runtime
from ..present_card_swap.views import swap_half
from . import model
from .wiring import BASE, gather_for, wiring_sha256


def _recipe_identity() -> tuple[str, dict[str, Any]]:
    from . import config

    return config.METHOD, config.recipe()


def _identity(slot: str, source: Mapping[str, str], run_id: str, epoch: int, bce: float,
              initial: str, geometry: Mapping[str, str]) -> dict[str, Any]:
    wiring = wiring_sha256(slot)
    identity = pure_weights.identity(pure_config.SLOT, source, run_id, epoch, bce, initial,
                                     geometry)
    method, recipe = _recipe_identity()
    identity.update(method=method, slot=slot, recipe=recipe, wiring_sha256=wiring)
    return identity


def _require_identity(slot: str, metadata: dict[str, Any]) -> None:
    try:
        expected = _identity(slot, metadata['source'], metadata['run_id'], metadata['epoch'],
                             metadata['bce'], metadata['initial_sha256'], metadata['geometry'])
        if canonical(expected) != canonical(metadata):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        raise ValueError('CHECKPOINT_IDENTITY') from None


class Runtime(swap_runtime.Runtime):
    _require_identity = staticmethod(_require_identity)
    backbone_initial_sha = ''
    fusion_initial_sha = ''

    def _input_domain(self, slot: str) -> tuple[Task, None]:
        gather_for(slot)
        return super()._input_domain(BASE)

    def _fresh_cpu(self) -> pure_model.Classifier:
        net = pure_model.build()
        self.backbone_initial_sha = initial_sha256(net)
        self.geometry = pure_model.geometry(net)
        model.add_fusion(net, self.slot)
        suffix = net.pair.shared.suffix
        self.fusion_initial_sha = (initial_sha256(suffix.fusion.projection)
                                   if isinstance(suffix, model.FusedSuffix) else '')
        self._initial = initial_sha256(net)
        self._parameters = sum(p.numel() for p in net.parameters())
        return net

    def batch_update(self, inputs: Floats, labels: Labels, meter: Meter) -> float:
        if len(inputs) != pure_config.BATCH_SIZE:
            raise ValueError('TRAIN_BATCH_SIZE')
        self._validate(inputs)
        inputs = swap_half(inputs, seed=swap_runtime._SWAP_SEED, step=self._local_step(meter))
        return pure_runtime.Runtime.batch_update(self, inputs, labels, meter)

    def _require_model_identity(self, metadata: dict[str, Any]) -> None:
        if self.geometry != metadata['geometry']:
            raise ValueError('MODEL_GEOMETRY_IDENTITY')
        if self._initial != metadata['initial_sha256']:
            raise ValueError('MODEL_INITIAL_IDENTITY')

    def dump_bound(self, metadata: dict[str, Any]) -> bytes:
        self._require_identity(self.slot, metadata)
        self._require_model_identity(metadata)
        return super().dump_bound(metadata)

    def restore_bound(self, raw: bytes, metadata: dict[str, Any]) -> None:
        super().restore_bound(raw, metadata)
        try:
            self._require_model_identity(metadata)
        except ValueError:
            self.models.clear()
            self._state = 'EMPTY'
            raise


class FitRuntime(swap_runtime.FitRuntime):
    """Bind all three slots to the same pure-card fit lifecycle."""

    _runtime: Runtime

    def __init__(self, slot: str, arm: dict[str, object], train: bool, *,
                 source: dict[str, str], run_id: str, origin: int) -> None:
        self._runtime = Runtime(slot, origin=origin)
        self._slot, self._arm, self._source, self._run_id = slot, arm, dict(source), run_id
        if train:
            self._runtime.start()

    @property
    def backbone_initial_sha(self) -> str:
        return self._runtime.backbone_initial_sha

    @property
    def backbone_geometry(self) -> dict[str, str]:
        return self.geometry

    @property
    def fusion_initial_sha(self) -> str:
        return self._runtime.fusion_initial_sha

    def _metadata(self, point: object) -> dict[str, Any]:
        if not isinstance(point, dict) or type(point.get('epoch')) is not int:
            raise ValueError('MISSING_DEV_SELECTION')
        step = point.get('step')
        if type(step) is not int or step != pure_config._steps(point['epoch']):
            raise ValueError('INVALID_DEV_STEP')
        return _identity(self._slot, self._source, self._run_id, point['epoch'], point['bce'],
                         cast(str, self._arm['initial_sha256']),
                         cast('dict[str, str]', self._arm['geometry']))
