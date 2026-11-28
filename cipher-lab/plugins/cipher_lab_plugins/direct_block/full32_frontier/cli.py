"""Bind the fixed full-word recipe to the common run owner."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter

from ...innovation1.prefix_bridge.cli import _device, _main, _run_owned
from ..bit_structure.source import source_identity as shared_identity
from . import config


def source_identity() -> dict[str, str]:
    result = shared_identity()
    prefix = 'cipher_lab_plugins/direct_block/full32_frontier/'
    modules = ('__init__', 'cli', 'config', 'data', 'metrics', 'model', 'report', 'training')
    if any(f'{prefix}{name}.py' not in result for name in modules):
        raise ArtifactError('SOURCE_UNAVAILABLE')
    return result


def _run(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks) -> int:
    from .report import report_files

    def execute(meter: Meter, store: RunStore, summary: dict[str, object]) -> None:
        import torch

        from ..cube_compare.checkpoint import dump
        from ..cube_compare.config import GPU_DEVICE
        from ..cube_compare.engine import TorchEngine, configure_runtime
        from ..cube_compare.preflight import gpu_peaks
        from ..split8.kat import production
        from . import data, model, training

        summary['runtime'] = configure_runtime()

        def make(depth: int, slot: str, arm: dict[str, object], seed: int) -> TorchEngine:
            network = model.build(slot.rsplit('-', 1)[1], seed, rounds=depth)
            arm['parameters'] = sum(p.numel() for p in network.parameters() if p.requires_grad)
            arm['initial_sha256'] = sha256(dump(network, identity='initial')).hexdigest()
            return TorchEngine(network, torch.device(GPU_DEVICE),
                               identity=f'{config.METHOD}/{slot}', width=32)

        training.run(meter, store, summary, gate=production, make_panels=data.production_panels,
                     make=make, read_snapshot=lambda: read_saved(root, run_id), peaks=gpu_peaks)

    return _run_owned(root, run_id, recipe, policy, clocks, slots=config.SLOTS,
                      source_identity=source_identity, device=_device,
                      execute=execute, report=report_files)


def main(argv: list[str]) -> int:
    return _main(argv, prog='simon32-full32-frontier', recipe=config.recipe(),
                 wall=config.WALL_SECONDS, cpu=config.CPU_SECONDS, rss=config.RSS_BYTES,
                 output=config.OUTPUT_BYTES, caps=config.CAPS, clocks_factory=Clocks, runner=_run)
