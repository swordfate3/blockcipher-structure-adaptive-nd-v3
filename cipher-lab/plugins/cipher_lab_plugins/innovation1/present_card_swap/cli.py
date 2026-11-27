"""Exact stdlib check and scoped two-fit admission."""
from __future__ import annotations

from pathlib import Path

from cipher_lab.artifacts import RunStore
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter

from ..prefix_bridge.cli import _device, _main, _run_owned
from ..present_card_fresh import config as pure
from . import config


def _run(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks) -> int:
    from .report import report_files
    from .source import source_identity

    def execute(meter: Meter, store: RunStore, summary: dict[str, object]) -> None:
        from cipher_lab_plugins.direct_block.cube_compare.preflight import gpu_peaks

        from ..mask_aux import kat
        from ..mask_aux_single import data
        from ..prefix_bridge.runtime import configure
        from . import training
        from .runtime import FitRuntime

        source = store.source_snapshot()
        summary['runtime'] = configure()

        def make(slot: str, arm: dict[str, object], train: bool, origin: int) -> FitRuntime:
            return FitRuntime(slot, arm, train, source=source, run_id=run_id, origin=origin)

        def panels_for(meter: Meter) -> data.Panels:
            return data.production_panels(meter, _card_recipe=config)

        training.run(meter, store, summary, gate=kat.production, panels_for=panels_for,
                     make=make, peaks=gpu_peaks)

    return _run_owned(root, run_id, recipe, policy, clocks, slots=config.SLOTS,
                      source_identity=source_identity, device=_device,
                      execute=execute, report=report_files)


def main(argv: list[str]) -> int:
    return _main(argv, prog='present-card-swap', recipe=config.recipe(),
                 wall=pure.WALL_SECONDS, cpu=pure.CPU_SECONDS, rss=pure.RSS_BYTES,
                 output=pure.OUTPUT_BYTES, caps=config.CAPS, clocks_factory=Clocks, runner=_run)
