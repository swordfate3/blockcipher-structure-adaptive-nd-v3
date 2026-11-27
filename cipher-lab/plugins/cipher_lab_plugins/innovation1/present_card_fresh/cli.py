"""Explicit stdlib check and concrete single-owner run admission."""
from __future__ import annotations

from pathlib import Path

from cipher_lab.artifacts import RunStore
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Meter

from ..prefix_bridge.cli import _device, _main, _run_owned
from . import config


def _run(root: Path, run_id: str, recipe: dict[str, object], policy: Policy, clocks: Clocks) -> int:
    from .source import source_identity

    def execute(meter: Meter, store: RunStore, summary: dict[str, object]) -> None:
        from cipher_lab_plugins.direct_block.cube_compare.preflight import gpu_peaks

        from ..mask_aux import kat
        from ..prefix_bridge.runtime import configure
        from . import training
        from .data import production_panels
        from .runtime import FitRuntime

        summary['runtime'] = configure()
        source = store.source_snapshot()

        def make(slot: str, arm: dict[str, object], train: bool) -> FitRuntime:
            return FitRuntime(slot, arm, train, source=source, run_id=run_id)

        training.run(meter, store, summary, gate=kat.production, make=make,
                     make_panels=production_panels, peaks=gpu_peaks)

    def report(saved: dict[str, object]) -> dict[str, bytes]:
        from .report import report_files

        return report_files(saved)

    return _run_owned(root, run_id, recipe, policy, clocks, slots=config.SLOTS,
                      source_identity=source_identity, device=_device,
                      execute=execute, report=report)


def main(argv: list[str]) -> int:
    return _main(argv, prog='present-card-fresh-1m', recipe=config.recipe(),
                 wall=config.WALL_SECONDS, cpu=config.CPU_SECONDS, rss=config.RSS_BYTES,
                 output=config.OUTPUT_BYTES, caps=config.CAPS, clocks_factory=Clocks, runner=_run)
