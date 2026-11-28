"""Static recipe admission and owned run assembly."""
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

        from ..prefix_bridge.runtime import configure
        from ..present_card_swap import training
        from . import data, gate
        from .runtime import FitRuntime

        source = store.source_snapshot()
        summary['runtime'] = configure()

        def make(slot: str, arm: dict[str, object], train: bool, origin: int) -> FitRuntime:
            return FitRuntime(slot, arm, train, source=source, run_id=run_id, origin=origin)

        training.run(meter, store, summary, gate=gate.production, panels_for=data.production_panels,
                     make=make, peaks=gpu_peaks, _recipe=config, separate_train=True,
                     train_dev_hashes=config.TRAIN_DEV_HASHES,
                     gate_name=config.KAT_NAME, gate_blocks=config.KAT_BLOCKS)

    return _run_owned(root, run_id, recipe, policy, clocks, slots=config.SLOTS,
                      source_identity=source_identity, device=_device,
                      execute=execute, report=report_files)


def main(argv: list[str]) -> int:
    return _main(argv, prog='present-card-per-pair-key', recipe=config.recipe(),
                 wall=pure.WALL_SECONDS, cpu=pure.CPU_SECONDS, rss=pure.RSS_BYTES,
                 output=pure.OUTPUT_BYTES, caps=config.CAPS, clocks_factory=Clocks, runner=_run)
