"""Fixed plugin entry; static check does not import the runtime or claim a run."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from ..single_train.cli import Execute, Render, Source, finish_run, fixed_cli
from .config import RUN_ID, SLOTS, LayoutRecipe, load_config


def _wiring(recipe: LayoutRecipe) -> tuple[Execute, Source, Render]:
    from .application import execute
    from .report import render
    from .source import source_identity

    return partial(execute, recipe=recipe), source_identity, render


def _finish(store: Any, root: Path, meter: Any) -> tuple[int, str]:
    from .report import render

    return finish_run(store, root, meter, RUN_ID, render)


def main(argv: Sequence[str] | None = None) -> int:
    return fixed_cli(argv, name='layout_probe',
                     description='Fixed SKINNY single-pair A/A-prime scan comparison; no resume.',
                     load=load_config, run_id=lambda recipe: recipe.run_id,
                     slots=SLOTS, wiring=_wiring)
