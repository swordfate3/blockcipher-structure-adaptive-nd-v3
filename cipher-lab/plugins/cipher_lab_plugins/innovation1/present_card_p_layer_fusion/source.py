"""Pin the production source closure for the fusion experiment."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _source

from ..present_card_positions.source import source_identity as positions
from ..present_card_swap.source import source_identity as swap
from ..source import no_reparse


def source_identity() -> dict[str, str]:
    here = Path(__file__).absolute().parent
    root = here.parents[1]
    out = positions() | swap()
    modules = ('__init__', 'cli', 'config', 'model', 'report', 'runtime', 'source', 'wiring')
    try:
        for module in modules:
            path = here / f'{module}.py'
            out[f'{root.name}/{path.relative_to(root).as_posix()}'] = sha256(
                no_reparse(path).read_bytes()).hexdigest()
        _source(out)
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(out.items()))
