"""Extend the swap closure with the paired owner and reused masking components."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _source

from ..present_card_swap.source import source_identity as swap
from ..source import no_reparse


def source_identity() -> dict[str, str]:
    here = Path(__file__).absolute().parent
    root = here.parents[1]
    paths = [here / f'{name}.py' for name in (
        '__init__', 'cli', 'config', 'report', 'runtime', 'source',
    )]
    paths.extend(here.parent / 'present_card_mask' / f'{name}.py' for name in (
        '__init__', 'config', 'masking', 'model', 'runtime', 'weights',
    ))
    paths.append(here.parent / 'mask_aux' / 'masking.py')
    out = swap()
    try:
        for path in paths:
            name = path.relative_to(root).as_posix()
            out[f'{root.name}/{name}'] = sha256(no_reparse(path).read_bytes()).hexdigest()
        _source(out)
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(out.items()))
