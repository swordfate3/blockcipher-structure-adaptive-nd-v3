"""The reviewed card closure plus the exact auxiliary modules used by this recipe."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _source

from ..present_card_fresh.source import source_identity as pure_identity
from ..source import no_reparse

_OWNED = ('__init__', 'cli', 'config', 'masking', 'model', 'report', 'runtime', 'source', 'weights')


def source_identity() -> dict[str, str]:
    tree = Path(__file__).absolute().parent
    result = pure_identity()
    extras = (*(tree / (name + '.py') for name in _OWNED),
              tree.parent / 'mask_aux' / 'masking.py',
              tree.parent / 'present_card_fresh' / 'report.py')
    try:
        for path in extras:
            key = path.relative_to(tree.parents[1]).as_posix()
            source_key = f'{tree.parents[1].name}/{key}'
            result[source_key] = sha256(no_reparse(path).read_bytes()).hexdigest()
        _source(result)
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
