"""Explicit source closure: existing shared dependencies plus this owner."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _source

from ..mask_aux.source import source_identity as group_identity
from ..source import no_reparse

_OWNER = ('__init__', 'cli', 'config', 'data', 'model', 'report',
          'runtime', 'source', 'training', 'weights')


def source_identity() -> dict[str, str]:
    result = group_identity()
    try:
        owner = Path(__file__).absolute().parent
        for module in _OWNER:
            filename = f'{module}.py'
            name = f'cipher_lab_plugins/innovation1/mask_aux_single/{filename}'
            result[name] = sha256(no_reparse(owner / filename).read_bytes()).hexdigest()
        _source(result)
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
