"""Extend the released closure without changing its source identity function."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _source

from ..single_train.source import no_reparse
from ..single_train.source import source_identity as single_source

_FILES = ('__init__', 'application', 'cli', 'config', 'data', 'layout', 'metrics', 'models',
          'report', 'runtime', 'source', 'weights')


def source_identity() -> dict[str, str]:
    result = single_source()
    here = Path(__file__).absolute().parent
    try:
        no_reparse(here)
        for name in _FILES:
            path = no_reparse(here / f'{name}.py')
            result[path.relative_to(here.parents[2]).as_posix()] = sha256(
                path.read_bytes()).hexdigest()
        _source(result)
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
