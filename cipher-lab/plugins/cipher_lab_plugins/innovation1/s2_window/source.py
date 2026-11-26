"""Claim-time and fresh production closure, extending the actual prefix owner."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _source

from ..prefix_bridge.source import source_identity as prefix_identity
from ..source import no_reparse

_OWNER = ('__init__', 'cli', 'config', 'data', 'kat', 'report', 'runtime', 'source',
          'training', 'weights')


def source_identity() -> dict[str, str]:
    result = prefix_identity()
    # The prefix closure hashes the actual core tree (including the new cipher).
    if 'cipher_lab/ciphers/present_s2.py' not in result:
        raise ArtifactError('SOURCE_UNAVAILABLE')
    try:
        owner = Path(__file__).absolute().parent
        for module in _OWNER:
            filename = f'{module}.py'
            name = f'cipher_lab_plugins/innovation1/s2_window/{filename}'
            result[name] = sha256(no_reparse(owner / filename).read_bytes()).hexdigest()
        _source(result)
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
