"""Claim and fresh production closure, including the cross-owner fit/runtime codecs."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _source
from cipher_lab_plugins.direct_block.source import source_identity as direct_identity

from ..source import no_reparse
from ..source import source_identity as innovation_identity

_OWNER = ('__init__', 'config', 'kat', 'data', 'model', 'runtime', 'weights', 'metrics',
          'training', 'cli', 'report', 'source')
_REQUIRED = ('cipher_lab/ciphers/present_prefix.py',
             'cipher_lab_plugins/direct_block/fit.py',
             'cipher_lab_plugins/direct_block/cube_compare/preflight.py',
             'cipher_lab_plugins/direct_block/cube_compare/resources.py',
             'cipher_lab_plugins/direct_block/cube_compare/config.py')


def source_identity() -> dict[str, str]:
    try:
        result = direct_identity()
        for name, digest in innovation_identity().items():
            if name in result and result[name] != digest:
                raise ArtifactError('SOURCE_CHANGED')
            result[name] = digest
        owner = Path(__file__).absolute().parent
        prefix = 'cipher_lab_plugins/innovation1/prefix_bridge/'
        for name in _OWNER:
            path = f'{name}.py'
            result[prefix + path] = sha256(no_reparse(owner / path).read_bytes()).hexdigest()
        if any(name not in result for name in _REQUIRED):
            raise ArtifactError('SOURCE_UNAVAILABLE')
        _source(result)
    except ArtifactError:
        raise
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
