"""Source closure for the owned runner and its cross-owner dependencies."""
from __future__ import annotations

from cipher_lab.artifacts import ArtifactError

from ...innovation1.prefix_bridge.source import source_identity as shared_identity


def source_identity() -> dict[str, str]:
    # Prefix bridge already merges direct_block and rejects inconsistent overlaps.
    result = shared_identity()
    prefix = 'cipher_lab_plugins/direct_block/bit_structure/'
    modules = ('__init__', 'cli', 'config', 'data', 'geometry', 'metrics', 'model',
               'report', 'source', 'training')
    required = {f'{prefix}{name}.py' for name in modules} | {
        'cipher_lab/ciphers/simon32.py',
        'cipher_lab_plugins/direct_block/split8/kat.py',
        'cipher_lab_plugins/direct_block/cube_compare/engine.py',
        'cipher_lab_plugins/direct_block/cube_compare/checkpoint.py',
    }
    if not required <= result.keys():
        raise ArtifactError('SOURCE_UNAVAILABLE')
    return result
