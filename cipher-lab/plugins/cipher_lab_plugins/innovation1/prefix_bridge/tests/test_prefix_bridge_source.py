from __future__ import annotations

import pytest
from cipher_lab_plugins.innovation1.prefix_bridge.source import source_identity

from cipher_lab.artifacts import ArtifactError


def test_identity_merges_production_dependency_closures() -> None:
    identity = source_identity()
    for name in (
        'cipher_lab/ciphers/present_prefix.py', 'cipher_lab/models/shared_m/components.py',
        'cipher_lab_plugins/direct_block/fit.py',
        'cipher_lab_plugins/direct_block/cube_compare/preflight.py',
        'cipher_lab_plugins/innovation1/models.py',
        'cipher_lab_plugins/innovation1/single_train/weights.py',
        'cipher_lab_plugins/innovation1/prefix_bridge/runtime.py',
    ):
        assert len(identity[name]) == 64
    assert identity == source_identity()
    assert not any('tests/' in key or '__pycache__' in key for key in identity)


def test_repeated_production_path_with_conflicting_hash_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.innovation1.prefix_bridge import source

    name = 'cipher_lab/ciphers/present_prefix.py'
    monkeypatch.setattr(source, 'direct_identity', lambda: {name: 'a' * 64})
    monkeypatch.setattr(source, 'innovation_identity', lambda: {name: 'b' * 64})
    with pytest.raises(ArtifactError, match='SOURCE_CHANGED'):
        source.source_identity()
