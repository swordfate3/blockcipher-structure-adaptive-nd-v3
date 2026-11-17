from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from cipher_lab_plugins.frontier.tests.frontier_fakes import plan, poison, raw_recipe, recipe

__all__ = ['poison']


def test_recipe_roundtrip_and_compact_spec() -> None:
    from cipher_lab_plugins.frontier.config import FrontierMLPSpec, parse_frontier
    value = recipe()
    assert parse_frontier(value.to_dict()) == value
    spec = FrontierMLPSpec(2, value.identity, plan())
    assert spec.to_dict() == {'hidden_width': 2, 'identity': value.identity.to_dict()}
    assert len(str(spec.to_dict())) < 1024


@pytest.mark.parametrize('field,value', [('rounds', True), ('rounds', 6),
    ('target_bit', 1), ('dimension', True), ('dimension', 90), ('order_sha256', 'a'*64),
    ('plan_sha256', 'A'*64), ('representation', 'simon-r3-residual13-v1')])
def test_recipe_rejects_unadmitted_identity(field: str, value: Any) -> None:
    from cipher_lab_plugins.frontier.config import parse_frontier
    node = raw_recipe()
    node['identity'][field] = value
    with pytest.raises(ValueError):
        parse_frontier(node)


@pytest.mark.parametrize('change', ['round', 'bit', 'extra', 'plan'])
def test_recipe_and_spec_binding(change: str) -> None:
    from cipher_lab_plugins.frontier.config import FrontierMLPSpec, parse_frontier
    node = raw_recipe()
    if change == 'plan':
        with pytest.raises(ValueError):
            FrontierMLPSpec(2, plan().identity, plan(5))
        return
    if change == 'extra':
        node['private_key'] = 0
    else:
        node['rounds' if change == 'round' else 'target_bit'] = 5 if change == 'round' else 1
    with pytest.raises(ValueError):
        parse_frontier(node)


def test_direct_dataclass_identity_is_not_a_trusted_boundary() -> None:
    from cipher_lab_plugins.frontier.config import FrontierMLPSpec
    identity = replace(plan().identity, dimension=True)
    with pytest.raises(ValueError):
        FrontierMLPSpec(2, identity, plan())


def test_legacy_probe_still_rejects_high_rounds() -> None:
    from cipher_lab_plugins.feature_probe.config import parse_feature_probe
    from cipher_lab_plugins.feature_probe.tests.feature_probe_fakes import raw_recipe as old_raw
    node = old_raw()
    node['rounds'] = 4
    with pytest.raises(ValueError):
        parse_feature_probe(node)


def test_config_import_chain_has_no_scientific_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh logical import in-process: evict/restore exact modules, no subprocess."""
    import importlib.abc
    import sys

    removed = {name: value for name, value in sys.modules.items()
               if name == 'cipher_lab.training_config'
               or name.startswith('cipher_lab.representations')}
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname.split('.')[0] in ('torch', 'numpy'):
                raise AssertionError('configuration attempted scientific import')
            return None
    # Removing existing scientific modules would disturb Torch state; instead
    # intercept Python's actual import calls even when modules are cached.
    import builtins
    original = builtins.__import__
    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split('.')[0] in ('torch', 'numpy'):
            raise AssertionError('configuration attempted scientific import')
        return original(name, *args, **kwargs)
    blocker = Block()
    try:
        for name in removed:
            sys.modules.pop(name)
        monkeypatch.setattr(builtins, '__import__', guarded)
        sys.meta_path.insert(0, blocker)
        from cipher_lab_plugins.frontier.config import parse_frontier
        assert parse_frontier(raw_recipe()).rounds == 4
    finally:
        sys.meta_path.remove(blocker)
        for name in tuple(sys.modules):
            if (name == 'cipher_lab.training_config'
                    or name.startswith('cipher_lab.representations')):
                sys.modules.pop(name)
        sys.modules.update(removed)
        # Restore parent attributes too: module identity remains stable for tests.
        import cipher_lab
        setattr(cipher_lab, 'training_config', removed['cipher_lab.training_config'])
        setattr(cipher_lab, 'representations', removed['cipher_lab.representations'])
