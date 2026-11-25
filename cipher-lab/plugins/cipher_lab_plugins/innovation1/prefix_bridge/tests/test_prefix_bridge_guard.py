from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest


def test_new_owner_rejects_cached_numerical_wrapper_probes(
    bridge_guard_probe: tuple[list[str], tuple[object, ...]],
) -> None:
    calls, (*handles, aliases) = bridge_guard_probe
    for wrapper in (*handles, *(getattr(aliases, name) for name in ('conv', 'norm', 'relu'))):
        with pytest.raises(AssertionError, match='PREFIX_BRIDGE_MANUAL_ONLY'):
            cast(Callable[[], object], wrapper)()
    assert calls == []
