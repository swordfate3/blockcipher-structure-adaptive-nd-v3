"""Owner-local fail-closed tests; never changes another owner's conftest."""
from __future__ import annotations

import socket
import subprocess
from collections.abc import Iterator

import pytest
from plugin_isolation import forbidden


@pytest.fixture(autouse=True)
def innovation1_guard(monkeypatch: pytest.MonkeyPatch,
                      request: pytest.FixtureRequest) -> Iterator[None]:
    # The globally loaded plugin_isolation supplies entropy/cipher/layer/Adam/RSS poison.
    request.getfixturevalue('default_science_poison')
    import torch

    for owner, names in ((socket, ('socket', 'create_connection')),
                         (subprocess, ('Popen', 'run', 'check_output'))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    # Even a dedicated forward that exits before a Torch layer is unauthorized.
    try:
        from cipher_lab.models.dedicated import FeistelSEExpert, PresentExpert, SkinnyExpert
    except ImportError:
        yield  # RED phase: missing implementation must fail the tests themselves.
        return
    for cls in (FeistelSEExpert, PresentExpert, SkinnyExpert):
        monkeypatch.setattr(cls, 'forward', forbidden)
    monkeypatch.setattr(torch.Tensor, 'cuda', forbidden)
    # Task2 resource owners must be poisoned at their entry, not only at an
    # underlying syscall whose public error sanitizer could hide the guard.
    from cipher_lab_plugins.innovation1 import device, resources
    monkeypatch.setattr(resources, 'production_rss_bytes', forbidden)
    monkeypatch.setattr(resources, 'gpu_sample', forbidden)
    monkeypatch.setattr(device, 'actual_attestor', forbidden)
    import ctypes
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    if hasattr(ctypes, 'WinDLL'):
        monkeypatch.setattr(ctypes, 'WinDLL', forbidden)
    yield
