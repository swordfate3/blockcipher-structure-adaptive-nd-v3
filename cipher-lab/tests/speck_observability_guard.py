"""Standalone fresh-process guard, installed before any project import.

Usage: python -B tests/speck_observability_guard.py [pytest test paths...]
Only explicit reviewed fake seams may replace traps. Not a hostile-code sandbox.
"""
from __future__ import annotations

import builtins
import ctypes
import importlib.abc
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

violations: list[str] = []


def forbidden(*args: Any, **kwargs: Any) -> Any:
    violations.append('forbidden-operation')
    raise AssertionError('observability guard forbids real science/native/entropy/process')


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if (fullname.split('.')[0] in ('torch', 'tensorflow', 'keras', 'numpy', 'h5py',
                                      'psutil', 'pynvml', 'win32com', 'comtypes')
                or fullname == 'cipher_lab.ciphers'
                or fullname.startswith(('cipher_lab.ciphers.', 'cipher_lab.training.',
                                         'cipher_lab.data.'))):
            forbidden()
        return None


# Initialize pytest before disabling entropy used by pytest's own temp naming;
# no project/NumPy is loaded here. Project import is guarded from first byte.
import pytest  # noqa: E402

assert not any(n == 'cipher_lab' or n.startswith('cipher_lab.') for n in sys.modules)
assert not any(n in sys.modules for n in ('numpy', 'torch', 'tensorflow'))
sys.dont_write_bytecode = True
sys.meta_path.insert(0, Guard())
for name in ('CDLL', 'WinDLL', 'OleDLL', 'PyDLL'):
    setattr(ctypes, name, forbidden)
for name in ('run', 'Popen'):
    setattr(subprocess, name, forbidden)
os.system = forbidden
os.urandom = forbidden
secrets.token_bytes = forbidden
secrets.randbits = forbidden
real_import = builtins.__import__


def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name.split('.')[0] in ('torch', 'tensorflow', 'numpy', 'h5py'):
        forbidden()
    return real_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
import cipher_lab.execution as execution  # noqa: E402

execution.production_rss_bytes = forbidden
sys.path.insert(0, str(Path(__file__).resolve().parent))
code = pytest.main(['-p', 'no:cacheprovider', *sys.argv[1:]])
print('observability_guard_violations=' + str(len(violations)))
raise SystemExit(code if not violations else 99)
