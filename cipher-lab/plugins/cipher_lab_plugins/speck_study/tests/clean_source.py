"""Test-owned production source roots; never clean or mutate the developer tree."""
from __future__ import annotations

import shutil
from pathlib import Path

from cipher_lab_plugins.speck_study.tests.locations import PLUGIN, ROOT


def clean_source(destination: Path) -> Path:
    """Copy core + this plugin + configs, without caches or collected test trees.

    Frozen positive tests need actual source bytes without developer import
    caches. Rejection tests continue to construct their own invalid source trees.
    The caller owns destination's lifetime; originals remain untouched.
    """
    destination.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', 'tests')
    shutil.copytree(ROOT/'src', destination/'src', ignore=ignore)
    shutil.copytree(ROOT/'configs', destination/'configs')
    namespace = destination/'plugins/cipher_lab_plugins'
    namespace.mkdir(parents=True)
    shutil.copyfile(ROOT/'plugins/cipher_lab_plugins/__init__.py', namespace/'__init__.py')
    shutil.copytree(PLUGIN, namespace/'speck_study', ignore=ignore)
    return destination
