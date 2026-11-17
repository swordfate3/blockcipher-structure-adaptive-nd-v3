"""Local source roots for the explicitly supplied core and plugin namespaces."""
from __future__ import annotations

from pathlib import Path

from cipher_lab_plugins.speck_study.runtime.source import source_path as source_path


def source_roots() -> tuple[Path, Path]:
    import cipher_lab

    core_file, plugin_file = Path(cipher_lab.__file__).absolute(), Path(__file__).absolute()
    for path in (core_file, plugin_file):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('source symlink')
    return core_file.resolve().parent.parent, plugin_file.resolve().parents[2]

