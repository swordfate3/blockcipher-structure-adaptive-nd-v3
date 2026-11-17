# ruff: noqa: UP007, UP045
"""Python3.9 source-only freeze and fixed trusted teacher entry (no TF on import).

Freeze: python -B plugins/cipher_lab_plugins/speck_study/runtime/freeze.py
    MANIFEST CORE_ROOT PLUGIN_ROOT
Provider: python -B plugins/cipher_lab_plugins/speck_study/runtime/freeze.py --teacher
    MANIFEST CORE_ROOT PLUGIN_ROOT [provider flags]
The manifest must be independently checked against the approved Git closure by
parent before launch. This is source identity verification, not a hostile sandbox.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

# Explicit py39-only provider closure. Never import execution/config/Torch.
CLOSURE = (
    'cipher_lab/__init__.py',
    'cipher_lab_plugins/speck_study/platform_resources/__init__.py',
    'cipher_lab_plugins/speck_study/platform_resources/dxgi.py',
    'cipher_lab_plugins/speck_study/gpu_telemetry.py',
    'cipher_lab_plugins/speck_study/platform_resources/gpu.py',
    'cipher_lab_plugins/speck_study/__init__.py',
    'cipher_lab_plugins/speck_study/teacher_protocol.py',
    'cipher_lab_plugins/speck_study/teacher_network.py',
    'cipher_lab_plugins/speck_study/teacher_provider.py',
    'cipher_lab_plugins/__init__.py',
)


def source_path(core_root: Path, plugin_root: Path, name: str) -> Path:
    parts = name.split('/')
    if (len(parts) < 2 or any(x in ('', '.', '..') for x in parts)
            or '\\' in name or ':' in name or not name.endswith('.py')
            or parts[0] not in ('cipher_lab', 'cipher_lab_plugins')):
        raise ValueError('invalid source closure path')
    return (core_root if parts[0] == 'cipher_lab' else plugin_root).joinpath(*parts)


def read_manifest(path: Path) -> dict[str, str]:
    def unique(pairs: list[tuple[str,str]]) -> dict[str,str]:
        result: dict[str,str] = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate provider manifest')
            result[key] = value
        return result
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise ValueError('oversize provider manifest')
    value = json.loads(raw,object_pairs_hook=unique)
    if type(value) is not dict:
        raise ValueError('provider manifest required')
    return value


def checked_sources(core_root: Path, plugin_root: Path, manifest: dict[str, str]) -> dict[str, bytes]:  # noqa: E501 - explicit migrated source/argv identity
    if set(manifest) != set(CLOSURE):
        raise ValueError('provider closure mismatch')
    sources: dict[str, bytes] = {}
    for name in CLOSURE:
        path = source_path(core_root,plugin_root,name)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('source symlink')
        data = path.read_bytes()  # no .pyc or importlib bytecode consulted
        if hashlib.sha256(data).hexdigest() != manifest[name]:
            raise ValueError('provider source mismatch')
        sources[name] = data
    return sources


class SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, core_root: Path, plugin_root: Path, sources: dict[str, bytes]) -> None:
        self.core_root, self.plugin_root, self.sources = core_root, plugin_root, sources

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if any(fullname == n or fullname.startswith(n+'.') for n in ('cipher_lab','cipher_lab_plugins')):  # noqa: E501 - explicit migrated source/argv identity
            name = fullname.replace('.', '/')
            for relative, package in ((name + '/__init__.py', True), (name + '.py', False)):
                if relative in self.sources:
                    return importlib.util.spec_from_loader(fullname, self, is_package=package)
            raise ImportError('outside frozen provider source closure')
        return None

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        name = module.__name__.replace('.', '/')
        relative = name + ('/__init__.py' if hasattr(module, '__path__') else '.py')
        module.__file__ = str(source_path(self.core_root,self.plugin_root,relative))
        exec(compile(self.sources[relative], module.__file__, 'exec'), module.__dict__)


def teacher(manifest: Path, core_root: Path, plugin_root: Path, args: list[str]) -> int:
    if any(any(n == prefix or n.startswith(prefix+'.') for prefix in ('cipher_lab','cipher_lab_plugins')) for n in sys.modules):  # noqa: E501 - explicit migrated source/argv identity
        raise RuntimeError('provider package imported before freeze')
    sources = checked_sources(core_root, plugin_root, read_manifest(manifest))
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, SourceOnly(core_root, plugin_root, sources))
    from cipher_lab_plugins.speck_study.platform_resources.gpu import actual_attestor
    from cipher_lab_plugins.speck_study.teacher_provider import main
    sys.argv = [str(Path(__file__).resolve())] + args
    return int(main(actual_attestor))


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == '--teacher':
        if len(args) < 4:
            raise ValueError('missing manifest/source root')
        return teacher(Path(args[1]), Path(args[2]), Path(args[3]), args[4:])
    if len(args) != 3:
        raise ValueError('expected MANIFEST CORE_ROOT PLUGIN_ROOT')
    destination, core_root, plugin_root = Path(args[0]), Path(args[1]), Path(args[2])
    snapshot = {n: hashlib.sha256(source_path(core_root,plugin_root,n).read_bytes()).hexdigest() for n in CLOSURE}  # noqa: E501 - explicit migrated source/argv identity
    checked_sources(core_root, plugin_root, snapshot)
    with destination.open('x', encoding='utf8') as stream:
        json.dump(snapshot, stream, sort_keys=True)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    # DeadlineIO workers must never delay provider exit on buffered pipe locks.
    try:
        code = main()
    except BaseException:
        code = 2
    os._exit(code)
