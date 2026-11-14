# ruff: noqa: UP007, UP045
"""Python3.9 source-only freeze and fixed trusted teacher entry (no TF on import).

Freeze: python -B tools/speck-study/freeze.py MANIFEST ROOT
Provider: python -B tools/speck-study/freeze.py --teacher MANIFEST ROOT [provider flags]
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
    'cipher_lab/__init__.py', 'cipher_lab/platform_resources/__init__.py',
    'cipher_lab/platform_resources/dxgi.py',
    'cipher_lab/speck_study/gpu_telemetry.py',
    'cipher_lab/platform_resources/gpu.py', 'cipher_lab/speck_study/__init__.py',
    'cipher_lab/speck_study/teacher_protocol.py',
    'cipher_lab/speck_study/teacher_network.py',
    'cipher_lab/speck_study/teacher_provider.py',
)


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


def checked_sources(root: Path, manifest: dict[str, str]) -> dict[str, bytes]:
    if set(manifest) != set(CLOSURE):
        raise ValueError('provider closure mismatch')
    sources: dict[str, bytes] = {}
    for name in CLOSURE:
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('source symlink')
        data = path.read_bytes()  # no .pyc or importlib bytecode consulted
        if hashlib.sha256(data).hexdigest() != manifest[name]:
            raise ValueError('provider source mismatch')
        sources[name] = data
    return sources


class SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root: Path, sources: dict[str, bytes]) -> None:
        self.root, self.sources = root, sources

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == 'cipher_lab' or fullname.startswith('cipher_lab.'):
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
        module.__file__ = str(self.root / relative)
        exec(compile(self.sources[relative], module.__file__, 'exec'), module.__dict__)


def teacher(manifest: Path, root: Path, args: list[str]) -> int:
    if any(n == 'cipher_lab' or n.startswith('cipher_lab.') for n in sys.modules):
        raise RuntimeError('provider package imported before freeze')
    sources = checked_sources(root, read_manifest(manifest))
    sys.meta_path.insert(0, SourceOnly(root, sources))
    from cipher_lab.platform_resources.gpu import actual_attestor
    from cipher_lab.speck_study.teacher_provider import main
    sys.argv = [str(Path(__file__).resolve())] + args
    return int(main(actual_attestor))


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == '--teacher':
        if len(args) < 3:
            raise ValueError('missing manifest/source root')
        return teacher(Path(args[1]), Path(args[2]), args[3:])
    if len(args) != 2:
        raise ValueError('expected MANIFEST ROOT')
    destination, root = Path(args[0]), Path(args[1])
    snapshot = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in CLOSURE}
    checked_sources(root, snapshot)
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
