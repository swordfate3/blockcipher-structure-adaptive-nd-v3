"""Main Python3.10 explicit source closure; no cipher/module imports to freeze.

Freeze: python -B plugins/cipher_lab_plugins/speck_study/runtime/source.py freeze
    MANIFEST CORE_ROOT PLUGIN_ROOT
Run: python -B plugins/cipher_lab_plugins/speck_study/runtime/source.py run
    MANIFEST CORE_ROOT PLUGIN_ROOT [cipher-lab CLI args]
Parent/operator must independently compare manifest with approved published Git
hashes; generating a manifest locally does not certify or authorize science.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

CLOSURE = (
    'cipher_lab_plugins/speck_study/cli.py',
    'cipher_lab/__init__.py',
    'cipher_lab/cli.py',
    'cipher_lab/plugin_cli.py',
    'cipher_lab/config.py',
    'cipher_lab/training_config.py',
    'cipher_lab/artifacts.py',
    'cipher_lab/execution.py',
    'cipher_lab/ciphers/__init__.py',
    'cipher_lab/ciphers/speck.py',
    'cipher_lab/ciphers/speck_tensor.py',
    'cipher_lab_plugins/__init__.py',
    'cipher_lab_plugins/speck_study/__init__.py',
    'cipher_lab_plugins/speck_study/application.py',
    'cipher_lab_plugins/speck_study/closing.py',
    'cipher_lab_plugins/speck_study/codec.py',
    'cipher_lab_plugins/speck_study/config.py',
    'cipher_lab_plugins/speck_study/data.py',
    'cipher_lab_plugins/speck_study/gpu_telemetry.py',
    'cipher_lab_plugins/speck_study/launch.py',
    'cipher_lab_plugins/speck_study/method.py',
    'cipher_lab_plugins/speck_study/native_control.py',
    'cipher_lab_plugins/speck_study/native_parent.py',
    'cipher_lab_plugins/speck_study/native_session.py',
    'cipher_lab_plugins/speck_study/paper.py',
    'cipher_lab_plugins/speck_study/parity.py',
    'cipher_lab_plugins/speck_study/paths.py',
    'cipher_lab_plugins/speck_study/platform_resources/__init__.py',
    'cipher_lab_plugins/speck_study/platform_resources/dxgi.py',
    'cipher_lab_plugins/speck_study/platform_resources/gpu.py',
    'cipher_lab_plugins/speck_study/platform_resources/windows.py',
    'cipher_lab_plugins/speck_study/report.py',
    'cipher_lab_plugins/speck_study/resources.py',
    'cipher_lab_plugins/speck_study/runtime/__init__.py',
    'cipher_lab_plugins/speck_study/runtime/source.py',
    'cipher_lab_plugins/speck_study/scoring.py',
    'cipher_lab_plugins/speck_study/search.py',
    'cipher_lab_plugins/speck_study/sources.py',
    'cipher_lab_plugins/speck_study/spectral.py',
    'cipher_lab_plugins/speck_study/storage.py',
    'cipher_lab_plugins/speck_study/teacher.py',
    'cipher_lab_plugins/speck_study/teacher_network.py',
    'cipher_lab_plugins/speck_study/teacher_protocol.py',
)


def source_path(core_root: Path, plugin_root: Path, name: str) -> Path:
    parts = name.split('/')
    if (len(parts) < 2 or any(x in ('', '.', '..') for x in parts)
            or '\\' in name or ':' in name or not name.endswith('.py')
            or parts[0] not in ('cipher_lab', 'cipher_lab_plugins')):
        raise ValueError('invalid source closure path')
    return (core_root if parts[0] == 'cipher_lab' else plugin_root).joinpath(*parts)


def read_manifest(path: Path) -> dict[str, str]:
    def unique(pairs: list[tuple[str, str]]) -> dict[str,str]:
        result: dict[str,str] = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate manifest field')
            result[key] = value
        return result
    raw = path.read_bytes()
    if len(raw) > 1024*1024:
        raise ValueError('oversize manifest')
    value = json.loads(raw,object_pairs_hook=unique)
    if type(value) is not dict or any(type(k) is not str or type(v) is not str
                                     for k,v in value.items()):
        raise ValueError('invalid source manifest')
    return value


def package_source_files(core_root: Path, plugin_root: Path) -> list[Path]:
    """Inspect only managed source trees; prune plugin tests before traversal."""
    def check(path: Path) -> None:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('source symlink')
        if path.suffix.lower() in ('.pyc', '.pyo') or path.name == '__pycache__':
            raise ValueError('not source-only')

    files = []
    for base, exclude_tests in ((core_root/'cipher_lab', False),
                                (plugin_root/'cipher_lab_plugins'/'speck_study', True)):
        check(base)
        pending = [base] if base.exists() else []
        while pending:
            for path in pending.pop().iterdir():
                if exclude_tests and path.name == 'tests':
                    continue
                check(path)
                if path.is_dir():
                    pending.append(path)
                elif path.suffix == '.py':
                    files.append(path)
    namespace_init = plugin_root/'cipher_lab_plugins'/'__init__.py'
    check(namespace_init)
    # -B prevents writes, not reads: the shared namespace initializer also
    # needs source-only protection, without inspecting sibling plugin trees.
    for path in namespace_init.parent.iterdir():
        if path.name in ('__init__.pyc', '__init__.pyo'):
            check(path)
        elif path.name == '__pycache__':
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError('source symlink')
            # Do not suppress enumeration errors or follow a cache-dir link.
            for cached in path.iterdir():
                if cached.name.startswith('__init__.'):
                    check(cached)
    files.append(namespace_init)
    return files


def package_snapshot(core_root: Path, plugin_root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(package_source_files(core_root, plugin_root)):
        root = core_root if path.is_relative_to(core_root/'cipher_lab') else plugin_root
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def freeze_all(destination: Path, core_root: Path, plugin_root: Path) -> None:
    tools = Path(__file__).resolve().parent
    namespace: dict[str,Any] = {'__name__':'_freeze_manifest', '__file__':str(tools/'freeze.py')}
    exec(compile((tools/'freeze.py').read_bytes(),str(tools/'freeze.py'),'exec'),namespace)
    provider = {n:hashlib.sha256(source_path(core_root,plugin_root,n).read_bytes()).hexdigest()
                for n in namespace['CLOSURE']}
    namespace['checked_sources'](core_root,plugin_root,provider)
    package = package_snapshot(core_root,plugin_root)
    boot = {n:hashlib.sha256((tools/n).read_bytes()).hexdigest()
            for n in ('source.py','freeze.py','supervise.py')}
    for path,value in ((destination,snapshot(core_root,plugin_root)),
                       (Path(str(destination)+'.provider.json'),provider),
                       (Path(str(destination)+'.bootstrap.json'),boot),
                       (Path(str(destination)+'.package.json'),package)):
        with path.open('x',encoding='utf8') as stream:
            json.dump(value,stream,sort_keys=True)
            stream.write('\n')


def checked(core_root: Path, plugin_root: Path, manifest: dict[str, str]) -> dict[str, bytes]:
    if set(manifest) != set(CLOSURE):
        raise ValueError('main closure mismatch')
    result = {}
    for name in CLOSURE:
        path = source_path(core_root,plugin_root,name)
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('source symlink')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[name]:
            raise ValueError('source identity mismatch')
        result[name] = raw
    return result


def snapshot(core_root: Path, plugin_root: Path) -> dict[str, str]:
    result = {name:hashlib.sha256(source_path(core_root,plugin_root,name).read_bytes()).hexdigest() for name in CLOSURE}  # noqa: E501 - explicit migrated source/argv identity
    checked(core_root,plugin_root,result)
    return result


class SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, core_root: Path, plugin_root: Path, sources: dict[str, bytes]) -> None:
        self.core_root,self.plugin_root,self.sources = core_root,plugin_root,sources

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if any(fullname == n or fullname.startswith(n+'.') for n in ('cipher_lab','cipher_lab_plugins')):  # noqa: E501 - explicit migrated source/argv identity
            name = fullname.replace('.','/')
            for relative,package in ((name+'/__init__.py',True),(name+'.py',False)):
                if relative in self.sources:
                    return importlib.util.spec_from_loader(fullname,self,is_package=package)
            raise ImportError('outside frozen main closure')
        return None

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        name = module.__name__.replace('.','/')
        relative = name+('/__init__.py' if hasattr(module,'__path__') else '.py')
        module.__file__ = str(source_path(self.core_root,self.plugin_root,relative))
        exec(compile(self.sources[relative],module.__file__,'exec'),module.__dict__)


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[0] not in ('freeze','run'):
        raise ValueError('expected freeze|run MANIFEST CORE_ROOT PLUGIN_ROOT [CLI args]')
    action,manifest,core_root,plugin_root = argv[0],Path(argv[1]),Path(argv[2]),Path(argv[3])
    if action == 'freeze':
        if len(argv) != 4:
            raise ValueError('unexpected freeze args')
        freeze_all(manifest,core_root,plugin_root)
        return 0
    if any(any(n == prefix or n.startswith(prefix+'.') for prefix in ('cipher_lab','cipher_lab_plugins')) for n in sys.modules):  # noqa: E501 - explicit migrated source/argv identity
        raise RuntimeError('package imported before freeze')
    sources = checked(core_root,plugin_root,read_manifest(manifest))
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0,SourceOnly(core_root,plugin_root,sources))
    from cipher_lab.cli import main as cli

    return int(cli(argv[4:]))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
