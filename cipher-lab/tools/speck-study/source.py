"""Main Python3.10 explicit source closure; no cipher/module imports to freeze.

Freeze: python -B tools/speck-study/source.py freeze MANIFEST SRC
Run: python -B tools/speck-study/source.py run MANIFEST SRC [cipher-lab CLI args]
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
    'cipher_lab/__init__.py', 'cipher_lab/cli.py', 'cipher_lab/config.py',
    'cipher_lab/training_config.py',
    'cipher_lab/speck_study_config.py', 'cipher_lab/speck_study_launch.py',
    'cipher_lab/artifacts.py', 'cipher_lab/execution.py',
    'cipher_lab/ciphers/__init__.py', 'cipher_lab/ciphers/speck.py',
    'cipher_lab/ciphers/speck_tensor.py', 'cipher_lab/platform_resources/__init__.py',
    'cipher_lab/platform_resources/gpu.py', 'cipher_lab/platform_resources/windows.py',
    'cipher_lab/speck_study/__init__.py', 'cipher_lab/speck_study/application.py',
    'cipher_lab/speck_study/closing.py', 'cipher_lab/speck_study/codec.py',
    'cipher_lab/speck_study/data.py',
    'cipher_lab/speck_study/method.py', 'cipher_lab/speck_study/paper.py',
    'cipher_lab/speck_study/parity.py', 'cipher_lab/speck_study/resources.py',
    'cipher_lab/speck_study/scoring.py', 'cipher_lab/speck_study/search.py',
    'cipher_lab/speck_study/spectral.py', 'cipher_lab/speck_study/storage.py',
    'cipher_lab/speck_study/report.py', 'cipher_lab/speck_study/teacher.py',
    'cipher_lab/speck_study/teacher_protocol.py',
    'cipher_lab/speck_study/teacher_network.py',
    'cipher_lab/platform_resources/dxgi.py',
    'cipher_lab/speck_study/gpu_telemetry.py',
    'cipher_lab/speck_study/native_control.py',
    'cipher_lab/speck_study/native_session.py',
    'cipher_lab/speck_study/native_parent.py',
)


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


def freeze_all(destination: Path, root: Path) -> None:
    tools = Path(__file__).resolve().parent
    namespace: dict[str,Any] = {'__name__':'_freeze_manifest', '__file__':str(tools/'freeze.py')}
    exec(compile((tools/'freeze.py').read_bytes(),str(tools/'freeze.py'),'exec'),namespace)
    provider = {n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                for n in namespace['CLOSURE']}
    namespace['checked_sources'](root,provider)
    package = {p.relative_to(root).as_posix():
               hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root/'cipher_lab').rglob('*.py'))}
    boot = {n:hashlib.sha256((tools/n).read_bytes()).hexdigest()
            for n in ('source.py','freeze.py','supervise.py')}
    for path,value in ((destination,snapshot(root)),
                       (Path(str(destination)+'.provider.json'),provider),
                       (Path(str(destination)+'.bootstrap.json'),boot),
                       (Path(str(destination)+'.package.json'),package)):
        with path.open('x',encoding='utf8') as stream:
            json.dump(value,stream,sort_keys=True)
            stream.write('\n')


def checked(root: Path, manifest: dict[str, str]) -> dict[str, bytes]:
    if set(manifest) != set(CLOSURE):
        raise ValueError('main closure mismatch')
    result = {}
    for name in CLOSURE:
        path = root/name
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('source symlink')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[name]:
            raise ValueError('source identity mismatch')
        result[name] = raw
    return result


def snapshot(root: Path) -> dict[str, str]:
    result = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CLOSURE}
    checked(root,result)
    return result


class SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root: Path, sources: dict[str, bytes]) -> None:
        self.root,self.sources = root,sources

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == 'cipher_lab' or fullname.startswith('cipher_lab.'):
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
        module.__file__ = str(self.root/relative)
        exec(compile(self.sources[relative],module.__file__,'exec'),module.__dict__)


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] not in ('freeze','run'):
        raise ValueError('expected freeze|run MANIFEST SRC [CLI args]')
    action,manifest,root = argv[0],Path(argv[1]),Path(argv[2])
    if action == 'freeze':
        if len(argv) != 3:
            raise ValueError('unexpected freeze args')
        freeze_all(manifest,root)
        return 0
    if any(n == 'cipher_lab' or n.startswith('cipher_lab.') for n in sys.modules):
        raise RuntimeError('package imported before freeze')
    sources = checked(root,read_manifest(manifest))
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0,SourceOnly(root,sources))
    from cipher_lab.cli import main as cli

    return int(cli(argv[3:]))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
