"""Reparse-safe paths and the source closure of this plugin plus the core."""
from __future__ import annotations

import os
import stat
from hashlib import sha256
from pathlib import Path

from cipher_lab import artifacts
from cipher_lab.artifacts import ArtifactError, _source


def no_reparse(path: Path) -> Path:
    if '..' in path.parts:
        raise ValueError('UNSAFE_PATH')
    absolute = path.absolute()
    for part in (*reversed(absolute.parents), absolute):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                or not (stat.S_ISDIR(info.st_mode)
                        or part == absolute and stat.S_ISREG(info.st_mode))):
            raise ValueError('UNSAFE_PATH')
    return absolute


def source_identity() -> dict[str, str]:
    result: dict[str, str] = {}
    owner = Path(__file__).absolute().parent

    def fail(error: OSError) -> None:
        raise error

    try:
        core = no_reparse(Path(artifacts.__file__).absolute().parent)
        for directory, dirs, files in os.walk(core, onerror=fail, followlinks=False):
            parent = Path(directory)
            for name in dirs:
                no_reparse(parent / name)
            for name in files:
                if name.endswith('.py'):
                    path = no_reparse(parent / name)
                    result['cipher_lab/' + path.relative_to(core).as_posix()] = sha256(
                        path.read_bytes()).hexdigest()
        # Explicit installed closure; missing obligatory files fail closed, not globbed away.
        modules = {
            '': ('__init__', 'application', 'batch', 'checkpoint', 'cli', 'config',
                 'contracts', 'data', 'dataset', 'metrics', 'models', 'report', 'resources',
                 'scoring', 'source', 'training'),
            'layout_probe': ('__init__', 'application', 'config', 'data', 'layout', 'metrics',
                             'models', 'report', 'reuse', 'runtime', 'weights'),
            'single_train': ('__init__', 'config', 'data', 'training', 'weights'),
            'single_pair': ('__init__', 'models'),
        }
        paths = [owner.parent / '__init__.py']
        paths += [owner / sub / f'{name}.py' for sub, names in modules.items() for name in names]
        for path in paths:
            key = path.relative_to(owner.parents[1]).as_posix()
            result[key] = sha256(no_reparse(path).read_bytes()).hexdigest()
        _source(result)
    except (ValueError, OSError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
