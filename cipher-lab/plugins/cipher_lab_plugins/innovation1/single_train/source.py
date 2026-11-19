"""Source binding for core, the innovation1 owner and the reused pair operators."""
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
    """Same core/top-owner closure, checking reparse points before descent/read."""
    result: dict[str, str] = {}
    here = Path(__file__).absolute().parent
    owner = here.parent
    names = ('__init__', 'application', 'cli', 'config', 'data', 'models', 'report',
             'source', 'training', 'weights')

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
                    key = 'cipher_lab/' + path.relative_to(core).as_posix()
                    result[key] = sha256(path.read_bytes()).hexdigest()
        no_reparse(owner)
        paths = [owner.parent / '__init__.py', *owner.glob('*.py')]
        paths += [here / f'{name}.py' for name in names]
        paths += [owner / 'single_pair' / name for name in ('__init__.py', 'models.py')]
        for path in paths:
            key = path.relative_to(here.parents[2]).as_posix()
            result[key] = sha256(no_reparse(path).read_bytes()).hexdigest()
        _source(result)
    except (ValueError, OSError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
