"""Claim-time and fresh identities for the core and this production owner only."""
from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError, _path


def _roots() -> tuple[Path, Path]:
    from cipher_lab import artifacts

    return Path(artifacts.__file__).absolute().parent, Path(__file__).absolute().parent


def source_identity() -> dict[str, str]:
    def fail(error: OSError) -> None:
        raise error

    try:
        core, owner = map(_path, _roots())
        parent_init = _path(owner.parent / '__init__.py')
        result = {'cipher_lab_plugins/__init__.py': sha256(parent_init.read_bytes()).hexdigest()}
        for prefix, root in (('cipher_lab/', core), ('cipher_lab_plugins/direct_block/', owner)):
            # Require each package entry; do not silently treat a missing root as empty.
            _path(root / '__init__.py').stat()
            for directory, dirs, files in os.walk(root, onerror=fail, followlinks=False):
                folder = Path(directory)
                dirs[:] = [d for d in dirs if d not in {'tests', '__pycache__'}]
                for d in dirs:
                    _path(folder / d)
                for name in files:
                    if name.endswith('.py') and not name.startswith('test_'):
                        path = _path(folder / name)
                        result[prefix + path.relative_to(root).as_posix()] = sha256(
                            path.read_bytes()).hexdigest()
        return dict(sorted(result.items()))
    except OSError:
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
