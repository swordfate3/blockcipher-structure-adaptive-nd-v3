"""Explicit source closure for the existing three-arm/paired implementation."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def source_identity() -> dict[str, str]:
    from cipher_lab.artifacts import ArtifactError
    from cipher_lab.artifacts import source_identity as core_identity

    def scan_error(error: OSError) -> None:
        raise error

    try:
        # Do not resolve: lexical ancestors must remain visible for link checks.
        location = Path(__file__).absolute()
        if any(path.is_symlink() for path in (location, *location.parents)):
            raise ArtifactError('SOURCE_CHANGED')
        root = location.parents[1]
        paths = [root / '__init__.py']
        # Common codecs and wrappers currently bind all three typed owners.
        for name in ('prediction_common', 'residual', 'feature_probe', 'frontier'):
            package = root / name
            if package.is_symlink() or not package.is_dir():
                raise ArtifactError('SOURCE_CHANGED')
            init = package / '__init__.py'
            if init.is_symlink() or not init.is_file():
                raise ArtifactError('SOURCE_CHANGED')
            for directory, dirs, files in os.walk(package, followlinks=False, onerror=scan_error):
                # Test-only links and unreadable test trees are not production dependencies.
                dirs[:] = [name for name in dirs if name != 'tests']
                current = Path(directory)
                for name in (*dirs, *files):
                    if name != 'tests' and (current / name).is_symlink():
                        raise ArtifactError('SOURCE_CHANGED')
                paths.extend(current / name for name in files if name.endswith('.py'))
        result = core_identity()
        for path in sorted(paths):
            if (path.is_symlink() or not path.is_file()
                    or any(parent.is_symlink() for parent in path.parents)):
                raise ArtifactError('SOURCE_CHANGED')
            result['cipher_lab_plugins/' + path.relative_to(root).as_posix()] = (
                hashlib.sha256(path.read_bytes()).hexdigest())
        return result
    except OSError:
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
