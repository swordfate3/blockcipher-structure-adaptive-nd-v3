"""Fresh source closure for core + Fourier only, never other plugins."""
from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import cipher_lab
from cipher_lab.artifacts import ArtifactError


def _source_roots() -> tuple[Path, Path]:
    assert cipher_lab.__file__ is not None
    return Path(cipher_lab.__file__).absolute().parent, Path(__file__).absolute().parent


def _safe(path: Path) -> None:
    # Reject symlinked ancestors too; don't hash bytes outside the claimed closure.
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ArtifactError('SOURCE_UNAVAILABLE')


def source_identity() -> dict[str, str]:
    """Hash Python sources with stable relative keys; missing/unreadable fails closed."""
    def scan_failed(error: OSError) -> None:
        raise ArtifactError('SOURCE_UNAVAILABLE') from None

    try:
        core, plugin = _source_roots()
        result: dict[str, str] = {}
        package_init = plugin.parent / '__init__.py'
        _safe(package_init)
        result['cipher_lab_plugins/__init__.py'] = sha256(package_init.read_bytes()).hexdigest()
        for root, prefix in ((core, 'cipher_lab'), (plugin, 'cipher_lab_plugins/fourier')):
            _safe(root)
            if not root.is_dir() or not (root / '__init__.py').is_file():
                raise ArtifactError('SOURCE_UNAVAILABLE')
            for directory, dirs, files in os.walk(root, onerror=scan_failed, followlinks=False):
                # Tests are not production dependencies: prune before checking links
                # or descending so an absent/unreadable/linked test tree is irrelevant.
                if root == plugin and Path(directory) == plugin:
                    dirs[:] = [name for name in dirs if name != 'tests']
                # Root/ancestors were checked once; walk never follows symlinks,
                # and each child is checked before descent/read. No source cache.
                for name in dirs:
                    if (Path(directory) / name).is_symlink():
                        raise ArtifactError('SOURCE_UNAVAILABLE')
                for name in sorted(files):
                    if name.endswith('.py'):
                        path = Path(directory) / name
                        if path.is_symlink():
                            raise ArtifactError('SOURCE_UNAVAILABLE')
                        key = prefix + '/' + path.relative_to(root).as_posix()
                        result[key] = sha256(path.read_bytes()).hexdigest()
        return dict(sorted(result.items()))
    except OSError:
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
