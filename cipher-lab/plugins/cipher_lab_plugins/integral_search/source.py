"""Fresh source closure for core + integral search only, never other plugins."""
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
        for name in ("ciphers", "data", "integral", "models", "training", "evaluation",
                     "representations"):
            required = core / name
            _safe(required)
            _safe(required / "__init__.py")
            if not required.is_dir() or not (required / "__init__.py").is_file():
                raise ArtifactError("SOURCE_UNAVAILABLE")
        package_init = plugin.parent / '__init__.py'
        _safe(package_init)
        result['cipher_lab_plugins/__init__.py'] = sha256(package_init.read_bytes()).hexdigest()
        for root, prefix in ((core, 'cipher_lab'), (plugin, 'cipher_lab_plugins/integral_search')):
            _safe(root)
            _safe(root / "__init__.py")
            if not root.is_dir() or not (root / '__init__.py').is_file():
                raise ArtifactError('SOURCE_UNAVAILABLE')
            for directory, dirs, files in os.walk(root, onerror=scan_failed, followlinks=False):
                # Root/ancestors were checked once; walk never follows symlinks,
                # and each child is checked before descent/read. No source cache.
                dirs[:] = [name for name in dirs if name not in ("tests", "__pycache__")]
                if any(name.endswith(".py") for name in files):
                    init = Path(directory) / "__init__.py"
                    _safe(init)
                    if not init.is_file():
                        raise ArtifactError("SOURCE_UNAVAILABLE")
                for name in dirs:
                    if (Path(directory) / name).is_symlink():
                        raise ArtifactError('SOURCE_UNAVAILABLE')
                for name in sorted(files):
                    if name == "tests":
                        continue
                    path = Path(directory) / name
                    if path.is_symlink():
                        raise ArtifactError('SOURCE_UNAVAILABLE')
                    if name.endswith('.py'):
                        key = prefix + '/' + path.relative_to(root).as_posix()
                        result[key] = sha256(path.read_bytes()).hexdigest()
        return dict(sorted(result.items()))
    except OSError:
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
