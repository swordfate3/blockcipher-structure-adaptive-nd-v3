"""Full core plus this owner and namespace initializer; no unrelated plugins."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab.artifacts import ArtifactError
from cipher_lab.artifacts import source_identity as core_source


def source_identity() -> dict[str, str]:
    result = core_source()
    owner = Path(__file__).absolute().parent
    try:
        paths = [owner.parent / '__init__.py', *sorted(owner.glob('*.py'))]
        for path in paths:
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError
            result['cipher_lab_plugins/' + path.relative_to(owner.parent).as_posix()] = sha256(
                path.read_bytes()).hexdigest()
    except Exception:
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
