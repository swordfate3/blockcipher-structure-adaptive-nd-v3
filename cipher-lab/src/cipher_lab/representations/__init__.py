"""Public geometry exports, loaded lazily so stdlib plans stay Torch-free."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cipher_lab.representations.input import public_backstep as public_backstep
    from cipher_lab.representations.tasks import public_features as public_features
    from cipher_lab.representations.tasks import validate_public as validate_public

__all__ = ["public_backstep", "public_features", "validate_public"]


def __getattr__(name: str) -> object:
    if name == 'public_backstep':
        from cipher_lab.representations.input import public_backstep
        return public_backstep
    if name in ('public_features', 'validate_public'):
        from cipher_lab.representations.tasks import public_features, validate_public
        return public_features if name == 'public_features' else validate_public
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
