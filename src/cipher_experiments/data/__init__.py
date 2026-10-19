"""CPU dataset generators, separate from model/training concerns."""

from .simon import make_simon
from .synthetic import make_synthetic

__all__ = ["make_simon", "make_synthetic"]
