"""Model wiring constants; specification types belong only to training_config."""
from cipher_lab.training_config import SharedMSpec

STEM_NAMES = ("local", "word", "nibble")

__all__ = ["SharedMSpec", "STEM_NAMES"]
