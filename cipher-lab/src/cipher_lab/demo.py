"""Fake engineering backend, NOT SIMON and NOT a random-function experiment.

The byte sequence is public/deterministic, merely passed through the collector's
private-input boundary. It contains no secrets. Separate 1/8/10/16 byte counters avoid
key reuse throughout a run; no production RNG, cipher or reduced observer width.
Fake null words are a counter sequence, not independent uniform words; tiny
samples may legitimately ALARM. Never override that outcome to advertise success.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DemoBytes:
    """Finite public engineering sequence, NOT suitable for keys or scientific null."""

    def __init__(self) -> None:
        self._counters = {1: 0, 8: 0, 10: 0, 16: 0}

    def __call__(self, size: int, /) -> bytes:
        if type(size) is not int or size not in self._counters:
            raise ValueError("demo supports only 1/8/10/16 byte engineering inputs")
        self._counters[size] += 1
        if size == 1:
            return bytes([self._counters[size] % 2])
        # A public fixed prefix and monotone suffix, never hidden/secret entropy.
        return b"DEMO" + self._counters[size].to_bytes(size - 4, "big")

    def bulk_words(self, amount: int) -> bytes:
        """1..65536 engineering words, exactly the continuous __call__(8) stream.

        Separate from byte-size requests: a 16-byte fake key is not two words.
        Other size counters, including the key counter, remain untouched.
        """
        if type(amount) is not int or not 1 <= amount <= 65536:
            raise ValueError("demo bulk amount must be in 1..65536 words")
        return b"".join(self(8) for _ in range(amount))


def fake_encrypt_many(points: tuple[int, ...], key_words: tuple[int, ...],
                      rounds: int) -> tuple[int, ...]:
    """Public identity-XOR function. Ignores fake key; no cipher claim whatsoever."""
    return tuple(point ^ rounds for point in points)


def fake_encrypt_batch(points: NDArray[np.uint64], key_words: tuple[int, ...],
                       rounds: int) -> NDArray[np.uint64]:
    """Explicit fake NumPy identity-XOR, not a cipher or scientific null."""
    import numpy as np

    return points ^ np.uint64(rounds)


def demo_rss_bytes() -> int:
    """Simulated RSS zero, deliberately NOT production resource monitoring."""
    return 0
