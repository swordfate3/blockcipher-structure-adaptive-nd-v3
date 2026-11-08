"""Fixed 64 units then 2016 distinct quadratic terms, in MSB0 order."""

from collections.abc import Iterable
from itertools import combinations

# Coordinate i occupies integer bit 2079-i; pairs are lexicographic i<j.
_TERM_MASKS = tuple(1 << (63 - i) for i in range(64)) + tuple(
    (1 << (63 - i)) | (1 << (63 - j)) for i, j in combinations(range(64), 2)
)


def aggregate_words(words: Iterable[int]) -> int:
    """XOR pointwise 2080-term observations of explicit unsigned 64-bit words.

    Empty input is zero. Units precede lexicographic distinct pairs; there is
    no constant term. In particular, quadratic products are evaluated BEFORE
    aggregation, not from the XOR of the input words. Consume an iterable once.
    Bool and integer subclasses are rejected, as are out-of-range words.
    """
    aggregate = 0
    for word in words:
        if type(word) is not int:
            raise TypeError("words must be built-in ints")
        if not 0 <= word < (1 << 64):
            raise ValueError("word outside unsigned 64-bit range")
        point = 0
        for term in _TERM_MASKS:
            point = (point << 1) | int(word & term == term)
        aggregate ^= point
    return aggregate
