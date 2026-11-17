"""Fixed 64 units then 2016 distinct quadratic terms, in MSB0 order."""

import sys
from collections.abc import Callable, Iterable
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

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


def aggregate_batch(
    words: "NDArray[np.uint64]", *, check: Callable[[], None] | None = None,
) -> int:
    """Aggregate the same 2080 MSB0 terms from at most 65536 native uint64s.

    Require an actual one-dimensional C-contiguous array, without coercion or
    mutation. Store 64 packed bit columns, not a point-by-term matrix; pair
    products use one reusable packed temporary. Packed tail padding is zero.
    Callback exceptions propagate unchanged; empty arrays still check twice.
    """
    import numpy as np

    if check is not None:
        check()
    if not isinstance(words, np.ndarray) or words.dtype != np.dtype(np.uint64):
        raise TypeError("words must be a native uint64 array")
    if words.ndim != 1 or not words.flags.c_contiguous:
        raise ValueError("words must be one-dimensional and C-contiguous")
    if words.size > 65536:
        raise ValueError("words must contain at most 65536 points")
    if words.size == 0:
        if check is not None:
            check()
        return 0

    packed_size = (words.size + 7) // 8
    columns = np.empty((64, packed_size), dtype=np.uint8)
    bits = np.empty(words.size, dtype=np.bool_)
    byte_view = words.view(np.uint8).reshape(words.size, 8)
    aggregate = 0
    for coordinate in range(64):
        # Native words may store their most significant byte at either end.
        byte_index = coordinate // 8 if sys.byteorder == "big" else 7 - coordinate // 8
        np.bitwise_and(
            byte_view[:, byte_index], np.uint8(1 << (7 - coordinate % 8)),
            out=bits, casting="unsafe",
        )
        columns[coordinate] = np.packbits(bits, bitorder="big")
        parity = int(np.bitwise_xor.reduce(columns[coordinate])).bit_count() & 1
        aggregate = (aggregate << 1) | parity
    del bits

    temporary = np.empty(packed_size, dtype=np.uint8)
    pairs = 0
    for left in range(64):
        for right in range(left + 1, 64):
            np.bitwise_and(columns[left], columns[right], out=temporary)
            parity = int(np.bitwise_xor.reduce(temporary)).bit_count() & 1
            aggregate = (aggregate << 1) | parity
            pairs += 1
            if check is not None and pairs % 64 == 0:
                check()
    if check is not None:
        check()
    return aggregate
