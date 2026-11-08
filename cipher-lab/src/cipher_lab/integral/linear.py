"""Packed GF(2) spaces with canonical full RREF, highest pivot first.

External coordinate i is integer bit width-1-i. Production spaces have all
2080 coordinates; smaller widths support artificial exhaustive checks only.
No sampling, configuration, scientific decisions or I/O belong here.
"""

from collections.abc import Iterable, Sequence

_WIDTH = 2080


def _validated(rows: Iterable[int], width: int) -> tuple[int, ...]:
    if type(width) is not int:
        raise TypeError("width must be a built-in int")
    if not 1 <= width <= _WIDTH:
        raise ValueError("width must be between 1 and 2080")
    result = tuple(rows)
    for row in result:
        if type(row) is not int:
            raise TypeError("vectors must be built-in ints")
        if not 0 <= row < (1 << width):
            raise ValueError("vector outside specified width")
    return result


def _canonical(rows: Iterable[int]) -> tuple[int, ...]:
    """Eliminate leading pivots, then remove each new pivot from prior rows."""
    pivots: dict[int, int] = {}
    for row in rows:
        # Descending existing pivots also clears non-leading pivot columns.
        for pivot in sorted(pivots, reverse=True):
            if row & (1 << pivot):
                row ^= pivots[pivot]
        if not row:
            continue
        pivot = row.bit_length() - 1
        for old_pivot, old_row in pivots.items():
            if old_row & (1 << pivot):
                pivots[old_pivot] = old_row ^ row
        pivots[pivot] = row
    return tuple(pivots[pivot] for pivot in sorted(pivots, reverse=True))


def canonical_basis(rows: Iterable[int], width: int = _WIDTH) -> tuple[int, ...]:
    """Unique full-RREF basis of span(rows), descending leading-bit order.

    Width is a built-in int in 1..2080. Rows must be built-in unsigned ints
    fitting that width. Zero/duplicate rows and one-shot iterables are allowed.
    """
    return _canonical(_validated(rows, width))


def kernel(rows: Iterable[int], width: int = _WIDTH) -> tuple[int, ...]:
    """Canonical basis of all v with even parity(row & v), for every row.

    Empty/zero constraints give the full width-dimensional space, not an empty
    basis. Full rank gives (). Small samples necessarily leave large empirical
    kernels; this mathematical fact is not a scientific discovery or proof.
    Input validation and vector conventions are those of canonical_basis.
    """
    reduced = canonical_basis(rows, width)
    pivot_mask = sum(1 << (row.bit_length() - 1) for row in reduced)
    basis: list[int] = []
    for bit in range(width):
        if pivot_mask & (1 << bit):
            continue
        vector = 1 << bit
        for row in reduced:
            if (row & vector).bit_count() % 2:
                vector |= 1 << (row.bit_length() - 1)
        basis.append(vector)
    return _canonical(basis)


def survivor_basis(discovery: Sequence[int], screen: Sequence[int]) -> tuple[int, ...]:
    """Return span(discovery) intersect kernel(screen), at full width 2080.

    discovery is a discovered relation basis, NOT discovery constraint rows.
    screen contains new constraint rows. Keep all surviving combinations, not
    merely the supplied basis vectors that individually pass. Inputs may be
    redundant or empty; validate BOTH sides even when the space is empty.
    """
    reduced = canonical_basis(discovery)
    constraints = _validated(screen, _WIDTH)
    if not reduced:
        return ()
    # Internal coefficient j uses LSB bit j, not an external MSB0 coordinate.
    coefficient_rows = tuple(
        sum(((row & vector).bit_count() % 2) << j for j, vector in enumerate(reduced))
        for row in constraints
    )
    survivors: list[int] = []
    for coefficients in kernel(coefficient_rows, len(reduced)):
        vector = 0
        for j, original in enumerate(reduced):
            if coefficients & (1 << j):
                vector ^= original
        survivors.append(vector)
    return _canonical(survivors)


def confirms(basis: Sequence[int], rows: Sequence[int]) -> bool:
    """Test all frozen relations against all rows without filtering/mutation.

    Both sides are validated as full-width vectors before any short circuit.
    Empty basis/rows are vacuously True in pure mathematics; the method owner
    must label an empty-space confirmation NOT_APPLICABLE, not scientific pass.
    """
    frozen = _validated(basis, _WIDTH)
    constraints = _validated(rows, _WIDTH)
    return all((vector & row).bit_count() % 2 == 0 for vector in frozen for row in constraints)
