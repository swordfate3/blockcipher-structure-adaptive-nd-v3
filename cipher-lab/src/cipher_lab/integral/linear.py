"""Packed GF(2) spaces with canonical full RREF, highest pivot first.

External coordinate i is integer bit width-1-i. Production spaces have all
2080 coordinates; smaller widths support artificial exhaustive checks only.
No sampling, configuration, scientific decisions or I/O belong here.
"""

from collections.abc import Callable, Iterable, Sequence

_WIDTH = 2080


class _Checks:
    """One counter per public call, shared by all its mathematical phases."""

    def __init__(self, check: Callable[[], None] | None) -> None:
        self.check = check
        self.steps = 0
        self.poll()

    def poll(self) -> None:
        if self.check is not None:
            self.check()
        self.steps = 0

    def step(self) -> None:
        if self.check is not None:
            self.steps += 1
            if self.steps == 64:
                self.poll()


def _validated(rows: Iterable[int], width: int, checks: _Checks) -> tuple[int, ...]:
    if type(width) is not int:
        raise TypeError("width must be a built-in int")
    if not 1 <= width <= _WIDTH:
        raise ValueError("width must be between 1 and 2080")
    if checks.check is None:
        # Preserve iterator-exception priority over vector errors without a hook.
        result = tuple(rows)
    else:
        collected: list[int] = []
        for row in rows:
            collected.append(row)
            checks.step()  # Do not exhaust a long generator without checking.
        result = tuple(collected)
    for row in result:
        if type(row) is not int:
            raise TypeError("vectors must be built-in ints")
        if not 0 <= row < (1 << width):
            raise ValueError("vector outside specified width")
        checks.step()
    return result


def _canonical(rows: Iterable[int], checks: _Checks) -> tuple[int, ...]:
    """Eliminate leading pivots, then remove each new pivot from prior rows."""
    pivots: dict[int, int] = {}
    for row in rows:
        checks.step()
        # Descending existing pivots also clears non-leading pivot columns.
        for pivot in sorted(pivots, reverse=True):
            checks.step()
            if row & (1 << pivot):
                row ^= pivots[pivot]
                checks.step()
        if not row:
            continue
        pivot = row.bit_length() - 1
        for old_pivot, old_row in pivots.items():
            checks.step()
            if old_row & (1 << pivot):
                pivots[old_pivot] = old_row ^ row
                checks.step()
        pivots[pivot] = row
    result: list[int] = []
    for pivot in sorted(pivots, reverse=True):
        result.append(pivots[pivot])
        checks.step()
    return tuple(result)


def canonical_basis(rows: Iterable[int], width: int = _WIDTH, *,
                    check: Callable[[], None] | None = None) -> tuple[int, ...]:
    """Unique full-RREF basis of span(rows), descending leading-bit order.

    Width is a built-in int in 1..2080. Rows must be built-in unsigned ints
    fitting that width. Zero/duplicate rows and one-shot iterables are allowed.
    check runs at entry, normal exit and at most every 64 loop steps (including
    inner XORs). Its exceptions propagate unchanged. This is cooperative, not a
    hard deadline: one iterator advance, integer operation or sort is atomic.
    Without check, original validation/iterator exception priority is preserved.
    """
    checks = _Checks(check)
    result = _canonical(_validated(rows, width, checks), checks)
    checks.poll()
    return result


def kernel(rows: Iterable[int], width: int = _WIDTH, *,
           check: Callable[[], None] | None = None) -> tuple[int, ...]:
    """Canonical basis of all v with even parity(row & v), for every row.

    Empty/zero constraints give the full width-dimensional space, not an empty
    basis. Full rank gives (). Small samples necessarily leave large empirical
    kernels; this mathematical fact is not a scientific discovery or proof.
    Input validation, vector and callback conventions are those of canonical_basis.
    """
    checks = _Checks(check)
    result = _kernel(_validated(rows, width, checks), width, checks)
    checks.poll()
    return result


def _kernel(rows: Iterable[int], width: int, checks: _Checks) -> tuple[int, ...]:
    """Kernel of already validated rows; keep the caller's counter across phases."""
    reduced = _canonical(rows, checks)
    pivot_mask = 0
    for row in reduced:
        pivot_mask += 1 << (row.bit_length() - 1)
        checks.step()
    basis: list[int] = []
    for bit in range(width):
        checks.step()
        if pivot_mask & (1 << bit):
            continue
        vector = 1 << bit
        for row in reduced:
            checks.step()
            if (row & vector).bit_count() % 2:
                vector |= 1 << (row.bit_length() - 1)
                checks.step()
        basis.append(vector)
    return _canonical(basis, checks)


def survivor_basis(discovery: Sequence[int], screen: Sequence[int], *,
                   check: Callable[[], None] | None = None) -> tuple[int, ...]:
    """Return span(discovery) intersect kernel(screen), at full width 2080.

    discovery is a discovered relation basis, NOT discovery constraint rows.
    screen contains new constraint rows. Keep all surviving combinations, not
    merely the supplied basis vectors that individually pass. Inputs may be
    redundant or empty; validate BOTH sides even when the space is empty.
    Callback conventions are those of canonical_basis.
    """
    checks = _Checks(check)
    reduced = _canonical(_validated(discovery, _WIDTH, checks), checks)
    constraints = _validated(screen, _WIDTH, checks)
    if not reduced:
        checks.poll()
        return ()
    # Internal coefficient j uses LSB bit j, not an external MSB0 coordinate.
    coefficient_rows: list[int] = []
    for row in constraints:
        checks.step()
        coefficients = 0
        for j, vector in enumerate(reduced):
            coefficients += ((row & vector).bit_count() % 2) << j
            checks.step()
        coefficient_rows.append(coefficients)
    survivors: list[int] = []
    for coefficients in _kernel(coefficient_rows, len(reduced), checks):
        checks.step()
        vector = 0
        for j, original in enumerate(reduced):
            checks.step()
            if coefficients & (1 << j):
                vector ^= original
                checks.step()
        survivors.append(vector)
    result = _canonical(survivors, checks)
    checks.poll()
    return result


def confirms(basis: Sequence[int], rows: Sequence[int], *,
             check: Callable[[], None] | None = None) -> bool:
    """Test all frozen relations against all rows without filtering/mutation.

    Both sides are validated as full-width vectors before any short circuit.
    Empty basis/rows are vacuously True in pure mathematics; the method owner
    must label an empty-space confirmation NOT_APPLICABLE, not scientific pass.
    Callback conventions are those of canonical_basis, including a final check
    before returning False; validation of both sides precedes any short circuit.
    """
    checks = _Checks(check)
    frozen = _validated(basis, _WIDTH, checks)
    constraints = _validated(rows, _WIDTH, checks)
    for vector in frozen:
        checks.step()
        for row in constraints:
            checks.step()
            if (vector & row).bit_count() % 2:
                checks.poll()
                return False
    checks.poll()
    return True
