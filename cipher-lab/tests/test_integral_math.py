"""Artificial truth tables only; no cipher, sampling or scientific panel."""

from collections.abc import Callable, Iterable
from typing import Any

import pytest

from cipher_lab.integral.linear import canonical_basis, confirms, kernel, survivor_basis
from cipher_lab.integral.observables import aggregate_words

WIDTH = 2080


def span(vectors: Iterable[int]) -> set[int]:
    """Enumerate a small span by subsets, independently of elimination."""
    result = {0}
    for vector in vectors:
        result |= {value ^ vector for value in result}
    return result


def truth_basis(space: set[int], width: int) -> tuple[int, ...]:
    """RREF rows are the least integers in each nonempty leading-bit class."""
    return tuple(
        min(candidates)
        for bit in reversed(range(width))
        if (candidates := [value for value in space if value.bit_length() == bit + 1])
    )


def truth_kernel(rows: Iterable[int], width: int) -> set[int]:
    constraints = tuple(rows)
    # Separate bit-by-bit dot product, not the implementation's packed parity.
    return {
        value for value in range(1 << width)
        if all(
            sum(((value >> bit) & 1) * ((row >> bit) & 1) for bit in range(width)) % 2 == 0
            for row in constraints
        )
    }


@pytest.mark.parametrize("bit", range(64))
def test_all_unit_coordinates_msb0(bit: int) -> None:
    assert aggregate_words([1 << (63 - bit)]) == 1 << (2079 - bit)


@pytest.mark.parametrize("left", range(63))
def test_all_distinct_pairs_in_lexicographic_order(left: int) -> None:
    # Each two-hot input has exactly its two units and one quadratic term.
    for right in range(left + 1, 64):
        index = 64 + left * (127 - left) // 2 + right - left - 1
        expected = (1 << (2079 - left)) | (1 << (2079 - right)) | (1 << (2079 - index))
        assert aggregate_words([(1 << (63 - left)) | (1 << (63 - right))]) == expected


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ((), 0), ((0,), 0), (((1 << 64) - 1,), (1 << 2080) - 1),
        ((1 << 63, 1 << 62), (1 << 2079) | (1 << 2078)),
        ((3, 3), 0),
        ((3, 2, 1), 1),  # XOR(words)=0 but the last quadratic term is one.
    ],
)
def test_pointwise_then_xor_not_product_of_xors(words: tuple[int, ...], expected: int) -> None:
    assert aggregate_words(iter(words)) == expected


def test_two_point_quadratic_counterexample() -> None:
    assert aggregate_words([2, 1]) == (3 << 2016)
    assert aggregate_words([3]) == (3 << 2016) | 1


@pytest.mark.parametrize("width", [1, 2, 3])
def test_exhaustive_independent_small_truth_tables(width: int) -> None:
    # Every subset of possible rows, including zero and the full-rank case.
    for selection in range(1 << (1 << width)):
        rows = tuple(value for value in range(1 << width) if selection & (1 << value))
        expected_span = span(rows)
        expected_kernel = truth_kernel(rows, width)
        assert canonical_basis(iter(rows), width) == truth_basis(expected_span, width)
        actual = kernel(iter(rows), width)
        assert actual == truth_basis(expected_kernel, width)
        assert span(actual) == expected_kernel
        assert len(actual) + len(canonical_basis(rows, width)) == width
        assert canonical_basis(reversed(rows + rows + (0,)), width) == truth_basis(
            expected_span, width
        )


@pytest.mark.parametrize(
    ("rows", "width", "reduced", "null"),
    [
        ((), 1, (), (1,)), ((0, 0), 1, (), (1,)), ((1, 1), 1, (1,), ()),
        ((0b110, 0b011), 3, (0b101, 0b011), (0b111,)),
        ((0b110, 0b101), 3, (0b101, 0b011), (0b111,)),
        ((0b100, 0b010, 0b001), 3, (4, 2, 1), ()),
        ((0b1001, 0b0011), 4, (0b1001, 0b0011), (0b1011, 0b0100)),
    ],
)
def test_literal_rref_and_kernel_fixtures(
    rows: tuple[int, ...], width: int, reduced: tuple[int, ...], null: tuple[int, ...]
) -> None:
    assert canonical_basis(rows, width) == reduced
    assert kernel(rows, width) == null


def test_full_2080_dimensions_are_not_reduced_to_64() -> None:
    units = tuple(1 << bit for bit in reversed(range(WIDTH)))
    assert kernel([]) == units
    assert kernel([0, 0]) == units
    assert len(kernel([1 << 2079])) == 2079
    assert kernel([1 << 2079]) == units[1:]
    assert kernel([1]) == units[:-1]
    assert kernel(iter(units)) == ()
    assert canonical_basis(iter(units)) == units
    assert canonical_basis([1, (1 << 2079) | 1, 1]) == (1 << 2079, 1)
    # A single measured constraint necessarily leaves a huge empirical space.
    one_constraint = kernel([(1 << 2080) - 1])
    assert len(one_constraint) == 2079
    assert all(value.bit_count() % 2 == 0 for value in one_constraint)


@pytest.mark.parametrize("offset", [0, 2077])
@pytest.mark.parametrize(
    "discovery",
    [(), (0,), (3, 5), (3, 5, 6, 3, 0), (1, 2, 4), (7,)],
)
@pytest.mark.parametrize("screen", [(), (0,), (1,), (3,), (1, 2), (1, 2, 4), (7, 7)])
def test_survivors_match_independent_intersection_truth_table(
    offset: int, discovery: tuple[int, ...], screen: tuple[int, ...]
) -> None:
    expected_space = span(discovery) & truth_kernel(screen, 3)
    expected = tuple(value << offset for value in truth_basis(expected_space, 3))
    assert survivor_basis(
        tuple(value << offset for value in discovery),
        tuple(value << offset for value in screen),
    ) == expected


@pytest.mark.parametrize("offset", [0, 2078])
def test_exhaustive_two_bit_intersections(offset: int) -> None:
    # All 16 subsets on each side, independently enumerating the intersection.
    subsets = [tuple(row for row in range(4) if selection & (1 << row))
               for selection in range(16)]
    for discovery in subsets:
        for screen in subsets:
            expected = span(discovery) & truth_kernel(screen, 2)
            actual = survivor_basis(
                tuple(value << offset for value in discovery),
                tuple(value << offset for value in screen),
            )
            assert actual == tuple(value << offset for value in truth_basis(expected, 2))


def test_screen_keeps_combinations_not_just_individual_surviving_basis_rows() -> None:
    assert survivor_basis((3, 5), (1,)) == (6,)
    assert survivor_basis((), ()) == ()
    assert survivor_basis((1,), (1,)) == ()


@pytest.mark.parametrize(
    ("basis", "rows", "expected"),
    [
        ((), (), True), ((), (1,), True), ((1,), (), True),
        ((0,), (1,), True), ((3, 5), (7,), True), ((3, 5), (1,), False),
        ((1 << 2079, 1), (1 << 2079,), False),
        (((1 << 2079) | 1,), ((1 << 2079) | 1,), True),
    ],
)
def test_confirmation_checks_every_frozen_basis_vector_without_mutation(
    basis: tuple[int, ...], rows: tuple[int, ...], expected: bool
) -> None:
    mutable_basis = list(basis)
    mutable_rows = list(rows)
    assert confirms(mutable_basis, mutable_rows) is expected
    assert mutable_basis == list(basis)
    assert mutable_rows == list(rows)


def test_single_row_flip_fails_without_rescuing_basis() -> None:
    basis = survivor_basis((3, 5), (7,))
    assert confirms(basis, [7])
    assert not confirms(basis, [6])
    assert basis == (5, 3)


class IntSubclass(int):
    pass


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", None, IntSubclass(1)])
def test_non_builtin_int_words_rejected(bad: Any) -> None:
    with pytest.raises(TypeError):
        aggregate_words([0, bad])


@pytest.mark.parametrize("bad", [-1, 1 << 64, (1 << 64) + 1])
def test_out_of_range_words_rejected(bad: int) -> None:
    with pytest.raises(ValueError):
        aggregate_words([bad])


@pytest.mark.parametrize("operation", [canonical_basis, kernel])
@pytest.mark.parametrize("bad", [True, False, 1.0, "3", None, IntSubclass(3)])
def test_non_builtin_width_rejected(
    operation: Callable[..., tuple[int, ...]], bad: Any
) -> None:
    with pytest.raises(TypeError):
        operation([], bad)


@pytest.mark.parametrize("operation", [canonical_basis, kernel])
@pytest.mark.parametrize("bad", [-1, 0, 2081])
def test_out_of_range_width_rejected(
    operation: Callable[..., tuple[int, ...]], bad: int
) -> None:
    with pytest.raises(ValueError):
        operation([], bad)


@pytest.mark.parametrize("operation", [canonical_basis, kernel])
@pytest.mark.parametrize("width", [1, 3, 2080])
@pytest.mark.parametrize("bad", [True, False, 1.0, "1", None, IntSubclass(1)])
def test_non_builtin_vectors_rejected(
    operation: Callable[..., tuple[int, ...]], width: int, bad: Any
) -> None:
    with pytest.raises(TypeError):
        operation(iter([0, bad]), width)


@pytest.mark.parametrize("operation", [canonical_basis, kernel])
@pytest.mark.parametrize("width", [1, 3, 2080])
def test_vector_range_boundaries(operation: Callable[..., tuple[int, ...]], width: int) -> None:
    assert operation([(1 << width) - 1], width) is not None
    for bad in (-1, 1 << width, (1 << width) + 1):
        with pytest.raises(ValueError):
            operation(iter([0, bad]), width)


@pytest.mark.parametrize("operation", [survivor_basis, confirms])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize(
    ("bad", "error"),
    [(True, TypeError), (False, TypeError), (1.0, TypeError), ("1", TypeError),
     (None, TypeError), (IntSubclass(1), TypeError), (-1, ValueError),
     (1 << 2080, ValueError)],
)
def test_no_empty_or_failure_shortcut_hides_invalid_vectors(
    operation: Callable[..., object], side: int, bad: Any, error: type[Exception]
) -> None:
    for other in ([], [1]):
        arguments = [other, other]
        arguments[side] = [1, bad]
        with pytest.raises(error):
            operation(*arguments)


def test_many_duplicate_constraints_and_one_shot_generators() -> None:
    assert canonical_basis((value for value in [7, 7, 0, 7]), 3) == (7,)
    assert kernel((7 for _ in range(4097)), 3) == (5, 3)
    assert aggregate_words(value for value in [3, 2, 1]) == 1


def test_full_2080_survivor_space_and_confirmation_edges() -> None:
    full = tuple(1 << bit for bit in reversed(range(WIDTH)))
    assert survivor_basis(full, ()) == full
    assert survivor_basis(full, (1 << 2079, 1)) == full[1:-1]
    assert confirms(full, [0, 0])
    assert not confirms(full, [1])
    assert not confirms(full, [1 << 2079])
    assert confirms((), full)  # Vacuous math truth; method must label NOT_APPLICABLE.
