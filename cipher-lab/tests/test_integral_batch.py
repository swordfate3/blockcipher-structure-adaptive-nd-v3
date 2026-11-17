"""Artificial uint64 observations only: no cipher, labels or optimizer updates."""

import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from cipher_lab.integral.observables import aggregate_batch, aggregate_words

LENGTHS = (0, 1, 3, 63, 64, 65, 1023)
LITERALS = (
    0, 1, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000, 0x4000000000000000,
    0x8000000000000001, 3, 2, 0x0123456789ABCDEF, 0xFEDCBA9876543210,
    0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xDEADBEEFCAFEBABE,
    0x13579BDF2468ACE0, 0xF00F3333CCCC0FF0,
)


@pytest.fixture(autouse=True)
def forbid_scientific_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    # No production cipher or training engine can be imported by these tests.
    monkeypatch.setitem(sys.modules, "cipher_lab.ciphers", None)
    monkeypatch.setitem(sys.modules, "torch", None)


def reference_terms(words: list[int]) -> Iterator[int]:
    """Independent literal MSB0 definition: units, then two nested pair loops."""
    for left in range(64):
        yield sum((word >> (63 - left)) & 1 for word in words) % 2
    for left in range(64):
        for right in range(left + 1, 64):
            yield sum(
                ((word >> (63 - left)) & 1) * ((word >> (63 - right)) & 1)
                for word in words
            ) % 2


def artificial_words(length: int, pattern: str) -> NDArray[np.uint64]:
    patterns = {
        "zero": (0,),
        "ones": (0xFFFFFFFFFFFFFFFF,),
        "high-low": (0x8000000000000000, 1, 0x8000000000000001),
        "repeats": (0xDEADBEEFCAFEBABE, 0xDEADBEEFCAFEBABE, 3, 3),
        "mixed": LITERALS,
    }
    values = patterns[pattern]
    return np.array([values[index % len(values)] for index in range(length)], dtype=np.uint64)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("pattern", ["zero", "ones", "high-low", "repeats", "mixed"])
def test_batch_matches_literal_all_2080_terms(length: int, pattern: str) -> None:
    # Catches bit reversal, pair order, padding parity and XOR-before-product.
    words = artificial_words(length, pattern)
    before = words.tobytes()
    result = aggregate_batch(words)
    assert type(result) is int
    assert 0 <= result < (1 << 2080)
    expected = list(reference_terms([int(word) for word in words]))
    assert len(expected) == 2080
    for coordinate, term in enumerate(expected):
        assert (result >> (2079 - coordinate)) & 1 == term, coordinate
    assert result == aggregate_words(int(word) for word in words)
    assert words.tobytes() == before


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], 0), ([0], 0), ([1], 1 << 2016),
        ([0x8000000000000000], 1 << 2079),
        ([0xFFFFFFFFFFFFFFFF], (1 << 2080) - 1),
        ([0x8000000000000000, 0x4000000000000000], (1 << 2079) | (1 << 2078)),
        ([3, 2, 1], 1), ([3, 3], 0),
    ],
)
def test_batch_hand_checked_fixtures(values: list[int], expected: int) -> None:
    assert aggregate_batch(np.array(values, dtype=np.uint64)) == expected


def test_batch_pair_product_is_not_product_of_xors() -> None:
    words = np.array([1 << 63, 1 << 62], dtype=np.uint64)
    assert (aggregate_batch(words) >> 2015) & 1 == 0  # First pair [0, 1].
    assert (aggregate_batch(np.array([(1 << 63) | (1 << 62)], dtype=np.uint64)) >> 2015) & 1 == 1


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("chunk", [1, 7, 8, 63, 64, 65, 257])
def test_batch_chunk_xor_matches_whole(length: int, chunk: int) -> None:
    words = artificial_words(length, "mixed")
    result = 0
    for start in range(0, length, chunk):
        result ^= aggregate_batch(words[start:start + chunk])
    assert result == aggregate_batch(words)


@pytest.mark.parametrize("value", [0, 0xFFFFFFFFFFFFFFFF])
def test_batch_inclusive_maximum_length(value: int) -> None:
    # Fixed correctness boundary, not a timing/RSS/performance scenario.
    words = np.full(65536, value, dtype=np.uint64)
    assert aggregate_batch(words) == 0
    words[-1] = 0
    expected = (1 << 2080) - 1 if value else 0
    assert aggregate_batch(words) == expected


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.uint32, np.float64, object])
def test_batch_rejects_wrong_dtype_without_coercion(dtype: Any) -> None:
    with pytest.raises(TypeError):
        aggregate_batch(np.array([0, 1], dtype=dtype))


@pytest.mark.parametrize("bad", [None, True, 1, [0, 1], (0, 1)])
def test_batch_rejects_non_arrays(bad: Any) -> None:
    with pytest.raises(TypeError):
        aggregate_batch(bad)


@pytest.mark.parametrize("shape", [(), (1, 1), (0, 2)])
def test_batch_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        aggregate_batch(np.zeros(shape, dtype=np.uint64))


def test_batch_rejects_noncontiguous_and_negative_stride() -> None:
    words = np.arange(10, dtype=np.uint64)
    for bad in (words[::2], words[::-1]):
        assert not bad.flags.c_contiguous
        with pytest.raises(ValueError):
            aggregate_batch(bad)


def test_batch_native_endian_only_and_byte_significance() -> None:
    values = [0x8000000000000000, 0x0100000000000000, 0x80, 1, 0x0123456789ABCDEF]
    native = np.array(values, dtype=np.dtype("=u8"))
    assert aggregate_batch(native) == aggregate_words(values)
    other = np.dtype(">u8" if sys.byteorder == "little" else "<u8")
    with pytest.raises(TypeError):
        aggregate_batch(np.array(values, dtype=other))


def test_batch_rejects_oversized_input() -> None:
    with pytest.raises(ValueError):
        aggregate_batch(np.zeros(65537, dtype=np.uint64))


def test_batch_does_not_expose_input_values_in_validation_error() -> None:
    canary = "SYNTHETIC_PRIVATE_VALUE_DO_NOT_ECHO"
    with pytest.raises(TypeError) as caught:
        aggregate_batch(np.array([canary], dtype=object))
    assert canary not in str(caught.value)


@pytest.mark.parametrize("length", [0, 65])
def test_batch_accepts_readonly_input_without_mutation(length: int) -> None:
    words = artificial_words(length, "mixed")
    before = words.tobytes()
    words.flags.writeable = False
    assert aggregate_batch(words) == aggregate_words(int(word) for word in words)
    assert words.tobytes() == before


@pytest.mark.parametrize("length", [0, 3])
def test_batch_callback_at_entry_pair_boundaries_and_exit(
    length: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Count real packed AND operations, not source text or private helper names.
    original = np.bitwise_and
    pairs = 0
    calls: list[int] = []

    def counted_and(*args: Any, **kwargs: Any) -> Any:
        nonlocal pairs
        if kwargs.get("out") is not None and kwargs["out"].dtype == np.dtype(np.uint8):
            pairs += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "bitwise_and", counted_and)
    aggregate_batch(artificial_words(length, "mixed"), check=lambda: calls.append(pairs))
    expected = [0, 0] if length == 0 else [0, *range(64, 2017, 64), 2016]
    assert calls == expected


class CallbackStop(BaseException):
    """Synthetic interruption must propagate without wrapping or partial result."""


@pytest.mark.parametrize("stop_at", [1, 2, 17, 33])
def test_batch_callback_exception_identity_and_input_unchanged(stop_at: int) -> None:
    words = artificial_words(65, "mixed")
    before = words.tobytes()
    calls = 0
    failure = CallbackStop("SYNTHETIC_CALLBACK_CANARY")

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls == stop_at:
            raise failure

    with pytest.raises(CallbackStop) as caught:
        aggregate_batch(words, check=check)
    assert caught.value is failure
    assert calls == stop_at
    assert words.tobytes() == before


@pytest.mark.parametrize("stop_at", [1, 2])
def test_empty_batch_callback_exception_identity(stop_at: int) -> None:
    failure = CallbackStop("SYNTHETIC_EMPTY_CALLBACK_CANARY")
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls == stop_at:
            raise failure

    with pytest.raises(CallbackStop) as caught:
        aggregate_batch(np.array([], dtype=np.uint64), check=check)
    assert caught.value is failure
    assert calls == stop_at


def test_scalar_import_and_operation_do_not_import_numpy() -> None:
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch'} or name.startswith('cipher_lab.ciphers'):
        raise AssertionError('forbidden scientific import')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from cipher_lab.integral.observables import aggregate_words
assert aggregate_words([3, 2, 1]) == 1
assert aggregate_words([]) == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
    )
    assert result.returncode == 0, result.stderr
