"""NO complete encryption calls, including validation/KAT; artificial one round only."""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from cipher_lab.ciphers import simon


@pytest.fixture(autouse=True)
def no_complete_cipher(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("complete cipher/KAT is outside T2 authorization")
    monkeypatch.setattr(simon, "encrypt_many", forbidden)
    # Static tests parse source, never execute this entry.
    monkeypatch.setattr(simon, "encrypt_batch", forbidden, raising=False)


def test_batch_interface_is_keyword_independent_and_numpy_lazy() -> None:
    tree = ast.parse(Path(inspect.getfile(simon)).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "encrypt_batch" in functions
    function = functions["encrypt_batch"]
    assert [arg.arg for arg in function.args.args] == ["points", "key_words", "rounds"]
    annotation = function.args.args[0].annotation
    assert annotation is not None and function.returns is not None
    assert ast.unparse(annotation) == "NDArray[np.uint64]"
    assert ast.unparse(function.returns) == "NDArray[np.uint64]"
    # Runtime module scope must remain zero NumPy; TYPE_CHECKING may hold annotations.
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                   "numpy" in ast.unparse(node) for node in tree.body)
    assert "_round_words" in functions


def test_artificial_single_round_rotations_and_xor_do_not_mutate() -> None:
    x = np.array([0, 1, 0x80000000, 0xFFFFFFFF, 0x12345678], dtype=np.uint32)
    y = np.array([1, 2, 3, 4, 0x87654321], dtype=np.uint32)
    x_before, y_before = x.copy(), y.copy()
    # Literal high-bit wrap, AND and XOR examples, NOT a real plaintext/key chain.
    left, right = simon._round_words(x, y, 0xA5A5A5A5)
    # Last word: rol1&rol8=24402810, rol2=48d159e0;
    # XOR with y=87654321 and the artificial round constant gives 4e519774.
    assert left.tolist() == [0xA5A5A5A4, 0xA5A5A5A3, 0xA5A5A5A4, 0xA5A5A5A1, 0x4E519774]
    assert right.tolist() == x.tolist()
    assert left.dtype == right.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)
    # No result aliases an input, including the right Feistel word.
    assert not any(np.shares_memory(result, value) for result in (left, right) for value in (x, y))


@pytest.mark.parametrize("left,right,key", [
    (np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.uint32), 0),
    (np.zeros((1, 2), dtype=np.uint32), np.zeros(2, dtype=np.uint32), 0),
    (np.zeros(3, dtype=np.uint32), np.zeros(2, dtype=np.uint32), 0),
    (np.zeros(4, dtype=np.uint32)[::2], np.zeros(2, dtype=np.uint32), 0),
    (np.zeros(2, dtype=np.uint32), np.zeros(2, dtype=np.uint32), True),
    (np.zeros(2, dtype=np.uint32), np.zeros(2, dtype=np.uint32), -1),
    (np.zeros(2, dtype=np.uint32), np.zeros(2, dtype=np.uint32), 1 << 32),
])
def test_artificial_round_rejects_malformed_words(left: object, right: object, key: int) -> None:
    from typing import cast

    from numpy.typing import NDArray

    with pytest.raises((TypeError, ValueError)):
        simon._round_words(cast(NDArray[np.uint32], left), cast(NDArray[np.uint32], right), key)
