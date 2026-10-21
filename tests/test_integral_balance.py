"""Exact cube-balance engine versus brute-force enumeration on real SIMON."""
from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from cipher_experiments.ciphers.simon import encrypt
from cipher_experiments.representations.integral import (
    CubeBalance,
    brute_force_cube_parity,
    cube_balance,
)
from cipher_experiments.representations.simon_cone import simon_cone


def _bit_oracle(rounds: int, target_bit: int, key: int) -> Callable[[int], int]:
    mask = 1 << (63 - target_bit)
    return lambda plaintext: int(bool(encrypt(plaintext, key, rounds) & mask))


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("active", [
    (32,), (0,), (32, 33), (0, 1), (0, 1, 2), (32, 33, 34),
    (1, 8), (0, 2, 31), (33, 40, 47),
])
def test_claimed_balance_holds_under_enumeration(rounds: int, active: tuple[int, ...]) -> None:
    """Soundness: whenever the engine claims balance, real encryption agrees.

    The engine over-approximates, so the reverse direction is not asserted here;
    ``test_engine_finds_conclusive_cases`` checks that it is not vacuous.
    """
    cone = simon_cone(rounds, 0)
    predicted = cube_balance(cone, set(active))
    assert isinstance(predicted, CubeBalance)
    if not predicted.balanced:
        return
    oracle = _bit_oracle(rounds, 0, random.Random(rounds * 7 + len(active)).getrandbits(128))
    for fixed in (0x0123456789ABCDEF, 0xFEDCBA9876543210, 0):
        parity = brute_force_cube_parity(cone, active, fixed=fixed, encrypt_bit=oracle)
        assert parity == 0


@pytest.mark.parametrize("target_bit", [0, 5, 31, 33, 63])
def test_claimed_balance_holds_for_every_key(target_bit: int) -> None:
    """A balance proof must survive any key, since keys enter as constants."""
    rounds, active = 2, (0, 1)
    cone = simon_cone(rounds, target_bit)
    predicted = cube_balance(cone, set(active))
    if not predicted.balanced:
        return
    for seed in (1, 2, 3):
        oracle = _bit_oracle(rounds, target_bit, random.Random(seed).getrandbits(128))
        parity = brute_force_cube_parity(cone, active, fixed=0x00FF00FF00FF00FF,
                                         encrypt_bit=oracle)
        assert parity == 0


def test_engine_finds_conclusive_cases() -> None:
    """The engine must actually decide some cubes, not silently abstain always."""
    decided = [act for act in [(32,), (33,), (0,), (32, 33), (33, 40), (0, 1)]
               if cube_balance(simon_cone(2, 33), set(act)).balanced]
    assert decided, "engine never claimed balance on a known-balanced family"


def test_engine_reports_unproven_when_cube_bit_is_the_output_function() -> None:
    """One-round output bit 33 is plaintext bit 1, so cubing bit 1 is not balanced."""
    cone = simon_cone(1, 33)
    assert not cube_balance(cone, {1}).balanced
    # Cubing a different bit is a constant over the cube, hence balanced.
    assert cube_balance(cone, {33}).balanced


def test_one_round_low_word_cube_is_balanced() -> None:
    """Round 1 outputs the previous high word as the low half, so that is balanced."""
    assert cube_balance(simon_cone(1, 33), {32}).balanced


def test_rejects_bad_input() -> None:
    cone = simon_cone(2, 0)
    with pytest.raises(ValueError, match="coordinates"):
        cube_balance(cone, {64})
    with pytest.raises(TypeError, match="SimonCone"):
        cube_balance("cone", {1})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SimonCone"):
        cube_balance("cone", {1})  # type: ignore[arg-type]


def test_engine_is_key_free() -> None:
    """No key material may enter this module: it must only import the cone type."""
    import ast
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1]
            / "src/cipher_experiments/representations/integral.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported.update(alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names)
    assert imported <= {"__future__", "dataclasses", "collections.abc", "simon_cone"}
