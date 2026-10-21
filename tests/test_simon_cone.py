"""Synthetic symbolic-graph checks; cipher evaluation stays in test code only."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cipher_experiments.ciphers.simon import _round_keys, encrypt
from cipher_experiments.models.prediction import support_indices
from cipher_experiments.representations.simon_cone import ConeNode, SimonCone, simon_cone


def _interpret(cone: SimonCone, plaintext: int, round_keys: tuple[int, ...]) -> int:
    """Independent tiny interpreter for test-only symbolic graph evaluation."""
    values: list[int] = []
    for node in cone.nodes:
        if node.kind == "plaintext":
            (bit,) = node.coordinate
            values.append((plaintext >> (63 - bit)) & 1)
        elif node.kind == "round_key":
            round_index, bit = node.coordinate
            values.append((round_keys[round_index] >> (31 - bit)) & 1)
        elif node.kind == "and":
            left, right = node.args
            values.append(values[left] & values[right])
        elif node.kind == "xor":
            left, right = node.args
            values.append(values[left] ^ values[right])
        else:
            raise AssertionError(f"unexpected graph node: {node.kind}")
    return values[cone.output]


@pytest.mark.parametrize("rounds", range(1, 7))
@pytest.mark.parametrize("target_bit", [0, 31, 32, 63])
def test_cone_evaluation_matches_scalar_simon(rounds: int, target_bit: int) -> None:
    samples = (
        (0x0123456789ABCDEF, 0x000102030405060708090A0B0C0D0E0F),
        (0xFEDCBA9876543210, 0x1B1A1918131211100B0A090803020100),
        (0xA5A55A5AC3C33C3C, 0xFFEEDDCCBBAA99887766554433221100),
    )
    cone = simon_cone(rounds, target_bit)
    for plaintext, key in samples:
        round_keys = _round_keys(key, rounds)
        expected = (encrypt(plaintext, key, rounds) >> (63 - target_bit)) & 1
        assert _interpret(cone, plaintext, round_keys) == expected


@pytest.mark.parametrize("rounds", range(1, 7))
@pytest.mark.parametrize("target_bit", range(64))
def test_cone_plaintext_leaves_match_public_support(rounds: int, target_bit: int) -> None:
    cone = simon_cone(rounds, target_bit)
    assert cone.plaintext_indices == support_indices(rounds, target_bit)
    assert cone.plaintext_indices == tuple(sorted(set(cone.plaintext_indices)))
    assert cone.round_key_coordinates == tuple(sorted(set(cone.round_key_coordinates)))


def test_one_round_high_bit_has_expected_minimal_cone_and_gate_counts() -> None:
    cone = simon_cone(1, 0)
    assert cone.plaintext_indices == (1, 2, 8, 32)
    assert cone.round_key_coordinates == ((0, 0),)
    assert cone.gate_counts == {"AND": 1, "XOR": 3}
    assert cone.nodes[cone.output].kind == "xor"


def test_cones_are_forward_dags_with_unique_memoized_terminals() -> None:
    cone = simon_cone(6, 0)
    assert cone.output == len(cone.nodes) - 1
    for index, node in enumerate(cone.nodes):
        if node.kind in ("and", "xor"):
            assert len(node.args) == 2
            assert all(argument < index for argument in node.args)
    plaintext_nodes = [node.coordinate for node in cone.nodes if node.kind == "plaintext"]
    key_nodes = [node.coordinate for node in cone.nodes if node.kind == "round_key"]
    assert len(plaintext_nodes) == len(set(plaintext_nodes))
    assert len(key_nodes) == len(set(key_nodes))


def test_production_module_has_no_cipher_or_experiment_runtime_dependencies() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/cipher_experiments/representations/simon_cone.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    roots.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert roots <= {"__future__", "dataclasses", "functools"}


def test_nodes_and_cones_are_immutable_and_reject_malformed_graphs() -> None:
    cone = simon_cone(1, 0)
    with pytest.raises(FrozenInstanceError):
        cone.output = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cone.nodes[0].kind = "xor"  # type: ignore[misc]

    with pytest.raises(ValueError, match="prior nodes"):
        SimonCone((ConeNode("xor", args=(0, 0)),), 0)
    with pytest.raises(ValueError, match="round-key coordinate"):
        SimonCone((ConeNode("round_key", coordinate=(16, 0)),), 0)
    with pytest.raises(ValueError, match="reachable"):
        SimonCone(
            (
                ConeNode("plaintext", coordinate=(0,)),
                ConeNode("plaintext", coordinate=(1,)),
            ),
            0,
        )


@pytest.mark.parametrize("rounds", [0, 17, True, 1.0])
def test_cone_rejects_invalid_round_contract(rounds: object) -> None:
    error = TypeError if type(rounds) is not int else ValueError
    with pytest.raises(error):
        simon_cone(rounds, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("target_bit", [-1, 64, True, 0.0])
def test_cone_rejects_invalid_target_bit_contract(target_bit: object) -> None:
    error = TypeError if type(target_bit) is not int else ValueError
    with pytest.raises(error):
        simon_cone(1, target_bit)  # type: ignore[arg-type]


@pytest.mark.parametrize("rounds", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("bit", [0, 5, 31])
def test_low_word_cone_is_the_previous_round_high_word(rounds: int, bit: int) -> None:
    """Feistel swap: ciphertext low word at round r is the round r-1 high word.

    So a cube over low-word plaintext bits is one round shallower than the same
    cube over high-word bits, and any low-word claim overstates rounds by one.
    """
    here = simon_cone(rounds, 32 + bit)
    previous = simon_cone(rounds - 1, bit)
    assert here.plaintext_indices == previous.plaintext_indices
    assert len(here.nodes) == len(previous.nodes)
    assert here.gate_counts == previous.gate_counts
    # Not merely isomorphic: the cone itself is identical, keys included.
    assert here.round_key_coordinates == previous.round_key_coordinates
    assert here.output == previous.output
    assert [(n.kind, n.args) for n in here.nodes] == [(n.kind, n.args) for n in previous.nodes]
