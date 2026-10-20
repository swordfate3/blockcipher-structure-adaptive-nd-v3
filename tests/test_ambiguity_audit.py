"""Identifiability audit helpers and the MSB-first round-key coordinate convention."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from cipher_experiments.ciphers.simon import encrypt, round_keys
from cipher_experiments.data.prediction import key_for_seed
from cipher_experiments.fitting.ambiguity import (
    assignment_bits,
    audit_assignment,
    evaluate_cone_bit,
    schedule_bits,
)
from cipher_experiments.representations.simon_cone import simon_cone

ROOT = Path(__file__).resolve().parents[1]


def probes(count: int = 256, seed: int = 5) -> tuple[int, ...]:
    generator = random.Random(seed)
    return tuple(generator.getrandbits(64) for _ in range(count))


@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
@pytest.mark.parametrize("target_bit", [0, 7, 40])
def test_msb_first_schedule_bits_reproduce_the_cipher(rounds: int, target_bit: int) -> None:
    key = key_for_seed(190502)
    cone = simon_cone(rounds, target_bit)
    bits = schedule_bits(round_keys(key, rounds), cone.round_key_coordinates)
    mask = 1 << (63 - target_bit)
    for plaintext in probes(64):
        assert evaluate_cone_bit(cone, plaintext, bits) == int(
            bool(encrypt(plaintext, key, rounds) & mask))


def test_one_coordinate_is_structurally_inert_and_therefore_unidentifiable() -> None:
    cone = simon_cone(4, 0)
    coordinates = cone.round_key_coordinates
    key = key_for_seed(1)
    bits = schedule_bits(round_keys(key, 4), coordinates)
    witness = audit_assignment(cone, probes(256, seed=11), bits, bits)
    assert witness.differing == ()
    assert witness.output_mismatches_vs_reference == 0
    assert (0, 2) in witness.individually_inert


def test_audit_detects_an_equivalent_flip_set() -> None:
    cone = simon_cone(4, 0)
    coordinates = cone.round_key_coordinates
    reference = schedule_bits(round_keys(key_for_seed(3), 4), coordinates)
    sample = probes(256, seed=13)
    base_out = [evaluate_cone_bit(cone, plaintext, reference) for plaintext in sample]

    # Search for a set of coordinates whose joint flip leaves the function intact.
    candidates = list(coordinates)[:6]
    found = None
    for first in candidates:
        for second in [None, *[c for c in candidates if c != first]]:
            probe = dict(reference)
            probe[first] ^= 1
            if second is not None:
                probe[second] ^= 1
            if [evaluate_cone_bit(cone, p, probe) for p in sample] == base_out:
                found = (first, second)
                break
        if found is not None:
            break
    if found is None:
        pytest.skip("no small equivalent flip set for this key; not a contract failure")
    learned = dict(reference)
    for coordinate in found:
        if coordinate is not None:
            learned[coordinate] ^= 1
    witness = audit_assignment(cone, sample, learned, reference)
    assert witness.differing
    assert witness.differing_set_flip_preserves_function
    assert witness.output_mismatches_vs_reference == 0


def test_audit_rejects_incomplete_assignments() -> None:
    cone = simon_cone(3, 0)
    coordinates = cone.round_key_coordinates
    bits = schedule_bits(round_keys(key_for_seed(2), 3), coordinates)
    partial = {c: bits[c] for c in coordinates[:-1]}
    with pytest.raises(ValueError, match="missing"):
        audit_assignment(cone, probes(8), partial, bits)
    with pytest.raises(ValueError, match="at least one"):
        audit_assignment(cone, (), bits, bits)
    with pytest.raises(ValueError, match="domain"):
        assignment_bits(-1, coordinates)


def test_tool_writes_report_for_a_synthetic_run(tmp_path: Path) -> None:
    cone = simon_cone(4, 0)
    coordinates = cone.round_key_coordinates
    key = key_for_seed(190502)
    reference = schedule_bits(round_keys(key, 4), coordinates)
    assignment = 0
    for index, coordinate in enumerate(coordinates):
        if reference[coordinate]:
            assignment |= 1 << index
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({
        "rounds": 4, "target_bit": 0, "key_seed": 190502, "label_mode": "cipher",
        "data_seed": 9}))
    (run / "fit.json").write_text(json.dumps({
        "assignment": assignment, "variable_coordinates": [list(c) for c in coordinates]}))
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/audit_hard_assignment.py"),
         "--run", str(run), "--output", str(output), "--probes", "128"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["learned_vs_real_cipher_mismatches"] == 0
    assert report["identifiable"] in (True, False)
    assert report["coordinates"] == len(coordinates)
