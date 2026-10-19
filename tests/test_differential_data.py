from __future__ import annotations

import pytest
import torch

from cipher_experiments.ciphers.expanded import encrypt_expanded
from cipher_experiments.ciphers.present80 import Present80
from cipher_experiments.ciphers.simeck64 import Simeck64_128
from cipher_experiments.ciphers.simon64 import Simon64_128
from cipher_experiments.ciphers.skinny64 import Skinny64_64
from cipher_experiments.ciphers.tasks import (
    DEFAULT_TASKS,
    TaskSpec,
    public_features,
    validate_public,
)
from cipher_experiments.data.differential import generate_batch, generate_material


@pytest.mark.parametrize(
    ("cipher", "key", "plaintext", "rounds", "expected"),
    [
        (
            "simon64",
            0x1B1A1918131211100B0A090803020100,
            0x656B696C20646E75,
            44,
            0x44C8FC20B9DFA07A,
        ),
        ("present80", 0, 0, 31, 0x5579C1387B228445),
        (
            "skinny64",
            0xF5269826FC681238,
            0x06034F957724D19D,
            32,
            0xBB39DFB2429B8AC7,
        ),
        (
            "simeck64",
            0x1B1A1918131211100B0A090803020100,
            0x656B696C20646E75,
            44,
            0x45CE69025F7AB7ED,
        ),
    ],
)
def test_cipher_known_answers(
    cipher: str, key: int, plaintext: int, rounds: int, expected: int
) -> None:
    scalar = {
        "simon64": Simon64_128,
        "present80": Present80,
        "skinny64": Skinny64_64,
        "simeck64": Simeck64_128,
    }[cipher]
    assert scalar(rounds, key).encrypt(plaintext) == expected
    assert encrypt_expanded(cipher, rounds, [key], [plaintext]) == [expected]


@pytest.mark.parametrize("spec", DEFAULT_TASKS)
def test_small_differential_batches_are_balanced_replayable_and_well_formed(
    spec: TaskSpec,
) -> None:
    for groups in (2, 4):
        batch = generate_batch(spec, groups=groups, seed=2026)
        replay = generate_batch(spec, groups=groups, seed=2026)
        assert len(batch) == groups
        assert batch.bits.shape == (groups, 4, 2, 64)
        assert batch.bits.dtype == torch.float32
        assert batch.label.shape == (groups,)
        assert batch.label.dtype == torch.float32
        assert set(batch.label.tolist()) == {0.0, 1.0}
        assert int(batch.label.sum()) == groups // 2
        assert torch.equal(batch.bits, replay.bits)
        assert torch.equal(batch.label, replay.label)
        assert bool(((batch.bits == 0) | (batch.bits == 1)).all())


def test_batch_encrypts_ordered_pairs_under_each_group_key() -> None:
    for spec in DEFAULT_TASKS:
        material = generate_material(spec, groups=4, seed=29)
        batch = generate_batch(spec, groups=4, seed=29)
        assert batch.label.tolist() == list(material.labels)
        for group, (key, first, second) in enumerate(
            zip(material.keys, material.first, material.second, strict=True)
        ):
            for pair in range(4):
                expected = encrypt_expanded(
                    spec.cipher,
                    spec.rounds,
                    [key, key],
                    [first[pair], second[pair]],
                )
                actual = [
                    int("".join(str(int(bit)) for bit in batch.bits[group, pair, side]), 2)
                    for side in range(2)
                ]
                assert actual == expected


def test_material_has_one_key_per_group_and_fixed_xor_only_for_positive_groups() -> None:
    spec = DEFAULT_TASKS[0]
    material = generate_material(spec, groups=8, seed=17)
    assert len(material.keys) == len(material.first) == len(material.second) == 8
    for key in material.keys:
        assert 0 <= key < 1 << spec.key_bits
    for first, second, label in zip(
        material.first, material.second, material.labels, strict=True
    ):
        assert len(first) == len(second) == 4
        assert label in (0, 1)
        if label:
            assert second == tuple(value ^ spec.input_difference for value in first)
        else:
            assert second != tuple(value ^ spec.input_difference for value in first)


def test_public_features_are_cipher_only_and_validate_reviewed_cards() -> None:
    assert len({public_features(task.name) for task in DEFAULT_TASKS}) == 4
    for task in DEFAULT_TASKS:
        public = torch.tensor(public_features(task.name), dtype=torch.float32)
        assert public.shape == (13,)
        validate_public(public)
    with pytest.raises(KeyError):
        public_features("unknown")
    with pytest.raises(ValueError):
        validate_public(torch.zeros(13))


@pytest.mark.parametrize(
    ("groups", "seed"),
    [(1, 0), (3, 0), (65_538, 0), (2.0, 0), (2, -1), (2, 1 << 32), (2, True)],
)
def test_material_rejects_invalid_size_and_seed(groups: int, seed: int) -> None:
    with pytest.raises(ValueError):
        generate_material(DEFAULT_TASKS[0], groups=groups, seed=seed)


def test_task_spec_rejects_invalid_public_task_values() -> None:
    with pytest.raises(ValueError):
        TaskSpec("unknown", 1, 1)
    with pytest.raises(ValueError):
        TaskSpec("simon64", 45, 1)
    with pytest.raises(ValueError):
        TaskSpec("simon64", 1, 0)
