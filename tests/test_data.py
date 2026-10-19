import random

import torch

from cipher_experiments.ciphers import encrypt
from cipher_experiments.data import make_simon, make_synthetic


def _assert_splits(data: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    assert tuple(data) == ("train", "dev", "test")
    all_inputs: list[tuple[float, ...]] = []
    for inputs, targets in data.values():
        assert inputs.device.type == targets.device.type == "cpu"
        assert inputs.dtype == targets.dtype == torch.float32
        assert inputs.ndim == 2 and inputs.shape[1] == 64
        assert targets.shape == (inputs.shape[0], 1)
        assert bool(((inputs == 0) | (inputs == 1)).all())
        assert bool(((targets == 0) | (targets == 1)).all())
        all_inputs.extend(tuple(row.tolist()) for row in inputs)
    assert len(set(all_inputs)) == len(all_inputs)


def test_synthetic_data_shapes_determinism_and_learnable_label() -> None:
    first = make_synthetic(8, 4, 3, seed=17)
    second = make_synthetic(8, 4, 3, seed=17)
    _assert_splits(first)
    for name in first:
        x, y = first[name]
        assert torch.equal(x, second[name][0])
        assert torch.equal(y, second[name][1])
        assert torch.equal(y[:, 0], x[:, 0])


def test_simon_data_is_deterministic_and_uses_msb_target_index() -> None:
    args = (8, 4, 3)
    kwargs = {"seed": 17, "key_seed": 21, "rounds": 4, "target_bit": 0}
    first = make_simon(*args, **kwargs)
    second = make_simon(*args, **kwargs)
    _assert_splits(first)
    key = random.Random(kwargs["key_seed"]).getrandbits(128)
    for name in first:
        inputs, targets = first[name]
        assert torch.equal(inputs, second[name][0])
        assert torch.equal(targets, second[name][1])
        for row, target in zip(inputs, targets[:, 0], strict=True):
            plaintext = 0
            for bit in row.to(torch.int64).tolist():
                plaintext = (plaintext << 1) | bit
            ciphertext = encrypt(plaintext, key, kwargs["rounds"])
            expected = (ciphertext >> (63 - kwargs["target_bit"])) & 1
            assert int(target.item()) == expected


def test_generators_reject_invalid_bounds() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_synthetic(0, 1, 1, seed=1)
    with pytest.raises(ValueError):
        make_simon(1, 1, 1, seed=1, key_seed=2, rounds=45, target_bit=0)
    with pytest.raises(ValueError):
        make_simon(1, 1, 1, seed=1, key_seed=2, rounds=1, target_bit=64)
