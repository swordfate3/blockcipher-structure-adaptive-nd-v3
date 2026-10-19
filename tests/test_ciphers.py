import pytest

from cipher_experiments.ciphers import encrypt


def test_simon64_128_known_answer_44_rounds() -> None:
    plaintext = 0x656B696C20646E75
    key = 0x1B1A1918131211100B0A090803020100
    assert encrypt(plaintext, key, 44) == 0x44C8FC20B9DFA07A


def test_simon_rejects_invalid_types_and_ranges() -> None:
    with pytest.raises(TypeError):
        encrypt(True, 0, 1)
    with pytest.raises(ValueError):
        encrypt(1 << 64, 0, 1)
    with pytest.raises(TypeError):
        encrypt(0, False, 1)
    with pytest.raises(ValueError):
        encrypt(0, 1 << 128, 1)
    with pytest.raises(ValueError):
        encrypt(0, 0, 0)
    with pytest.raises(ValueError):
        encrypt(0, 0, 45)
