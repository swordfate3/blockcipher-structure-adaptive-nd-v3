"""Pure SPECK32/64, official word order (k3,k2,k1,k0), round0 uses k0.

Block integer = (left16 << 16) | right16. No RNG, config, model or saved state.
Source: YuanYufeiISCAS/FourierNeuralDistinguishers main/core/speck.py;
only public round and schedule formulas adapted. Real KAT is a separate R0 gate.
"""
from __future__ import annotations


def _word(value: int) -> None:
    if type(value) is not int or not 0 <= value <= 65535:
        raise ValueError('expected uint16 word')


def rol16(x: int, shift: int) -> int:
    _word(x)
    if type(shift) is not int or not 0 <= shift < 16:
        raise ValueError('invalid rotation')
    return ((x << shift) | (x >> (16 - shift))) & 65535


def ror16(x: int, shift: int) -> int:
    _word(x)
    if type(shift) is not int or not 0 <= shift < 16:
        raise ValueError('invalid rotation')
    return ((x >> shift) | (x << (16 - shift))) & 65535


def enc_one_round(pair: tuple[int, int], key: int) -> tuple[int, int]:
    x, y = pair
    _word(x)
    _word(y)
    _word(key)
    x = ((ror16(x, 7) + y) & 65535) ^ key
    return x, rol16(y, 2) ^ x


def _rounds(rounds: int) -> None:
    if type(rounds) is not int or not 1 <= rounds <= 22:
        raise ValueError('rounds must be in 1..22')


def expand_key(key_words: tuple[int, int, int, int], rounds: int) -> tuple[int, ...]:
    _rounds(rounds)
    if type(key_words) is not tuple or len(key_words) != 4:
        raise ValueError('expected four key words in official high-to-low order')
    for word in key_words:
        _word(word)
    keys = [key_words[3]]
    lane = list(reversed(key_words[:3]))
    for i in range(rounds - 1):
        lane[i % 3], next_key = enc_one_round((lane[i % 3], keys[-1]), i)
        keys.append(next_key)
    return tuple(keys)


def encrypt_scalar(point: int, key_words: tuple[int, int, int, int], rounds: int) -> int:
    if type(point) is not int or not 0 <= point < 2**32:
        raise ValueError('point must be uint32')
    keys = expand_key(key_words, rounds)
    pair = point >> 16, point & 65535
    for key in keys:
        pair = enc_one_round(pair, key)
    return (pair[0] << 16) | pair[1]
