"""Scalar SIMON64/128 encryption owned by this clean cipher package.

This implementation exposes only complete-block encryption and the key
schedule needed for it; it does not retain or expose round traces.
"""

_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1
_Z3 = "11011011101011000110010111100000010010001010011100110100001111"


def _unsigned(value: int, bits: int, label: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{label} must be a built-in int")
    if not 0 <= value < (1 << bits):
        raise ValueError(f"{label} must fit in {bits} bits")


def _round_count(rounds: int) -> None:
    if type(rounds) is not int:
        raise TypeError("rounds must be a built-in int")
    if not 1 <= rounds <= 44:
        raise ValueError("SIMON64/128 rounds must be between 1 and 44")


def _ror32(word: int, distance: int) -> int:
    return ((word >> distance) | (word << (32 - distance))) & _MASK32


def _round_keys(key: int, rounds: int) -> tuple[int, ...]:
    """Generate the requested SIMON64/128 round-key prefix, low word first."""
    _unsigned(key, 128, "key")
    _round_count(rounds)
    words = [(key >> offset) & _MASK32 for offset in (0, 32, 64, 96)]
    for index in range(4, rounds):
        temporary = _ror32(words[index - 1], 3) ^ words[index - 3]
        temporary ^= _ror32(temporary, 1)
        words.append(
            words[index - 4]
            ^ temporary
            ^ 0xFFFFFFFC
            ^ int(_Z3[(index - 4) % 62])
        )
    return tuple(words[:rounds])


def round_keys(key: int, rounds: int) -> tuple[int, ...]:
    """Public read-only view of the SIMON64/128 round-key prefix.

    Exposed for post-hoc audits that compare a learned constant assignment with
    the true schedule. It never feeds model construction or training.
    """
    return _round_keys(key, rounds)


def encrypt(plaintext: int, key: int, rounds: int) -> int:
    """Encrypt one unsigned 64-bit block with a 128-bit key and 1..44 rounds.

    State words are packed as ``x`` in the high half and ``y`` in the low
    half. The ciphertext is the final SIMON state (no whitening is applied).
    Only built-in integers are accepted; booleans and integer subclasses are
    rejected to keep the public width contract explicit.
    """
    _unsigned(plaintext, 64, "plaintext")
    schedule = _round_keys(key, rounds)
    x, y = (plaintext >> 32) & _MASK32, plaintext & _MASK32
    for round_key in schedule:
        rotated1 = ((x << 1) | (x >> 31)) & _MASK32
        rotated8 = ((x << 8) | (x >> 24)) & _MASK32
        rotated2 = ((x << 2) | (x >> 30)) & _MASK32
        x, y = (y ^ (rotated1 & rotated8) ^ rotated2 ^ round_key) & _MASK32, x
    return ((x << 32) | y) & _MASK64
