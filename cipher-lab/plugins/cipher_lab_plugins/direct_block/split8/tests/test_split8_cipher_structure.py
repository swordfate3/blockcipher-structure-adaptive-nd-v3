from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from cipher_lab.execution import Meter


def test_split8_cipher_validation_only() -> None:
    from cipher_lab.ciphers import simon32

    key = (0, 0, 1, 65535)
    for n in (1, 65536, 0):
        points = np.arange(n, dtype=np.uint32)
        simon32._batch_input(points)
    for bad_array in (np.zeros(65537, np.uint32), np.zeros(10, np.uint64),
                      np.zeros((2, 2), np.uint32), np.zeros(10, np.uint32)[::2],
                      np.zeros(10, np.dtype(np.uint32).newbyteorder('S')), [1, 2]):
        with pytest.raises((ValueError, TypeError)):
            simon32._batch_input(bad_array)
    for bad_point in (True, -1, 2**32, 1., np.uint32(1)):
        with pytest.raises(ValueError):
            simon32._scalar_input(bad_point)
    for point in (0, 2**32 - 1):
        simon32._scalar_input(point)
    for bad_key, rounds in ((key, True), (key, 0), (key, 33), ((0, 0, 0, 65536), 1),
                            ((False, 0, 0, 0), 2), ([0] * 4, 1), ((0, 0, 0), 1)):
        with pytest.raises(ValueError):
            simon32._parameters(bad_key, rounds)
    simon32._parameters(key, 32)
    text = Path(simon32.__file__).read_text()
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert not any('encrypt_scalar' in ast.unparse(child) for child in
                   functions['encrypt_batch'].body)
    assert 'from cipher_lab' not in text and 'import torch' not in text
    assert '11111010001001010110000111001101111101000100101011000011100110' in text
    with pytest.raises(AssertionError):
        simon32.encrypt_scalar(0, key, 1)
    with pytest.raises(AssertionError):
        simon32.encrypt_batch(np.zeros(1, np.uint32), key, 1)


@pytest.mark.parametrize('fault', ['', 'scalar', 'batch', 'mutated'])
def test_split8_public_kat_only_manual(split8_meter: Meter, fault: str) -> None:
    from cipher_lab_plugins.direct_block.split8 import kat

    calls: list[str] = []
    public = (0x0100, 0x0908, 0x1110, 0x1918)

    def scalar(point: int, key: tuple[int, int, int, int], rounds: int) -> int:
        assert (point, key, rounds) == (0x65656877, public, 32)
        calls.append('scalar')
        return 0 if fault == 'scalar' else 0xc69be9bb

    def batch(points: kat.Words, key: tuple[int, int, int, int], rounds: int) -> kat.Words:
        assert points.tolist() == [0x65656877] and key == public and rounds == 32
        calls.append('batch')
        if fault == 'mutated':
            points[0] = 0
        return np.array([0 if fault == 'batch' else 0xc69be9bb], np.uint32)

    if fault:
        with pytest.raises(ValueError):
            kat.check(split8_meter, scalar, batch)
    else:
        kat.check(split8_meter, scalar, batch)
    assert calls == (['scalar'] if fault == 'scalar' else ['scalar', 'batch'])
    counts = split8_meter.snapshot()
    assert counts['attempted_real_blocks'] == len(calls)
    assert counts['completed_real_blocks'] == len(calls) - bool(fault)
    assert counts['attempted_forward_rows'] == counts['attempted_optimizer_updates'] == 0
