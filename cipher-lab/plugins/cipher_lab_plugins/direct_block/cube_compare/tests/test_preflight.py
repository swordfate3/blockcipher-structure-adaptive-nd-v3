from __future__ import annotations

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare.data import Words

from cipher_lab.ciphers import simon
from cipher_lab.execution import Meter


def test_public_checks_attempt_exactly_once(
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import preflight

    calls: list[tuple[str, tuple[int, ...], tuple[int, ...], int]] = []

    def scalar(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> tuple[int, ...]:
        calls.append(('scalar', points, key, rounds))
        return (0x44C8FC20B9DFA07A,) if rounds == 44 else points

    def batch(points: Words, key: tuple[int, ...], rounds: int) -> Words:
        words = tuple(map(int, points))
        calls.append(('batch', words, key, rounds))
        return np.array((0x44C8FC20B9DFA07A,) if rounds == 44 else words, dtype=np.uint64)

    monkeypatch.setattr(simon, 'encrypt_many', scalar)
    monkeypatch.setattr(simon, 'encrypt_batch', batch)
    preflight.public_cipher_checks(cube_meter)
    key = (0x03020100, 0x0B0A0908, 0x13121110, 0x1B1A1918)
    assert [(method, len(points), used, rounds) for method, points, used, rounds in calls] == [
        ('scalar', 1, key, 44), ('batch', 1, key, 44),
        ('scalar', 8, key, 7), ('batch', 8, key, 7),
    ]
    assert calls[0][1] == calls[1][1] == (0x656B696C20646E75,)
    assert calls[2][1] == calls[3][1] and len(set(calls[2][1])) == 8
    assert cube_meter.snapshot()['attempted_real_blocks'] == 18
    assert cube_meter.snapshot()['completed_real_blocks'] == 18


def test_uuid_conflict_rejected_before_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import preflight

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='GPU_IDENTITY'):
        preflight.gpu_gate()


@pytest.mark.parametrize('fault', ['', 'count', 'uuid', 'free', 'invalid-memory'])
def test_gpu_gate_never_falls_back(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from types import SimpleNamespace

    import torch
    from cipher_lab_plugins.direct_block.cube_compare import config, preflight

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', config.GPU_UUID)
    queries: list[str] = []

    def count() -> int:
        queries.append('count')
        return 2 if fault == 'count' else 1

    def props(device: str) -> SimpleNamespace:
        queries.append('properties')
        assert device == 'cuda:0'
        uuid = 'mismatch' if fault == 'uuid' else config.GPU_UUID.removeprefix('GPU-')
        return SimpleNamespace(uuid=uuid, name='Fake GPU', total_memory=32 * 2**30)

    def memory(device: str) -> tuple[int, int]:
        queries.append('memory')
        assert device == 'cuda:0'
        return (config.GPU_FREE_BYTES - (fault == 'free'),
                0 if fault == 'invalid-memory' else 32 * 2**30)

    monkeypatch.setattr(torch.cuda, 'device_count', count)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', props)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', memory)
    if fault:
        with pytest.raises(preflight.DeviceError):
            preflight.gpu_gate()
    else:
        result = preflight.gpu_gate()
        assert result['uuid'] == config.GPU_UUID and result['physical_gpu_index'] == 1
        assert result['visible_count'] == 1 and result['entry_free_bytes'] == config.GPU_FREE_BYTES
    expected = ['count', 'properties', 'memory']
    assert queries == expected[:{'count': 1, 'uuid': 2}.get(fault, 3)]
