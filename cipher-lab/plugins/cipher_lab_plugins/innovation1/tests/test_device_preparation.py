"""Exercise only the real setup function with an explicit fabricated Torch facade."""
from __future__ import annotations

import builtins
import os
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from cipher_lab_plugins.innovation1 import application, resources
from cipher_lab_plugins.innovation1.config import Recipe
from plugin_isolation import forbidden

from cipher_lab.execution import Clocks

_real_prepare = application.prepare_device  # autouse guard still poisons product call sites
_UUID = 'GPU-00000000-1111-2222-3333-444444444444'


@pytest.mark.parametrize('fault', [None, 'import', 'count', 'select', 'memory', 'fraction'])
def test_uuid_bound_before_slow_import_no_fallback(
    monkeypatch: pytest.MonkeyPatch, fault: str | None,
) -> None:
    recipe = Recipe()
    events: list[str] = []
    elapsed = [0.]
    fp32 = object()
    matmul = SimpleNamespace(allow_tf32=True)
    cudnn = SimpleNamespace(allow_tf32=True, benchmark=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'wrong-device')
    monkeypatch.delitem(sys.modules, 'torch')
    monkeypatch.setattr(application, 'Clocks', lambda: Clocks(
        lambda: elapsed[0], lambda: 0., lambda: datetime(2026, 1, 1, tzinfo=UTC)))
    monkeypatch.setattr(resources, 'production_rss_bytes', lambda: 7)
    ledger = application.make_resources(recipe)

    def record(event: str, actual: object, expected: object) -> None:
        events.append(event)
        assert actual == expected
        if fault == event:
            raise RuntimeError('raw device failure must not escape')

    def count() -> int:
        events.append('count')
        return 2 if fault == 'count' else 1

    def properties(device: int) -> Any:
        events.append('properties')
        assert device == 0
        return SimpleNamespace(total_memory=(1 if fault == 'memory' else 40 * 1024**3))

    fake = SimpleNamespace(float32=fp32, set_num_threads=lambda n: record('intra', n, 4),
        set_num_interop_threads=lambda n: record('inter', n, 1),
        set_default_dtype=lambda dtype: record('fp32', dtype, fp32),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
        cuda=SimpleNamespace(device_count=count, get_device_properties=properties,
                             set_device=lambda d: record('select', d, 0),
                             set_per_process_memory_fraction=lambda r, d:
                                 record('fraction', (r, d), (.5, 0))))
    original_import = builtins.__import__

    def import_fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name != 'torch':
            return original_import(name, *args, **kwargs)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == _UUID
        assert 'torch' not in sys.modules and not events
        events.append('import')
        elapsed[0] = 19.  # longer than the retired proof-age rule, no sleep or device
        if fault == 'import':
            raise ImportError('raw import failure')
        return fake

    monkeypatch.setattr(builtins, '__import__', import_fake)
    steps = ['import', 'count', 'intra', 'inter', 'fp32', 'select', 'properties', 'fraction']
    ledger.check()
    if fault is None:
        _real_prepare(recipe, _UUID)
        ledger.check()
        assert ledger.snapshot()['wall_elapsed_seconds'] == 19.
        assert matmul.allow_tf32 is cudnn.allow_tf32 is cudnn.benchmark is False
        assert events == steps
    else:
        with pytest.raises(resources.ResourceStop) as stopped:
            _real_prepare(recipe, _UUID)
        assert stopped.value.detail_code == {'import': 'TORCH_IMPORT_FAILED',
            'count': 'CUDA_COUNT_MISMATCH', 'select': 'CUDA_SELECT_FAILED',
            'memory': 'GPU_MEMORY_LIMIT', 'fraction': 'MEMORY_FRACTION_FAILED'}[fault]
        assert events == steps[:steps.index('properties' if fault == 'memory' else fault) + 1]


def test_device_guard_and_preimported_torch_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'unchanged')
    with pytest.raises(AssertionError, match='UNAUTHORIZED_SCIENCE'):
        application.prepare_device(Recipe(), _UUID)
    with pytest.raises(AssertionError, match='UNAUTHORIZED_SCIENCE'):
        resources.production_rss_bytes()
    monkeypatch.setattr(builtins, '__import__', forbidden)
    with pytest.raises(resources.ResourceStop, match='GPU_UNAVAILABLE') as stopped:
        _real_prepare(Recipe(), _UUID)
    assert stopped.value.detail_code == 'DEVICE_ENTRY_REJECTED'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'unchanged'
