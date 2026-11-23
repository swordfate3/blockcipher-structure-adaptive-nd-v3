from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from cipher_lab.execution import Stopped


def test_cube_windows_working_set_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import resources

    monkeypatch.setattr(resources, '_read_windows_working_set', lambda: 123456)
    assert resources.working_set_bytes(platform='win32') == 123456
    with pytest.raises(Stopped, match='RSS_UNSUPPORTED'):
        resources.working_set_bytes(platform='darwin')
    monkeypatch.setattr(resources, '_read_windows_working_set', lambda: 0)
    with pytest.raises(Stopped, match='RSS_UNAVAILABLE'):
        resources.working_set_bytes(platform='win32')

    def os_error() -> int:
        raise OSError('private')

    monkeypatch.setattr(resources, '_read_windows_working_set', os_error)
    with pytest.raises(Stopped, match='^RSS_UNAVAILABLE$'):
        resources.working_set_bytes(platform='win32')


def test_cube_gpu_identity_requires_real_uuid_match(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, resources

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', config.GPU_UUID)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(
        uuid=config.GPU_UUID.removeprefix('GPU-').upper(), name='Artificial GPU',
        total_memory=2**35))
    info = resources.device_identity()
    assert info == {'logical_device': 'cuda:0', 'physical_gpu_index': 1,
                    'uuid': config.GPU_UUID, 'name': 'Artificial GPU', 'total_bytes': 2**35}
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(
        uuid=None, name='Artificial GPU', total_memory=2**35))
    with pytest.raises(ValueError, match='GPU_IDENTITY'):
        resources.device_identity()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(ValueError, match='GPU_IDENTITY'):
        resources.device_identity()


def test_cube_windows_ffi_binds_once_and_samples_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import resources

    bindings: list[str] = []
    signatures: list[tuple[str, str]] = []
    records: list[ctypes.Structure] = []
    samples = iter([98765, 54321, None])

    class Function:
        def __setattr__(self, name: str, value: object) -> None:
            signatures.append((type(self).__name__, name))
            super().__setattr__(name, value)

    class Handle(Function):
        def __call__(self) -> int:
            return 17

    class Info(Function):
        def __call__(self, handle: int, pointer: object, size: int) -> int:
            assert handle == 17
            assert isinstance(pointer, type(ctypes.byref(ctypes.c_size_t())))
            record = getattr(pointer, '_obj')
            assert record.cb == size == ctypes.sizeof(record)
            records.append(record)
            record.PeakWorkingSetSize = 123456789
            working = next(samples)
            if working is None:
                return 0
            record.WorkingSetSize = working
            return 1

    def library(name: str, *, use_last_error: bool) -> object:
        assert use_last_error
        bindings.append(name)
        if name == 'kernel32':
            return SimpleNamespace(GetCurrentProcess=Handle())
        assert name == 'psapi'
        return SimpleNamespace(GetProcessMemoryInfo=Info())

    monkeypatch.setattr(ctypes, 'WinDLL', library, raising=False)
    monkeypatch.setattr(resources, '_windows_binding', None, raising=False)
    with pytest.raises(Stopped, match='^RSS_UNSUPPORTED$'):
        resources.working_set_bytes(platform='linux')
    assert bindings == []
    assert resources.working_set_bytes(platform='win32') == 98765
    assert resources.working_set_bytes(platform='win32') == 54321
    with pytest.raises(Stopped, match='^RSS_UNAVAILABLE$'):
        resources.working_set_bytes(platform='win32')
    assert bindings == ['kernel32', 'psapi']
    assert signatures == [('Handle', 'argtypes'), ('Handle', 'restype'),
                          ('Info', 'argtypes'), ('Info', 'restype')]
    assert len(records) == len({id(record) for record in records}) == 3
    assert len({type(record) for record in records}) == 1
