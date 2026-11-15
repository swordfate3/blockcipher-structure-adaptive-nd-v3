"""Only fake errors/Job/clock; protected per-process before project imports."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_speck_resources import resource_guard  # noqa: F401

from cipher_lab.speck_study import native_control, native_parent
from cipher_lab.speck_study.resources import ResourceStop, Supervisor


def test_adopt_then_supervisor_failure_saved_before_kill(tmp_path: Path) -> None:
    diagnostics = native_parent.StopDiagnostics(tmp_path)
    server = native_control.ControlServer(tmp_path, lambda kind, **kw: {},
                                         on_stage=diagnostics.set_stage)
    for seq, kind in enumerate(('check', 'verify', 'expect-provider', 'adopt-provider'), 1):
        native_control.write_message(tmp_path/'request.json', dict(seq=seq, kind=kind,
                                                                 payload={}))
        server.poll()
    killed = []

    def kill() -> None:
        saved = native_control.read_message(tmp_path/'parent-stop.json')
        assert saved['first_cause']['code'] == 'GPU_UNVERIFIED'
        assert saved['first_cause']['stage'] == 'control-adopt-provider'
        killed.append(True)
        raise ResourceStop('CLEANUP_UNKNOWN')

    def fail(**kwargs: Any) -> None:
        raise ResourceStop('GPU_UNVERIFIED')

    resources = SimpleNamespace(check=fail, tree=SimpleNamespace(close=kill, dispose=lambda: None))
    supervisor = Supervisor(cast(Any, resources),
                            before_cleanup=lambda error: diagnostics.record('supervisor', error))
    with pytest.raises(ResourceStop, match='GPU_UNVERIFIED'):
        supervisor.tick()
    assert killed == [True]
    assert supervisor.error is not None
    assert isinstance(supervisor.cleanup_errors[0], ResourceStop)
    assert supervisor.cleanup_errors[0].reason == 'CLEANUP_UNKNOWN'
    diagnostics.record('parent', ResourceStop('CLEANUP_UNKNOWN'))
    diagnostics.record('cleanup', supervisor.cleanup_errors[0])
    saved = native_control.read_message(tmp_path/'parent-stop.json')
    assert saved['first_cause']['code'] == 'GPU_UNVERIFIED'
    assert saved['errors']['parent']['code'] == 'CLEANUP_UNKNOWN'
    assert saved['errors']['cleanup']['code'] == 'CLEANUP_UNKNOWN'


def test_unknown_chain_sanitized_bounded_and_no_dynamic_stage(tmp_path: Path) -> None:
    diagnostics = native_parent.StopDiagnostics(tmp_path)
    secret = 'PRIVATE-KEY-MALFORMED-FRAME-DO-NOT-SAVE'
    error = KeyboardInterrupt(secret)
    error.__cause__ = ResourceStop('RSS_LIMIT')
    error.__cause__.__cause__ = error
    diagnostics.set_stage(secret)
    diagnostics.record('control', error)
    raw = (tmp_path/'parent-stop.json').read_bytes()
    assert secret.encode() not in raw
    saved = json.loads(raw)
    assert saved['first_cause']['code'] == 'UNEXPECTED_ERROR'
    assert saved['first_cause']['stage'] == 'parent-setup'
    assert saved['first_cause']['chain'] == ['UNEXPECTED_ERROR', 'RSS_LIMIT']


def test_failed_diagnostic_write_does_not_block_supervisor_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = native_parent.StopDiagnostics(tmp_path)
    def broken(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt('DO-NOT-SAVE')
    monkeypatch.setattr(native_parent, 'write_diagnostic', broken)
    killed = []
    resources = SimpleNamespace(check=broken, tree=SimpleNamespace(close=lambda: killed.append(1)))
    supervisor = Supervisor(cast(Any, resources),
                            before_cleanup=lambda error: diagnostics.record('supervisor', error))
    with pytest.raises(KeyboardInterrupt):
        supervisor.tick()
    assert killed == [1]
    assert diagnostics.write_failed


def test_diagnostic_atomic_readback_and_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path/'diagnostic.json'
    native_control.write_diagnostic(path, {'code': 'GPU_UNVERIFIED'})
    assert json.loads(path.read_bytes()) == {'code': 'GPU_UNVERIFIED'}
    assert not path.with_name(path.name+'.pending').exists()
    with pytest.raises(ValueError):
        native_control.write_diagnostic(path, {'origin': float('nan')})
    monkeypatch.setattr(native_control, '_read_frame_bytes', lambda *a: b'{}')
    with pytest.raises(ResourceStop, match='FINALIZE_FAILED'):
        native_control.write_diagnostic(path, {'code': 'RSS_LIMIT'})


def test_control_failure_callback_is_sanitized_and_does_not_replace_error(tmp_path: Path) -> None:
    diagnostics = native_parent.StopDiagnostics(tmp_path)
    server = native_control.ControlServer(tmp_path, lambda kind, **kw: {},
                                         on_error=lambda error: diagnostics.record(
                                             'control', error))
    (tmp_path/'request.json').write_bytes(b'{"private":"DO-NOT-SAVE"}')
    server.start()
    assert server.thread is not None
    server.thread.join(1.)
    assert server.error is not None
    with pytest.raises(ResourceStop):
        server.stop()
    raw = (tmp_path/'parent-stop.json').read_bytes()
    assert b'DO-NOT-SAVE' not in raw
    assert json.loads(raw)['errors']['control']['code'] == 'UNEXPECTED_ERROR'


def test_public_gpu_error_code_not_exception_text(tmp_path: Path) -> None:
    from cipher_lab.platform_resources.gpu import GpuError
    diagnostics = native_parent.StopDiagnostics(tmp_path)
    diagnostics.record('publisher', GpuError())
    saved = native_control.read_message(tmp_path/'parent-stop.json')
    assert saved['errors']['publisher']['code'] == 'GPU_UNVERIFIED'


def test_parent_does_not_own_stderr_capture_lifecycle() -> None:
    # Reverting capture removes disk/thread/pipe failure surfaces completely;
    # the launch matrix below verifies this contract at the actual spawn seam.
    assert not hasattr(native_parent, 'MainStderr')
