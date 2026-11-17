"""Task2: synthetic IPC only; no real weights/TF/cipher/CUDA."""
from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from cipher_lab_plugins.speck_study import teacher_protocol as wire


def test_literal_protocol_roundtrip() -> None:
    payload = wire.pack_points([0, 0x0123456789ABCDEF, (1 << 64) - 1])
    assert payload == bytes(8) + bytes.fromhex('0123456789abcdef') + bytes([255]) * 8
    stream = io.BytesIO()
    wire.write_frame(stream, wire.Frame(wire.REQUEST, 1, 3, payload), time.monotonic() + 2)
    stream.seek(0)
    frame = wire.read_frame(stream, time.monotonic() + 2, 50000,
                            expected_kind=wire.REQUEST, expected_seq=1)
    assert frame.payload == payload
    assert wire.unpack_points(payload) == (0, 0x0123456789ABCDEF, (1 << 64) - 1)
    reply = wire.pack_reply([0., .5, 1.])
    assert wire.unpack_reply(reply, 3) == ((0., .5, 1.), (0, 0, 1))


@pytest.mark.parametrize('payload', [b'', b'\0', struct.pack('>fB', float('nan'), 0),
                                    struct.pack('>fB', 1.1, 1),
                                    struct.pack('>fB', .5, 1), struct.pack('>fB', .75, 0)])
def test_reply_rejects(payload: bytes) -> None:
    with pytest.raises(wire.ProtocolError):
        wire.unpack_reply(payload, 1)


def test_bit_order() -> None:
    from cipher_lab_plugins.speck_study.teacher_network import points_to_bits
    bits = points_to_bits(wire.pack_points([0x8001400280048008]))
    assert bits.shape == (1, 64)
    assert bits[0].tolist() == [int(x) for x in f'{0x8001400280048008:064b}']


def test_network_exact_spec() -> None:
    from cipher_lab_plugins.speck_study.teacher_network import make_official_teacher
    from cipher_lab_plugins.speck_study.tests.speck_teacher_fakes import FakeTF
    tf = FakeTF()
    make_official_teacher(tf)
    assert tf.calls == [
        ('Input', (), {'shape': (64,)}), ('Reshape', ((4, 16),), {}),
        ('Permute', ((2, 1),), {}),
        ('Conv1D', (32,), {'kernel_size': 1, 'padding': 'same', 'kernel_regularizer': .0001}),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}),
        ('Conv1D', (32,), {'kernel_size': 3, 'padding': 'same', 'kernel_regularizer': .0001}),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}),
        ('Conv1D', (32,), {'kernel_size': 3, 'padding': 'same', 'kernel_regularizer': .0001}),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}), ('Add', (), {}),
        ('Flatten', (), {}), ('Dense', (64,), {'kernel_regularizer': .0001}),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}),
        ('Dense', (64,), {'kernel_regularizer': .0001}),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}),
        ('Dense', (1,), {'activation': 'sigmoid', 'kernel_regularizer': .0001}),
    ]


def test_frame_rejects_mutation() -> None:
    stream = io.BytesIO()
    wire.write_frame(stream, wire.Frame(wire.REQUEST, 1, 1, bytes(8)), time.monotonic() + 2)
    good = stream.getvalue()
    for raw in (b'', good[:3], good[:-1], b'X' + good[1:], good[:-1] + b'X'):
        with pytest.raises(wire.ProtocolError):
            wire.read_frame(io.BytesIO(raw), time.monotonic() + 2, 50000,
                            expected_kind=wire.REQUEST, expected_seq=1)
    for kwargs in ({'expected_seq': 2}, {'expected_kind': wire.REPLY}):
        with pytest.raises(wire.ProtocolError):
            wire.read_frame(io.BytesIO(good), time.monotonic() + 2, 50000, **kwargs)


def test_frame_limits_and_deadline() -> None:
    for points in ([], [True], [-1], [1 << 64], [0] * 50001):
        with pytest.raises(wire.ProtocolError):
            wire.pack_points(points)
    with pytest.raises(TimeoutError):
        wire.read_frame(io.BytesIO(), time.monotonic() - 1, 50000)


@pytest.mark.parametrize('kind,seq,count,length,version', [
    (wire.REQUEST, 1, 50001, 400008, 1), (wire.REQUEST, 1, 1, 0xFFFFFFFF, 1),
    (wire.REQUEST, 0, 1, 8, 1), (wire.REQUEST, 1, 0, 0, 1),
    (wire.READY, 0, 0, wire.MAX_READY + 1, 1), (wire.READY, 1, 0, 1, 1),
    (99, 1, 1, 8, 1), (wire.REQUEST, 1, 1, 8, 2),
])
def test_header_rejected_before_payload(kind: int, seq: int, count: int,
                                        length: int, version: int) -> None:
    raw = wire.HEADER.pack(wire.MAGIC, version, kind, seq, count, length, bytes(32))
    with pytest.raises(wire.ProtocolError):
        wire.read_frame(io.BytesIO(raw), time.monotonic() + 2)


def test_short_writes_and_max_frame() -> None:
    class ShortWriter(io.BytesIO):
        def write(self, data: Any) -> int:
            return super().write(data[:3])
    stream = ShortWriter()
    payload = wire.pack_points([0] * 50000)
    wire.write_frame(stream, wire.Frame(wire.REQUEST, 1, 50000, payload), time.monotonic() + 3)
    stream.seek(0)
    assert wire.read_frame(stream, time.monotonic() + 3).payload == payload
    reply = wire.pack_reply([.5] * 50000)
    assert len(reply) == 250000
    assert len(wire.unpack_reply(reply, 50000)[0]) == 50000


def test_gpu_setup_fails_closed_without_proof() -> None:
    from cipher_lab_plugins.speck_study.teacher_network import configure_tensorflow
    from cipher_lab_plugins.speck_study.tests.speck_teacher_fakes import FakeTF
    tf = FakeTF()
    with pytest.raises(RuntimeError, match='GPU_UNVERIFIED'):
        configure_tensorflow(tf, 'GPU-test', '0000:01:00.0', None)
    assert tf.gpu_calls == []


def test_fake_type_contract() -> None:
    from cipher_lab_plugins.speck_study.tests.speck_teacher_fakes import FakeTF
    tf: Any = FakeTF()
    assert not hasattr(tf.keras, 'load_model')


def fake_verifier(evidence: dict[str, Any], pid: int, uuid: str, pci: str) -> bool:
    # Synthetic ONLY, not a production verifier.
    return evidence == {'fake_only': True, 'pid': pid, 'gpu_uuid': uuid, 'pci_bus_id': pci}


def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str = 'normal',
           seconds: float = 6.) -> Any:
    from cipher_lab_plugins.speck_study.teacher import TeacherClient
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([
        str(Path(__file__).resolve().parent), str(__import__('cipher_lab').__path__[0]+'/..'), str(Path(__file__).resolve().parents[3])]))  # noqa: E501 - explicit migrated source/argv identity
    monkeypatch.setenv('SPECK_FAKE_MODE', mode)
    return TeacherClient(sys.executable, 'speck_teacher_fakes', 'never-open-real-weights.h5',
                         'a' * 64, 'GPU-test', tmp_path, time.monotonic() + seconds,
                         gpu_pci_bus_id='0000:01:00.0', verify_gpu=fake_verifier,
                         close_grace=1.)


def test_client_formula_serial_and_reap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = client(tmp_path, monkeypatch)
    process = teacher.process
    with teacher:
        assert teacher.predict([0, 1, 2, 3, (1 << 64) - 1]) == (
            (0., .25, .5, .75, .75), (0, 0, 0, 1, 1))
        assert teacher.predict([2]) == ((.5,), (0,))
        assert teacher.seq == 2
        assert teacher.ready['tensorflow'] == 'synthetic-no-TF'
    assert process.returncode is not None
    teacher.close()
    assert not any(t.name.startswith('teacher-') for t in threading.enumerate())


@pytest.mark.parametrize('mode', ['partial', 'hang', 'bad-seq', 'bad-reply', 'wrong-count'])
def test_failed_reply_never_retries_and_reaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              mode: str) -> None:
    teacher = client(tmp_path, monkeypatch, mode)
    teacher.deadline = time.monotonic() + .15
    start = time.monotonic()
    with pytest.raises((wire.ProtocolError, TimeoutError)):
        teacher.predict([0, 1])
    assert time.monotonic() - start < 2.5
    assert teacher.seq == 1
    assert teacher.closed and teacher.process.returncode is not None
    assert not any(t.name.startswith('teacher-') for t in threading.enumerate())


@pytest.mark.parametrize('mode', ['partial-ready', 'hang-ready', 'early-exit', 'bad-ready'])
def test_constructor_failure_reaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   mode: str) -> None:
    real_popen = subprocess.Popen
    spawned: list[Any] = []

    def track(*args: Any, **kwargs: Any) -> Any:
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc
    monkeypatch.setattr(subprocess, 'Popen', track)
    with pytest.raises((wire.ProtocolError, TimeoutError)):
        client(tmp_path, monkeypatch, mode, seconds=2.5)
    assert len(spawned) == 1
    assert spawned[0].returncode is not None
    assert not any(t.name.startswith('teacher-') for t in threading.enumerate())


def test_stderr_flood_bounded_and_no_deadlock(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.speck_study.teacher import STDERR_LIMIT
    with client(tmp_path, monkeypatch, 'stderr-flood') as teacher:
        assert teacher.predict([3]) == ((.75,), (1,))
    assert (tmp_path / 'teacher-stderr.log').stat().st_size == STDERR_LIMIT


def test_one_inflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = client(tmp_path, monkeypatch)
    teacher.lock.acquire()
    try:
        with pytest.raises(wire.ProtocolError, match='one teacher request'):
            teacher.predict([0])
        assert teacher.seq == 0
    finally:
        teacher.lock.release()
        teacher.close()


def test_missing_executable_and_missing_proof(tmp_path: Path) -> None:
    from cipher_lab_plugins.speck_study.teacher import TeacherClient
    args = ('/no/such/python', 'fake', 'not-opened.h5', 'a' * 64, 'GPU-test', tmp_path,
            time.monotonic() + 3)
    with pytest.raises(wire.ProtocolError, match='GPU_UNVERIFIED'):
        TeacherClient(*args, gpu_pci_bus_id='pci')
    with pytest.raises(FileNotFoundError):
        TeacherClient(*args, gpu_pci_bus_id='pci', verify_gpu=fake_verifier)
    assert not any(t.name.startswith('teacher-') for t in threading.enumerate())


def test_unrequested_extra_bytes_poison_next_request(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    with client(tmp_path, monkeypatch, 'extra') as teacher:
        assert teacher.predict([0]) == ((.5,), (0,))
        teacher.deadline = time.monotonic() + .15
        with pytest.raises((wire.ProtocolError, TimeoutError)):
            teacher.predict([0])
    assert teacher.process.returncode is not None


@pytest.mark.parametrize('frame', [wire.Frame(True, 1, 1, bytes(8)),
                                   wire.Frame(wire.READY, 0, False, b'x')])
def test_exact_header_field_types(frame: wire.Frame) -> None:
    with pytest.raises(wire.ProtocolError):
        wire.write_frame(io.BytesIO(), frame, time.monotonic() + 2)


def test_repeated_close_must_recheck_thread_join(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = client(tmp_path, monkeypatch)
    # Inject a join failure ONLY after real OS child is terminated/reaped.
    assert teacher.reader is not None
    real_join = teacher.reader.join
    monkeypatch.setattr(teacher.reader, 'join', lambda end: False)
    with pytest.raises(RuntimeError, match='cleanup UNKNOWN'):
        teacher.close()
    assert teacher.process.returncode is not None
    calls: list[float] = []

    def join(end: float) -> bool:
        calls.append(end)
        return bool(real_join(end))
    monkeypatch.setattr(teacher.reader, 'join', join)
    teacher.close()
    assert calls, 'must recheck cleanup after UNKNOWN even if process is dead'


@pytest.fixture(autouse=True)
def task2_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.speck_study.tests.speck_teacher_fakes import install_teacher_guards
    install_teacher_guards(monkeypatch)


def test_provider_import_closure_and_missing_attestation(tmp_path: Path) -> None:
    from cipher_lab_plugins.speck_study.teacher_provider import parser, run_provider
    args = parser().parse_args(['--weights', 'NEVER-OPEN.h5', '--sha256', 'a' * 64,
                                '--gpu-uuid', 'GPU-test', '--gpu-pci', 'pci',
                                '--deadline', str(time.monotonic() + 2)])
    # Must fail before opening any weights or importing TF (both are trapped).
    with pytest.raises(RuntimeError, match='runtime attestor required'):
        run_provider(args, io.BytesIO(), io.BytesIO())


def test_allocator_before_init_and_actual_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from cipher_lab_plugins.speck_study.teacher_network import configure_tensorflow
    calls: list[str] = []

    def event(name: str, result: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            if name == 'limit':
                assert kwargs == {'memory_limit': 8192}
            return result
        return call
    config = SimpleNamespace(list_physical_devices=event('physical', ['physical0']),
                             set_visible_devices=event('visible', None),
                             LogicalDeviceConfiguration=event('limit', '8GiB'),
                             set_logical_device_configuration=event('configure', None),
                             list_logical_devices=event('initialize', ['logical0']))
    tf = SimpleNamespace(config=config)

    def attestor(tf: Any) -> dict[str, Any]:
        assert calls == ['physical', 'visible', 'limit', 'configure', 'initialize']
        calls.append('attest')
        return {'gpu_uuid': 'GPU-test', 'pci_bus_id': 'pci', 'pid': os.getpid(),
                'logical_device': 0, 'evidence': {'synthetic_only': True}}
    evidence = configure_tensorflow(tf, 'GPU-test', 'pci', attestor)
    assert evidence['gpu_uuid'] == 'GPU-test'
    calls.clear()
    with pytest.raises(RuntimeError, match='runtime identity mismatch'):
        configure_tensorflow(tf, 'GPU-other', 'pci', attestor)


def test_provider_serve_partial_eof_and_sequence() -> None:
    from cipher_lab_plugins.speck_study.teacher_provider import serve
    data = io.BytesIO()
    wire.write_frame(data, wire.Frame(wire.REQUEST, 1, 2, wire.pack_points([0, 1])),
                     time.monotonic() + 2)
    data.seek(0)
    output = io.BytesIO()
    seen: list[bytes] = []

    def predict(payload: bytes) -> list[float]:
        seen.append(payload)
        return [.5, .75]
    with pytest.raises(wire.ProtocolError, match='EOF'):
        serve(data, output, time.monotonic() + 2, {'fake': True}, predict)
    assert len(seen) == 1
    output.seek(0)
    assert wire.read_frame(output, time.monotonic() + 2).kind == wire.READY
    reply = wire.read_frame(output, time.monotonic() + 2, expected_seq=1)
    assert wire.unpack_reply(reply.payload, 2) == ((.5, .75), (0, 1))


def test_close_term_exception_still_kills_reaps(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = client(tmp_path, monkeypatch, 'hang')

    def fail_term() -> None:
        raise OSError('synthetic TERM failure')
    monkeypatch.setattr(teacher.process, 'terminate', fail_term)
    teacher.close()
    assert teacher.process.returncode is not None
    assert teacher.cleanup_complete


@pytest.mark.parametrize('mode', ['no-read', 'stdout-flood', 'exit-request'])
def test_blocked_write_and_stdout_flood_reaped(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch,
                                               mode: str) -> None:
    teacher = client(tmp_path, monkeypatch, mode)
    teacher.deadline = time.monotonic() + .15
    with pytest.raises((wire.ProtocolError, TimeoutError, OSError)):
        teacher.predict([0] * (50000 if mode == 'no-read' else 2))
    assert teacher.process.returncode is not None
    assert teacher.cleanup_complete
    assert not any(t.name.startswith('teacher-') for t in threading.enumerate())


def test_parent_rejects_string_echo_verification(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:

    from cipher_lab_plugins.speck_study.teacher import TeacherClient
    teacher = object.__new__(TeacherClient)
    teacher.process = None
    # No default verifier means no process can even launch based on echo strings.
    with pytest.raises(wire.ProtocolError, match='GPU_UNVERIFIED'):
        TeacherClient(sys.executable, 'fake', 'unopened.h5', 'a' * 64, 'GPU-test',
                      tmp_path, time.monotonic() + 2, gpu_pci_bus_id='pci')


def test_gpu_environment_before_tf_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.speck_study.teacher_network import prepare_environment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-other')
    with pytest.raises(RuntimeError, match='inherited GPU binding mismatch'):
        prepare_environment('GPU-test')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    prepare_environment('GPU-test')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
    monkeypatch.setitem(sys.modules, 'tensorflow', object())
    with pytest.raises(RuntimeError, match='already imported'):
        prepare_environment('GPU-test')


def test_provider_imports_only_audited_closure() -> None:
    import ast
    base = Path(__file__).resolve().parents[1]
    for name in ('teacher_protocol', 'teacher_network', 'teacher_provider'):
        tree = ast.parse((base / (name + '.py')).read_text(), feature_version=(3, 9))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.module in ('teacher_network', 'teacher_protocol')
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([alias.name for alias in node.names]
                           if isinstance(node, ast.Import) else [node.module or ''])
                assert not any(part in module.split('.')
                               for module in modules
                               for part in ('config', 'execution', 'speck_study_config'))


def test_provider_only_load_weights_with_fake_tf(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    from types import SimpleNamespace

    import cipher_lab_plugins.speck_study.teacher_provider as provider
    path = tmp_path / 'synthetic-weights.bin'
    path.write_bytes(b'synthetic-not-H5')
    args = provider.parser().parse_args([
        '--weights', str(path), '--sha256', hashlib.sha256(path.read_bytes()).hexdigest(),
        '--gpu-uuid', 'GPU-test', '--gpu-pci', 'pci',
        '--deadline', str(time.monotonic() + 2)])
    loads: list[str] = []

    class FakeModel:
        def load_weights(self, name: str) -> None:
            loads.append(name)

    monkeypatch.setattr(provider, 'make_official_teacher', lambda tf: FakeModel())
    monkeypatch.setattr(provider, 'configure_tensorflow', lambda *args: {'fake_only': True})
    # Empty request stream: no fake or actual NN forward, just weights-only wiring.
    output = io.BytesIO()
    with pytest.raises(wire.ProtocolError, match='EOF'):
        provider.run_provider(args, io.BytesIO(), output, lambda tf: {},
                              SimpleNamespace(__version__='fake-TF'))
    assert loads == [str(path)]
    assert wire.read_frame(io.BytesIO(output.getvalue()), time.monotonic() + 2).kind == wire.READY


def test_ready_strict_identity_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import copy
    import json
    with client(tmp_path, monkeypatch) as teacher:
        good = teacher.ready
        variants = []
        for key, value in [('python', '3.12.0'), ('tensorflow', ''), ('weights_sha256', 'b' * 64),
                           ('allocator_mib', 8192.), ('extra', True)]:
            bad = copy.deepcopy(good)
            bad[key] = value
            variants.append(json.dumps(bad).encode())
        for key, value in [('depth', True), ('d1', 256)]:
            bad = copy.deepcopy(good)
            bad['model'][key] = value
            variants.append(json.dumps(bad).encode())
        for key, value in [('pid', teacher.pid + 1), ('gpu_uuid', 'GPU-wrong'),
                           ('pci_bus_id', 'wrong')]:
            bad = copy.deepcopy(good)
            bad['gpu'][key] = value
            variants.append(json.dumps(bad).encode())
        variants += [b'{"python":1,"python":2}', b'NaN', b'[]', b'\xff']
        for raw in variants:
            with pytest.raises(wire.ProtocolError):
                # Even an incorrectly permissive injected verifier cannot bypass
                # required PID/UUID/PCI field comparisons and model identity.
                teacher._validate_ready(raw, 'a' * 64, 'GPU-test', '0000:01:00.0',
                                        lambda *args: True)


def test_float32_tie_is_derived_after_rounding() -> None:
    reply = wire.pack_reply([.5 + 1e-10, .5 - 1e-10])
    assert wire.unpack_reply(reply, 2) == ((.5, .5), (0, 0))


def test_provider_rejects_reordered_request_before_predict() -> None:
    from cipher_lab_plugins.speck_study.teacher_provider import serve
    data = io.BytesIO()
    wire.write_frame(data, wire.Frame(wire.REQUEST, 2, 1, wire.pack_points([1])),
                     time.monotonic() + 2)
    data.seek(0)

    def never_predict(payload: bytes) -> list[float]:
        raise AssertionError('must reject sequence before any forward')
    with pytest.raises(wire.ProtocolError, match='wrong sequence'):
        serve(data, io.BytesIO(), time.monotonic() + 2, {'fake': True}, never_predict)


def test_provider_partial_request_exits_by_own_deadline(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise provider serve's own blocking-read timeout, not just client kill.
    teacher = client(tmp_path, monkeypatch, seconds=1.5)
    try:
        assert teacher.process.stdin is not None
        teacher.process.stdin.write(b'SPK')
        teacher.process.stdin.flush()
        teacher.process.wait(timeout=2.5)
        assert teacher.process.returncode == 2
    finally:
        teacher.close()
    assert teacher.cleanup_complete


def test_guards_are_installed_in_this_process() -> None:
    import importlib
    import secrets

    import numpy as np
    for operation in (lambda: os.urandom(8), lambda: secrets.token_bytes(8),
                      lambda: np.random.default_rng(), lambda: importlib.import_module('torch'),
                      lambda: importlib.import_module('tensorflow'),
                      lambda: importlib.import_module('cipher_lab.ciphers.speck')):
        with pytest.raises(AssertionError, match='Task2 forbids'):
            operation()


def test_client_explicit_qpc_epoch_not_local_timeout_epoch(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Native deadlines stay QPC; cleanup transports join on their local clock."""
    import json

    from cipher_lab_plugins.speck_study.teacher import TeacherClient
    from cipher_lab_plugins.speck_study.teacher_network import ALLOCATOR_MIB, MODEL_SPEC
    ready: dict[str,Any] = dict(python='3.9.synthetic',tensorflow='fake-TF',
        model=MODEL_SPEC,weights_sha256='a'*64,allocator_mib=ALLOCATOR_MIB,
        gpu=dict(pid=123,gpu_uuid='GPU-test',pci_bus_id='0000:01:00.0'))
    stream = io.BytesIO()
    wire.write_frame(stream,wire.Frame(wire.READY,0,0,json.dumps(ready).encode()),
                     time.monotonic()+1)
    stream.seek(0)
    class Process:
        pid = 123
        returncode = None
        stdin = io.BytesIO()
        stdout = stream
        stderr = io.BytesIO()
        def poll(self) -> Any:
            return self.returncode
        def terminate(self) -> None:
            self.returncode = 0
        def kill(self) -> None:
            self.returncode = 1
        def wait(self, timeout: float) -> int:
            assert 0 < timeout <= 2
            return 0
    def spawn(*args: Any, **kwargs: Any) -> Any:
        return Process()
    teacher = TeacherClient('fake-python','fake_provider','fake.bin','a'*64,'GPU-test',
                            tmp_path,101.,gpu_pci_bus_id='0000:01:00.0',
                            verify_gpu=lambda *args: True,spawn=spawn,
                            clock=lambda: 100.)
    teacher.close()
    assert teacher.cleanup_complete
    with pytest.raises(ValueError):
        TeacherClient('fake','fake','fake','a'*64,'GPU-test',tmp_path,100.,
                      gpu_pci_bus_id='0000:01:00.0',clock=lambda: 100.)


def test_frozen_provider_main_own_guards_telemetry_before_fake_tf(tmp_path: Path) -> None:
    """Exec owns guards, real freeze/main/weight-SHA/READY; no DLL/TF/H5/cipher."""
    import hashlib
    import json
    import subprocess

    from cipher_lab_plugins.speck_study.tests.locations import ROOT, RUNTIME
    root = ROOT
    weight = tmp_path/'synthetic.bin'
    weight.write_bytes(b'FAKE-NOT-H5')
    manifest = tmp_path/'provider.json'
    script = r'''
import sys,os,ctypes,importlib.abc,importlib.util,json,hashlib,io
from pathlib import Path
from types import SimpleNamespace
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('torch','tensorflow','numpy','h5py','keras') or any(
            name==x or name.startswith(x+'.') for x in
            ('cipher_lab.ciphers','cipher_lab.execution','cipher_lab.data')):
            raise AssertionError('provider child forbidden scientific import '+name)
sys.meta_path.insert(0,Guard())
def forbidden(*args,**kwargs):
    raise AssertionError('provider child native/scientific call')
os.urandom=forbidden
ctypes.CDLL=forbidden
ctypes.WinDLL=forbidden
import subprocess
subprocess.run=forbidden
subprocess.Popen=forbidden
entry,source,plugins,manifest,weight=map(Path,sys.argv[1:6])
spec=importlib.util.spec_from_file_location('own_provider_wrapper',entry)
wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
manifest.write_text(json.dumps({n:hashlib.sha256(wrapper.source_path(source,plugins,n).read_bytes()).hexdigest()
                               for n in wrapper.CLOSURE}))
events=[]
original=wrapper.SourceOnly
class Guarded(original):
    def exec_module(self,module):
        super().exec_module(module)
        if module.__name__=='cipher_lab_plugins.speck_study.teacher_provider':
            install(module)
wrapper.SourceOnly=Guarded
os.environ.update(SPECK_STUDY_NATIVE='1',SPECK_STUDY_ORIGIN=str(10**12),
    SPECK_STUDY_SCIENCE_DEADLINE=str(10**12+28200),SPECK_STUDY_CONTROL_ROOT=str(weight.parent),
    CUDA_VISIBLE_DEVICES='GPU-test')
def install(provider):
    from cipher_lab_plugins.speck_study.platform_resources import dxgi
    from cipher_lab_plugins.speck_study import gpu_telemetry as telemetry
    class Publisher:
        def __init__(self,*args,**kwargs):
            assert kwargs['pid']==77 and kwargs['created']==770
        def publish_once(self):
            events.append('measured-zero')
        def start(self):
            events.append('publisher-start')
        def check(self):
            assert events[:2]==['measured-zero','publisher-start']
    telemetry.TelemetryPublisher=Publisher
    telemetry.current_process_identity=lambda:(77,770)
    telemetry.windows_monotonic=lambda:10**12+1
    dxgi.DxgiUsage=lambda:SimpleNamespace(sample=lambda:[])
    provider.configure_tensorflow=lambda *args: {'fake_only':True}
    class Model:
        def load_weights(self,path):
            assert events[:2]==['measured-zero','publisher-start']
            events.append('fake-weights-only')
    provider.make_official_teacher=lambda tf:Model()
    real_run=provider.run_provider
    def run(args,reader,writer,attestor,**kwargs):
        assert kwargs['clock']()==10**12+1
        # The empty bounded stream deliberately stops after real READY, before
        # any model forward. It exercises real serve and QPC DeadlineIO.
        return real_run(args,io.BytesIO(),writer,attestor,
                        SimpleNamespace(__version__='2.5.fake-only'),**kwargs)
    provider.run_provider=run
code=wrapper.teacher(manifest,source,plugins,['--weights',str(weight),'--sha256',sys.argv[6],
    '--gpu-uuid','GPU-test','--gpu-pci','0000:02:00.0','--deadline',str(10**12+28200)])
assert code==2 and events==['measured-zero','publisher-start','fake-weights-only']
for name,module in tuple(sys.modules.items()):
    if name=='cipher_lab' or name.startswith('cipher_lab.'):
        assert isinstance(module.__loader__,Guarded),name
'''
    result = subprocess.run([sys.executable,'-B','-c',script,
        str(RUNTIME/'freeze.py'),str(root/'src'),str(root/'plugins'),str(manifest),str(weight),
        hashlib.sha256(weight.read_bytes()).hexdigest()],capture_output=True,timeout=10,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode == 0, result.stderr.decode()
    frame = wire.read_frame(io.BytesIO(result.stdout),time.monotonic()+1)
    assert frame.kind == wire.READY
    ready = json.loads(frame.payload)
    assert ready['tensorflow'] == '2.5.fake-only'
    assert ready['weights_sha256'] == hashlib.sha256(weight.read_bytes()).hexdigest()
