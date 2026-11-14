"""Pure fake telemetry. The parent never obtains expectations from a child frame."""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from test_speck_resources import resource_guard  # noqa: F401

from cipher_lab.platform_resources.gpu import GpuBinding, GpuError
from cipher_lab.speck_study.gpu_telemetry import (
    TelemetryGpu,
    TelemetryPublisher,
    read_sample,
)

LUID = '0100000000000000'
OTHER = '0200000000000000'
_LOCAL_POPEN = subprocess.Popen
BINDING = GpuBinding(1, 'GPU-one', '0000:02:00.0')


def rows(used: int = 0, other: int = 0) -> list[dict[str, Any]]:
    return [dict(luid=value, description='identical model',
                 nodes=[dict(node=0, current_usage=usage)])
            for value, usage in [(LUID, used), (OTHER, other)]]


def identity() -> dict[str, Any]:
    return dict(gpu_uuid='GPU-one', pci_bus_id=BINDING.pci_bus_id, pid=1,
                logical_device=0, evidence='cuda-current-context', luid=LUID,
                device_node_mask=1)


def setup(tmp_path: Path) -> tuple[TelemetryPublisher, TelemetryGpu, list[float], list[Any]]:
    now = [100.]
    members: list[Any] = [dict(pid=1, created=10)]
    publisher = TelemetryPublisher(tmp_path, tmp_path / 'one.frame', pid=1, created=10,
                                   sample=lambda: rows(7), clock=lambda: now[0])
    gpu = TelemetryGpu(tmp_path, {1: (10, tmp_path / 'one.frame')},
                       members=lambda: members, creation_time=lambda pid: 10,
                       inventory={LUID: 1, OTHER: 1}, scientific_pids={1},
                       resolve=lambda binding: dict(luid=LUID, device_node_mask=1),
                       clock=lambda: now[0], science_deadline=28300.)
    return publisher, gpu, now, members


def test_bootstrap_sum_not_scientific_ready_then_strict_luid(tmp_path: Path) -> None:
    pub, gpu, _, _ = setup(tmp_path)
    pub.publish_once()
    assert gpu.processes(BINDING) == {1: 7}
    assert gpu.ready is False
    assert gpu.verify(identity(), 1, BINDING.gpu_uuid, BINDING.pci_bus_id)
    assert gpu.ready is True
    assert gpu.processes(BINDING) == {1: 7}  # same fresh seq is legal between ticks


@pytest.mark.parametrize('fault', ['missing', 'expired', 'future', 'pid', 'created',
                                  'reuse', 'unknown', 'members', 'sequence', 'tick',
                                  'other', 'wronggpu', 'wrongluid', 'late'])
def test_parent_independent_expectations_and_sticky_failure(tmp_path: Path, fault: str) -> None:
    pub, gpu, now, members = setup(tmp_path)
    pub.publish_once()
    gpu.processes(BINDING)
    if fault == 'missing':
        (tmp_path / 'one.frame').unlink()
    elif fault == 'expired':
        now[0] += 2.
    elif fault == 'future':
        pub.clock = lambda: 101.
        pub.publish_once()
    elif fault == 'reuse':
        gpu.creation_time = lambda pid: 11
    elif fault == 'unknown':
        members.append(dict(pid=2, created=20))
    elif fault == 'members':
        members.clear()
    elif fault in ('sequence', 'tick'):
        pub.publish_once()
        gpu.processes(BINDING)
        if fault == 'sequence':
            pub.seq = 0
        else:
            pub.clock = lambda: 99.9
            pub.last_time = None  # synthesize bad producer; production publisher rejects regression
        pub.publish_once()
    elif fault in ('pid', 'created'):
        setattr(pub, fault, 42)
        pub.publish_once()
    else:
        evidence = identity()
        if fault == 'other':
            pub.sample = lambda: rows(7, 1)
            pub.publish_once()
        elif fault == 'wronggpu':
            evidence['gpu_uuid'] = 'GPU-zero'
        elif fault == 'wrongluid':
            evidence['luid'] = OTHER
        else:
            now[0] = 28300.
        with pytest.raises(GpuError):
            gpu.verify(evidence, 1, BINDING.gpu_uuid, BINDING.pci_bus_id)
        with pytest.raises(GpuError):
            gpu.processes(BINDING)
        return
    with pytest.raises(GpuError):
        gpu.processes(BINDING)
    with pytest.raises(GpuError):
        gpu.processes(BINDING)


@pytest.mark.parametrize('fault', ['sha', 'size', 'oversize', 'nan', 'duplicate', 'truncated'])
def test_frame_bounded_strict_json_checksum(tmp_path: Path, fault: str) -> None:
    pub, _, _, _ = setup(tmp_path)
    pub.publish_once()
    path = tmp_path / 'one.frame'
    envelope = json.loads(path.read_bytes())
    if fault == 'sha':
        envelope['sha256'] = '0' * 64
    elif fault == 'size':
        envelope['size'] += 1
    elif fault in ('nan', 'duplicate'):
        raw = envelope['payload'].encode()
        raw = raw.replace(b'"tick":1', b'"tick":NaN' if fault == 'nan'
                          else b'"tick":1,"tick":1')
        envelope.update(payload=raw.decode(), size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    data = json.dumps(envelope).encode()
    if fault == 'oversize':
        data = b' ' * 65537
    elif fault == 'truncated':
        data = data[:-1]
    path.write_bytes(data)
    with pytest.raises(GpuError):
        read_sample(tmp_path, path)


def test_sampling_thread_independent_of_prediction_and_failure_sticky(tmp_path: Path) -> None:
    pub, _, _, _ = setup(tmp_path)
    called = threading.Event()
    def sample() -> list[dict[str, Any]]:
        called.set()
        raise OSError('native sample failed')
    pub.sample = sample
    pub.start()
    assert called.wait(1.)
    assert pub.thread is not None
    pub.thread.join(1.)
    with pytest.raises(GpuError):
        pub.check()
    with pytest.raises(GpuError):
        pub.stop(timeout=.1)


def test_stop_join_bounded_blocked_native_is_not_live(tmp_path: Path) -> None:
    pub, _, _, _ = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def sample() -> list[dict[str, Any]]:
        entered.set()
        release.wait(2.)
        return rows()
    pub.sample = sample
    pub.start()
    assert entered.wait(1.)
    try:
        with pytest.raises(GpuError):
            pub.stop(timeout=.01)
        with pytest.raises(GpuError):
            pub.check()
    finally:
        release.set()
        assert pub.thread is not None
        pub.thread.join(1.)
    assert not (tmp_path / 'one.frame').exists()  # no late publish after stop


@pytest.mark.parametrize('name', ['../escape', 'sub/../escape', '/absolute'])
def test_paths_parent_supplied_no_traversal(tmp_path: Path, name: str) -> None:
    with pytest.raises(GpuError):
        TelemetryPublisher(tmp_path, tmp_path / name, pid=1, created=10, sample=rows)


def test_reparse_or_link_never_followed(tmp_path: Path) -> None:
    outside = tmp_path / 'outside'
    outside.mkdir()
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(GpuError):
        TelemetryPublisher(root, root / 'link' / 'frame', pid=1, created=10, sample=rows)


def test_thread_keeps_publishing_while_owner_waits_and_stop_removes_frame(tmp_path: Path) -> None:
    pub, _, _, _ = setup(tmp_path)
    twice = threading.Event()
    count = [0]
    def sample() -> list[dict[str, Any]]:
        count[0] += 1
        if count[0] >= 2:
            twice.set()
        return rows()
    pub.sample, pub.interval = sample, .01
    pub.start()
    try:
        assert twice.wait(1.)  # owner is not calling publisher/check/predict helpers
        pub.check()
    finally:
        pub.stop(timeout=1.)
    assert not (tmp_path / 'one.frame').exists()
    with pytest.raises(GpuError):
        pub.check()


def test_slow_identity_resolver_never_holds_sampler_lock(tmp_path: Path) -> None:
    pub, gpu, _, _ = setup(tmp_path)
    pub.publish_once()
    entered, release, sampled = threading.Event(), threading.Event(), threading.Event()
    errors: list[BaseException] = []
    def resolve(binding: GpuBinding) -> dict[str, Any]:
        entered.set()
        assert release.wait(2.)
        return dict(luid=LUID, device_node_mask=1)
    gpu.resolve = resolve
    def verify() -> None:
        try:
            gpu.verify(identity(), 1, BINDING.gpu_uuid, BINDING.pci_bus_id)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=verify)
    worker.start()
    assert entered.wait(1.)
    def monitor() -> None:
        try:
            assert gpu.processes(BINDING) == {1: 7}
            sampled.set()
        except BaseException as error:
            errors.append(error)
    sampler = threading.Thread(target=monitor)
    sampler.start()
    try:
        assert sampled.wait(1.)
    finally:
        release.set()
        worker.join(1.)
        sampler.join(1.)
    assert not errors


def test_actual_aggregate_20gib_uses_study_resource_limit(tmp_path: Path) -> None:
    from cipher_lab.speck_study.resources import ResourceStop, StudyResources
    pub, gpu, _, _ = setup(tmp_path)
    pub.sample = lambda: rows(11 * 2**30, 10 * 2**30)
    pub.publish_once()
    class Tree:
        def sample(self) -> list[dict[str, Any]]:
            return [dict(pid=1, created=10, rss=1, cpu=0.)]
    resources = StudyResources(BINDING, tmp_path, 100., 28300., 28900., Tree(),
                               gpu=gpu, clock=lambda: 100., output_bytes=lambda: 0)
    with pytest.raises(ResourceStop, match='GPU_LIMIT'):
        resources.check(science=False)
    assert resources.latest == {}


def test_no_fresh_first_sample_and_same_seq_changed_payload(tmp_path: Path) -> None:
    _, gpu, _, _ = setup(tmp_path)
    with pytest.raises(GpuError):
        gpu.processes(BINDING)
    pub, gpu, _, _ = setup(tmp_path)
    pub.publish_once()
    gpu.processes(BINDING)
    pub.seq = 0
    pub.sample = lambda: rows(9)
    pub.publish_once()
    with pytest.raises(GpuError):
        gpu.processes(BINDING)


def test_adapter_coverage_must_match_parent_inventory(tmp_path: Path) -> None:
    pub, gpu, _, _ = setup(tmp_path)
    pub.sample = lambda: rows()[:1]
    pub.publish_once()
    with pytest.raises(GpuError):
        gpu.processes(BINDING)


def test_bootstrap_cannot_admit_scientific_work(tmp_path: Path) -> None:
    from cipher_lab.speck_study.resources import ResourceStop, StudyResources
    pub, gpu, _, _ = setup(tmp_path)
    pub.publish_once()
    class Tree:
        def sample(self) -> list[dict[str, Any]]:
            return [dict(pid=1, created=10, rss=1, cpu=0.)]
    resources = StudyResources(BINDING, tmp_path, 100., 28300., 28900., Tree(),
                               gpu=gpu, clock=lambda: 100., output_bytes=lambda: 0)
    resources.check(science=False)
    with pytest.raises(ResourceStop, match='GPU_UNVERIFIED'):
        resources.check()


def test_child_process_installs_own_import_dll_entropy_guards(tmp_path: Path) -> None:
    """Exec is not assumed to inherit monkeypatches; traps are installed IN child."""
    import os
    import subprocess
    import sys
    import textwrap
    # Call only a saved real launcher with a fixed local Python, no shell.
    # Autouse guard poisons subprocess.run/Popen in THIS process, so obtain the
    # trusted fixture originals via MonkeyPatch.context undo is not acceptable.
    # subprocess is launched through the original fixture seam captured below.
    script = textwrap.dedent('''
        import builtins, ctypes, os, subprocess, sys
        original = builtins.__import__
        def denied(*args, **kwargs):
            raise AssertionError('child actual probe/entropy forbidden')
        def guarded(name, *args, **kwargs):
            if (name.split('.')[0] in ('torch', 'tensorflow', 'numpy', 'psutil')
                    or name.startswith('cipher_lab.ciphers')):
                raise AssertionError('child scientific import forbidden')
            return original(name, *args, **kwargs)
        builtins.__import__ = guarded
        ctypes.CDLL = ctypes.WinDLL = denied
        subprocess.run = subprocess.Popen = denied
        os.urandom = denied
        from pathlib import Path
        from cipher_lab.platform_resources.dxgi import DxgiUsage
        from cipher_lab.speck_study.gpu_telemetry import TelemetryPublisher, read_sample
        class COM:
            def adapters(self): return [1]
            def description(self, adapter): return ('0100000000000000', 'bootstrap only')
            def node_count(self, identity): return 1
            def query(self, adapter, node): return 0
            def release(self, adapter): pass
            def close(self): pass
        root = Path(sys.argv[1])
        pub = TelemetryPublisher(root, root/'child.frame', pid=7, created=9,
            sample=DxgiUsage(api_factory=COM).sample, clock=lambda: 100.)
        pub.publish_once()
        frame = read_sample(root, root/'child.frame')
        assert frame['pid'] == 7 and frame['rows'][0]['nodes'][0]['current_usage'] == 0
        print('child-own-guards fake explicit-zero OK')
    ''')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(
        Path(__file__).resolve().parents[1] / 'src'))
    process = _LOCAL_POPEN([sys.executable, '-B', '-c', script, str(tmp_path)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        stdout, stderr = process.communicate(timeout=10.)
    except BaseException:
        process.kill()
        process.wait(timeout=2.)
        raise
    assert process.returncode == 0, stderr.decode()
    assert stdout == b'child-own-guards fake explicit-zero OK\n'


def test_own_pid_creationtime_api_not_frame_echo() -> None:
    from cipher_lab.speck_study.gpu_telemetry import current_process_identity
    class Kernel:
        def GetCurrentProcess(self) -> int:
            return -1
        def GetCurrentProcessId(self) -> int:
            return 71
        def GetProcessTimes(self, handle: Any, created: Any, *other: Any) -> int:
            assert handle == -1
            created._obj.low, created._obj.high = 3, 2
            return 1
    assert current_process_identity(Kernel()) == (71, (2 << 32) | 3)
    class Failed(Kernel):
        def GetProcessTimes(self, *args: Any) -> int:
            return 0
    with pytest.raises(GpuError):
        current_process_identity(Failed())


def test_stop_timeout_in_file_fsync_does_not_wait_on_publication_lock(tmp_path: Path,
                                                                   monkeypatch: Any) -> None:
    pub, _, _, _ = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def fsync(fd: int) -> None:
        entered.set()
        assert release.wait(2.)
    monkeypatch.setattr('os.fsync', fsync)
    pub.start()
    assert entered.wait(1.)
    try:
        with pytest.raises(GpuError):
            pub.stop(timeout=.01)
    finally:
        release.set()
        assert pub.thread is not None
        pub.thread.join(1.)
    assert pub.failure and not (tmp_path / 'one.frame').exists()
    assert not (tmp_path / 'one.frame.pending').exists()


def test_publisher_pending_and_frame_both_inside_output_accounting(tmp_path: Path) -> None:
    from cipher_lab.speck_study.resources import directory_bytes
    pub, _, _, _ = setup(tmp_path)
    pub.publish_once()
    size = (tmp_path / 'one.frame').stat().st_size
    assert directory_bytes(tmp_path) == size and size <= 65536
    assert not (tmp_path / 'one.frame.pending').exists()


def test_slow_parent_frame_read_checks_ttl_again(tmp_path: Path, monkeypatch: Any) -> None:
    from cipher_lab.speck_study import gpu_telemetry
    pub, gpu, now, _ = setup(tmp_path)
    pub.publish_once()
    original = gpu_telemetry.read_sample
    def slow(root: Path, path: Path) -> dict[str, Any]:
        value = original(root, path)
        now[0] += 2.
        return value
    monkeypatch.setattr(gpu_telemetry, 'read_sample', slow)
    with pytest.raises(GpuError):
        gpu.processes(BINDING)


def test_live_thread_stale_sample_does_not_claim_live(tmp_path: Path) -> None:
    pub, _, now, _ = setup(tmp_path)
    first, blocked, release = threading.Event(), threading.Event(), threading.Event()
    count = [0]
    def sample() -> list[dict[str, Any]]:
        count[0] += 1
        if count[0] == 1:
            first.set()
        else:
            blocked.set()
            assert release.wait(2.)
        return rows()
    pub.sample, pub.interval = sample, .01
    pub.start()
    assert first.wait(1.) and blocked.wait(1.)
    now[0] += 2.
    try:
        with pytest.raises(GpuError):
            pub.check()
    finally:
        release.set()
        assert pub.thread is not None
        pub.thread.join(1.)
        try:
            pub.stop(timeout=1.)
        except GpuError:
            pass


@pytest.mark.parametrize('name', ['stream:ads', 'back\\slash', 'trailing.', 'trailing ', 'CON'])
def test_no_windows_namespace_alias_or_ads(tmp_path: Path, name: str) -> None:
    with pytest.raises(GpuError):
        TelemetryPublisher(tmp_path, tmp_path / name, pid=1, created=10, sample=rows)


def test_concurrent_publication_after_parent_tick_is_not_future(tmp_path: Path,
                                                               monkeypatch: Any) -> None:
    from cipher_lab.speck_study import gpu_telemetry
    pub, gpu, now, _ = setup(tmp_path)
    pub.publish_once()
    original = gpu_telemetry.read_sample
    def concurrent(root: Path, path: Path) -> dict[str, Any]:
        now[0] += .1  # parent started at100; publisher ran at100.1 before read
        pub.publish_once()
        return original(root, path)
    monkeypatch.setattr(gpu_telemetry, 'read_sample', concurrent)
    assert gpu.processes(BINDING) == {1: 7}


def test_windows_qpc_shared_epoch_not_python39_monotonic_offset() -> None:
    from cipher_lab.speck_study.gpu_telemetry import windows_monotonic
    class Kernel:
        def QueryPerformanceFrequency(self, ptr: Any) -> int:
            ptr._obj.value = 10
            return 1
        def QueryPerformanceCounter(self, ptr: Any) -> int:
            ptr._obj.value = 1234
            return 1
    assert windows_monotonic(Kernel()) == 123.4
    class Bad(Kernel):
        def QueryPerformanceFrequency(self, ptr: Any) -> int:
            return 0
    with pytest.raises(GpuError):
        windows_monotonic(Bad())


def test_parent_science_roles_not_any_attested_pid(tmp_path: Path) -> None:
    pub, gpu, _, _ = setup(tmp_path)
    pub.publish_once()
    gpu.scientific_pids = {1, 2}  # parent-owned roles, never obtained from frame
    gpu.verify(identity(), 1, BINDING.gpu_uuid, BINDING.pci_bus_id)
    assert gpu.ready is False


class ShareDeleteKernel:
    """WinAPI-shaped fake; sharing is enforced, not POSIX rename semantics."""
    def __init__(self, data: bytes) -> None:
        self.data, self.position = data, 0
        self.share = 0
        self.closed = False
        self.attributes = 0x80
        self.links = 1
        self.file_type = 1
        self.fail = ''
        self.on_read: Any = lambda: None
        self.read_sizes: list[int] = []

    def CreateFileW(self, path: str, access: int, share: int, security: Any,
                    disposition: int, flags: int, template: Any) -> int:
        assert access == 0x80000000 and disposition == 3
        assert flags & 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT
        self.share = share
        return -1 if self.fail == 'open' else 71

    def GetFileType(self, handle: Any) -> int:
        return self.file_type

    def GetFileInformationByHandle(self, handle: Any, pointer: Any) -> int:
        info = pointer._obj
        info.attributes, info.links = self.attributes, self.links
        info.size_low, info.size_high = len(self.data) & 0xffffffff, len(self.data) >> 32
        return 0 if self.fail == 'info' else 1

    def ReadFile(self, handle: Any, buffer: Any, size: int, read: Any, overlap: Any) -> int:
        self.read_sizes.append(size)
        self.on_read()
        chunk = self.data[self.position:self.position + min(size, 17)]
        buffer[:len(chunk)] = chunk
        self.position += len(chunk)
        read._obj.value = len(chunk)
        return 0 if self.fail == 'read' else 1

    def CloseHandle(self, handle: Any) -> int:
        self.closed = True
        return 0 if self.fail == 'close' else 1

    def require_delete_share(self) -> None:
        if not self.share & 4:
            raise OSError('fake Windows ERROR_SHARING_VIOLATION')


@pytest.mark.parametrize('operation', ['replace', 'stop-unlink'])
def test_fix1_windows_reader_shares_delete_with_publisher(tmp_path: Path, operation: str) -> None:
    from cipher_lab.speck_study.gpu_telemetry import _read_frame_bytes
    pub, _, _, _ = setup(tmp_path)
    pub.publish_once()
    previous = (tmp_path / 'one.frame').read_bytes()
    kernel = ShareDeleteKernel(previous)
    once = [False]
    def concurrent() -> None:
        if once[0]:
            return
        once[0] = True
        kernel.require_delete_share()  # fake Win contract BEFORE invoking local file operations
        if operation == 'replace':
            pub.sample = lambda: rows(9)
            pub.publish_once()
        else:
            pub.stop(timeout=.1)
    kernel.on_read = concurrent
    assert _read_frame_bytes(tmp_path, tmp_path / 'one.frame', kernel=kernel) == previous
    assert kernel.share == 7 and kernel.closed
    assert max(kernel.read_sizes) <= 65537
    assert (tmp_path / 'one.frame').exists() is (operation == 'replace')


@pytest.mark.parametrize('fault', ['open', 'info', 'read', 'close', 'reparse',
                                  'directory', 'hardlink', 'pipe', 'oversize'])
def test_fix1_windows_native_reader_faults_closed(tmp_path: Path, fault: str) -> None:
    from cipher_lab.speck_study.gpu_telemetry import _read_frame_bytes
    pub, _, _, _ = setup(tmp_path)
    pub.publish_once()
    kernel = ShareDeleteKernel(b'x' * (65537 if fault == 'oversize' else 10))
    if fault in ('open', 'info', 'read', 'close'):
        kernel.fail = fault
    elif fault == 'reparse':
        kernel.attributes = 0x400
    elif fault == 'directory':
        kernel.attributes = 0x10
    elif fault == 'hardlink':
        kernel.links = 2
    elif fault == 'pipe':
        kernel.file_type = 3
    with pytest.raises(GpuError):
        _read_frame_bytes(tmp_path, tmp_path / 'one.frame', kernel=kernel)
    assert kernel.closed is (fault != 'open')


def test_fix1_read_sample_windows_dispatch_not_builtin_open(tmp_path: Path,
                                                          monkeypatch: Any) -> None:
    import ctypes
    from types import SimpleNamespace

    from cipher_lab.speck_study import gpu_telemetry as module
    pub, _, _, _ = setup(tmp_path)
    pub.publish_once()
    kernel = ShareDeleteKernel((tmp_path / 'one.frame').read_bytes())
    monkeypatch.setattr(module, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw: kernel, raising=False)
    # ctypes signature declaration is exercised by a callable fake wrapper.
    for name in ('CreateFileW', 'GetFileType', 'GetFileInformationByHandle', 'ReadFile',
                 'CloseHandle'):
        function = getattr(kernel, name)
        setattr(kernel, name, lambda *args, fn=function: fn(*args))
    assert read_sample(tmp_path, tmp_path / 'one.frame')['pid'] == 1
    assert kernel.share == 7 and kernel.closed


def test_fix1_publisher_check_t1_then_publication_t2_not_future(tmp_path: Path) -> None:
    pub, _, now, _ = setup(tmp_path)
    pub.publish_once()
    entered, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []
    pub.started = True
    def clock() -> float:
        if threading.current_thread() is pub.thread:
            entered.set()
            assert release.wait(2.)
            return 100.  # check took t1, publication t2 occurs before it returns
        return now[0]
    pub.clock = clock
    def check() -> None:
        try:
            pub.check()
        except BaseException as error:
            errors.append(error)
    pub.thread = threading.Thread(target=check)
    pub.thread.start()
    assert entered.wait(1.)
    try:
        now[0] = 101.
        pub.publish_once()
    finally:
        release.set()
        pub.thread.join(1.)
    assert not errors and not pub.failure


@pytest.mark.parametrize('fault', ['ttl', 'future', 'death', 'sticky'])
def test_fix1_publisher_check_preserves_failure_gates(tmp_path: Path, fault: str) -> None:
    pub, _, now, _ = setup(tmp_path)
    pub.publish_once()
    release, entered = threading.Event(), threading.Event()
    def alive() -> None:
        entered.set()
        release.wait(2.)
    pub.started = True
    pub.thread = threading.Thread(target=alive)
    pub.thread.start()
    assert entered.wait(1.)
    if fault == 'ttl':
        now[0] = 102.
    elif fault == 'future':
        pub.last_time = 101.
    elif fault == 'sticky':
        pub.failure = True
    else:
        release.set()
        pub.thread.join(1.)
    try:
        with pytest.raises(GpuError):
            pub.check()
        with pytest.raises(GpuError):
            pub.check()
    finally:
        release.set()
        pub.thread.join(1.)


def test_fix1_owner_check_never_waits_for_fsync_publication_lock(tmp_path: Path) -> None:
    pub, _, _, _ = setup(tmp_path)
    pub.publish_once()
    entered, release, checked = threading.Event(), threading.Event(), threading.Event()
    errors: list[BaseException] = []
    def alive() -> None:
        entered.set()
        release.wait(2.)
    pub.thread, pub.started = threading.Thread(target=alive), True
    pub.thread.start()
    assert entered.wait(1.)
    def owner() -> None:
        try:
            pub.check()
            checked.set()
        except BaseException as error:
            errors.append(error)
    with pub.lock:  # simulate another thread stuck in publication fsync
        worker = threading.Thread(target=owner)
        worker.start()
        try:
            assert checked.wait(1.)
        finally:
            release.set()
    worker.join(1.)
    pub.thread.join(1.)
    assert not errors
