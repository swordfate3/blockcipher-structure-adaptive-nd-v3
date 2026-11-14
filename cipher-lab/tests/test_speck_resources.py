"""Task3 fake clocks/RSS/GPU only; never read this host's resources."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cipher_lab.execution import Clocks, Meter, Stopped
from cipher_lab.platform_resources.gpu import GpuBinding, GpuError, verify_runtime
from cipher_lab.speck_study.resources import ResourceStop, StudyResources


@pytest.fixture(autouse=True)
def resource_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tripwires: a mistaken real probe must fail, not silently test this host."""
    import builtins
    import ctypes
    original = builtins.__import__
    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split('.')[0] in ('torch', 'tensorflow') or name.startswith('cipher_lab.ciphers'):
            raise AssertionError('Task3 forbids scientific imports')
        return original(name, *args, **kwargs)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('Task3 forbids actual GPU/RSS/process probes')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(ctypes, 'WinDLL', forbidden, raising=False)
    monkeypatch.setattr('os.urandom', forbidden)
    monkeypatch.setattr('subprocess.run', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.setattr('cipher_lab.execution.production_rss_bytes', forbidden)


class Tree:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = [dict(pid=1, created=10, rss=1, cpu=2.)]
        self.killed = False

    def sample(self) -> list[dict[str, Any]]:
        return self.rows

    def close(self) -> None:
        self.killed = True


class GPU:
    def __init__(self) -> None:
        self.used: int | None = 1

    def processes(self, binding: GpuBinding) -> dict[int, int | None]:
        return {1: self.used}


@pytest.fixture
def binding() -> GpuBinding:
    return GpuBinding.from_snapshot(1, [
        dict(index=0, gpu_uuid='GPU-zero', pci_bus_id='00000000:01:00.0'),
        dict(index=1, gpu_uuid='GPU-one', pci_bus_id='00000000:02:00.0')])


def test_binding_not_enumeration_or_environment(binding: GpuBinding) -> None:
    assert binding.gpu_uuid == 'GPU-one'
    good = dict(gpu_uuid='GPU-one', pci_bus_id=binding.pci_bus_id, pid=1,
                logical_device=0, evidence='cuda-current-context')
    verify_runtime(binding, good)
    for change in (dict(gpu_uuid='GPU-zero'), dict(pci_bus_id='0000:03:00.0'),
                   dict(logical_device=1), dict(evidence='CUDA_VISIBLE_DEVICES')):
        with pytest.raises(GpuError):
            verify_runtime(binding, {**good, **change})


def make_resources(tmp_path: Path, binding: GpuBinding
                   ) -> tuple[StudyResources, Tree, GPU, list[float]]:
    tree, gpu, now = Tree(), GPU(), [100.]
    r = StudyResources(binding, tmp_path, 100., 28300., 28900., tree,
                       gpu=gpu, clock=lambda: now[0], output_bytes=lambda: 0)
    return r, tree, gpu, now


@pytest.mark.parametrize(('kind', 'reason'), [
    ('gpu', 'GPU_LIMIT'), ('wddm', 'GPU_UNVERIFIED'), ('rss', 'RSS_LIMIT'),
    ('missingrss', 'RSS_UNAVAILABLE'), ('cpu', 'CPU_UNAVAILABLE'),
    ('science', 'SCIENCE_DEADLINE'), ('close', 'CLOSE_DEADLINE'), ('output', 'OUTPUT_LIMIT')])
def test_resource_fail_closed(tmp_path: Path, binding: GpuBinding, kind: str, reason: str) -> None:
    r, tree, gpu, now = make_resources(tmp_path, binding)
    if kind == 'gpu':
        gpu.used = 20 * 2**30 + 1
    elif kind == 'wddm':
        gpu.used = None
    elif kind == 'rss':
        tree.rows[0]['rss'] = 32 * 2**30 + 1
    elif kind == 'missingrss':
        tree.rows[0]['rss'] = None
    elif kind == 'cpu':
        tree.rows[0]['cpu'] = None
    elif kind == 'science':
        now[0] = 28300.
    elif kind == 'close':
        now[0] = 28900.
    else:
        r.output_bytes = lambda: 4 * 2**30 - 64 * 2**20 + 1
    with pytest.raises(ResourceStop, match=reason):
        r.check()
    with pytest.raises(ResourceStop, match=reason):
        r.check()


def test_tree_cpu_never_resets_on_exit(tmp_path: Path, binding: GpuBinding) -> None:
    r, tree, _, _ = make_resources(tmp_path, binding)
    r.check()
    tree.rows[0]['cpu'] = 9.
    r.check()
    tree.rows = []
    r.check()
    assert r.cpu_seconds == 9.


def test_actual_sum_not_allocator_caps(tmp_path: Path, binding: GpuBinding) -> None:
    r, tree, gpu, _ = make_resources(tmp_path, binding)
    tree.rows.append(dict(pid=2, created=20, rss=1, cpu=1.))
    gpu.processes = lambda binding: {1: 11 * 2**30, 2: 10 * 2**30}  # type: ignore[method-assign]
    with pytest.raises(ResourceStop, match='GPU_LIMIT'):
        r.check()


def test_cpu_keyword_default_stops_false_still_samples() -> None:
    from cipher_lab.config import Policy
    p = Policy(2, 100., 1., '2030-01-01T00:00:20+00:00', 12, 16, 1000, 1000, 0, 0)
    cpu = [0.]
    clocks = Clocks(lambda: 0., lambda: cpu[0],
                    lambda: datetime(2030, 1, 1, tzinfo=timezone.utc))
    old = Meter(p, clocks, lambda: 1)
    new = Meter(p, clocks, lambda: 1, enforce_cpu=False)
    cpu[0] = 2.
    with pytest.raises(Stopped, match='CPU_LIMIT'):
        old.check()
    new.check()
    assert new.snapshot()['cpu_elapsed_seconds'] == 2.


def test_study_stop_never_completes(tmp_path: Path, binding: GpuBinding) -> None:
    from test_execution import FakeTime

    from cipher_lab.speck_study.resources import StudyMeter
    resources, _, _, now = make_resources(tmp_path, binding)
    meter = StudyMeter(FakeTime().meter(), resources)
    meter.reserve('teacher', 0)
    now[0] = resources.science_deadline
    with pytest.raises(ResourceStop):
        meter.complete('teacher', 0)


class Driver:
    """ctypes-shaped fake current CUDA context; no CUDA library loaded."""
    def __init__(self, current: int = 1, device: int = 0) -> None:
        self.current, self.device = current, device

    def cuCtxGetCurrent(self, pointer: Any) -> int:
        pointer._obj.value = self.current
        return 0

    def cuCtxGetDevice(self, pointer: Any) -> int:
        pointer._obj.value = self.device
        return 0

    def cuDeviceGetUuid(self, pointer: Any, device: Any) -> int:
        for i in range(16):
            pointer._obj[i] = i
        return 0

    def cuDeviceGetPCIBusId(self, buffer: Any, size: Any, device: Any) -> int:
        buffer.value = b'00000000:02:00.0'
        return 0

    def cuDeviceGetLuid(self, buffer: Any, mask: Any, device: Any) -> int:
        buffer[0] = b'\x01'
        mask._obj.value = 1
        return 0


@pytest.mark.parametrize(('context', 'device'), [(0, 0), (1, 1)])
def test_actual_attestor_no_context_or_wrong_logical(context: int, device: int) -> None:
    from cipher_lab.platform_resources.gpu import actual_attestor
    with pytest.raises(GpuError):
        actual_attestor(driver=Driver(context, device))


def test_actual_attestor_reads_driver_not_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.platform_resources.gpu import actual_attestor
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-deadbeef')
    evidence = actual_attestor(driver=Driver())
    assert evidence['gpu_uuid'] == 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f'
    assert evidence['pci_bus_id'] == '0000:02:00.0'


class FakeNvml:
    def __init__(self, used: int = 9) -> None:
        self.used = used

    def nvmlInit_v2(self) -> int:
        return 0

    def nvmlDeviceGetHandleByUUID(self, uuid: Any, pointer: Any) -> int:
        pointer._obj.value = 5
        return 0

    def nvmlDeviceGetComputeRunningProcesses_v2(self, h: Any, count: Any, rows: Any) -> int:
        count._obj.value = 1
        rows[0].pid, rows[0].used = 1, self.used
        return 0

    def nvmlDeviceGetGraphicsRunningProcesses_v2(self, h: Any, count: Any, rows: Any) -> int:
        return self.nvmlDeviceGetComputeRunningProcesses_v2(h, count, rows)


def test_nvml_stable_abi_union_no_double_count(binding: GpuBinding) -> None:
    from cipher_lab.platform_resources.gpu import Nvml
    assert Nvml(FakeNvml()).processes(binding) == {1: 9}
    assert Nvml(FakeNvml(2**64 - 1)).processes(binding) == {1: None}


def test_main_torch_actual_binding_and_allocator(monkeypatch: pytest.MonkeyPatch,
                                               binding: GpuBinding) -> None:
    from cipher_lab.platform_resources.gpu import configure_torch
    events: list[Any] = []
    class Cuda:
        def device_count(self) -> int:
            return 1
        def set_device(self, index: int) -> None:
            events.append(('device', index))
        def get_device_properties(self, index: int) -> Any:
            class Props:
                total_memory = 48 * 2**30
            return Props()
        def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
            events.append(('allocator', fraction, device))
        def init(self) -> None:
            events.append('init')
    class Torch:
        cuda = Cuda()
    def attest(runtime: Any) -> dict[str, Any]:
        return dict(gpu_uuid=binding.gpu_uuid, pci_bus_id=binding.pci_bus_id,
                    logical_device=0, pid=1, evidence='cuda-current-context')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', binding.gpu_uuid)
    configure_torch(Torch(), binding, attestor=attest)
    assert events == [('device', 0), ('allocator', 1/6, 0), 'init']
    def wrong(runtime: Any) -> dict[str, Any]:
        return {**attest(runtime), 'gpu_uuid': 'GPU-zero'}
    with pytest.raises(GpuError):
        configure_torch(Torch(), binding, attestor=wrong)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(GpuError):
        configure_torch(Torch(), binding, attestor=attest)


def test_parent_verifier_not_ready_echo(binding: GpuBinding,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab.platform_resources.gpu import Nvml
    evidence = dict(gpu_uuid=binding.gpu_uuid, pci_bus_id=binding.pci_bus_id,
                    logical_device=0, pid=1, evidence='cuda-current-context')
    class Snapshot:
        stdout = 'GPU-one, 00000000:02:00.0\n'
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: Snapshot())
    assert Nvml(FakeNvml()).verify(evidence, 1, binding.gpu_uuid, binding.pci_bus_id)
    assert not Nvml(FakeNvml(2**64 - 1)).verify(
        evidence, 1, binding.gpu_uuid, binding.pci_bus_id)
    Snapshot.stdout = 'GPU-one, 00000000:03:00.0\n'
    with pytest.raises(GpuError):
        Nvml(FakeNvml()).verify(evidence, 1, binding.gpu_uuid, binding.pci_bus_id)


def test_parent_proof_requires_owned_creation_identity(tmp_path: Path, binding: GpuBinding) -> None:
    r, tree, gpu, _ = make_resources(tmp_path, binding)
    evidence = dict(pid=1)
    gpu.verify = lambda *args: True  # type: ignore[attr-defined]
    assert r.verify_teacher(evidence, 1, binding.gpu_uuid, binding.pci_bus_id)
    def reused(*args: Any) -> bool:
        tree.rows[0]['created'] = 99
        return True
    gpu.verify = reused  # type: ignore[attr-defined]
    assert not r.verify_teacher(evidence, 1, binding.gpu_uuid, binding.pci_bus_id)


@pytest.mark.parametrize('fault', ['clock', 'job', 'gpulost', 'cpubackward'])
def test_additional_monitor_faults(tmp_path: Path, binding: GpuBinding, fault: str) -> None:
    r, tree, gpu, now = make_resources(tmp_path, binding)
    r.check()
    if fault == 'clock':
        now[0] -= 1
        reason = 'CLOCK_UNAVAILABLE'
    elif fault == 'job':
        def failed() -> Any:
            raise OSError('fake Job query')
        tree.sample = failed  # type: ignore[method-assign]
        reason = 'PROCESS_UNVERIFIED'
    elif fault == 'gpulost':
        gpu.processes = lambda binding: {}  # type: ignore[method-assign]
        reason = 'GPU_UNVERIFIED'
    else:
        tree.rows[0]['cpu'] = 1.
        reason = 'CPU_UNAVAILABLE'
    with pytest.raises(ResourceStop, match=reason):
        r.check()


def test_blocked_parent_proof_does_not_block_sampler(tmp_path: Path, binding: GpuBinding) -> None:
    import threading
    r, _, gpu, _ = make_resources(tmp_path, binding)
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    def verifier(*args: Any) -> bool:
        entered.set()
        assert release.wait(2.)
        return True
    gpu.verify = verifier  # type: ignore[attr-defined]
    worker = threading.Thread(target=lambda: r.verify_teacher({}, 1, binding.gpu_uuid,
                                                              binding.pci_bus_id))
    def sample() -> None:
        r.check()
        done.set()
    sampler = threading.Thread(target=sample)
    worker.start()
    try:
        assert entered.wait(1.)
        sampler.start()
        assert done.wait(1.)
    finally:
        release.set()
        worker.join(2.)
        sampler.join(2.)
    assert not worker.is_alive() and not sampler.is_alive()


def test_terminal_reserve_only_inside_close(tmp_path: Path, binding: GpuBinding) -> None:
    r, _, _, _ = make_resources(tmp_path, binding)
    r.output_bytes = lambda: 4 * 2**30 - 1
    r.begin_closing()
    r.check(science=False)
    r.output_bytes = lambda: 4 * 2**30 + 1
    with pytest.raises(ResourceStop, match='OUTPUT_LIMIT'):
        r.check(science=False)


def test_fix1_child_closing_blocks_meter_attempts_and_completion(
        tmp_path: Path, binding: GpuBinding) -> None:
    from test_execution import FakeTime

    from cipher_lab.speck_study.resources import StudyMeter
    r, _, gpu, _ = make_resources(tmp_path, binding)
    meter = StudyMeter(FakeTime().meter(), r)
    meter.reserve('scoring', 4)
    r.begin_closing()  # child quiesces before its deadline, then signals ACK
    with pytest.raises(ResourceStop, match='SCIENCE_DEADLINE'):
        meter.reserve('scoring', 1)
    with pytest.raises(ResourceStop, match='SCIENCE_DEADLINE'):
        meter.complete('scoring', 4)
    assert meter.counts['scoring'] == [4, 0]
    r.check(science=False)
    gpu.used = 20 * 2**30 + 1
    with pytest.raises(ResourceStop, match='GPU_LIMIT'):
        r.check(science=False)  # cooperation never disables hard resource checks
