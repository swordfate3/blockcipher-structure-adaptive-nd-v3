"""Task3 ownership/deadline/storage faults; all Windows/GPU/process APIs faked."""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

import pytest

from cipher_lab_plugins.speck_study.platform_resources.windows import ProcessError, WindowsJob
from cipher_lab_plugins.speck_study.resources import (
    ResourceStop,
    Supervisor,
    durable_receipt,
    verify_sources,
)
from cipher_lab_plugins.speck_study.tests.test_speck_resources import (  # noqa: F401
    make_resources,
    resource_guard,
)


@pytest.fixture
def binding() -> Any:
    from cipher_lab_plugins.speck_study.platform_resources.gpu import GpuBinding
    return GpuBinding(1, 'GPU-one', '0000:02:00.0')


@pytest.fixture
def source_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Handwritten dual roots include the namespace required by source checks."""
    core, plugins = tmp_path / 'src', tmp_path / 'plugins'
    for root, name in ((core, 'cipher_lab'), (plugins, 'cipher_lab_plugins'),
                       (plugins, 'cipher_lab_plugins/speck_study'),
                       (plugins, 'cipher_lab_plugins/speck_study/runtime')):
        package = root / name
        package.mkdir(parents=True, exist_ok=True)
        (package / '__init__.py').write_bytes(b'')
    return core, plugins


class API:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.created = 10
        self.active: list[int] = []
        self.member = True
        self.assign_failure = False

    def create_job(self) -> int:
        self.events.append('job-kill-on-close')
        return 100

    def open(self, pid: int) -> int:
        self.events.append('open')
        return pid

    def times(self, handle: Any) -> tuple[int, float]:
        return self.created, 4.

    def belongs(self, job: Any, handle: Any) -> bool:
        return self.member

    def assign(self, job: Any, handle: Any) -> None:
        self.events.append('assign')
        if self.assign_failure:
            raise OSError('fake assign failure')
        self.active = [1]

    def resume(self, handle: Any) -> None:
        self.events.append('resume')

    def pids(self, job: Any) -> list[int]:
        return self.active

    def rss(self, handle: Any) -> int:
        return 100

    def cpu(self, job: Any) -> float:
        return 99.

    def close_handle(self, handle: Any) -> None:
        self.events.append('close-handle')

    def kill_job(self, handle: Any) -> None:
        self.events.append('kill-job')
        self.active = []


class Process:
    pid = 1
    _handle = 200
    stdin: Any = None
    stdout: Any = None
    stderr: Any = None

    def __init__(self, events: list[str], flags: int) -> None:
        assert flags == 4
        self.events = events
        events.append('spawn-suspended')

    def kill(self) -> None:
        self.events.append('kill-owned-handle')

    def wait(self, timeout: float) -> int:
        self.events.append('wait')
        return 1


def test_job_precedes_spawn_assign_precedes_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    api = API()
    monkeypatch.setattr('subprocess.Popen',
                        lambda args, creationflags, **kw: Process(api.events, creationflags))
    job = WindowsJob(api)
    job.spawn(['fake'])
    assert api.events[:5] == ['job-kill-on-close', 'spawn-suspended', 'assign', 'open', 'resume']
    assert job.cpu_seconds() == 99.  # exited-child accounting retained
    job.close()
    assert job.empty()
    job.dispose()


def test_assign_fail_never_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    api = API()
    api.assign_failure = True
    monkeypatch.setattr('subprocess.Popen',
                        lambda args, creationflags, **kw: Process(api.events, creationflags))
    job = WindowsJob(api)
    with pytest.raises(ProcessError):
        job.spawn(['fake'])
    assert 'resume' not in api.events
    assert api.events[-2:] == ['kill-owned-handle', 'wait']


@pytest.mark.parametrize('reuse', [True, False])
def test_reused_or_nonmember_pid_never_killed(reuse: bool) -> None:
    api = API()
    job = WindowsJob(api)
    api.active = [1]
    job.sample()
    if reuse:
        api.created += 1
    else:
        api.member = False
    with pytest.raises(ProcessError):
        job.sample()
    assert 'kill-owned-handle' not in api.events
    job.close()  # only original Job, not reused/unrelated PID
    assert api.events[-1] == 'kill-job'


def test_independent_sampling_while_constructor_blocks(tmp_path: Path, binding: Any) -> None:
    r, tree, gpu, _ = make_resources(tmp_path, binding)
    s = Supervisor(r)
    sampled = threading.Event()
    def missing(binding: Any) -> Any:
        sampled.set()
        return {1: None}
    s.start()
    gpu.processes = missing  # type: ignore[method-assign]
    # Main thread waits, as if in blocked teacher constructor/ready/predict.
    assert sampled.wait(2.)
    assert s.thread is not None
    s.thread.join(1.)
    assert tree.killed and isinstance(s.error, ResourceStop)
    assert not s.complete


@pytest.mark.parametrize('fault', ['late', 'afterrename', 'partial', 'missingexit'])
def test_finalization_cannot_complete(tmp_path: Path, binding: Any, fault: str) -> None:
    r, tree, _, now = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    s = Supervisor(r)
    def write(path: Path, record: dict[str, Any]) -> None:
        if fault == 'afterrename':
            def broken(parent: Path) -> None:
                raise OSError('fake directory flush failure AFTER rename')
            durable_receipt(path, record, sync_directory=broken)
        elif fault == 'late':
            now[0] = r.close_deadline
    with pytest.raises((OSError, ResourceStop)):
        s.finish(tmp_path / 'resource.json', child_exit=None if fault == 'missingexit' else 0,
                 child_terminal=fault != 'partial', write=write)
    assert not s.complete and tree.killed
    if fault == 'afterrename':
        assert (tmp_path / 'resource.json').exists()  # existence is not successful close


def test_science_close_reserve_and_success(tmp_path: Path, binding: Any) -> None:
    r, tree, _, now = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    now[0] = r.science_deadline
    s = Supervisor(r, request_closing=lambda: None, closing_acknowledged=lambda: True)
    s.finish(tmp_path / 'resource.json', child_exit=0, child_terminal=True)
    assert s.complete
    assert 'RESOURCE_CLOSED' in (tmp_path / 'resource.json').read_text()


def test_source_only_rejects_pyc_even_with_valid_source(source_tree: tuple[Path, Path]) -> None:
    import hashlib
    core, plugins = source_tree
    source = core / 'cipher_lab/x.py'
    source.write_text('x=1\n')
    manifest = {'cipher_lab/x.py': hashlib.sha256(source.read_bytes()).hexdigest()}
    verify_sources(core, plugins, manifest)
    source.with_suffix('.pyc').write_bytes(b'fake stale source bypass')
    with pytest.raises(ValueError, match='source-only'):
        verify_sources(core, plugins, manifest)


def test_wrapper_compile_source_no_bytecode(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / 'runtime/freeze.py'
    spec = importlib.util.spec_from_file_location('task3_freeze_test', script)
    assert spec and spec.loader
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    import hashlib
    for relative in wrapper.CLOSURE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source\n')
        path.with_suffix('.pyc').write_bytes(b'not executable')
    manifest = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
                for n in wrapper.CLOSURE}
    assert len(wrapper.checked_sources(tmp_path, tmp_path, manifest)) == len(wrapper.CLOSURE)
    (tmp_path / wrapper.CLOSURE[0]).unlink()
    with pytest.raises(FileNotFoundError):
        wrapper.checked_sources(tmp_path, tmp_path, manifest)


def test_trusted_launcher_rewrites_only_fixed_provider(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / 'runtime/supervise.py'
    spec = importlib.util.spec_from_file_location('task3_supervise_test', script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[Any] = []
    class Job:
        def spawn(self, args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return 'fake-process'
    spawn = module.TrustedTeacherSpawn(Job(), tmp_path / 'manifest.json', tmp_path, tmp_path)
    result = spawn(['python39', '-B', '-m', 'cipher_lab_plugins.speck_study.teacher_provider',
                    '--weights', 'fake'], env={'fake': 'yes'})
    assert result == 'fake-process'
    args = calls[0][0]
    assert args[1] == '-B' and args[3] == '--teacher'
    assert Path(args[2]).name == 'freeze.py'
    assert args[-2:] == ['--weights', 'fake']
    with pytest.raises(ValueError):
        spawn(['python39', '-B', '-m', 'arbitrary.plugin'])


def test_late_thread_join_never_complete(tmp_path: Path, binding: Any) -> None:
    r, tree, _, now = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    class LateThread:
        def join(self, timeout: float) -> None:
            now[0] = r.close_deadline
        def is_alive(self) -> bool:
            return False
    supervisor = Supervisor(r)
    supervisor.thread = LateThread()  # type: ignore[assignment]
    with pytest.raises(ResourceStop, match='CLOSE_DEADLINE'):
        supervisor.finish(tmp_path / 'resource.json', child_exit=0, child_terminal=True)
    assert not supervisor.complete


def test_teacher_client_uses_explicit_spawn_seam(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import json
    import time

    from cipher_lab_plugins.speck_study.teacher import TeacherClient
    from cipher_lab_plugins.speck_study.teacher_network import MODEL_SPEC
    from cipher_lab_plugins.speck_study.teacher_protocol import READY, Frame, write_frame
    payload = dict(python='3.9.synthetic', tensorflow='synthetic-no-TF', model=MODEL_SPEC,
                   weights_sha256='a'*64, allocator_mib=8192,
                   gpu=dict(pid=1, gpu_uuid='GPU-one', pci_bus_id='0000:02:00.0'))
    wire = io.BytesIO()
    write_frame(wire, Frame(READY, 0, 0, json.dumps(payload).encode()), time.monotonic()+10)
    wire.seek(0)
    class Child:
        pid = 1
        stdin = io.BytesIO()
        stdout = wire
        stderr = io.BytesIO()
        def poll(self) -> int:
            return 0
        def wait(self, timeout: float) -> int:
            return 0
    calls: list[Any] = []
    def spawn(args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return Child()
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('must use Job spawn seam, not default Popen')
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    teacher = TeacherClient('fake39', 'cipher_lab_plugins.speck_study.teacher_provider', 'unopened.h5',  # noqa: E501 - explicit migrated source/argv identity
                            'a'*64, 'GPU-one', tmp_path, time.monotonic()+10,
                            gpu_pci_bus_id='0000:02:00.0', verify_gpu=lambda *a: True,
                            spawn=spawn)
    teacher.close()
    assert len(calls) == 1 and teacher.cleanup_complete


def test_output_root_link_and_missing_fail_closed(tmp_path: Path) -> None:
    from cipher_lab_plugins.speck_study.resources import directory_bytes
    with pytest.raises(ResourceStop, match='OUTPUT_UNAVAILABLE'):
        directory_bytes(tmp_path / 'missing')
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'x').write_bytes(b'public fake')
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ResourceStop, match='OUTPUT_UNAVAILABLE'):
        directory_bytes(alias)


def test_final_file_fsync_failure_never_complete(tmp_path: Path, binding: Any,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    r, tree, _, _ = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    supervisor = Supervisor(r)
    def fail(fd: int) -> None:
        raise OSError('synthetic file flush failure')
    monkeypatch.setattr('os.fsync', fail)
    with pytest.raises(OSError):
        supervisor.finish(tmp_path / 'resource.json', child_exit=0, child_terminal=True)
    assert not supervisor.complete
    assert not (tmp_path / 'resource.json').exists()


def test_py39_fixed_wrapper_is_lazy_source_closure() -> None:
    import ast
    script = Path(__file__).parents[1] / 'runtime/freeze.py'
    tree = ast.parse(script.read_text(), feature_version=(3, 9))
    # Top-level imports cannot initialize TF/CUDA/Torch; trusted imports occur
    # only inside teacher AFTER its explicit source-byte manifest check.
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not n.name.startswith(('torch', 'tensorflow', 'cipher_lab'))
                       for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith(('torch', 'tensorflow', 'cipher_lab'))
    text = script.read_text()
    assert 'main(actual_attestor)' in text and 'os._exit(code)' in text


@pytest.mark.parametrize(('exit_code', 'terminal'), [(False, True), (0, 'partial')])
def test_finalization_exact_exit_and_terminal_types(tmp_path: Path, binding: Any,
                                                   exit_code: Any, terminal: Any) -> None:
    r, tree, _, _ = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    supervisor = Supervisor(r)
    with pytest.raises(ResourceStop, match='CLEANUP_UNKNOWN'):
        supervisor.finish(tmp_path / 'resource.json', child_exit=exit_code,
                          child_terminal=terminal)
    assert not supervisor.complete


def test_fix1_closing_handshake_preserves_save_window(tmp_path: Path, binding: Any) -> None:
    r, tree, _, now = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    requested: list[bool] = []
    ack = [False]
    supervisor = Supervisor(r, request_closing=lambda: requested.append(True),
                            closing_acknowledged=lambda: ack[0])
    # Child stops accepting science BEFORE the deadline and acknowledges after
    # its single writer has quiesced. Saving uses resources only, not science.
    r.begin_closing()
    ack[0] = True
    now[0] = r.science_deadline
    supervisor.tick()
    assert requested == [True] and not tree.killed
    with pytest.raises(ResourceStop, match='SCIENCE_DEADLINE'):
        r.check()
    assert r.failure is None
    now[0] += 200
    supervisor.finish(tmp_path / 'closing.json', child_exit=0, child_terminal=True)
    assert supervisor.complete and not tree.killed


@pytest.mark.parametrize('fault', ['missingack', 'missingterminal', 'timeout'])
def test_fix1_closing_unknown_never_complete(tmp_path: Path, binding: Any, fault: str) -> None:
    r, tree, _, now = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    supervisor = Supervisor(r, request_closing=lambda: None,
                            closing_acknowledged=lambda: fault != 'missingack')
    now[0] = r.science_deadline
    supervisor.tick()
    assert not tree.killed
    if fault == 'timeout':
        now[0] = r.close_deadline
        with pytest.raises(ResourceStop, match='CLOSE_DEADLINE'):
            supervisor.tick()
        assert tree.killed
    with pytest.raises(ResourceStop):
        supervisor.finish(tmp_path / 'unknown.json', child_exit=0,
                          child_terminal=fault != 'missingterminal')
    assert not supervisor.complete


@pytest.mark.parametrize('stage', ['assign', 'open', 'resume'])
@pytest.mark.parametrize('cleanup_fault', ['kill', 'wait', 'stream', 'none'])
def test_fix1_spawn_rollback_retains_owner_and_retries(monkeypatch: pytest.MonkeyPatch,
                                                      stage: str, cleanup_fault: str) -> None:
    api = API()
    events = api.events
    class BrokenAPI(API):
        def assign(self, job: Any, handle: Any) -> None:
            if stage == 'assign':
                raise OSError('assign')
            super().assign(job, handle)
        def open(self, pid: int) -> Any:
            if stage == 'open':
                raise OSError('open')
            return super().open(pid)
        def resume(self, handle: Any) -> None:
            raise OSError('resume')
    badapi = BrokenAPI()
    badapi.events = events
    faults = [True]
    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name
        def close(self) -> None:
            events.append(self.name)
            if self.name == 'stdin-close' and cleanup_fault == 'stream' and faults[0]:
                raise OSError('stream')
    class Child(Process):
        def __init__(self) -> None:
            super().__init__(events, 4)
            self.stdin, self.stdout, self.stderr = (Stream(n + '-close')
                                                    for n in ('stdin', 'stdout', 'stderr'))
        def kill(self) -> None:
            events.append('kill-owned-handle')
            if cleanup_fault == 'kill' and faults[0]:
                raise OSError('kill')
        def wait(self, timeout: float) -> int:
            events.append('wait')
            if cleanup_fault in ('kill', 'wait') and faults[0]:
                raise TimeoutError('wait')
            return 1
    child = Child()
    monkeypatch.setattr('subprocess.Popen', lambda *a, **kw: child)
    job = WindowsJob(badapi)
    with pytest.raises(ProcessError):
        job.spawn(['fake'])
    assert all(n in events for n in ('wait', 'stdin-close', 'stdout-close', 'stderr-close'))
    if cleanup_fault != 'none':
        assert job.cleanup_unknown
        assert job.orphans[0].process is child  # no PID reopening, Popen handle retained
        with pytest.raises(ProcessError):
            job.retry_cleanup()
    faults[0] = False
    job.retry_cleanup()
    assert not job.cleanup_unknown
    assert job.orphans == []


def test_fix1_terminate_failure_immediate_closehandle(tmp_path: Path, binding: Any) -> None:
    r, _, _, now = make_resources(tmp_path, binding)
    api = API()
    def fail(handle: Any) -> None:
        api.events.append('terminate-failed')
        raise OSError('terminate')
    api.kill_job = fail  # type: ignore[method-assign]
    job = WindowsJob(api)
    r.tree = job
    now[0] = r.close_deadline
    supervisor = Supervisor(r)
    with pytest.raises(ResourceStop):
        supervisor.tick()
    assert api.events[-2:] == ['terminate-failed', 'close-handle']
    assert not supervisor.complete


def test_fix1_launch_cleanup_failure_still_waits_and_joins(
        tmp_path: Path, binding: Any, monkeypatch: pytest.MonkeyPatch,
        source_tree: tuple[Path, Path]) -> None:
    import hashlib
    script = Path(__file__).parents[1] / 'runtime/supervise.py'
    spec = importlib.util.spec_from_file_location('task3_fix1_launch_test', script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r, tree, _, _ = make_resources(tmp_path, binding)
    events: list[str] = []
    class Thread:
        def join(self, timeout: float) -> None:
            events.append('join')
        def is_alive(self) -> bool:
            return False
    def start(s: Any) -> None:
        s.thread = Thread()
    monkeypatch.setattr(module.Supervisor, 'start', start)
    class Child:
        returncode = 1
        def poll(self) -> int:
            return 1
        def wait(self, timeout: float) -> int:
            events.append('wait')
            raise TimeoutError('wait')
    def spawn(*args: Any, **kwargs: Any) -> Any:
        return Child()
    def close() -> None:
        events.append('terminate')
        raise OSError('terminate')
    def dispose() -> None:
        events.append('dispose')
        raise OSError('closehandle')
    tree.spawn = spawn  # type: ignore[attr-defined]
    tree.close = close  # type: ignore[method-assign]
    tree.dispose = dispose  # type: ignore[attr-defined]
    tree.empty = lambda: True  # type: ignore[attr-defined]
    core, plugins = source_tree
    source = core / 'cipher_lab/fake.py'
    source.write_text('# public fake\n')
    with pytest.raises(ResourceStop):
        module.launch(r, ['fakepython', '-B', '-m', 'cipher_lab.cli', 'run', '--plugin', 'cipher_lab_plugins.speck_study.cli'],  # noqa: E501 - explicit migrated source/argv identity
                      source_root=core, plugin_root=plugins, source_manifest={'cipher_lab/fake.py': hashlib.sha256(  # noqa: E501 - explicit migrated source/argv identity
                          source.read_bytes()).hexdigest()}, receipt=tmp_path / 'receipt.json',
                      terminal_verified=lambda: False, clock=lambda: 100.,
                      request_closing=lambda: None, closing_acknowledged=lambda: True)
    assert 'wait' in events and 'join' in events and 'dispose' in events


def test_fix1_launch_cooperative_close_before_deadline(
        tmp_path: Path, binding: Any, monkeypatch: pytest.MonkeyPatch,
        source_tree: tuple[Path, Path]) -> None:
    import hashlib
    script = Path(__file__).parents[1] / 'runtime/supervise.py'
    spec = importlib.util.spec_from_file_location('task3_fix1_cooperate_test', script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, tree, _, now = make_resources(tmp_path, binding)
    child, _, _, child_now = make_resources(tmp_path, binding)
    events: list[str] = []
    ack = [False]
    class Child:
        returncode = 0
        polls = 0
        def poll(self) -> int | None:
            self.polls += 1
            if self.polls == 1:
                # child quiesces before 28200 then saves across that boundary
                child_now[0] = child.science_deadline - .1
                child.begin_closing()
                ack[0] = True
                now[0] = parent.science_deadline
                return None
            assert 'request' in events and not tree.killed
            child_now[0] = now[0] = parent.science_deadline + 100
            child.check(science=False)
            with pytest.raises(ResourceStop, match='SCIENCE_DEADLINE'):
                child.check()
            events.append('child-saved-terminal')
            return 0
        def wait(self, timeout: float) -> int:
            events.append('wait')
            return 0
    def spawn(*args: Any, **kwargs: Any) -> Any:
        return Child()
    tree.spawn = spawn  # type: ignore[attr-defined]
    tree.dispose = lambda: events.append('dispose')  # type: ignore[attr-defined]
    tree.empty = lambda: True  # type: ignore[attr-defined]
    # Keep this launch scenario deterministic; tick is exercised by launch loop.
    monkeypatch.setattr(module.Supervisor, 'start', lambda s: s.tick())
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    core, plugins = source_tree
    source = core / 'cipher_lab/fake.py'
    source.write_text('# fake source\n')
    result = module.launch(parent, ['fake', '-B', '-m', 'cipher_lab.cli', 'run', '--plugin', 'cipher_lab_plugins.speck_study.cli'],  # noqa: E501 - explicit migrated source/argv identity
                           source_root=core, plugin_root=plugins, source_manifest={'cipher_lab/fake.py': hashlib.sha256(  # noqa: E501 - explicit migrated source/argv identity
                               source.read_bytes()).hexdigest()}, receipt=tmp_path / 'closed.json',
                           terminal_verified=lambda: 'child-saved-terminal' in events,
                           request_closing=lambda: events.append('request'),
                           closing_acknowledged=lambda: ack[0], clock=lambda: now[0])
    assert result.complete and events == ['request', 'child-saved-terminal', 'wait', 'dispose']


def test_fix1_orphan_cleanup_dispose_retries_after_job_handle_closed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    api = API()
    api.assign_failure = True
    fail = [True]
    class Child(Process):
        def wait(self, timeout: float) -> int:
            if fail[0]:
                raise TimeoutError('synthetic wait')
            return 0
    child = Child(api.events, 4)
    monkeypatch.setattr('subprocess.Popen', lambda *a, **kw: child)
    job = WindowsJob(api)
    with pytest.raises(ProcessError):
        job.spawn(['fake'])
    with pytest.raises(ProcessError):
        job.dispose()
    assert job.closed and job.cleanup_unknown
    fail[0] = False
    job.dispose()
    assert not job.cleanup_unknown


def test_fix1_unreaped_spawn_prevents_new_child(monkeypatch: pytest.MonkeyPatch) -> None:
    api = API()
    api.assign_failure = True
    events = api.events
    class Child(Process):
        def wait(self, timeout: float) -> int:
            raise TimeoutError('unreaped')
    def popen(*args: Any, **kwargs: Any) -> Any:
        return Child(events, 4)
    monkeypatch.setattr('subprocess.Popen', popen)
    job = WindowsJob(api)
    with pytest.raises(ProcessError):
        job.spawn(['fake'])
    assert job.cleanup_unknown
    with pytest.raises(ProcessError):
        job.spawn(['fake-again'])
    assert events.count('spawn-suspended') == 1


def test_fix2_receipt_tick_requires_final_ack(tmp_path: Path, binding: Any) -> None:
    r, tree, _, _ = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    receipt_entered, tick_done = threading.Event(), threading.Event()
    requests: list[bool] = []
    supervisor = Supervisor(r, request_closing=lambda: requests.append(True),
                            closing_acknowledged=lambda: False)
    def background_tick() -> None:
        assert receipt_entered.wait(2.)
        try:
            supervisor.tick()
        finally:
            tick_done.set()
    supervisor.thread = threading.Thread(target=background_tick)
    supervisor.thread.start()
    def write(path: Path, record: dict[str, Any]) -> None:
        # finish passed its first ACK gate before background tick sends request.
        assert not supervisor.closing_requested
        receipt_entered.set()
        assert tick_done.wait(2.)
    try:
        with pytest.raises(ResourceStop, match='CLEANUP_UNKNOWN'):
            supervisor.finish(tmp_path / 'receipt.json', child_exit=0,
                              child_terminal=True, write=write)
    finally:
        receipt_entered.set()
        supervisor.thread.join(2.)
    assert requests == [True] and not supervisor.complete


@pytest.mark.parametrize('fault', ['throw', 'nan', 'backward', 'deadline', 'waitthrow'])
def test_fix2_final_clock_fault_revokes_complete(
        tmp_path: Path, binding: Any, monkeypatch: pytest.MonkeyPatch,
        fault: str, source_tree: tuple[Path, Path]) -> None:
    import hashlib
    script = Path(__file__).parents[1] / 'runtime/supervise.py'
    spec = importlib.util.spec_from_file_location('task3_fix2_launch_test', script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r, tree, _, _ = make_resources(tmp_path, binding)
    events: list[str] = []
    owners: list[Any] = []
    class Thread:
        def join(self, timeout: float) -> None:
            events.append('join')
        def is_alive(self) -> bool:
            return False
    def start(s: Any) -> None:
        owners.append(s)
        s.thread = Thread()
        s.tick()
    monkeypatch.setattr(module.Supervisor, 'start', start)
    class Child:
        returncode = 0
        def poll(self) -> int:
            return 0
        def wait(self, timeout: float) -> int:
            events.append('wait')
            return 0
    tree.spawn = lambda *a, **kw: Child()  # type: ignore[attr-defined]
    tree.empty = lambda: True  # type: ignore[attr-defined]
    tree.dispose = lambda: events.append('dispose')  # type: ignore[attr-defined]
    clock_calls = [0]
    def clock() -> float:
        clock_calls[0] += 1
        if clock_calls[0] == 1:  # cleanup wait timeout sample
            if fault == 'waitthrow':
                raise RuntimeError('private wait clock failure')
            return 101.
        if fault == 'waitthrow':
            return 101.
        assert owners[0].complete  # finish succeeded, final guard must revoke it
        if fault == 'throw':
            raise RuntimeError('private clock failure')
        return {'nan': float('nan'), 'backward': 100.5,
                'deadline': r.close_deadline}[fault]
    core, plugins = source_tree
    source = core / 'cipher_lab/fake.py'
    source.write_text('# fake source\n')
    with pytest.raises(ResourceStop, match='CLEANUP_UNKNOWN'):
        module.launch(r, ['fake', '-B', '-m', 'cipher_lab.cli', 'run', '--plugin', 'cipher_lab_plugins.speck_study.cli'],  # noqa: E501 - explicit migrated source/argv identity
                      source_root=core, plugin_root=plugins, source_manifest={'cipher_lab/fake.py': hashlib.sha256(  # noqa: E501 - explicit migrated source/argv identity
                          source.read_bytes()).hexdigest()}, receipt=tmp_path / 'closed.json',
                      terminal_verified=lambda: True, clock=clock,
                      request_closing=lambda: None, closing_acknowledged=lambda: True)
    assert not owners[0].complete
    assert events.count('join') == 2 and 'wait' in events and 'dispose' in events
    assert owners[0].error.reason == 'CLEANUP_UNKNOWN'


def test_fix3_unordered_sampler_read_does_not_reject_parent_clock(
        tmp_path: Path, binding: Any, monkeypatch: pytest.MonkeyPatch,
        source_tree: tuple[Path, Path]) -> None:
    import hashlib
    script = Path(__file__).parents[1] / 'runtime/supervise.py'
    spec = importlib.util.spec_from_file_location('task3_fix3_clock_test', script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r, tree, _, now = make_resources(tmp_path, binding)
    parent_sampled, sampler_done = threading.Event(), threading.Event()
    def sampler() -> None:
        assert parent_sampled.wait(2.)
        try:
            now[0] = 102.  # t2: later sample must not invalidate earlier parent t1
            r.check(science=False)
        finally:
            sampler_done.set()
    worker = threading.Thread(target=sampler)
    worker.start()
    calls = [0]
    def clock() -> float:
        calls[0] += 1
        if calls[0] == 1:
            t1 = 101.
            parent_sampled.set()
            assert sampler_done.wait(2.)  # resume comparison only after t2 published
            assert r.last_wall == 102.
            return t1
        return 103.
    events: list[str] = []
    class Child:
        returncode = 0
        def poll(self) -> int:
            return 0
        def wait(self, timeout: float) -> int:
            events.append('wait')
            return 0
    tree.spawn = lambda *a, **kw: Child()  # type: ignore[attr-defined]
    tree.empty = lambda: True  # type: ignore[attr-defined]
    tree.dispose = lambda: events.append('dispose')  # type: ignore[attr-defined]
    monkeypatch.setattr(module.Supervisor, 'start', lambda s: s.tick())
    core, plugins = source_tree
    source = core / 'cipher_lab/fake.py'
    source.write_text('# fake source\n')
    try:
        result = module.launch(r, ['fake', '-B', '-m', 'cipher_lab.cli', 'run', '--plugin', 'cipher_lab_plugins.speck_study.cli'],  # noqa: E501 - explicit migrated source/argv identity
                               source_root=core, plugin_root=plugins, source_manifest={'cipher_lab/fake.py': hashlib.sha256(  # noqa: E501 - explicit migrated source/argv identity
                                   source.read_bytes()).hexdigest()},
                               receipt=tmp_path / 'closed.json', terminal_verified=lambda: True,
                               clock=clock, request_closing=lambda: None,
                               closing_acknowledged=lambda: True)
        assert result.complete and result.error is None
        assert events == ['wait', 'dispose']
    finally:
        parent_sampled.set()
        worker.join(2.)
    assert not worker.is_alive()


def test_task5a_windows_receipt_readback_not_powerloss(tmp_path: Path) -> None:
    import hashlib
    import json
    path = tmp_path / 'receipt.json'
    proof = durable_receipt(path, dict(status='RESOURCE_CLOSED'), platform='nt')
    raw = path.read_bytes()
    assert proof['target_size'] == len(raw)
    assert proof['target_sha256'] == hashlib.sha256(raw).hexdigest()
    saved = json.loads(raw)
    assert saved['namespace_powerloss'] == 'UNVERIFIED'
    assert saved['write_protocol'] == 'file-fsync+replace+target-readback-len-sha256'


def test_task5a_receipt_bad_readback_never_complete(tmp_path: Path, binding: Any,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes
    def bad(path: Path) -> bytes:
        if path.name == 'receipt.json':
            return b'corrupted'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', bad)
    r, tree, _, _ = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    supervisor = Supervisor(r)
    with pytest.raises(ResourceStop):
        supervisor.finish(tmp_path / 'receipt.json', child_exit=0, child_terminal=True)
    assert not supervisor.complete


def test_task5a_supervisor_retains_readback_evidence(tmp_path: Path, binding: Any) -> None:
    r, tree, _, _ = make_resources(tmp_path, binding)
    tree.empty = lambda: True  # type: ignore[attr-defined]
    supervisor = Supervisor(r)
    supervisor.finish(tmp_path / 'receipt.json', child_exit=0, child_terminal=True)
    assert supervisor.complete
    assert supervisor.receipt_evidence['namespace_powerloss'] == 'UNVERIFIED'
    assert supervisor.receipt_evidence['target_size'] > 0
