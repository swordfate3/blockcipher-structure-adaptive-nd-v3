"""5B lifecycle tests: only fake OS/clock/resources; never native/science calls."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest
from test_speck_resources import resource_guard  # noqa: F401

from cipher_lab.speck_study.teacher_protocol import DeadlineIO


def test_qpc_ipc_different_epoch_and_local_join() -> None:
    transport = DeadlineIO(io.BytesIO(b'x'), clock=lambda: 10**12)
    assert transport.run(lambda: transport.stream.read(1), 10**12 + 1) == b'x'
    # join is a LOCAL timeout clock, not the frozen formal clock.
    import time
    assert transport.join(time.monotonic() + 1)
    with pytest.raises(TimeoutError):
        transport.run(lambda: b'', 10**12)


def test_old_default_ipc_clock_still_expires() -> None:
    with pytest.raises(TimeoutError):
        DeadlineIO(io.BytesIO()).run(lambda: b'', 0)


def test_closing_requires_request_and_two_whitelisted_manual_events() -> None:
    from cipher_lab.speck_study.closing import ClosingChannel, EventPair
    class API:
        def __init__(self) -> None:
            self.calls: list[Any] = []
            self.signalled: set[int] = set()
        def create(self) -> int:
            self.calls.append('manual-reset')
            return len(self.calls)
        def close(self, handle: int) -> None:
            self.calls.append(('close', handle))
        def poll(self, handle: int) -> bool:
            return handle in self.signalled
        def set(self, handle: int) -> None:
            self.signalled.add(handle)
    api = API()
    pair = EventPair(api)
    assert api.calls == ['manual-reset', 'manual-reset']
    assert pair.spawn_options()['close_fds'] is True
    assert pair.handle_list == [1, 2]
    channel = ClosingChannel(api, 1, 2)
    with pytest.raises(RuntimeError):
        channel.acknowledge(saved=True, quiesced=True)
    channel.request()
    channel.acknowledge(saved=True, quiesced=True)
    assert channel.acknowledged()
    pair.close()
    assert api.calls[-2:] == [('close', 1), ('close', 2)]


def test_control_mailbox_strict_no_closed_channel_success(tmp_path: Path) -> None:
    from cipher_lab.speck_study.native_control import read_message, write_message
    p = tmp_path/'message'
    write_message(p, {'seq': 1, 'kind': 'check'})
    assert read_message(p) == {'seq': 1, 'kind': 'check'}
    p.write_bytes(b'{"seq":1,"seq":2}')
    with pytest.raises(ValueError):
        read_message(p)
    p.write_bytes(b'x'*65537)
    with pytest.raises(ValueError):
        read_message(p)


def test_child_science_stop_is_irreversible_and_no_unsolicited_ack() -> None:
    from cipher_lab.speck_study.native_control import ChildResources
    from cipher_lab.speck_study.resources import ResourceStop
    class Channel:
        requested_value = False
        def requested(self) -> bool:
            return self.requested_value
        def acknowledge(self, **kwargs: Any) -> None:
            raise AssertionError('unsolicited ACK')
    channel = Channel()
    calls = []
    def rpc(kind: str, **payload: Any) -> dict[str, Any]:
        calls.append((kind,payload))
        return {'wall': 100., 'science_deadline': 28200., 'latest': {}}
    resources = ChildResources(rpc, channel, lambda: None)
    resources.check()
    channel.requested_value = True
    with pytest.raises(ResourceStop):
        resources.check()
    channel.requested_value = False
    with pytest.raises(ResourceStop):
        resources.check()
    resources.acknowledge_closing()  # no requested signal, no ACK
    assert calls[0] == ('check', {'science': True})


def test_bootstrap_missing_first_frame_is_bounded_not_zero(tmp_path: Path) -> None:
    from cipher_lab.platform_resources.gpu import GpuBinding, GpuError
    from cipher_lab.speck_study.gpu_telemetry import BootstrapPending, TelemetryGpu
    now = [100.]
    def members() -> list[dict[str,Any]]:
        return [dict(pid=1,created=10)]
    gpu = TelemetryGpu(tmp_path, {}, members=members, creation_time=lambda p: 10,
                       science_deadline=28300., inventory={'0100000000000000':1},
                       clock=lambda: now[0])
    gpu.register(1,10,tmp_path/'first.json', first_frame_deadline=105.)
    with pytest.raises(BootstrapPending):
        gpu.processes(GpuBinding(1,'GPU-one','0000:02:00.0'))
    assert not gpu.failure and not gpu.ready
    now[0] = 105.
    with pytest.raises(GpuError):
        gpu.processes(GpuBinding(1,'GPU-one','0000:02:00.0'))
    assert gpu.failure


def test_job_external_suspended_adopt_register_before_resume() -> None:
    from cipher_lab.platform_resources.windows import WindowsJob
    class API:
        def __init__(self) -> None:
            self.events: list[str] = []
        def create_job(self) -> int:
            return 99
        def open(self, pid: int) -> int:
            self.events.append('open')
            return pid
        def belongs(self, job: int, handle: int) -> bool:
            return True
        def times(self, handle: int) -> tuple[int,float]:
            return 123,0.
        def resume(self, handle: int) -> None:
            self.events.append('resume')
        def close_handle(self, handle: int) -> None:
            self.events.append('close')
    api = API()
    job = WindowsJob(api)
    def before(pid: int, created: int) -> None:
        assert (pid,created) == (8,123)
        api.events.append('register-slot')
    job.adopt_suspended(8, before_resume=before)
    assert api.events == ['open','register-slot','resume']
    assert job.creation_time(8) == 123


def test_parent_inventory_science_roles_and_reap_retirement(tmp_path: Path) -> None:
    from cipher_lab.speck_study.native_session import OwnedTree
    class API:
        def __init__(self) -> None:
            self.created = {10:100,20:200,30:300}
        def open(self, pid: int) -> int:
            return pid
        def times(self, handle: int) -> tuple[int,float]:
            return self.created[handle], float(handle)
        def rss(self, handle: int) -> int:
            return handle
        def exited(self, handle: int) -> bool:
            return handle == 30
        def close_handle(self, handle: int) -> None:
            pass
    class Job:
        def __init__(self) -> None:
            self.api = API()
            self.handles = {20:(200,20),30:(300,30)}
            self.rows = [dict(pid=20,created=200,cpu=20.,rss=20),
                         dict(pid=30,created=300,cpu=30.,rss=30)]
        def sample(self) -> list[dict[str,Any]]:
            return self.rows
        def creation_time(self, pid: int) -> int:
            return self.handles[pid][0]
        def cpu_seconds(self) -> float:
            return 50.
    job = Job()
    tree = OwnedTree(job,10,100,tmp_path,clock=lambda: 100.)
    class GPU:
        def __init__(self) -> None:
            self.slots: dict[int,tuple[int,Path]] = {}
            self.scientific_pids: set[int] = set()
            self.seen = {30:(1,100.,'fake')}
        def register(self, pid: int, created: int, path: Path, **kwargs: Any) -> None:
            self.slots[pid] = created,path
        def unregister(self, pid: int) -> None:
            del self.slots[pid]
            self.scientific_pids.discard(pid)
    gpu = GPU()
    tree.gpu = gpu
    tree.register(10,100,'parent')
    tree.register(20,200,'main')
    tree.register(30,300,'provider')
    assert gpu.scientific_pids == {20,30}
    assert {r['pid'] for r in tree.sample()} == {10,20,30}
    assert tree.cpu_seconds() == 60.
    job.rows.pop()
    assert {r['pid'] for r in tree.sample()} == {10,20}
    assert gpu.scientific_pids == {20} and 30 not in gpu.slots
    job.api.created[20] = 201
    with pytest.raises(Exception):
        tree.creation_time(20)


def test_child_begin_closing_requests_parent_before_owner_ack() -> None:
    from cipher_lab.speck_study.native_control import ChildResources
    calls: list[str] = []
    class Channel:
        requested_value = False
        def requested(self) -> bool:
            return self.requested_value
        def acknowledge(self, **kwargs: Any) -> None:
            assert calls == ['closing','saved']
            calls.append('ack')
    channel = Channel()
    def rpc(kind: str, **payload: Any) -> dict[str,Any]:
        calls.append(kind)
        channel.requested_value = True
        return {}
    resources = ChildResources(rpc,channel,lambda: None)
    resources.begin_closing()
    calls.append('saved')
    resources.acknowledge_closing()
    assert calls == ['closing','saved','ack']


def test_parent_closing_does_not_poison_control_or_allow_science() -> None:
    from types import SimpleNamespace

    from cipher_lab.speck_study.native_parent import ParentControl
    class Resources:
        closing = False
        last_wall = 100.
        science_deadline = 28200.
        latest: dict[str,Any] = {}
        def clock(self) -> float:
            return 100.
        def begin_closing(self) -> None:
            self.closing = True
        def check(self, *, science: bool = True) -> None:
            assert not science
    class Channel:
        def requested(self) -> bool:
            return True
        def request(self) -> None:
            pass
    control = ParentControl(cast(Any,SimpleNamespace()),cast(Any,Resources()),Channel())
    value = control.dispatch('check',science=True)
    assert value['closing'] is True
    assert control.dispatch('closing') == {}
    assert control.dispatch('check',science=False)['wall'] == 100.


@pytest.mark.parametrize('fault',['closed','sequence','changed','dispatch'])
def test_control_server_fault_is_not_success(tmp_path: Path, fault: str) -> None:
    from cipher_lab.speck_study.native_control import ControlServer, write_message
    def dispatch(kind: str, **payload: Any) -> dict[str,Any]:
        if fault == 'dispatch':
            raise OSError('fake closed proof channel')
        return {}
    server = ControlServer(tmp_path,dispatch)
    if fault == 'closed':
        (tmp_path/'request.json').write_bytes(b'')
    else:
        write_message(tmp_path/'request.json',dict(seq=2 if fault == 'sequence' else 1,
                                                 kind='check',payload={}))
    if fault == 'changed':
        server.poll()
        write_message(tmp_path/'request.json',dict(seq=1,kind='changed',payload={}))
    with pytest.raises(Exception):
        server.poll()


def test_strict_parent_terminal_requires_source_policy_and_registered_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from uuid import UUID

    import cipher_lab.artifacts as artifacts
    from cipher_lab.speck_study.codec import StudyPredictor, StudyTerm, encode_predictor
    from cipher_lab.speck_study.native_parent import verify_terminal
    from cipher_lab.speck_study.storage import source_digest
    from cipher_lab.speck_study_config import SpeckStudyRecipe
    monkeypatch.setattr(artifacts,'uuid4',lambda: UUID(int=1))
    recipe = SpeckStudyRecipe.synthetic_recipe(r_coef_size=8)
    policy = SimpleNamespace(to_dict=lambda: {'test':'fake-only'})
    owner = artifacts.RunStore.claim(tmp_path,'terminal',output_bytes=10_000_000,needs_models=True)
    source = owner.source_snapshot()
    owner.write_snapshot({'recipe':recipe.to_dict(),'policy':policy.to_dict(),
                          'summary':{'backend_identity':'speck-study'}})
    model = StudyPredictor('R','teacher-reproduction',source_digest(source),recipe,
                           (StudyTerm(1<<63,8),),8)
    owner.write_model('A-best',encode_predictor(model))
    owner.finalize({'status':'SUCCEEDED'})
    assert verify_terminal(tmp_path,'terminal',recipe,policy,source)
    assert not verify_terminal(tmp_path,'terminal',recipe,policy,{'cipher_lab/x.py':'0'*64})
    assert not verify_terminal(tmp_path,'terminal',recipe,SimpleNamespace(to_dict=lambda:{}),source)
    path = tmp_path/'models/terminal/A-best.pt'
    path.write_bytes(b'bad codec')
    assert not verify_terminal(tmp_path,'terminal',recipe,policy,source)


@pytest.mark.parametrize('fault',['none','constructorblocked','unrequestedACK','missingACK',
                                  'termjobthrow','reapfail','lateclock','wronggpu',
                                  'missingpublisher','stale','partialsave'])
def test_actual_parent_launch_fake_job_cleanup_fault_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    import os
    import threading
    from types import SimpleNamespace

    import cipher_lab.speck_study.native_parent as parent
    from cipher_lab.platform_resources.gpu import GpuBinding
    from cipher_lab.platform_resources.windows import WindowsJob
    from cipher_lab.speck_study.gpu_telemetry import TelemetryPublisher
    from cipher_lab.speck_study.native_session import OwnedTree
    events: list[str] = []
    now = [100.]
    active: list[int] = []
    channel = SimpleNamespace(requested_value=False,ack_value=False)
    def requested() -> bool:
        return bool(channel.requested_value)
    def request() -> None:
        channel.requested_value = True
    def acknowledged() -> bool:
        return bool(channel.ack_value)
    channel.requested,channel.request,channel.acknowledged = requested,request,acknowledged
    class Pair:
        def __init__(self) -> None:
            self.channel = channel
            self.handle_list = [11,12]
        def spawn_options(self) -> dict[str,Any]:
            return {'close_fds':True}
        def close(self) -> None:
            events.append('events-closed')
    class API:
        def create_job(self) -> int:
            return 99
        def open(self, pid: int) -> int:
            return pid
        def times(self, handle: int) -> tuple[int,float]:
            return handle*10,1.
        def belongs(self, job: int, handle: int) -> bool:
            return True
        def rss(self, handle: int) -> int:
            return 100
        def cpu(self, job: int) -> float:
            return 1.
        def pids(self, job: int) -> list[int]:
            return list(active)
        def assign(self, job: int, handle: int) -> None:
            active.append(2)
            events.append('assign')
        def resume(self, handle: int) -> None:
            events.append('resume')
            # Before resume, the independent PID/creation slot already exists.
            assert 2 in tree_holder[0].gpu.slots
            publisher = TelemetryPublisher(control,control/'main-telemetry.json',pid=2,
                created=20,sample=rows,clock=lambda: now[0])
            if fault != 'missingpublisher':
                publisher.publish_once()
        def exited(self, handle: int) -> bool:
            return handle not in active
        def close_handle(self, handle: int) -> None:
            events.append('handle-closed')
            if handle == 99:
                active.clear()
        def kill_job(self, job: int) -> None:
            events.append('termjob')
            active.clear()
            if fault == 'termjobthrow':
                raise OSError('fake TerminateJob failure')
    control = tmp_path/'supervision-fake'
    def rows() -> list[dict[str,Any]]:
        return [dict(luid='0100000000000000',description='fake',
                     nodes=[dict(node=0,current_usage=0)])]
    class Process:
        pid = 2
        _handle = 2
        returncode: int | None = None
        stdin = stdout = stderr = None
        polls = 0
        def poll(self) -> int | None:
            self.polls += 1
            if fault == 'constructorblocked' and self.polls == 1:
                # No child cooperative call; parent deadline independently kills.
                now[0] = 28900.
                return None
            if fault == 'wronggpu' and self.polls == 1:
                gpu = tree_holder[0].gpu
                with pytest.raises(Exception):
                    gpu.verify(dict(pid=2,gpu_uuid='GPU-zero',pci_bus_id='0000:01:00.0',
                                    logical_device=0,evidence='cuda-current-context',
                                    luid='0100000000000000',device_node_mask=1),
                               2,'GPU-one','0000:02:00.0')
            if fault == 'missingpublisher' and self.polls == 1:
                now[0] = 106.
                return None
            if fault == 'stale' and self.polls == 1:
                now[0] = 102.
                return None
            if fault != 'wronggpu':
                tree_holder[0].gpu.processes(GpuBinding(1,'GPU-one','0000:02:00.0'))
            channel.requested_value = fault != 'unrequestedACK'
            channel.ack_value = fault != 'missingACK'
            active.clear()
            self.returncode = 0
            return 0
        def wait(self, timeout: float) -> int:
            events.append('wait')
            if fault == 'reapfail':
                raise OSError('fake wait failure')
            self.returncode = 0
            return 0
        def kill(self) -> None:
            active.clear()
            self.returncode = 1
    process = Process()
    monkeypatch.setattr('subprocess.Popen',lambda *args,**kwargs: process)
    monkeypatch.setattr(parent,'os',SimpleNamespace(name='nt',environ=os.environ))
    monkeypatch.setattr(parent,'EventPair',Pair)
    monkeypatch.setattr(parent,'WindowsJob',lambda: WindowsJob(API()))
    monkeypatch.setattr(parent,'current_process_identity',lambda: (1,10))
    monkeypatch.setattr(parent,'windows_monotonic',lambda: now[0])
    monkeypatch.setattr(parent,'DxgiUsage',lambda: SimpleNamespace(sample=rows))
    monkeypatch.setattr('cipher_lab.speck_study.gpu_telemetry.DxgiUsage',
                        lambda: SimpleNamespace(sample=rows))
    monkeypatch.setattr(GpuBinding,'from_snapshot',lambda index: GpuBinding(
        1,'GPU-one','0000:02:00.0'))
    tree_holder: list[Any] = []
    def owned(*args: Any) -> Any:
        tree = OwnedTree(*args,clock=lambda: now[0])
        tree_holder.append(tree)
        return tree
    monkeypatch.setattr(parent,'OwnedTree',owned)
    def publisher(*args: Any, **kwargs: Any) -> Any:
        return TelemetryPublisher(*args,**kwargs,clock=lambda: now[0])
    monkeypatch.setattr(parent,'TelemetryPublisher',publisher)
    manifest = tmp_path/'package.json'
    manifest.write_text('{}')
    monkeypatch.setattr(parent,'frozen_launch_files',lambda *args: dict(
        wrapper=tmp_path/'source.py',main=manifest,provider=manifest,package=manifest))
    monkeypatch.setattr(parent,'verify_terminal',lambda *args: fault != 'partialsave')
    # Force final ordered clock after all cleanup to cross close deadline.
    if fault == 'lateclock':
        class LatePair(Pair):
            def close(self) -> None:
                super().close()
                now[0] = 28900.
        monkeypatch.setattr(parent,'EventPair',LatePair)
    args = SimpleNamespace(root=tmp_path,id='fake',native_argv=['run'])
    settings = SimpleNamespace(gpu_uuid='GPU-one',gpu_pci='0000:02:00.0')
    code = parent.launch_native(args,SimpleNamespace(),SimpleNamespace(),settings)
    from cipher_lab.speck_study.native_control import read_message
    record = read_message(control/'parent-final.json')
    assert code == (0 if fault == 'none' else 7)
    assert record['status'] == ('COMPLETE' if fault == 'none' else 'UNKNOWN')
    assert events.index('assign') < events.index('resume')
    assert 'wait' in events and 'termjob' in events and 'events-closed' in events
    assert not active
    assert not any(t.name in ('dxgi-telemetry','study-control','study-resource-supervisor')
                   for t in threading.enumerate())


def test_main_context_alone_not_r_admission_and_reap_required_for_b(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from cipher_lab.speck_study.native_session import LockedGpu
    tree = SimpleNamespace(roles={1:'parent',2:'main'},provider_reaped=False)
    gpu = SimpleNamespace(ready=True)
    adapter = LockedGpu(cast(Any,tree),gpu)
    assert adapter.ready is False
    tree.roles[3] = 'provider'
    assert adapter.ready is True
    del tree.roles[3]
    assert adapter.ready is False
    tree.provider_reaped = True
    assert adapter.ready is True


def test_reaped_pid_without_observed_publisher_cannot_disappear(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from cipher_lab.speck_study.native_session import OwnedTree
    api = SimpleNamespace(open=lambda pid:pid,times=lambda h:(h*10,0.),
                          exited=lambda h:True,rss=lambda h:0)
    job = SimpleNamespace(api=api,handles={2:(20,2)},sample=lambda:[],cpu_seconds=lambda:0.)
    tree = OwnedTree(job,1,10,tmp_path,clock=lambda:100.)
    tree.roles = {1:'parent',2:'main'}
    tree.gpu = SimpleNamespace(seen={},unregister=lambda p:None)
    with pytest.raises(Exception):
        tree.sample()


def test_event_creation_inherit_failure_still_closes_new_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab.speck_study.closing import WindowsEvents
    closed = []
    class Library:
        def CreateEventW(self, *args: Any) -> int:
            assert args == (None,True,False,None)
            return 44
        def CloseHandle(self, handle: int) -> bool:
            closed.append(handle)
            return True
    def fail(*args: Any) -> None:
        raise OSError('fake handle flag failure')
    monkeypatch.setattr('os.set_handle_inheritable',fail,raising=False)
    with pytest.raises(OSError):
        WindowsEvents(Library()).create()
    assert closed == [44]


@pytest.mark.parametrize('exit_at',['before-gpu','inside-collect'])
@pytest.mark.parametrize('proof',['fresh','unseen','reuse'])
def test_fix1_exit_snapshot_not_missing_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           exit_at: str, proof: str) -> None:
    from types import SimpleNamespace

    from cipher_lab.platform_resources.gpu import GpuBinding
    from cipher_lab.speck_study.gpu_telemetry import TelemetryGpu, TelemetryPublisher
    from cipher_lab.speck_study.native_parent import ParentControl
    from cipher_lab.speck_study.native_session import LockedGpu, OwnedTree
    from cipher_lab.speck_study.resources import StudyResources
    binding = GpuBinding(1,'GPU-one','0000:02:00.0')
    active = [2,3]
    created = {1:10,2:20,3:30}
    def rows() -> list[dict[str,Any]]:
        return [dict(pid=p,created=created[p],cpu=1.,rss=10) for p in active]
    api = SimpleNamespace(open=lambda p:p,times=lambda h:(created[h],1.),rss=lambda h:10,
                          exited=lambda h:h not in active)
    job = SimpleNamespace(api=api,handles={2:(20,2),3:(30,3)},sample=rows,
                          cpu_seconds=lambda:2.)
    tree = OwnedTree(job,1,10,tmp_path,clock=lambda:100.)
    gpu = TelemetryGpu(tmp_path,{},members=tree.sample,creation_time=tree.creation_time,
                       science_deadline=28300.,inventory={'0100000000000000':1},
                       clock=lambda:100.)
    tree.gpu = gpu
    gpu.lock = tree.lock
    for pid,role in ((1,'parent'),(2,'main'),(3,'provider')):
        tree.register(pid,pid*10,role)
        TelemetryPublisher(tmp_path,tree.path(role),pid=pid,created=pid*10,
            sample=lambda:[dict(luid='0100000000000000',description='fake',
                                nodes=[dict(node=0,current_usage=7)])],
            clock=lambda:100.).publish_once()
    gpu.bound = dict(binding=binding,luid='0100000000000000',device_node_mask=1)
    gpu.attested = {2,3}
    gpu.processes(binding)
    if proof == 'unseen':
        gpu.seen.pop(3)
    def exit_provider() -> None:
        active.remove(3)
        if proof == 'reuse':
            created[3] = 31
    if exit_at == 'before-gpu':
        real_sample = tree.sample
        once = [True]
        def sample_then_exit() -> list[dict[str,Any]]:
            result = real_sample()
            if once[0]:
                once[0] = False
                exit_provider()  # exact gap: resource tree rows already returned
            return result
        monkeypatch.setattr(tree,'sample',sample_then_exit)
    else:
        import cipher_lab.speck_study.gpu_telemetry as telemetry
        original_read = telemetry.read_sample
        fired = [False]
        def read_then_exit(root: Path, path: Path) -> dict[str,Any]:
            frame = original_read(root,path)
            if path == tree.path('provider') and not fired[0]:
                fired[0] = True
                exit_provider()  # between _collect's two member snapshots
            return frame
        monkeypatch.setattr(telemetry,'read_sample',read_then_exit)
        # Unseen prior sample must NOT be legitimized by this same collection
        # when its exit identity was never established before the round.
    resources = StudyResources(binding,tmp_path,100.,28300.,28900.,tree,
                               gpu=LockedGpu(tree,gpu),clock=lambda:100.)
    if proof != 'fresh':
        with pytest.raises(Exception):
            resources.check(science=False)
        return
    resources.check(science=False)
    assert resources.failure is None and not gpu.failure
    assert resources.latest['gpu_bytes'] == 14  # two measured live PIDs, not missing0
    control = ParentControl(tree,resources,SimpleNamespace(requested=lambda:False))
    control.provider_pid = 3
    assert control.dispatch('provider-reaped') == {'reaped':True}
    resources.check()  # B can proceed ONLY after independent parent reaped gate


def test_fix1_fresh_but_late_first_frame_sticky(tmp_path: Path) -> None:
    from cipher_lab.platform_resources.gpu import GpuBinding, GpuError
    from cipher_lab.speck_study.gpu_telemetry import TelemetryGpu, TelemetryPublisher
    now = [100.]
    gpu = TelemetryGpu(tmp_path,{},members=lambda:[dict(pid=1,created=10)],
                       creation_time=lambda p:10,science_deadline=28300.,
                       inventory={'0100000000000000':1},clock=lambda:now[0])
    gpu.register(1,10,tmp_path/'first.json',first_frame_deadline=105.)
    now[0] = 105.1
    TelemetryPublisher(tmp_path,tmp_path/'first.json',pid=1,created=10,
        sample=lambda:[dict(luid='0100000000000000',description='fake',
                            nodes=[dict(node=0,current_usage=0)])],
        clock=lambda:now[0]).publish_once()
    now[0] = 105.2
    with pytest.raises(GpuError):
        gpu.processes(GpuBinding(1,'GPU-one','0000:02:00.0'))
    assert gpu.failure


@pytest.mark.parametrize('fallback',[False,True])
def test_fix1_cleanup_waits_all_retained_after_main_first_wait_failure(fallback: bool) -> None:
    from cipher_lab.platform_resources.windows import WindowsJob
    from cipher_lab.speck_study.native_parent import reap_owned
    now = [0.]
    events: list[str] = []
    live = {2,3}
    class API:
        def create_job(self) -> int:
            return 99
        def times(self, h: int) -> tuple[int,float]:
            return h*10,0.
        def kill_job(self, h: int) -> None:
            events.append('terminate-asynchronous')
        def pids(self, h: int) -> list[int]:
            assert h == 99 and 'job-closed' not in events
            return sorted(live)
        def exited(self, h: int) -> bool:
            # TERM is async: main first, provider later, no fake immediate reap.
            if now[0] >= (0.1 if h == 2 else 0.3):
                live.discard(h)
            events.append('poll-'+str(h))
            return h not in live
        def close_handle(self, h: int) -> None:
            if h == 99:
                events.append('job-closed')
            else:
                assert h not in live, 'process identity released BEFORE confirmed exit'
                events.append('released-'+str(h))
    class Process:
        waits = 0
        def wait(self, timeout: float) -> int:
            self.waits += 1
            events.append('main-wait-'+str(self.waits))
            if self.waits == 1 and fallback:
                raise TimeoutError('fake first main wait')
            if not fallback:
                live.discard(2)
            if 2 in live:
                raise TimeoutError('fake async main')
            return 0
    job = WindowsJob(API())
    job.handles = {2:(20,2),3:(30,3)}
    job.close()
    if fallback:
        job.close_job_handle()  # kill-on-close but identity handles stay retained
        assert set(job.handles) == {2,3}
    def sleep(seconds: float) -> None:
        now[0] += seconds
    evidence, errors = reap_owned(job,Process(),timeout=1.,clock=lambda:now[0],sleep=sleep)
    assert not live and not errors
    assert evidence['exited_pids'] == [2,3]
    assert evidence['job_empty'] is (None if fallback else True)
    if fallback:
        assert 'main-wait-2' in events
    assert set(job.handles) == {2,3}  # proof retained until final disposal
    job.dispose()
    assert 'released-2' in events and 'released-3' in events


def test_fix1_reap_timeout_attempts_provider_even_if_main_clock_or_wait_fails() -> None:
    from cipher_lab.platform_resources.windows import WindowsJob
    from cipher_lab.speck_study.native_parent import reap_owned
    events: list[int] = []
    now = [0.]
    class API:
        def create_job(self) -> int:
            return 99
        def times(self, handle: int) -> tuple[int,float]:
            return handle*10,0.
        def exited(self, handle: int) -> bool:
            events.append(handle)
            return False
        def pids(self, handle: int) -> list[int]:
            return [2,3]
    job = WindowsJob(API())
    job.handles = {2:(20,2),3:(30,3)}
    def sleep(seconds: float) -> None:
        now[0] += seconds
    evidence, errors = reap_owned(job,None,timeout=.1,clock=lambda:now[0],sleep=sleep)
    assert errors and set(events) == {2,3}
    assert evidence['exited_pids'] == [] and evidence['job_empty'] is False
    assert set(job.handles) == {2,3}


def test_fix1_reap_clock_failure_still_polls_every_identity() -> None:
    from types import SimpleNamespace

    from cipher_lab.platform_resources.windows import WindowsJob
    from cipher_lab.speck_study.native_parent import reap_owned
    polled = []
    def exited(handle: int) -> bool:
        polled.append(handle)
        return True
    api = SimpleNamespace(create_job=lambda:99,times=lambda h:(h*10,0.),
                          exited=exited,pids=lambda h:[])
    job = WindowsJob(api)
    job.handles = {2:(20,2),3:(30,3)}
    def fail() -> float:
        raise OSError('fake local clock failure')
    evidence, errors = reap_owned(job,None,clock=fail)
    assert errors and polled == [2,3] and evidence['exited_pids'] == [2,3]


def test_fix1_dispose_retains_unreaped_identity_for_final_backstop() -> None:
    from types import SimpleNamespace

    from cipher_lab.platform_resources.windows import ProcessError, WindowsJob
    closed = []
    api = SimpleNamespace(create_job=lambda:99,times=lambda h:(h*10,0.),
                          exited=lambda h:h==2,close_handle=lambda h:closed.append(h))
    job = WindowsJob(api)
    job.handles = {2:(20,2),3:(30,3)}
    with pytest.raises(ProcessError):
        job.dispose(require_exit=True)
    assert closed == [99,2] and set(job.handles) == {3}


def test_fix2_main_reaped_provider_only_exits_after_jobclose_bounded_wait() -> None:
    from cipher_lab.platform_resources.windows import WindowsJob
    from cipher_lab.speck_study.native_parent import reap_owned
    now = [0.]
    closed_at: list[float] = []
    events: list[tuple[str,int,float]] = []
    provider_live = [True]
    class API:
        def create_job(self) -> int:
            return 99
        def times(self, h: int) -> tuple[int,float]:
            return h*10,0.
        def pids(self, h: int) -> list[int]:
            assert not closed_at
            return [3] if provider_live[0] else []
        def exited(self, h: int) -> bool:
            events.append(('poll',h,now[0]))
            if h == 3 and closed_at and now[0] >= closed_at[0]+.05:
                provider_live[0] = False
            return h == 2 or not provider_live[0]
        def close_handle(self, h: int) -> None:
            events.append(('close',h,now[0]))
            if h == 99:
                closed_at.append(now[0])
            else:
                assert h == 2 or not provider_live[0], 'released provider BEFORE exit'
    class Process:
        def wait(self, timeout: float) -> int:
            events.append(('popen-wait',2,now[0]))
            return 0  # main first wait succeeds; does NOT prove provider exit
    job = WindowsJob(API())
    job.handles = {2:(20,2),3:(30,3)}
    def sleep(seconds: float) -> None:
        now[0] += seconds
    evidence, errors = reap_owned(job,Process(),timeout=.4,clock=lambda:now[0],sleep=sleep)
    assert closed_at and closed_at[0] < .4
    assert not errors and not provider_live[0]
    assert evidence == {'exited_pids':[2,3],'job_empty':None}
    assert any(kind == 'poll' and pid == 3 and tick >= closed_at[0]+.05
               for kind,pid,tick in events)
    assert any(kind == 'popen-wait' and tick > closed_at[0] for kind,pid,tick in events)
    assert now[0] <= .4 and set(job.handles) == {2,3}
    job.dispose(require_exit=True)
    assert any(kind == 'close' and pid == 3 and tick >= closed_at[0]+.05
               for kind,pid,tick in events)
