"""Concrete local Windows parent. Owns resources/Job, never child RunStore.

The only launch input is the already validated CLI command and explicit frozen
source files. No SSH, Git, task package, retries or scientific implementation.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cipher_lab.platform_resources.dxgi import DxgiUsage
from cipher_lab.platform_resources.gpu import GpuBinding
from cipher_lab.platform_resources.windows import WindowsJob
from cipher_lab.speck_study.closing import EventPair
from cipher_lab.speck_study.gpu_telemetry import (
    TelemetryGpu,
    TelemetryPublisher,
    current_process_identity,
    windows_monotonic,
)
from cipher_lab.speck_study.native_control import ControlServer, read_message, write_message
from cipher_lab.speck_study.native_session import LockedGpu, OwnedTree
from cipher_lab.speck_study.resources import (
    ResourceStop,
    StudyResources,
    Supervisor,
    directory_bytes,
)


class ParentControl:
    def __init__(self, tree: OwnedTree, resources: StudyResources, channel: Any) -> None:
        self.tree, self.resources, self.channel = tree,resources,channel
        self.main_pid: int | None = None
        self.provider_pid: int | None = None

    def dispatch(self, kind: str, **payload: Any) -> dict[str, Any]:
        if kind == 'check' and set(payload) == {'science'} and type(payload['science']) is bool:
            # Closing request is an admission stop, not a poisoned transport.
            science = payload['science']
            if science and (self.resources.closing or self.channel.requested()
                            or self.resources.clock() >= self.resources.science_deadline):
                self.resources.begin_closing()
                self.channel.request()
                self.resources.check(science=False)
                return dict(wall=self.resources.last_wall,
                            science_deadline=self.resources.science_deadline,
                            latest=self.resources.latest,closing=True)
            try:
                self.resources.check(science=science)
            except ResourceStop as error:
                if error.reason != 'SCIENCE_DEADLINE':
                    raise
                self.resources.begin_closing()
                self.channel.request()
                self.resources.check(science=False)
                return dict(wall=self.resources.last_wall,
                            science_deadline=self.resources.science_deadline,
                            latest=self.resources.latest,closing=True)
            return dict(wall=self.resources.last_wall,
                        science_deadline=self.resources.science_deadline,
                        latest=self.resources.latest)
        if kind == 'closing' and not payload:
            self.resources.begin_closing()
            self.channel.request()
            return {}
        if kind == 'expect-provider' and not payload:
            with self.tree.lock:
                if self.provider_pid is not None or self.tree.expected_provider:
                    raise ResourceStop('PROCESS_UNVERIFIED')
                self.tree.expected_provider = True
                self.tree.provider_deadline = self.resources.clock()+5.
            return {}
        if kind == 'adopt-provider' and set(payload) == {'pid'}:
            pid = payload['pid']
            with self.tree.lock:
                if type(pid) is not int or pid <= 0 or self.provider_pid is not None:
                    raise ResourceStop('PROCESS_UNVERIFIED')
                # sample may already have registered this suspended process;
                # identity always comes from independently opened parent handle.
                def registered(p: int, created: int) -> None:
                    if p not in self.tree.roles:
                        if not self.tree.expected_provider:
                            raise ResourceStop('PROCESS_UNVERIFIED')
                        self.tree.register(p,created,'provider')
                    if self.tree.roles[p] != 'provider':
                        raise ResourceStop('PROCESS_UNVERIFIED')
                    self.tree.expected_provider = False
                self.tree.job.adopt_suspended(pid,before_resume=registered)
                self.provider_pid = pid
            return {}
        if kind == 'verify' and set(payload) == {'evidence','role'}:
            role,evidence = payload['role'],payload['evidence']
            expected = self.main_pid if role == 'main' else self.provider_pid
            if (role not in ('main','provider') or expected is None
                    or type(evidence) is not dict or evidence.get('pid') != expected):
                raise ResourceStop('GPU_UNVERIFIED')
            binding = self.resources.binding
            if self.resources.verify_teacher(evidence,expected,binding.gpu_uuid,
                                             binding.pci_bus_id) is not True:
                raise ResourceStop('GPU_UNVERIFIED')
            return {'verified':True}
        if kind == 'provider-reaped' and not payload:
            with self.tree.lock:
                if self.provider_pid is None:
                    raise ResourceStop('PROCESS_UNVERIFIED')
                rows = self.tree.sample()
                if self.provider_pid in {r['pid'] for r in rows}:
                    raise ResourceStop('CLEANUP_UNKNOWN')
                if not self.tree.job.api.exited(self.tree.job.handles[self.provider_pid][1]):
                    raise ResourceStop('CLEANUP_UNKNOWN')
                self.tree.provider_reaped = True
            return {'reaped':True}
        raise ResourceStop('PROCESS_UNVERIFIED')


def verify_terminal(root: Path, run_id: str, recipe: Any, policy: Any,
                    source: dict[str,str]) -> bool:
    """Independent strict saved read. No owner acquisition, recomputation or forward."""
    from cipher_lab.artifacts import read_binary, read_saved
    from cipher_lab.speck_study.codec import decode_predictor
    from cipher_lab.speck_study.storage import source_digest
    try:
        saved = read_saved(root,run_id)
        if (saved['recipe'] != recipe.to_dict() or saved['policy'] != policy.to_dict()
                or saved['source'] != source or saved['run_id'] != run_id
                or saved['status'] != 'SUCCEEDED'):
            return False
        summary = saved['summary']
        if not isinstance(summary,dict) or summary.get('backend_identity') != 'speck-study':
            return False
        # Registered models, if present, require strict external identity as
        # well as read_saved's registered binary hashes. No in-memory fallback.
        artifacts = saved['artifacts']
        if not isinstance(artifacts,dict):
            return False
        for ref, entry in artifacts.items():
            if entry.get('kind') == 'model':
                if ref not in ('model/A-best.pt','model/B-best.pt'):
                    return False
                kind = 'R' if Path(ref).name == 'A-best.pt' else 'B'
                raw = read_binary(root,run_id,ref,entry['sha256'],256*1024)
                decode_predictor(raw,expected_recipe=recipe,expected_identity=dict(
                    kind=kind,task='teacher-reproduction' if kind == 'R'
                    else 'absolute-bit-prediction',source_sha256=source_digest(source),
                    method=recipe.method,synthetic=recipe.synthetic))
        return True
    except Exception:
        return False


def frozen_launch_files(settings: Any, source_root: Path) -> dict[str,Path]:
    """Main/provider/bootstrap/package manifests are separately reviewed inputs."""
    main = settings.source_manifest
    files = dict(main=main,provider=Path(str(main)+'.provider.json'),
                 bootstrap=Path(str(main)+'.bootstrap.json'),
                 package=Path(str(main)+'.package.json'))
    from cipher_lab.artifacts import source_identity
    loader = globals().get('__loader__')
    if type(loader).__name__ != 'SourceOnly':
        raise ValueError('native parent requires frozen source-only entry')
    tools = source_root.parent/'tools'/'speck-study'
    main_namespace: dict[str,Any] = {'__name__':'_main_freeze',
                                    '__file__':str(tools/'source.py')}
    # Wrapper identity is checked below before executing these bytes.
    if read_message(files['package']) != source_identity():
        raise ValueError('package snapshot mismatch')
    bootstrap = read_message(files['bootstrap'])
    if set(bootstrap) != {'source.py','freeze.py','supervise.py'}:
        raise ValueError('bootstrap closure mismatch')
    for name,digest in bootstrap.items():
        path = tools/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('bootstrap source mismatch')
    exec(compile((tools/'source.py').read_bytes(),str(tools/'source.py'),'exec'),main_namespace)
    main_namespace['checked'](source_root,read_message(files['main']))
    # Source-only wrapper has already checked main. Provider exact source list
    # is checked without importing or running the provider/TF.
    namespace: dict[str,Any] = {'__name__':'_checked_provider',
                               '__file__':str(tools/'freeze.py')}
    exec(compile((tools/'freeze.py').read_bytes(),str(tools/'freeze.py'),'exec'),namespace)
    namespace['checked_sources'](source_root,read_message(files['provider']))
    files['wrapper'] = tools/'source.py'
    return files


def reap_owned(job: WindowsJob, process: Any, *, timeout: float = 2.,
               clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep
               ) -> tuple[dict[str, Any], list[BaseException]]:
    """Bounded LOCAL elapsed wait, independent of frozen QPC science deadlines.

    Every retained handle is polled even after main wait failure. Kill-on-close
    fallback keeps those handles and triggers a second main wait. Closing the
    Job prevents an independent Jobempty query, so evidence stays explicitly
    unknown in that case even when every known member exit is confirmed.
    Reserve the latter half of the SAME timeout for post-Job-close polling if
    TERM leaves any member alive, even when main's first Popen wait succeeded.
    """
    errors: list[BaseException] = []
    evidence: dict[str, Any] = dict(exited_pids=[], job_empty=None)
    exited: set[int] = set()
    main_reaped = process is None
    end = None
    try:
        start = clock()
        if not isinstance(start, (int, float)) or not math.isfinite(start):
            raise ResourceStop('CLOCK_UNAVAILABLE')
        end = start+timeout
    except BaseException as error:
        errors.append(error)  # still poll EVERY owned handle at least once
    if process is not None:
        try:
            process.wait(timeout=max(.01,min(.25,timeout/4)))
            main_reaped = True
        except BaseException:
            try:
                job.close_job_handle()  # no process handle release before waits
            except BaseException as error:
                errors.append(error)
    # Bound both elapsed time and attempts if an injected/broken clock stalls.
    for _ in range(max(1,int(timeout/.01)+1)):
        with job.lock:
            handles = list(job.handles.items())
        for pid, (created, handle) in handles:
            if pid in exited:
                continue
            try:
                if job.api.times(handle)[0] != created:
                    raise ResourceStop('PROCESS_UNVERIFIED')
                if job.api.exited(handle):
                    exited.add(pid)
            except BaseException as error:
                errors.append(error)
        if process is not None and not main_reaped:
            try:
                process.wait(timeout=.01)
                main_reaped = True
            except BaseException:
                pass  # handle polling continues for EVERY owned member
        if not job.closed:
            try:
                # Catch descendants observed during asynchronous termination;
                # never infer Jobempty just because main Popen has exited.
                job.retain_members()
                evidence['job_empty'] = job.empty()
            except BaseException as error:
                errors.append(error)
        evidence['exited_pids'] = sorted(exited)
        if (set(job.handles) <= exited and main_reaped
                and (job.closed or evidence['job_empty'] is True)):
            break
        if end is None:
            break
        try:
            now = clock()
            if not isinstance(now, (int, float)) or not math.isfinite(now) or now < start:
                raise ResourceStop('CLOCK_UNAVAILABLE')
            if not job.closed and now >= start+timeout/2:
                # Main exit is not tree exit. Reserve remaining cleanup budget
                # for kill-on-close AND asynchronous descendant exit proof.
                job.close_job_handle()
                evidence['job_empty'] = None  # closed Job cannot be queried
                main_reaped = process is None  # explicitly re-wait Popen too
            if now >= end:
                break
            sleep(min(.01,max(0.,end-now)))
        except BaseException as error:
            errors.append(error)
            break
    if (not set(job.handles) <= exited or not main_reaped
            or (not job.closed and evidence['job_empty'] is not True)):
        errors.append(ResourceStop('CLEANUP_UNKNOWN'))
    return evidence, errors


def launch_native(args: Any, recipe: Any, policy: Any, settings: Any) -> int:
    """Parent has no RunStore; child claims exactly once after frozen bootstrap."""
    if os.name != 'nt':
        raise RuntimeError('ENVIRONMENT_BLOCKED')
    source_root = Path(__file__).resolve().parents[2]
    if args.native_argv[:1] != ['run']:
        raise ValueError('native run argv required')
    files = frozen_launch_files(settings,source_root)
    binding = GpuBinding.from_snapshot(1)  # only pre-origin GPU gate, no CUDA
    if (binding.gpu_uuid,binding.pci_bus_id) != (settings.gpu_uuid,settings.gpu_pci):
        raise ResourceStop('GPU_UNVERIFIED')
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.id) is None:
        raise ValueError('invalid native run id')
    root = args.root.resolve(strict=True)
    control = root/('supervision-'+args.id)
    control.mkdir(exist_ok=False)
    pair = job = tree = own = resources = server = supervisor = None
    process = None
    origin = None
    complete = False
    errors: list[BaseException] = []
    try:
        pair = EventPair()
        job = WindowsJob()
        pid,created = current_process_identity()
        # One formal origin BEFORE child imports Torch or provider imports TF.
        origin = windows_monotonic()
        tree = OwnedTree(job,pid,created,control)
        own = TelemetryPublisher(control,tree.path('parent'),pid=pid,created=created,
                                 sample=DxgiUsage().sample)
        gpu = TelemetryGpu(control,{},members=tree.sample,creation_time=tree.creation_time,
                           science_deadline=origin+28200,clock=windows_monotonic)
        tree.gpu = gpu
        gpu.lock = tree.lock  # one lifecycle lock; no GPU->tree inverse lock order
        tree.register(pid,created,'parent')
        own.publish_once()  # CPU-only 0 must be actually observed
        own.start()
        events = pair.channel
        owned = tree
        class ParentResources(StudyResources):
            def check(self, *, science: bool = True) -> None:
                if events.acknowledged() and not events.requested():
                    raise ResourceStop('CLEANUP_UNKNOWN')
                with owned.lock:
                    super().check(science=science)
        resources = ParentResources(binding,root,origin,origin+28200,origin+28800,tree,
                                   gpu=LockedGpu(tree,gpu),clock=windows_monotonic)
        control_owner = ParentControl(tree,resources,pair.channel)
        server = ControlServer(control,control_owner.dispatch)
        supervisor = Supervisor(resources,request_closing=pair.channel.request,
                                closing_acknowledged=pair.channel.acknowledged)
        env = os.environ.copy()
        env.update(SPECK_STUDY_NATIVE='1',SPECK_STUDY_CONTROL_ROOT=str(control),
                   SPECK_STUDY_ORIGIN=repr(origin),SPECK_STUDY_SCIENCE_DEADLINE=repr(origin+28200),
                   SPECK_STUDY_CLOSE_DEADLINE=repr(origin+28800),
                   SPECK_STUDY_REQUEST_HANDLE=str(pair.handle_list[0]),
                   SPECK_STUDY_ACK_HANDLE=str(pair.handle_list[1]),
                   SPECK_STUDY_PROVIDER_MANIFEST=str(files['provider']),
                   CUDA_VISIBLE_DEVICES=binding.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PYTHONDONTWRITEBYTECODE='1')
        for key in ('TMP','TEMP','TMPDIR','CUDA_CACHE_PATH','KERAS_HOME','XDG_CACHE_HOME'):
            path = control/('main-'+key.lower())
            path.mkdir()
            env[key] = str(path)
        command = [sys.executable,'-B',str(files['wrapper']),'run',str(files['main']),
                   str(source_root)]
        # CLI called through source.py retains wrapper args in sys.argv; explicitly
        # pass the parsed original CLI argv, never infer command text or shell it.
        command += list(args.native_argv)
        supervisor.start()
        server.start()
        def registered(p: int, c: int) -> None:
            tree.register(p,c,'main')
            control_owner.main_pid = p
        with tree.lock:
            process = job.spawn(command,env=env,before_resume=registered,
                                **pair.spawn_options())
        while process.poll() is None:
            own.check()
            if server.error or supervisor.error:
                raise ResourceStop('CLEANUP_UNKNOWN')
            if windows_monotonic() >= resources.close_deadline:
                raise ResourceStop('CLOSE_DEADLINE')
            time.sleep(.05)
        process.wait(timeout=1.)
        tree.sample()  # independently retire exited slots before exact GPU check
        if pair.channel.acknowledged() and not pair.channel.requested():
            raise ResourceStop('CLEANUP_UNKNOWN')
        # Closing is always explicitly requested before child save, including
        # early scientific gates. Never finish without the requested ACK.
        if not pair.channel.requested() or not pair.channel.acknowledged():
            raise ResourceStop('CLEANUP_UNKNOWN')
        supervisor.closing_requested = True
        supervisor.finish(control/'resource-receipt.json',child_exit=process.returncode,
                          child_terminal=verify_terminal(root,args.id,recipe,policy,
                                                        read_message(files['package'])))
    except BaseException as error:
        errors.append(error)
        if supervisor is not None:
            supervisor.complete = False
    finally:
        if job is not None:
            try:
                job.retain_members()  # capture owned identities before async TERM
            except BaseException as error:
                errors.append(error)
        if supervisor is not None:
            supervisor.stop_event.set()
            errors.extend(supervisor.cleanup_tree())
        elif job is not None:
            try:
                job.close()
            except BaseException as error:
                errors.append(error)
                try:
                    job.dispose(release_processes=False)  # keep identities for final reap
                except BaseException as fallback:
                    errors.append(fallback)
        reap_evidence: dict[str,Any] = {}
        if job is not None:
            reap_evidence, reap_errors = reap_owned(job,process)
            errors.extend(reap_errors)
            if reap_evidence.get('job_empty') is not True:
                errors.append(ResourceStop('CLEANUP_UNKNOWN'))
        actions = []
        if server is not None:
            actions.append(server.stop)
        if supervisor is not None and supervisor.thread:
            sampler = supervisor.thread
            actions.append(lambda: sampler.join(1.))
        if job is not None:
            final_job = job
            actions.append(lambda: final_job.dispose(require_exit=True))
        if own is not None:
            actions.append(own.stop)
        if tree is not None:
            actions.append(tree.close_own_handle)
        if pair is not None:
            actions.append(pair.close)
        for action in actions:
            try:
                action()
            except BaseException as error:
                errors.append(error)
        if supervisor is not None and supervisor.thread and supervisor.thread.is_alive():
            errors.append(ResourceStop('CLEANUP_UNKNOWN'))
        try:
            if resources is not None and windows_monotonic() >= resources.close_deadline:
                errors.append(ResourceStop('CLOSE_DEADLINE'))
        except BaseException as error:
            errors.append(error)
        complete = supervisor is not None and supervisor.complete and not errors
        write_message(control/'parent-final.json',dict(status='COMPLETE' if complete else 'UNKNOWN',
                      child_exit=None if process is None else process.returncode,
                      receipt_evidence={} if supervisor is None else supervisor.receipt_evidence,
                      reap_evidence=reap_evidence,
                      evidence='dxgi-self-per-process+parent-luid',origin=origin,
                      science_deadline=None if origin is None else origin+28200,
                      close_deadline=None if origin is None else origin+28800))
        # Receipt/final-write elapsed time is part of the SAME close window.
        # A late/corrupt write cannot release an in-memory successful exit.
        try:
            if (origin is None or windows_monotonic() >= origin+28800
                    or directory_bytes(root) > 4*2**30):
                complete = False
        except BaseException:
            complete = False
        if not complete:
            write_message(control/'parent-final.json',dict(status='UNKNOWN',
                          child_exit=None if process is None else process.returncode,
                          origin=origin,reap_evidence=reap_evidence,
                          evidence='dxgi-self-per-process+parent-luid'))
    return 0 if complete else 7
