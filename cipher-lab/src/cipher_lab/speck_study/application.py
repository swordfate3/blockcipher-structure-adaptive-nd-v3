"""SPECK application assembly, not scientific algorithms or platform policy.

Native child consumes the parent-owned Job/resource contract and preserves the
original single scientific owner. Unsupported or unverified environments stop;
there is no synthetic production backend.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from cipher_lab.config import Policy
from cipher_lab.speck_study_config import SpeckStudyRecipe
from cipher_lab.speck_study_launch import LaunchSettings


class CipherAdapter:
    """Ephemeral NumPy uint32/uint16 -> caller-owned Torch device adapter.

    Keys remain arguments from StudyData. Neither adapter nor cipher retains a
    key, hash, round schedule or other output bit for the model/teacher.
    """
    def __init__(self, torch: Any, device: str) -> None:
        self.torch, self.device = torch, device

    def __call__(self, points: Any, keys: Any, rounds: int) -> Any:
        import numpy as np

        from cipher_lab.ciphers.speck_tensor import encrypt_tensor

        if (type(points) is not np.ndarray or points.dtype != np.dtype(np.uint32)
                or points.ndim != 1 or not 1 <= len(points) <= 50000
                or type(keys) is not np.ndarray or keys.dtype != np.dtype(np.uint16)
                or keys.shape not in ((4,), (len(points),4))):
            raise ValueError('invalid cipher adapter input')
        p = self.torch.tensor(points.astype(np.int64), dtype=self.torch.int64, device=self.device)
        k = self.torch.tensor(keys.astype(np.int64), dtype=self.torch.int64, device=self.device)
        result = encrypt_tensor(p, k, rounds)
        return result.detach().cpu().numpy().astype(np.uint32)

    def public_r0(self, points: Any, keys: Any, rounds: int, *, implementation: str) -> Any:
        import numpy as np

        if implementation == 'batch':
            return self(points, keys, rounds)
        if implementation != 'reference':
            raise ValueError('unknown R0 implementation')
        from cipher_lab.ciphers.speck import encrypt_scalar

        if keys.shape != (4,):
            raise ValueError('R0 reference requires public single key')
        public_key = (int(keys[0]),int(keys[1]),int(keys[2]),int(keys[3]))
        return np.array([encrypt_scalar(int(p), public_key, rounds) for p in points],
                        dtype=np.uint32)


def formal_r0(data: Any) -> bool:
    """Counted real KAT/equality hook; invoke ONLY after formal origin/admission.

    Four public blocks per implementation (8 total), never private entropy. The
    owner's public_r0 enforces <=256 each/<=512 overall and charges attempts.
    Runtime/weights/UUID verification must precede this hook in the assembly.
    """
    import numpy as np

    key = np.array([0x1918,0x1110,0x0908,0x0100], dtype=np.uint16)
    kat = np.array([0x6574694c], dtype=np.uint32)
    expected = np.array([0xa86842f2], dtype=np.uint32)
    for implementation in ('reference','batch'):
        if not np.array_equal(data.public_r0(kat,key,22,implementation=implementation), expected):
            return False
    public = np.array([0,0x12345678,0xffffffff],dtype=np.uint32)
    reference = data.public_r0(public,key,8,implementation='reference')
    batch = data.public_r0(public,key,8,implementation='batch')
    return bool(np.array_equal(reference,batch))


def assemble(recipe: SpeckStudyRecipe, store: Any, cipher: Any,
             private_random: Callable[[int], bytes], sampler: Any, meter: Any,
             resources: Any, teacher: Any) -> tuple[Any, Callable[[], object]]:
    """One concrete data owner and fixed method; no math override/fake fallback."""
    from cipher_lab.speck_study.data import StudyData
    from cipher_lab.speck_study.method import run_study

    data = StudyData(recipe,cipher,private_random,sampler,meter,store)
    return data, lambda: run_study(recipe,data,teacher,None,resources,store)


def require_native_launch() -> None:
    import os
    import sys
    if (os.name != 'nt' or os.environ.get('SPECK_STUDY_NATIVE') != '1'
            or sys.version_info[:2] != (3,10)):
        raise RuntimeError('ENVIRONMENT_BLOCKED')


def run_application(recipe: SpeckStudyRecipe, policy: Policy, owner: Any,
                    root: Path, run_id: str, settings: LaunchSettings) -> None:
    """Child-only assembly; original owner is the sole scientific writer.

    Parent has already frozen origin, assigned this suspended child to its Job,
    and independently registered its identity before resume. No developer probe.
    """
    import math
    import os
    import subprocess
    import time

    # Unsupported direct CLI remains an owner-saved stop without probing any
    # platform/scientific package. Windows formal path continues below.
    try:
        require_native_launch()
    except RuntimeError:
        owner.finalize({'status':'STOPPED','summary':{
            'method':recipe.method,'backend_identity':'speck-study','synthetic':False,
            'scientific_validation':False,'prediction_positive':False,
            'reproduction_matched':False,'test':None,'dev':None,
            'reason':'ENVIRONMENT_BLOCKED',
            'weights_revision':settings.revision,'weights_sha256':settings.weights_sha256}})
        return

    from cipher_lab.execution import Clocks, Meter
    from cipher_lab.platform_resources.dxgi import DxgiUsage
    from cipher_lab.platform_resources.gpu import GpuBinding, configure_torch
    from cipher_lab.speck_study.closing import ClosingChannel, WindowsEvents
    from cipher_lab.speck_study.data import PublicSampler
    from cipher_lab.speck_study.gpu_telemetry import (
        TelemetryPublisher,
        current_process_identity,
        windows_monotonic,
    )
    from cipher_lab.speck_study.native_control import ChildResources, ControlClient
    from cipher_lab.speck_study.resources import StudyMeter
    from cipher_lab.speck_study.storage import StudyStore
    from cipher_lab.speck_study.teacher import TeacherClient

    publisher = None
    teacher: Any = None
    invoked = False
    resources = None
    try:
        require_native_launch()
        origin = float(os.environ['SPECK_STUDY_ORIGIN'])
        deadline = float(os.environ['SPECK_STUDY_SCIENCE_DEADLINE'])
        close = float(os.environ['SPECK_STUDY_CLOSE_DEADLINE'])
        if (not all(math.isfinite(x) for x in (origin,deadline,close))
                or deadline != origin+28200 or close != origin+28800
                or not origin <= windows_monotonic() < deadline):
            raise RuntimeError('ENVIRONMENT_BLOCKED')
        control = Path(os.environ['SPECK_STUDY_CONTROL_ROOT'])
        channel = ClosingChannel(WindowsEvents(),int(os.environ['SPECK_STUDY_REQUEST_HANDLE']),
                                 int(os.environ['SPECK_STUDY_ACK_HANDLE']))
        rpc = ControlClient(control,close)
        pid,created = current_process_identity()
        publisher = TelemetryPublisher(control,control/'main-telemetry.json',pid=pid,
                                       created=created,sample=DxgiUsage().sample)
        publisher.publish_once()
        publisher.start()
        resources = ChildResources(rpc,channel,publisher.check)
        # No science check before actual context binding; parent continues
        # bootstrap accounting, but can never turn bootstrap into admission.
        resources.check(science=False)
        import torch
        binding = GpuBinding(1,settings.gpu_uuid,settings.gpu_pci)
        evidence = configure_torch(torch,binding)
        if rpc('verify',role='main',evidence=evidence).get('verified') is not True:
            raise RuntimeError('GPU_UNVERIFIED')
        source_root = Path(__file__).resolve().parents[2]
        wrapper = source_root.parent/'tools'/'speck-study'/'freeze.py'
        manifest = os.environ['SPECK_STUDY_PROVIDER_MANIFEST']

        def spawn(args: list[str], **kwargs: Any) -> Any:
            if args[1:4] != ['-B','-m','cipher_lab.speck_study.teacher_provider']:
                raise RuntimeError('untrusted teacher entry')
            rpc('expect-provider')
            command = [args[0],'-B',str(wrapper),'--teacher',manifest,str(source_root)]+args[4:]
            process = subprocess.Popen(command,creationflags=0x4,close_fds=True,**kwargs)
            try:
                rpc('adopt-provider',pid=process.pid)  # parent independent handle+resume
                return process
            except BaseException:
                # Constructor has not received Popen yet; retain cleanup here.
                errors: list[BaseException] = []
                try:
                    process.kill()
                except BaseException as error:
                    errors.append(error)
                reaped = False
                try:
                    process.wait(timeout=2.)
                    reaped = True
                except BaseException as error:
                    errors.append(error)
                if reaped:
                    for name in ('stdin','stdout','stderr'):
                        try:
                            stream = getattr(process,name)
                            if stream is not None:
                                stream.close()
                        except BaseException as error:
                            errors.append(error)
                if errors:
                    raise RuntimeError('CLEANUP_UNKNOWN') from errors[0]
                raise

        def verify(e: dict[str,Any], p: int, uuid: str, pci: str) -> bool:
            return (e.get('pid') == p and uuid == settings.gpu_uuid
                    and pci == settings.gpu_pci
                    and rpc('verify',role='provider',evidence=e).get('verified') is True)

        teacher = TeacherClient(str(settings.teacher_python),
            'cipher_lab.speck_study.teacher_provider',str(settings.weights),
            settings.weights_sha256,settings.gpu_uuid,control,deadline,
            gpu_pci_bus_id=settings.gpu_pci,verify_gpu=verify,spawn=spawn,
            clock=windows_monotonic)
        client = teacher
        class ReapedTeacher:
            ready = client.ready
            def predict(self, points: Any) -> Any:
                return client.predict(points)
            def close(self) -> None:
                client.close()
                if rpc('provider-reaped').get('reaped') is not True:
                    raise RuntimeError('CLEANUP_UNKNOWN')
        teacher = ReapedTeacher()
        resources.check()  # BOTH main and provider attested before R0
        def r0(data: Any) -> bool:
            resources.check()
            ready = client.ready
            if (ready['weights_sha256'] != settings.weights_sha256
                    or not ready['python'].startswith('3.9.')
                    or not ready['tensorflow'].startswith('2.5.')
                    or not str(torch.__version__).startswith('2.7.')
                    or str(torch.version.cuda) != '11.8'):
                return False
            return formal_r0(data)
        owner.write_payload('formal-environment',dict(
            python=__import__('platform').python_version(),torch=str(torch.__version__),
            cuda=str(torch.version.cuda),teacher=client.ready,origin=origin,
            science_deadline=deadline,close_deadline=close,
            weights_revision=settings.revision,weights_sha256=settings.weights_sha256))
        resources.r0 = r0
        ledger = Meter(policy,Clocks(wall=windows_monotonic,cpu=time.process_time),
                       lambda: int(resources.latest['rss_bytes']),enforce_cpu=False)
        study_meter = StudyMeter(ledger,cast(Any,resources))
        class NativeStore(StudyStore):
            def write_snapshot(self, value: dict[str,Any]) -> None:
                # Task4 snapshots carry arithmetic estimates under policy; the
                # native owner retains actual CLI Policy, estimates separately.
                saved = dict(value)
                saved['policy'] = policy.to_dict()
                self.owner.write_snapshot(saved)
        from cipher_lab.speck_study_config import estimate_study
        owner.write_payload('study-estimate',estimate_study(recipe))
        _, invoke = assemble(recipe,NativeStore(owner,root,run_id),CipherAdapter(torch,'cuda:0'),
                             os.urandom,PublicSampler(),study_meter,resources,teacher)
        invoked = True
        invoke()  # fixed Task4 method, math=None, owns finalize and requested ACK
    except Exception:
        if invoked:
            raise  # no retry/finalize after the method or ArtifactError
        if resources is not None:
            resources.begin_closing()
        if teacher is not None:
            teacher.close()
        owner.finalize({'status':'STOPPED','summary':{
            'method':recipe.method,'backend_identity':'speck-study','synthetic':False,
            'scientific_validation':False,'prediction_positive':False,
            'reproduction_matched':False,'test':None,'dev':None,
            'reason':'ENVIRONMENT_BLOCKED',
            'weights_revision':settings.revision,'weights_sha256':settings.weights_sha256}})
        if resources is not None:
            resources.acknowledge_closing()
    finally:
        # Publisher lives through single-owner save/ACK; native owner exits next.
        # Do not unlink while the parent still counts this live PID. Process
        # exit stops the daemon; parent retires the slot only after handle reap.
        if publisher is not None:
            publisher.check()
