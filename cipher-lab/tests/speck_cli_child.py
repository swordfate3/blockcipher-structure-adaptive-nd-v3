"""Executed driver: installs its own guards BEFORE any project/science imports.

Only fake identity cipher/constant teacher/manual public stream. No real KAT,
private entropy, TF/Torch/NN/CUDA/RSS/SSH/Git/installation or performance work.
"""
from __future__ import annotations

import ctypes
import importlib.abc
import os
import secrets
import sys
from pathlib import Path
from uuid import UUID


def forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError('Task5 clean-source child scientific call forbidden')


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if (fullname.split('.')[0] in ('torch','tensorflow','h5py')
                or fullname == 'cipher_lab.ciphers' or fullname.startswith('cipher_lab.ciphers.')
                or fullname.startswith('cipher_lab.training.')):
            forbidden()
        return None


# TEST ONLY: select the fake child assembly BEFORE CLI dispatch on Windows.
# This is not a native-parent attestation and must never be used by production.
os.environ['SPECK_STUDY_NATIVE'] = '1'
setattr(ctypes,'CDLL',forbidden)
setattr(ctypes,'WinDLL',forbidden)
guard = Guard()
sys.meta_path.insert(0,guard)

# Load the actual source-only ENTRY before any project/scientific import. The
# snapshot is test-local, not a publication attestation. No ordinary import path
# may rescue a module missing from its fixed manifest.
import importlib.util  # noqa: E402
import json  # noqa: E402

entry, source_root, manifest_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('task5_frozen_entry',entry)
assert spec and spec.loader
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
manifest_path.write_text(json.dumps(bootstrap.snapshot(source_root)),encoding='utf8')
assert not any(n == 'cipher_lab' or n.startswith('cipher_lab.') for n in sys.modules)

import subprocess  # noqa: E402

subprocess.run = forbidden  # type: ignore[assignment]
subprocess.Popen = forbidden  # type: ignore[misc,assignment]
os.system = forbidden  # type: ignore[assignment]
os.urandom = lambda size: bytes(size)
secrets.randbits = lambda k: 0
import numpy as np  # noqa: E402
import numpy.random  # noqa: E402

os.urandom = forbidden  # type: ignore[assignment]
secrets.token_bytes = forbidden  # type: ignore[assignment]
secrets.randbits = forbidden  # type: ignore[assignment]
for name in ('default_rng','PCG64','seed','rand','randn','randint','random',
             'RandomState','Generator'):
    setattr(np.random,name,forbidden)

from collections import defaultdict  # noqa: E402
from typing import Any  # noqa: E402

# The guarded loader stays in front of SourceOnly, so a frozen cipher cannot
# bypass the child science guard. Project modules never fall back to sys.path.
native_application: Any = None
original_source_only = bootstrap.SourceOnly
class GuardedSourceOnly(original_source_only):  # type: ignore[valid-type,misc]
    def exec_module(self, module: Any) -> None:
        # Test the pre-dispatch marker even on Linux (which skips the Windows
        # parent branch). No inherited harness flag may stand in for this step.
        assert os.environ.get('SPECK_STUDY_NATIVE') == '1'
        guard.find_spec(module.__name__)
        super().exec_module(module)
        if module.__name__ == 'cipher_lab.cli':
            install_fake_application()

setattr(bootstrap,'SourceOnly',GuardedSourceOnly)


def install_fake_application() -> None:
    import cipher_lab.artifacts as artifacts
    import cipher_lab.execution as execution
    from cipher_lab.speck_study import application
    # Fail here (outside CLI's sanitized method boundary) if the real frozen
    # dependency graph is incomplete; neither ordinary import nor pyc may help.
    for name in ('cipher_lab.speck_study.spectral','cipher_lab.speck_study.storage'):
        module = importlib.import_module(name)
        assert isinstance(module.__loader__,GuardedSourceOnly)

    # Explicit synthetic identity, NOT a claim about this host. platform.system
    # may launch `ver` on Windows, conflicting with this child's subprocess trap.
    # Keep production _environment_identity untouched outside this fake driver.
    setattr(artifacts, '_environment_identity', lambda: {
        'python_implementation': {'cpython': 'CPython', 'pypy': 'PyPy'}.get(
            sys.implementation.name, sys.implementation.name),
        'python_version': '.'.join(str(v) for v in sys.version_info[:3]),
        'os': 'Windows', 'architecture': 'AMD64',
    })
    setattr(artifacts,'uuid4',lambda: UUID(int=1))
    setattr(execution,'production_rss_bytes',forbidden)
    global native_application
    native_application = application.run_application
    def diagnosed(*args: Any, **kwargs: Any) -> None:
        try:
            fake_application(*args,**kwargs)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
    application.run_application = diagnosed


class Resources:
    closing = False
    def check(self, *, science: bool = True) -> None:
        if science and self.closing:
            raise AssertionError('new science after closing')
    def begin_closing(self) -> None:
        self.closing = True
    def acknowledge_closing(self) -> None:
        assert self.closing


class Meter:
    def __init__(self, resources: Resources) -> None:
        self.meter = self
        self.resources = resources
        self.counts: Any = defaultdict(lambda:[0,0])
    def check(self) -> None:
        self.resources.check()
    def reserve(self, kind: str, rows: int) -> None:
        self.check()
        self.counts[kind][0] += rows
    def complete(self, kind: str, rows: int) -> None:
        self.check()
        self.counts[kind][1] += rows
        assert self.counts[kind][1] <= self.counts[kind][0]
    def snapshot(self) -> dict[str,int]:
        return {f'{status}_{kind}':values[i] for kind,values in self.counts.items()
                for i,status in enumerate(('attempted','completed'))}


class Teacher:
    closed = False
    def predict(self, points: Any) -> tuple[Any,Any]:
        assert not self.closed
        return (0.5,)*len(points),(0,)*len(points)
    def close(self) -> None:
        self.closed = True


def cipher(points: Any, keys: Any, rounds: int) -> Any:
    assert rounds == 0
    return points.copy()  # not encryption


def sample(seed: int, domain: str, offset: int, rows: int, lanes: int) -> Any:
    return np.arange(offset*lanes,(offset+rows)*lanes,dtype=np.uint64).reshape(rows,lanes)


def fake_application(recipe: Any, policy: Any, owner: Any, root: Path,
                     run_id: str, settings: Any) -> None:
    import cipher_lab.artifacts as artifacts
    from cipher_lab.speck_study import application
    from cipher_lab.speck_study.codec import StudyPredictor, StudyTerm
    from cipher_lab.speck_study.storage import StudyStore, durable_model, source_digest
    from cipher_lab.speck_study_config import SpeckStudyRecipe

    synthetic = SpeckStudyRecipe.synthetic_recipe(r_bits=3,b_bits=3,initial_depth=1,
        pairs_per_layer=8,r_classification_size=8,r_coef_size=8,r_agreement_size=8,
        b_coef_size=8,b_dev_size=8,b_test_size=8)
    store = StudyStore(owner,root,run_id)
    store.write_snapshot({'recipe':synthetic.to_dict()})
    model = StudyPredictor('R','teacher-reproduction',source_digest(store.source_snapshot()),
                           synthetic,(StudyTerm(1<<63,8),),8)
    loaded = durable_model(store,model,expected_kind='R',expected_recipe=synthetic,
                           expected_source=store.source_snapshot(),expected_backend='speck-study')
    assert loaded == model
    # Enter REAL native application assembly. Only OS/GPU/cipher/public sampler
    # and provider process are fakes; TeacherClient/IPC/ChildResources/Meter/
    # StudyData/method/RunStore/source-only owner remain production code.
    import io
    from types import SimpleNamespace

    from cipher_lab.platform_resources import dxgi, gpu
    from cipher_lab.speck_study import closing, data, gpu_telemetry, native_control
    from cipher_lab.speck_study import teacher_protocol as wire
    from cipher_lab.speck_study.teacher_network import MODEL_SPEC
    events: list[str] = []
    requested = [False]
    attested: set[str] = set()
    reaped = [False]
    publisher_checks = [0]
    class FakePublisher:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs['pid'] == 7 and kwargs['created'] == 70
        def publish_once(self) -> None:
            events.append('main-measured-zero')
        def start(self) -> None:
            events.append('publisher-start')
        def check(self) -> None:
            publisher_checks[0] += 1
    class FakeEvents:
        def poll(self, handle: int) -> bool:
            return requested[0] if handle == 1 else 'ack' in events
        def set(self, handle: int) -> None:
            assert handle == 2 and requested[0]
            assert artifacts.read_saved(root,run_id)['status'] == 'SUCCEEDED'
            events.append('ack')
    evidence = dict(pid=9,gpu_uuid=settings.gpu_uuid,pci_bus_id=settings.gpu_pci,
                    logical_device=0,evidence='cuda-current-context',
                    luid='0100000000000000',device_node_mask=1)
    ready = dict(python='3.9.synthetic',tensorflow='2.5.synthetic',model=MODEL_SPEC,
                 weights_sha256=settings.weights_sha256,gpu=evidence,allocator_mib=8192)
    stream = io.BytesIO()
    for frame in (wire.Frame(wire.READY,0,0,json.dumps(ready).encode()),
                  wire.Frame(wire.REPLY,1,8,wire.pack_reply([.5]*8))):
        wire.write_frame(wire.DeadlineIO(stream,clock=lambda: 10**12+1),
                         frame,10**12+28200)
    stream.seek(0)
    class Process:
        pid = 9
        returncode = None
        stdin = io.BytesIO()
        stdout = stream
        stderr = io.BytesIO()
        def poll(self) -> Any:
            return self.returncode
        def terminate(self) -> None:
            events.append('teacher-terminate')
            self.returncode = 0
        def kill(self) -> None:
            self.returncode = 1
        def wait(self, timeout: float) -> int:
            assert self.returncode is not None
            reaped[0] = True
            return self.returncode
    def popen(args: Any, **kwargs: Any) -> Any:
        assert Path(args[2]).name == 'freeze.py'
        assert args[3] == '--teacher'
        assert kwargs['creationflags'] == 4 and kwargs['close_fds'] is True
        assert events[-1] == 'expect-provider'
        events.append('suspended')
        return Process()
    class RPC:
        def __init__(self, *args: Any) -> None:
            pass
        def __call__(self, kind: str, **payload: Any) -> dict[str,Any]:
            if kind == 'expect-provider':
                events.append(kind)
                return {}
            if kind == 'adopt-provider':
                assert events[-1] == 'suspended' and payload['pid'] == 9
                events.extend(['independent-created-slot','resume'])
                return {}
            if kind == 'verify':
                attested.add(payload['role'])
                return {'verified':True}
            if kind == 'check':
                if payload['science']:
                    assert attested == {'main','provider'}
                return dict(wall=10**12+1,science_deadline=10**12+28200,
                            latest={'rss_bytes':123})
            if kind == 'provider-reaped':
                assert reaped[0]
                events.append('provider-retired')
                return {'reaped':True}
            if kind == 'closing':
                requested[0] = True
                return {}
            raise AssertionError(kind)
    class FakeCipher:
        def __init__(self, *args: Any) -> None:
            pass
        def __call__(self, points: Any, keys: Any, rounds: int) -> Any:
            return cipher(points,keys,rounds)
    setattr(application,'require_native_launch',lambda: None)  # ENGINEERING only
    setattr(application,'CipherAdapter',FakeCipher)
    setattr(gpu,'configure_torch',lambda *args: dict(evidence,pid=7))
    setattr(dxgi,'DxgiUsage',lambda: SimpleNamespace(sample=lambda: []))
    setattr(gpu_telemetry,'current_process_identity',lambda: (7,70))
    setattr(gpu_telemetry,'windows_monotonic',lambda: 10**12+1)
    setattr(gpu_telemetry,'TelemetryPublisher',FakePublisher)
    setattr(closing,'WindowsEvents',FakeEvents)
    setattr(native_control,'ControlClient',RPC)
    setattr(data,'PublicSampler',lambda: sample)
    setattr(subprocess,'Popen',popen)
    setattr(os,'urandom',lambda size: bytes(size))  # explicit fake private owner input
    from types import ModuleType
    fake_torch = ModuleType('torch')
    setattr(fake_torch,'__version__','2.7.synthetic')
    setattr(fake_torch,'version',SimpleNamespace(cuda='11.8'))
    sys.modules['torch'] = fake_torch
    os.environ.update(SPECK_STUDY_NATIVE='1',SPECK_STUDY_ORIGIN=str(10**12),
        SPECK_STUDY_SCIENCE_DEADLINE=str(10**12+28200),
        SPECK_STUDY_CLOSE_DEADLINE=str(10**12+28800),SPECK_STUDY_CONTROL_ROOT=str(root),
        SPECK_STUDY_REQUEST_HANDLE='1',SPECK_STUDY_ACK_HANDLE='2',
        SPECK_STUDY_PROVIDER_MANIFEST=str(root/'fake-provider-manifest'))
    # install_fake_application saved the production callable before replacing CLI
    native_application(synthetic,policy,owner,root,run_id,settings)
    assert 'independent-created-slot' in events and 'resume' in events
    assert events.index('suspended') < events.index('independent-created-slot')
    assert events.index('independent-created-slot') < events.index('resume')
    assert 'provider-retired' in events and events[-1] == 'ack'
    assert publisher_checks[0] > 5
    saved = artifacts.read_saved(root,run_id)
    assert isinstance(saved['summary'],dict)
    assert saved['summary']['scientific_validation'] is False
    assert saved['summary']['reason'] == 'REPRODUCTION_MISMATCH'
    assert saved['status'] == 'SUCCEEDED'
    assert saved['policy'] == policy.to_dict()
    from cipher_lab.speck_study.native_parent import verify_terminal
    assert verify_terminal(root,run_id,synthetic,policy,store.source_snapshot())
    for name, module in tuple(sys.modules.items()):
        if name == 'cipher_lab' or name.startswith('cipher_lab.'):
            assert isinstance(module.__loader__,GuardedSourceOnly), name
    assert isinstance(sys.modules['cipher_lab.speck_study.codec'].__loader__,GuardedSourceOnly)


raise SystemExit(bootstrap.main(['run',str(manifest_path),str(source_root)]+sys.argv[4:]))
