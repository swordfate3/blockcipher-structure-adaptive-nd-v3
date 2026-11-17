"""Device setup diagnostics: artificial drivers/clocks only, no scientific work."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cipher_lab_plugins.innovation1 import application, resources
from cipher_lab_plugins.innovation1 import device as device_module
from cipher_lab_plugins.innovation1.config import Recipe

from cipher_lab.execution import Clocks, Stopped

ROOT = Path(__file__).resolve().parents[4]
TOOL = ROOT/'tools/innovation1/device_check.py'
_POP = subprocess.Popen
_SAMPLE = resources.gpu_sample
_ATTEST = device_module.actual_attestor  # original body, fake driver only under guards


@pytest.fixture(autouse=True)
def device_only_guard(monkeypatch: Any) -> None:
    """Stricter than default: this owner file allows no model construction at all."""
    import torch
    from cipher_lab_plugins.innovation1 import models
    from plugin_isolation import forbidden
    monkeypatch.setattr(torch.nn.Module, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    monkeypatch.setattr(models.TorchRuntime, '__init__', forbidden)
    for name in ('device_count', 'set_device', 'current_device', 'get_device_properties',
                 'set_per_process_memory_fraction', 'memory_allocated'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def tool() -> Any:
    assert TOOL.is_file(), 'missing independent device check'
    spec = importlib.util.spec_from_file_location('innovation1_device_check_tool', TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reason_detail_compatible_and_whitelisted() -> None:
    error = resources.ResourceStop('GPU_LIMIT', detail_code='GPU_MEMORY_LIMIT')
    assert str(error) == error.reason == 'GPU_LIMIT'
    assert error.detail_code == 'GPU_MEMORY_LIMIT'
    with pytest.raises(ValueError):
        resources.ResourceStop(detail_code='raw private error')


@pytest.mark.parametrize('kind,detail', [('stale', 'GPU_PROOF_STALE'),
                                        ('missing', 'GPU_SAMPLE_INVALID')])
def test_gpu_detail_no_new_sampling(kind: str, detail: str) -> None:
    now = [0.]
    samples: list[Any] = []
    def sample() -> Any:
        samples.append(1)
        return dict(uuid='GPU-manual', pci='0000:3b:00.0', bytes=0 if kind == 'stale' else None)
    gate = resources.GpuGate('GPU-manual', '0000:3b:00.0', 20, sample, lambda: now[0])
    if kind == 'stale':
        gate.check()
        now[0] = 6.
    with pytest.raises(resources.ResourceStop) as caught:
        gate.check()
    assert caught.value.detail_code == detail
    assert len(samples) == 1


def fake_setup(monkeypatch: Any, failure: Exception | None = None) -> tuple[Any, list[Any]]:
    events: list[Any] = []
    gate = SimpleNamespace(last=1., latest={'uuid': 'GPU-manual'}, failure=None)
    monkeypatch.setattr(application, 'GpuGate', lambda *a: gate)
    class FakeResources:
        def __init__(self, meter: Any, gpu: Any) -> None:
            self.gpu = gpu
        def check(self) -> None:
            events.append('original-check')
            if failure:
                raise failure
    monkeypatch.setattr(application, 'Resources', FakeResources)
    return gate, events


def test_original_stop_and_internal_gate_proof_survive_observer(monkeypatch: Any) -> None:
    original = Stopped('RSS_LIMIT')
    gate, calls = fake_setup(monkeypatch, original)
    observed: list[Any] = []
    # Explicit fake sys.modules membership, not real Torch/CUDA bootstrap.
    monkeypatch.delitem(__import__('sys').modules, 'torch')
    with pytest.raises(Stopped) as caught:
        application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0',
                                         observer=observed.append)
    assert caught.value is original
    assert calls == ['original-check']
    assert any(e['stage'] == 'initial-resource-check' and e['event'] == 'stop' for e in observed)
    proof = next(e for e in observed if e.get('gate_last') == 1.)
    proof['gate_latest']['uuid'] = 'mutated-copy'
    assert gate.latest['uuid'] == 'GPU-manual'


def test_observer_cannot_swallow_original_stop(monkeypatch: Any) -> None:
    original = resources.ResourceStop('GPU_LIMIT')
    fake_setup(monkeypatch, original)
    monkeypatch.delitem(__import__('sys').modules, 'torch')
    def broken(event: Any) -> None:
        if event['event'] == 'stop':
            raise RuntimeError('OBSERVER_PRIVATE')
    with pytest.raises(resources.ResourceStop) as caught:
        application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0',
                                         observer=broken)
    assert caught.value is original


def test_device_tool_science_zero_internal_failure_evidence(tmp_path: Path) -> None:
    t = tool()
    def setup(recipe: Any, dev: str, uuid: str, bus: str, *, observer: Any) -> Any:
        observer(dict(stage='cuda-init', event='begin', elapsed_seconds=1.,
                      gate_last=0.5, gate_latest={'uuid': uuid}))
        raise resources.ResourceStop(detail_code='CUDA_INIT_FAILED')
    result = t.run_check(ROOT/'configs/innovation1-fresh.json', tmp_path/'device-check',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=setup, clocks=Clocks(lambda: 0., lambda: 0.), rss=lambda: 0)
    saved = json.loads((tmp_path/'device-check/device-check.json').read_bytes())
    assert result['command_status'] == 'STOPPED' and saved['reason'] == 'GPU_UNVERIFIED'
    assert saved['detail_code'] == 'CUDA_INIT_FAILED'
    assert saved['events'][0]['gate_last'] == 0.5
    for kind in ('real_blocks', 'null_words', 'forward_rows', 'optimizer_updates'):
        assert saved['counts'][f'attempted_{kind}'] == saved['counts'][f'completed_{kind}'] == 0
    assert saved['limits']['wall_seconds'] == 120 and saved['limits']['cpu_seconds'] == 480
    assert 'tools/innovation1/device_check.py' in saved['source']
    assert not saved['scientific_validation']


def test_gpu_preserves_original_stopped_and_caches_without_retry() -> None:
    original = Stopped('CPU_LIMIT')
    calls: list[Any] = []
    def sample() -> Any:
        calls.append(1)
        raise original
    gate = resources.GpuGate('GPU-manual', '0000:3b:00.0', 20, sample, lambda: 0.)
    for _ in range(2):
        with pytest.raises(Stopped) as caught:
            gate.check()
        assert caught.value is original
    assert len(calls) == 1


def artificial_torch(monkeypatch: Any, *, failure: str | None = None) -> tuple[Any, list[str]]:
    import builtins
    from copy import deepcopy

    events: list[str] = []
    def operation(name: str, value: Any = None) -> Any:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            events.append(name)
            if failure == name:
                raise RuntimeError('PRIVATE_TORCH_FAILURE')
            return value
        return invoke
    fake = SimpleNamespace(cuda=SimpleNamespace(
        device_count=operation('count', 1), set_device=operation('select'),
        init=operation('init'), get_device_properties=operation('properties',
            SimpleNamespace(total_memory=24*2**30)),
        set_per_process_memory_fraction=operation('fraction')),
        set_num_threads=operation('intra'), set_num_interop_threads=operation('inter'),
        set_default_dtype=operation('dtype'), float32='FLOAT32',
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                 cudnn=SimpleNamespace(allow_tf32=True, benchmark=True)))
    real_import = builtins.__import__
    def imported(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'torch':
            return fake
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imported)
    monkeypatch.delitem(__import__('sys').modules, 'torch')
    return deepcopy(fake), events


@pytest.mark.parametrize('where,stage,detail', [
    ('init', 'cuda-init', 'CUDA_INIT_FAILED'),
    ('select', 'cuda-select', 'CUDA_SELECT_FAILED'),
    ('properties', 'device-properties', 'DEVICE_PROPERTIES_FAILED'),
    ('fraction', 'memory-fraction', 'MEMORY_FRACTION_FAILED'),
    ('intra', 'torch-configure', 'TORCH_CONFIGURE_FAILED'),
])
def test_device_stage_fixed_detail_and_gate_proof_before_return(
        monkeypatch: Any, where: str, stage: str, detail: str) -> None:
    from cipher_lab_plugins.innovation1 import device
    gate, checks = fake_setup(monkeypatch)
    _, calls = artificial_torch(monkeypatch, failure=where)
    monkeypatch.setattr(device, 'actual_attestor', lambda *a: dict(
        gpu_uuid='GPU-manual', pci_bus_id='0000:3b:00.0', logical_device=0,
        pid=os.getpid(), evidence='cuda-current-context'))
    observed: list[Any] = []
    with pytest.raises(resources.ResourceStop) as caught:
        application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0',
                                         observer=observed.append)
    assert caught.value.reason == 'GPU_UNVERIFIED' and caught.value.detail_code == detail
    assert observed[-1]['stage'] == stage and observed[-1]['event'] == 'stop'
    assert observed[-1]['gate_last'] == gate.last
    assert calls and checks == ['original-check']
    assert 'PRIVATE_TORCH' not in json.dumps(observed)


def test_observer_default_no_extra_checks_and_failure_never_success(monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1 import device
    _, checks = fake_setup(monkeypatch)
    artificial_torch(monkeypatch)
    monkeypatch.setattr(device, 'actual_attestor', lambda *a: dict(
        gpu_uuid='GPU-manual', pci_bus_id='0000:3b:00.0', logical_device=0,
        pid=os.getpid(), evidence='cuda-current-context'))
    application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0')
    assert checks == ['original-check', 'original-check']
    checks.clear()
    observed: list[Any] = []
    def broken(event: Any) -> None:
        observed.append(event)
        if event['stage'] == 'final-resource-check' and event['event'] == 'success':
            raise RuntimeError('PRIVATE_OBSERVER_FAILURE')
    with pytest.raises(resources.ResourceStop) as caught:
        application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0',
                                         observer=broken)
    assert caught.value.detail_code == 'OBSERVER_FAILED'
    assert checks == ['original-check', 'original-check']
    assert observed[-1]['event'] == 'success'  # no fabricated return after observer failure


@pytest.mark.parametrize('kind,reason', [('wall', 'WALL_LIMIT'), ('cpu', 'CPU_LIMIT'),
                                        ('rss', 'RSS_LIMIT')])
def test_tool_observer_outer_meter_only_and_budget_includes_setup(
        tmp_path: Path, kind: str, reason: str) -> None:
    from cipher_lab.execution import Clocks
    t = tool()
    now = dict(wall=0., cpu=0., rss=0.)
    events: list[Any] = []
    def setup(*args: Any, observer: Any) -> Any:
        events.append('setup')
        observer(dict(stage='cuda-init', event='begin', elapsed_seconds=0., gate_last=1.))
        now[kind] = 10**15
        observer(dict(stage='cuda-init', event='success', elapsed_seconds=1., gate_last=1.))
        pytest.fail('outer budget must stop before any more setup')
    result = t.run_check(ROOT/'configs/innovation1-fresh.json', tmp_path/'device-check',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI, setup=setup,
        clocks=Clocks(lambda: now['wall'], lambda: now['cpu']), rss=lambda: int(now['rss']))
    assert result['reason'] == reason and result['command_status'] == 'STOPPED'
    assert events == ['setup']
    saved = json.loads((tmp_path/'device-check/device-check.json').read_bytes())
    assert saved['stage'] == 'cuda-init' and len(saved['events']) == 2


def test_tool_normal_device_only_postwrite_stop_and_exclusive_output(tmp_path: Path,
                                                                   monkeypatch: Any) -> None:
    from cipher_lab.execution import Clocks
    t = tool()
    now = [0.]
    calls: list[Any] = []
    def setup(*args: Any, observer: Any) -> Any:
        calls.append('device-only')
        observer(dict(stage='returned', event='success', elapsed_seconds=1., gate_last=0.5,
                      attestation=dict(gpu_uuid=t.GPU_UUID)))
        return SimpleNamespace(check=lambda: pytest.fail('no extra GPU check'))
    original = t._write
    def write(path: Path, raw: bytes) -> None:
        original(path, raw)
        now[0] = 121.
    monkeypatch.setattr(t, '_write', write)
    output = tmp_path/'device-check'
    result = t.run_check(ROOT/'configs/innovation1-fresh.json', output,
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI, setup=setup,
        clocks=Clocks(lambda: now[0], lambda: 0.), rss=lambda: 0)
    saved = json.loads((output/'device-check.json').read_bytes())
    assert saved['status'] == 'PASS' and result['evidence_status'] == 'PASS'
    assert result['command_status'] == 'STOPPED' and result['reason'] == 'WALL_LIMIT'
    assert result['exit_code'] == 3 and calls == ['device-only']
    assert saved['device'] == {'gpu_uuid': t.GPU_UUID}
    second = t.run_check(ROOT/'configs/innovation1-fresh.json', output,
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI, setup=setup,
        clocks=Clocks(lambda: 0., lambda: 0.), rss=lambda: 0)
    assert second['reason'] == 'OUTPUT_OCCUPIED' and calls == ['device-only']
    assert json.loads((output/'device-check.json').read_bytes()) == saved


@pytest.mark.parametrize('failure,detail', [
    ('cuCtxGetCurrent', 'CONTEXT_QUERY_FAILED'), ('no-context', 'NO_CURRENT_CONTEXT'),
    ('cuCtxGetDevice', 'CONTEXT_DEVICE_FAILED'), ('ordinal', 'CONTEXT_ORDINAL_MISMATCH'),
    ('cuDeviceGetUuid', 'DEVICE_UUID_FAILED'), ('cuDeviceGetPCIBusId', 'DEVICE_PCI_FAILED'),
    ('cuDeviceGetLuid', 'DEVICE_LUID_FAILED'), ('luid-zero', 'DEVICE_LUID_INVALID')])
def test_attestor_fake_driver_fixed_details_no_actual_device(monkeypatch: Any,
                                                           failure: str, detail: str) -> None:
    from cipher_lab_plugins.innovation1 import device
    class Driver:
        def cuCtxGetCurrent(self, ptr: Any) -> int:
            ptr._obj.value = 0 if failure == 'no-context' else 1
            return 1 if failure == 'cuCtxGetCurrent' else 0
        def cuCtxGetDevice(self, ptr: Any) -> int:
            ptr._obj.value = 1 if failure == 'ordinal' else 0
            return 1 if failure == 'cuCtxGetDevice' else 0
        def cuDeviceGetUuid(self, ptr: Any, dev: Any) -> int:
            return 1 if failure == 'cuDeviceGetUuid' else 0
        def cuDeviceGetPCIBusId(self, bus: Any, length: int, dev: Any) -> int:
            bus.value = b'0000:3b:00.0'
            return 1 if failure == 'cuDeviceGetPCIBusId' else 0
        def cuDeviceGetLuid(self, local: Any, mask: Any, dev: Any) -> int:
            local[0] = b'1'
            mask._obj.value = 0 if failure == 'luid-zero' else 1
            return 1 if failure == 'cuDeviceGetLuid' else 0
    # Replace only module's os reference, never process-global os.name/Path behavior.
    monkeypatch.setattr(device, 'os', SimpleNamespace(name='nt', getpid=os.getpid))
    with pytest.raises(resources.ResourceStop) as caught:
        _ATTEST(driver=Driver())
    assert caught.value.detail_code == detail


def test_device_tool_guarded_children_help_and_manual(tmp_path: Path) -> None:
    import subprocess
    import sys
    # Parent constructor captured before owner subprocess poison (see module constant).
    script = '''import sys,importlib.abc,runpy
if sys.argv[1]=='help':
 class Block(importlib.abc.MetaPathFinder):
  def find_spec(self,name,*a,**k):
   if name.split('.')[0] in ('torch','numpy','cipher_lab','cipher_lab_plugins'):
    raise AssertionError('NO_SCIENCE_IMPORT')
 sys.meta_path.insert(0,Block())
 def audit(event,args):
  if event in ('os.mkdir','subprocess.Popen','socket.__new__'):
   raise AssertionError('NO_EFFECT')
  if event=='open' and isinstance(args[1],str) and any(c in args[1] for c in 'wax+'):
   raise AssertionError('NO_WRITE')
 sys.addaudithook(audit)
 path=sys.argv[2];sys.argv=[path,'--help'];runpy.run_path(path,run_name='__main__')
else:
 from checkpoint_child_guard import install_checkpoint_guard,forbidden
 install_checkpoint_guard()
 import pytest,socket,subprocess,ctypes
 patch=pytest.MonkeyPatch()
 for owner,names in ((socket,('socket','create_connection')),
                     (subprocess,('Popen','run','check_output'))):
  for name in names:patch.setattr(owner,name,forbidden)
 patch.setattr(ctypes,'CDLL',forbidden)
 if hasattr(ctypes,'WinDLL'):patch.setattr(ctypes,'WinDLL',forbidden)
 import torch
 patch.setattr(torch.nn.Module,'__init__',forbidden)
 patch.setattr(torch.cuda,'set_per_process_memory_fraction',forbidden)
 from cipher_lab_plugins.innovation1 import resources,device,application
 for module,name in ((resources,'production_rss_bytes'),(resources,'gpu_sample'),
                     (device,'actual_attestor'),(application,'production_rss_bytes'),
                     (application,'gpu_sample'),(application,'production_resources')):
  patch.setattr(module,name,forbidden)
 class Block(importlib.abc.MetaPathFinder):
  def find_spec(self,name,*a,**k):
   if name.startswith(('cipher_lab.models','cipher_lab_plugins.innovation1.models',
       'cipher_lab_plugins.innovation1.training','cipher_lab_plugins.innovation1.data',
       'cipher_lab.ciphers','torch.optim')):
    forbidden()
 sys.meta_path.insert(0,Block())
 from test_innovation1_device_check import tool,ROOT
 from cipher_lab.execution import Clocks
 from pathlib import Path
 t=tool()
 def setup(*a,observer):
  observer(dict(stage='returned',event='success',elapsed_seconds=0.,gate_last=None))
 result=t.run_check(ROOT/'configs/innovation1-fresh.json',Path(sys.argv[3]),
  device='cuda:0',gpu_uuid=t.GPU_UUID,gpu_pci=t.GPU_PCI,setup=setup,
  clocks=Clocks(lambda:0.,lambda:0.),rss=lambda:0)
 assert result['command_status']=='PASS'
 assert result['counts']['completed_forward_rows']==0
 print('DEVICE_ONLY_MANUAL_PASS')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(str(p) for p in (
        ROOT/'src', ROOT/'plugins', ROOT/'tests', Path(__file__).parent)))
    for mode in ('help', 'manual'):
        proc = _POP([sys.executable, '-c', script, mode, str(TOOL), str(tmp_path/'device-check')],
                    cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        assert proc.returncode == 0, (mode, out, err)
        assert '--output-dir' in out if mode == 'help' else 'DEVICE_ONLY_MANUAL_PASS' in out


def test_observer_budget_at_begin_prevents_operation(monkeypatch: Any) -> None:
    _, checks = fake_setup(monkeypatch)
    _, calls = artificial_torch(monkeypatch)
    original = Stopped('WALL_LIMIT')
    def observer(event: Any) -> None:
        if event['stage'] == 'cuda-init' and event['event'] == 'begin':
            raise original
    with pytest.raises(Stopped) as caught:
        application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0',
                                         observer=observer)
    assert caught.value is original
    assert 'init' not in calls
    assert checks == ['original-check']


def test_telemetry_command_and_format_details(monkeypatch: Any) -> None:
    import subprocess
    original_sample = _SAMPLE
    def failed(*a: Any, **k: Any) -> Any:
        raise OSError('PRIVATE_COMMAND')
    monkeypatch.setattr(subprocess, 'run', failed)
    with pytest.raises(resources.ResourceStop) as caught:
        original_sample('GPU-manual')
    assert caught.value.detail_code == 'GPU_SAMPLE_COMMAND_FAILED'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='N/A'))
    with pytest.raises(resources.ResourceStop) as caught:
        original_sample('GPU-manual')
    assert caught.value.detail_code == 'GPU_SAMPLE_FORMAT_INVALID'


def test_tool_does_not_overwrite_original_stop_at_prewrite(tmp_path: Path) -> None:
    t = tool()
    now = [0.]
    def setup(*a: Any, observer: Any) -> Any:
        observer(dict(stage='cuda-init', event='begin', elapsed_seconds=0., gate_last=1.))
        now[0] = 121.
        raise resources.ResourceStop(detail_code='CUDA_INIT_FAILED')
    result = t.run_check(ROOT/'configs/innovation1-fresh.json', tmp_path/'device-check',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI, setup=setup,
        clocks=Clocks(lambda: now[0], lambda: 0.), rss=lambda: 0)
    saved = json.loads((tmp_path/'device-check/device-check.json').read_bytes())
    assert saved['reason'] == 'GPU_UNVERIFIED' and saved['detail_code'] == 'CUDA_INIT_FAILED'
    assert result['reason'] == 'GPU_UNVERIFIED'
    assert result['postwrite_reason'] == result['prewrite_reason'] == 'WALL_LIMIT'


def test_device_internal_meter_stop_has_gate_and_stage(monkeypatch: Any) -> None:
    monkeypatch.delitem(__import__('sys').modules, 'torch')
    original = Stopped('RSS_UNAVAILABLE')
    fake_setup(monkeypatch)
    def fail(*a: Any, **k: Any) -> Any:
        raise original
    monkeypatch.setattr(application, 'Meter', fail)
    events: list[Any] = []
    with pytest.raises(Stopped) as caught:
        application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0',
                                         observer=events.append)
    assert caught.value is original
    assert events[-1]['stage'] == 'resource-setup' and events[-1]['event'] == 'stop'
    assert events[-1]['gate_last'] == 1.


def test_tool_write_failure_is_unknown_and_no_raw_exception(tmp_path: Path,
                                                          monkeypatch: Any) -> None:
    t = tool()
    def fail(*a: Any) -> None:
        raise OSError('PRIVATE_WRITE_ERROR')
    monkeypatch.setattr(t, '_write', fail)
    result = t.run_check(ROOT/'configs/innovation1-fresh.json', tmp_path/'device-check',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *a, **k: None, clocks=Clocks(lambda: 0., lambda: 0.), rss=lambda: 0)
    assert result['exit_code'] == 6 and result['evidence_status'] == 'UNKNOWN'
    assert result['reason'] == 'EVIDENCE_WRITE_FAILED'
    assert 'PRIVATE' not in json.dumps(result)


def test_device_tool_no_budget_or_fallback_cli_switches(capsys: Any) -> None:
    t = tool()
    for option in ('--wall-seconds', '--allow-stale', '--fallback', '--cpu-seconds'):
        with pytest.raises(SystemExit) as caught:
            t.main(['--config', 'unused', '--output-dir', 'unused', '--device', 'cuda:0',
                    '--gpu-uuid', 'unused', '--gpu-pci', 'unused', option, '1'])
        assert caught.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_attestor_evidence_available_on_its_success_before_budget_stop(monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1 import device
    fake_setup(monkeypatch)
    artificial_torch(monkeypatch)
    attested = dict(gpu_uuid='GPU-manual', pci_bus_id='0000:3b:00.0',
                    logical_device=0, pid=os.getpid(), evidence='cuda-current-context')
    monkeypatch.setattr(device, 'actual_attestor', lambda *a: attested)
    events: list[Any] = []
    def observer(event: Any) -> None:
        events.append(event)
        if event['stage'] == 'context-attest' and event['event'] == 'success':
            raise Stopped('WALL_LIMIT')
    with pytest.raises(Stopped):
        application.production_resources(Recipe(), 'cuda:0', 'GPU-manual', '0000:3b:00.0',
                                         observer=observer)
    assert events[-1]['attestation'] == attested
    assert events[-1]['attestation'] is not attested


def test_driver_load_original_stop_not_sanitized(monkeypatch: Any) -> None:
    import ctypes
    original = Stopped('CPU_LIMIT')
    monkeypatch.setattr(device_module, 'os', SimpleNamespace(name='posix', getpid=os.getpid))
    def load(*a: Any, **k: Any) -> Any:
        raise original
    monkeypatch.setattr(ctypes, 'CDLL', load)
    with pytest.raises(Stopped) as caught:
        _ATTEST()
    assert caught.value is original


def test_tool_real_assembly_with_fakes_no_scientific_entry(monkeypatch: Any,
                                                        tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1 import device
    t = tool()
    fake_setup(monkeypatch)
    artificial_torch(monkeypatch)
    monkeypatch.setattr(device, 'actual_attestor', lambda *a: dict(
        gpu_uuid=t.GPU_UUID, pci_bus_id=t.GPU_PCI, logical_device=0,
        pid=os.getpid(), evidence='cuda-current-context'))
    result = t.run_check(ROOT/'configs/innovation1-fresh.json', tmp_path/'device-check',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        clocks=Clocks(lambda: 0., lambda: 0.), rss=lambda: 0)
    saved = json.loads((tmp_path/'device-check/device-check.json').read_bytes())
    assert result['command_status'] == 'PASS'
    assert saved['events'][-1]['stage'] == 'returned'
    assert saved['device']['gpu_uuid'] == t.GPU_UUID


@pytest.mark.parametrize('original', [
    Stopped('RSS_UNAVAILABLE'),
    resources.ResourceStop('GPU_UNVERIFIED', detail_code='CUDA_INIT_FAILED'),
], ids=['rss-unavailable', 'cuda-init-failed'])
def test_tool_write_failure_preserves_original_diagnostic(
        tmp_path: Path, monkeypatch: Any, original: Exception) -> None:
    t = tool()
    writes: list[Path] = []
    def setup(*args: Any, observer: Any) -> Any:
        observer(dict(stage='cuda-init', event='begin', elapsed_seconds=0., gate_last=1.))
        raise original
    def fail_write(path: Path, raw: bytes) -> None:
        writes.append(path)
        raise OSError('PRIVATE_WRITE_ERROR')
    monkeypatch.setattr(t, '_write', fail_write)
    output = tmp_path/'device-check'
    result = t.run_check(ROOT/'configs/innovation1-fresh.json', output,
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI, setup=setup,
        clocks=Clocks(lambda: 0., lambda: 0.), rss=lambda: 0)
    assert result['command_status'] == 'FAILED' and result['exit_code'] == 6
    assert result['evidence_status'] == 'UNKNOWN'
    assert result['reason'] == result['write_reason'] == 'EVIDENCE_WRITE_FAILED'
    assert result['diagnostic_status'] == 'STOPPED'
    assert result['diagnostic_reason'] == str(original)
    assert result['detail_code'] == getattr(original, 'detail_code', None)
    assert result['stage'] == 'cuda-init'
    assert result['prewrite_reason'] is result['postwrite_reason'] is None
    assert writes == [output/'device-check.json']
    assert list(output.iterdir()) == []  # no retry, rewrite or secondary evidence
    assert 'PRIVATE' not in json.dumps(result)
