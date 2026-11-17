"""Independent preflight: all numerical/device operations are explicit manual fakes."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from cipher_lab.execution import Clocks, Stopped

ROOT = Path(__file__).resolve().parents[4]
TOOL = ROOT / 'tools/innovation1/preflight.py'
_POP = subprocess.Popen


def tool() -> Any:
    assert TOOL.is_file(), 'missing independent preflight tool'
    spec = importlib.util.spec_from_file_location('innovation1_preflight_tool', TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Manual:
    kind = 'manual-test-only'
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.evidence = {'uuid': 'GPU-manual', 'pci': '0000:3b:00.0',
                         'evidence': 'ARTIFICIAL_NOT_DEVICE_PROOF'}
        self.gpu = None

    def encrypt(self, vector: dict[str, Any]) -> tuple[int, ...]:
        self.calls.append(('kat', vector['cipher'], vector['rounds'], vector['key_words']))
        return (int(vector['expected'], 16),)

    def build(self, recipe: Any) -> None:
        self.calls.append(('build',))

    def evaluate(self, device: str, slot: str, task: Any) -> list[list[float]]:
        self.calls.append(('eval', device, slot, task.cipher))
        return [[-0.25], [0.5]]

    def update(self, slot: str, task: Any, lr: float, resources: Any) -> float:
        self.calls.append(('update', slot, task.cipher, lr))
        resources.reserve('forward_rows', 2)
        resources.complete('forward_rows', 2)
        return 0.7


def clock(now: dict[str, float]) -> Clocks:
    return Clocks(lambda: now['wall'], lambda: now['cpu'],
                  lambda: datetime(2020, 1, 1, tzinfo=UTC))


def run_manual(tmp_path: Path, *, runtime: Any = None, setup: Any = None,
               now: dict[str, float] | None = None) -> tuple[Any, ...]:
    t = tool()
    runtime = runtime or Manual()
    now = now or {'wall': 0., 'cpu': 0., 'rss': 0.}
    output = tmp_path / 'one' / 'preflight'
    result = t.run_preflight(ROOT / 'configs/innovation1-fresh.json', output,
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=setup or (lambda *args: runtime), clocks=clock(now), rss=lambda: int(now['rss']))
    return t, result, runtime, output


def test_preflight_exact_work_order_and_single_public_evidence(tmp_path: Path) -> None:
    t, result, runtime, output = run_manual(tmp_path)
    assert result['command_status'] == result['evidence_status'] == 'PASS'
    evidence = json.loads((output / 'preflight.json').read_bytes())
    counts = evidence['counts']
    for kind, amount in [('real_blocks', 8), ('forward_rows', 48), ('optimizer_updates', 8)]:
        assert counts[f'attempted_{kind}'] == counts[f'completed_{kind}'] == amount
    assert len([x for x in runtime.calls if x[0] == 'kat']) == 8
    evals = [x for x in runtime.calls if x[0] == 'eval']
    assert len(evals) == 16
    assert len([x for x in evals if x[1] == 'cpu']) == 8
    assert len([x for x in evals if x[1] == 'cuda:0']) == 8
    updates = [x for x in runtime.calls if x[0] == 'update']
    assert [x[1] for x in updates] == [s for cipher in (
        'SIMECK', 'PRESENT', 'SIMON', 'SKINNY') for s in ('M-best', f'D-{cipher}-best')]
    assert all(x[3] == 0.001 for x in updates)
    assert evidence['runtime_kind'] == 'manual-test-only'
    assert evidence['scientific_validation'] is False
    assert evidence['source']['tools/innovation1/preflight.py'] == sha256(
        TOOL.read_bytes()).hexdigest()
    assert evidence['config_sha256'] == sha256(
        (ROOT/'configs/innovation1-fresh.json').read_bytes()).hexdigest()
    assert evidence['fixture_sha256'] == sha256(t.FIXTURE_BYTES).hexdigest()
    assert evidence['limits'] == dict(wall_seconds=600, cpu_seconds=2400,
        rss_bytes=16*2**30, gpu_bytes=20*2**30, artifact_bytes=16*2**20,
        real_blocks=8, forward_rows=48, optimizer_updates=8)
    assert sorted(p.name for p in output.iterdir()) == ['preflight.json']
    assert evidence['tolerance'] == {'atol': 1e-5, 'rtol': 1e-4}
    assert len(evidence['operations']) == 32


def test_preflight_frozen_vectors_exact_low_word_first() -> None:
    t = tool()
    vectors = json.loads(t.FIXTURE_BYTES)
    expected = {'SIMON': (44, '44c8fc20b9dfa07a', 12, '3b2bc82c320b0062'),
                'PRESENT': (31, '5579c1387b228445', 8, '6ba06c48b513e6cc'),
                'SKINNY': (32, 'bb39dfb2429b8ac7', 7, '276128939906f992'),
                'SIMECK': (44, '45ce69025f7ab7ed', 8, 'a14867236133b40e')}
    assert len(vectors) == 8
    for family, (full, answer, reduced, replay) in expected.items():
        pair = [x for x in vectors if x['cipher'] == family]
        assert [(x['rounds'], x['expected']) for x in pair] == [(full, answer), (reduced, replay)]
        assert pair[0]['level'] == 'full-public-kat'
        assert pair[1]['level'] == 'reduced-replay-not-independent-official'
        if family in ('SIMON', 'SIMECK'):
            assert pair[0]['key_words'] == pair[1]['key_words'] == [
                0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918]
        elif family == 'SKINNY':
            assert pair[0]['key_words'] == pair[1]['key_words'] == [0xfc681238, 0xf5269826]
        else:
            assert pair[0]['key_words'] == pair[1]['key_words'] == [0, 0, 0]


@pytest.mark.parametrize('failure', ['kat', 'eval', 'compare', 'update'])
def test_preflight_first_failure_partial_no_retry(tmp_path: Path, failure: str) -> None:
    runtime = Manual()
    if failure == 'kat':
        runtime.encrypt = lambda vector: (0,)  # type: ignore[method-assign]
    elif failure == 'eval':
        def fail(*args: Any) -> Any:
            raise RuntimeError('PRIVATE_EXCEPTION_MUST_NOT_APPEAR')
        setattr(runtime, 'evaluate', fail)
    elif failure == 'compare':
        runtime.evaluate = lambda device, *a: [[-0.25], [0.5 if device == 'cpu' else 0.6]]  # type: ignore[method-assign]
    else:
        original = runtime.update
        def update(slot: str, *args: Any) -> float:
            if slot != 'M-best':
                raise RuntimeError('PRIVATE_EXCEPTION_MUST_NOT_APPEAR')
            return original(slot, *args)
        setattr(runtime, 'update', update)
    _, result, _, output = run_manual(tmp_path, runtime=runtime)
    saved = json.loads((output/'preflight.json').read_bytes())
    assert result['command_status'] == 'FAILED' and saved['status'] == 'FAILED'
    assert saved['stage'] == failure
    assert 'PRIVATE_EXCEPTION' not in json.dumps(saved)
    if failure == 'kat':
        assert saved['counts']['attempted_real_blocks'] == 1
        assert saved['counts']['completed_real_blocks'] == 0
    elif failure == 'update':
        assert saved['counts']['attempted_optimizer_updates'] == 2
        assert saved['counts']['completed_optimizer_updates'] == 1
    assert not any(p.name in ('data', 'models', 'result') for p in output.parent.iterdir())


@pytest.mark.parametrize('kind,reason', [('wall', 'WALL_LIMIT'), ('cpu', 'CPU_LIMIT'),
                                        ('rss', 'RSS_LIMIT')])
def test_preflight_budget_includes_setup(tmp_path: Path, kind: str, reason: str) -> None:
    now = {'wall': 0., 'cpu': 0., 'rss': 0.}
    def setup(*args: Any) -> Manual:
        now[kind] = float(10**15)
        return Manual()
    _, result, runtime, output = run_manual(tmp_path, setup=setup, now=now)
    saved = json.loads((output/'preflight.json').read_bytes())
    assert result['command_status'] == 'STOPPED' and saved['reason'] == reason
    assert saved['stage'] == 'device-setup'
    assert not runtime.calls
    assert saved['counts']['attempted_real_blocks'] == 0


def test_preflight_exclusive_occupied_output_and_postwrite_status(tmp_path: Path,
                                                               monkeypatch: Any) -> None:
    t = tool()
    output = tmp_path/'one'/'preflight'
    output.mkdir(parents=True)
    (output/'keep').write_text('untouched')
    def forbidden(*args: Any) -> Any:
        raise AssertionError('occupied must not setup')
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', output,
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI, setup=forbidden,
        clocks=clock({'wall': 0., 'cpu': 0.}), rss=lambda: 0)
    assert result['command_status'] == 'FAILED' and result['reason'] == 'OUTPUT_OCCUPIED'
    assert (output/'keep').read_text() == 'untouched'
    assert not (output/'preflight.json').exists()


def test_preflight_invalid_device_and_telemetry_no_fallback(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.resources import ResourceStop
    t = tool()
    calls: list[Any] = []
    def setup(*args: Any) -> Any:
        calls.append(args)
        raise ResourceStop()
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI, setup=setup,
        clocks=clock({'wall': 0., 'cpu': 0.}), rss=lambda: 0)
    assert result['command_status'] == 'STOPPED' and result['reason'] == 'GPU_UNVERIFIED'
    assert len(calls) == 1


def test_preflight_help_guarded_stdlib_child(tmp_path: Path) -> None:
    assert TOOL.is_file()
    script = '''import sys,runpy,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*a,**k):
  if name.split('.')[0] in ('torch','numpy','cipher_lab','cipher_lab_plugins'):
   raise AssertionError('UNAUTHORIZED_IMPORT')
sys.meta_path.insert(0,Block())
def audit(event,args):
 if event in ('os.mkdir','subprocess.Popen','socket.__new__'):
  raise AssertionError('UNAUTHORIZED_EFFECT')
 if event=='open' and isinstance(args[1],str) and any(c in args[1] for c in 'wax+'):
  raise AssertionError('UNAUTHORIZED_WRITE')
sys.addaudithook(audit)
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    proc = _POP([sys.executable, '-c', script, str(TOOL)], cwd=tmp_path,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate(timeout=20)
    assert proc.returncode == 0, (out, err)
    assert '--output-dir' in out


def test_preflight_postwrite_stop_does_not_relabel_saved_pass(tmp_path: Path,
                                                           monkeypatch: Any) -> None:
    t = tool()
    now = {'wall': 0., 'cpu': 0., 'rss': 0.}
    original = t._write_evidence
    def write(path: Path, raw: bytes) -> None:
        original(path, raw)
        now['wall'] = 601.
    monkeypatch.setattr(t, '_write_evidence', write)
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *args: Manual(), clocks=clock(now), rss=lambda: 0)
    assert result['command_status'] == 'STOPPED' and result['reason'] == 'WALL_LIMIT'
    assert result['evidence_status'] == 'PASS'
    assert json.loads((tmp_path/'preflight/preflight.json').read_bytes())['status'] == 'PASS'
    assert result['counts']['wall_elapsed_seconds'] == 601.


def test_preflight_rss_unavailable_before_setup_stops(tmp_path: Path) -> None:
    t = tool()
    def rss() -> int:
        raise Stopped('RSS_UNSUPPORTED')
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *args: pytest.fail('no numerical setup'),
        clocks=clock({'wall': 0., 'cpu': 0.}), rss=rss)
    assert result['command_status'] == 'STOPPED' and result['reason'] == 'RSS_UNSUPPORTED'
    saved = json.loads((tmp_path/'preflight/preflight.json').read_bytes())
    assert saved['stage'] == 'settings'


def test_preflight_clock_setup_failure_has_public_cli_status(tmp_path: Path) -> None:
    t = tool()
    def bad() -> float:
        raise RuntimeError('PRIVATE_CLOCK_DETAILS')
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        clocks=Clocks(bad, bad, lambda: datetime(2020, 1, 1, tzinfo=UTC)), rss=lambda: 0)
    assert result['command_status'] == 'STOPPED' and result['reason'] == 'CLOCK_UNAVAILABLE'
    assert 'PRIVATE' not in json.dumps(result)


def test_preflight_source_changed_and_write_failure_fixed_codes(tmp_path: Path,
                                                              monkeypatch: Any) -> None:
    t = tool()
    original = t._source
    calls = [0]
    def source() -> Any:
        calls[0] += 1
        return original() if calls[0] == 1 else {'tools/changed.py': 'a'*64}
    monkeypatch.setattr(t, '_source', source)
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *args: Manual(), clocks=clock({'wall': 0., 'cpu': 0.}), rss=lambda: 0)
    assert result['command_status'] == 'FAILED' and result['reason'] == 'SOURCE_CHANGED'
    def write(*args: Any) -> None:
        raise OSError('PRIVATE_DISK_DETAILS')
    monkeypatch.setattr(t, '_write_evidence', write)
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight2',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *args: Manual(), clocks=clock({'wall': 0., 'cpu': 0.}), rss=lambda: 0)
    assert result['reason'] == 'EVIDENCE_WRITE_FAILED' and result['evidence_status'] == 'UNKNOWN'
    assert 'PRIVATE' not in json.dumps(result)


def test_preflight_gpu_gate_is_original_and_staleness_not_reset(tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.resources import GpuGate
    t = tool()
    now = {'wall': 0., 'cpu': 0., 'rss': 0.}
    runtime = Manual()
    gate = GpuGate(t.GPU_UUID, t.GPU_PCI, 20*2**30,
        lambda: dict(uuid=t.GPU_UUID, pci=t.GPU_PCI, bytes=0), lambda: now['wall'])
    gate.check()
    runtime.gpu = gate  # type: ignore[assignment]
    def setup(*args: Any) -> Manual:
        now['wall'] = 6.
        return runtime
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=setup, clocks=clock(now), rss=lambda: 0)
    assert result['command_status'] == 'STOPPED' and result['reason'] == 'GPU_UNVERIFIED'
    assert not runtime.calls


def test_preflight_production_device_assembly_reuses_real_boundary(monkeypatch: Any) -> None:
    from types import SimpleNamespace

    from cipher_lab_plugins.innovation1 import application, device
    t = tool()
    gate = object()
    events: list[Any] = []
    resources = SimpleNamespace(gpu=None, check=lambda: events.append('check'))
    def setup(recipe: Any, dev: str, uuid: str, pci: str) -> Any:
        events.append((recipe, dev, uuid, pci))
        return SimpleNamespace(gpu=gate)
    evidence = dict(gpu_uuid=t.GPU_UUID, pci_bus_id=t.GPU_PCI, logical_device=0,
                    pid=os.getpid(), evidence='cuda-current-context')
    monkeypatch.setattr(application, 'production_resources', setup)
    monkeypatch.setattr(device, 'actual_attestor', lambda: evidence)
    runtime = t.Production('RECIPE', resources, 'cuda:0', t.GPU_UUID, t.GPU_PCI)
    assert resources.gpu is runtime.gpu is gate
    assert runtime.evidence == evidence
    assert events == [('RECIPE', 'cuda:0', t.GPU_UUID, t.GPU_PCI), 'check']


def test_preflight_production_cpu_gpu_same_state_synthetic_only(monkeypatch: Any) -> None:
    import torch
    from cipher_lab_plugins.innovation1 import models
    from cipher_lab_plugins.innovation1.config import Recipe
    t = tool()
    events: list[Any] = []
    class ArtificialTensor:
        device = 'cpu'
        def __init__(self, value: Any) -> None:
            self.value = value
        def to(self, target: str) -> Any:
            events.append(('tensor-to', target))
            return self
        def detach(self) -> Any:
            return self
        def cpu(self) -> Any:
            return self
        def clone(self) -> Any:
            return self
    public_state = {'manual': ArtificialTensor(42)}
    class ArtificialModel:
        def state_dict(self) -> Any:
            return public_state
        def load_state_dict(self, state: Any, *, strict: bool) -> None:
            assert state == public_state and strict
            events.append('same-state')
        def eval(self) -> Any:
            return self
    class ArtificialRuntime:
        def __init__(self, recipe: Any, dev: str) -> None:
            assert dev == 'cuda:0'
            self.recipe, self.models = recipe, {'M-best': ArtificialModel()}
            events.append('production-construct')
        def predict(self, slot: str, task: Any, x: Any) -> Any:
            assert len(x.value) == 2 and len(x.value[0]) == 4
            events.append(('predict', slot, task.cipher))
            return [[0.], [0.]]
        def update(self, slot: str, task: Any, x: Any, y: Any, lr: float, resources: Any) -> float:
            assert y.value == [[0.], [1.]] and lr == 0.001
            events.append('production-update')
            return 0.7
    monkeypatch.setattr(models, 'TorchRuntime', ArtificialRuntime)
    monkeypatch.setattr(models, 'construct', lambda *a: ArtificialModel())
    monkeypatch.setattr(torch, 'tensor', lambda data, **kw: ArtificialTensor(data))
    runtime = object.__new__(t.Production)
    runtime.build(Recipe())
    task = Recipe().tasks[0]
    assert runtime.evaluate('cpu', 'M-best', task) == runtime.evaluate('cuda:0', 'M-best', task)
    assert runtime.update('M-best', task, 0.001, 'METER') == 0.7
    assert events.count('same-state') == 1 and events.count('production-update') == 1
    assert len([e for e in events if isinstance(e, tuple) and e[0] == 'predict']) == 2


def test_preflight_guarded_child_manual_full_cycle(tmp_path: Path) -> None:
    script = '''import sys,importlib.util,uuid,itertools,socket,subprocess,ctypes
from checkpoint_child_guard import install_checkpoint_guard,forbidden
install_checkpoint_guard()
import pytest
p=pytest.MonkeyPatch()
for owner,names in ((socket,('socket','create_connection')),
                    (subprocess,('Popen','run','check_output'))):
 for name in names:p.setattr(owner,name,forbidden)
p.setattr(ctypes,'CDLL',forbidden)
if hasattr(ctypes,'WinDLL'):p.setattr(ctypes,'WinDLL',forbidden)
from test_innovation1_preflight import run_manual
from pathlib import Path
_,result,_,output=run_manual(Path(sys.argv[1]))
assert result['command_status']=='PASS'
assert result['counts']['completed_forward_rows']==48
assert result['counts']['completed_real_blocks']==8
assert result['counts']['completed_optimizer_updates']==8
print('GUARDED_PREFLIGHT_MANUAL_PASS')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(str(p) for p in (
        ROOT/'src', ROOT/'plugins', ROOT/'tests', TOOL.parents[0], Path(__file__).parent)))
    proc = _POP([sys.executable, '-c', script, str(tmp_path)], cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, (out, err)
    assert 'GUARDED_PREFLIGHT_MANUAL_PASS' in out


def test_preflight_output_symlink_parent_rejected(tmp_path: Path) -> None:
    t = tool()
    real = tmp_path/'real'
    real.mkdir()
    (tmp_path/'link').symlink_to(real, target_is_directory=True)
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'link/preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *a: Manual(), clocks=clock({'wall': 0., 'cpu': 0.}), rss=lambda: 0)
    assert result['reason'] == 'OUTPUT_CLAIM_FAILED'
    assert not list(real.iterdir())


def test_preflight_budget_stop_before_evidence_write_and_size_cap(tmp_path: Path,
                                                                monkeypatch: Any) -> None:
    t = tool()
    assert len(t.INPUT_FIXTURE_BYTES) == 1026
    assert t.INPUT_FIXTURE_BYTES[-2:] == bytes([0, 1])
    with pytest.raises(ValueError):
        t._write_evidence(tmp_path/'too-big.json', b'x'*(16*2**20+1))
    assert not (tmp_path/'too-big.json').exists()
    now = {'wall': 0., 'cpu': 0., 'rss': 0.}
    original_dumps = t.json.dumps
    def dumps(value: Any, **kwargs: Any) -> str:
        if (isinstance(value, dict) and value.get('method')
                == 'innovation1-engineering-preflight-v1'):
            now['wall'] = 601.
        return str(original_dumps(value, **kwargs))
    monkeypatch.setattr(t.json, 'dumps', dumps)
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *a: Manual(), clocks=clock(now), rss=lambda: 0)
    assert result['command_status'] == 'STOPPED'
    assert result['evidence_status'] == 'STOPPED'
    assert json.loads((tmp_path/'preflight/preflight.json').read_bytes())['status'] == 'STOPPED'


def test_preflight_production_cipher_wrapper_only_public_vectors(monkeypatch: Any) -> None:
    from cipher_lab.ciphers import present, simeck, simon, skinny
    t = tool()
    calls: list[Any] = []
    vectors = json.loads(t.FIXTURE_BYTES)
    for family, module in [('SIMON', simon), ('SIMECK', simeck),
                           ('PRESENT', present), ('SKINNY', skinny)]:
        def oracle(points: Any, key: Any, rounds: int, *, name: str = family) -> Any:
            match = next(v for v in vectors if v['cipher'] == name and v['rounds'] == rounds)
            assert points == (int(match['plaintext'], 16),) and key == tuple(match['key_words'])
            calls.append((name, rounds))
            return (int(match['expected'], 16),)
        monkeypatch.setattr(module, 'encrypt_many', oracle)
    runtime = object.__new__(t.Production)
    for vector in vectors:
        assert runtime.encrypt(vector) == (int(vector['expected'], 16),)
    assert len(calls) == 8


def test_preflight_ledger_cannot_exceed_declared_budget(tmp_path: Path) -> None:
    class OverBudget(Manual):
        def update(self, slot: str, task: Any, lr: float, resources: Any) -> float:
            resources.reserve('optimizer_updates', 9)
            raise AssertionError('must stop before update')
    _, result, _, output = run_manual(tmp_path, runtime=OverBudget())
    assert result['command_status'] == 'STOPPED'
    assert result['reason'] == 'OPTIMIZER_UPDATES_LIMIT'
    saved = json.loads((output/'preflight.json').read_bytes())
    assert saved['counts']['attempted_optimizer_updates'] == 1
    assert saved['counts']['completed_optimizer_updates'] == 0


def test_preflight_device_mismatch_rejected_before_numerical_setup(tmp_path: Path) -> None:
    t = tool()
    def forbidden(*args: Any) -> Any:
        raise AssertionError('no wrong-device setup')
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid='GPU-wrong', gpu_pci=t.GPU_PCI, setup=forbidden,
        clocks=clock({'wall': 0., 'cpu': 0.}), rss=lambda: 0)
    assert result['reason'] == 'DEVICE_IDENTITY' and result['command_status'] == 'FAILED'
    assert result['counts']['attempted_real_blocks'] == 0


def test_preflight_write_failure_still_samples_postwrite_budget(tmp_path: Path,
                                                              monkeypatch: Any) -> None:
    t = tool()
    now = {'wall': 0., 'cpu': 0., 'rss': 0.}
    def write(*args: Any) -> None:
        now['wall'] = 601.
        raise OSError('PRIVATE_WRITE_DETAILS')
    monkeypatch.setattr(t, '_write_evidence', write)
    result = t.run_preflight(ROOT/'configs/innovation1-fresh.json', tmp_path/'preflight',
        device='cuda:0', gpu_uuid=t.GPU_UUID, gpu_pci=t.GPU_PCI,
        setup=lambda *a: Manual(), clocks=clock(now), rss=lambda: 0)
    assert result['counts']['wall_elapsed_seconds'] == 601.
    assert result['command_status'] == 'STOPPED' and result['reason'] == 'WALL_LIMIT'
    assert result['evidence_status'] == 'UNKNOWN'
    assert result['write_reason'] == 'EVIDENCE_WRITE_FAILED'


def test_preflight_attestor_unavailable_is_public_stop(monkeypatch: Any) -> None:
    from types import SimpleNamespace

    from cipher_lab_plugins.innovation1 import application, device
    from cipher_lab_plugins.innovation1.resources import ResourceStop
    t = tool()
    monkeypatch.setattr(application, 'production_resources',
                        lambda *a: SimpleNamespace(gpu=object()))
    def fail() -> Any:
        raise OSError('PRIVATE_DRIVER_DETAIL')
    monkeypatch.setattr(device, 'actual_attestor', fail)
    with pytest.raises(ResourceStop, match='GPU_UNVERIFIED'):
        t.Production('RECIPE', SimpleNamespace(check=lambda: None), 'cuda:0',
                     t.GPU_UUID, t.GPU_PCI)


@pytest.mark.parametrize('value', [[[float('nan')], [0.]], [[0., 1.]], [[True], [0.]]])
def test_preflight_output_strict_shape_finite_no_bool(value: Any) -> None:
    t = tool()
    with pytest.raises(t.Failure, match='OUTPUT_INVALID'):
        t._logits(value)
