# ruff: noqa: UP007, UP045
"""Stdlib launcher + synthetic selfcheck, Python 3.9 syntax safe.

Provider uses only a tiny test array adapter: NO NumPy runtime/DLL bootstrap.
Main's existing fake application still uses CPU NumPy under its own guards.

Examples (root MUST be new, exclusively owned by this invocation):
  python -B tests/speck_portable_selfcheck.py --mode main --rootowned <new-path>
  python -B tests/speck_portable_selfcheck.py --mode provider --rootowned <new-path>
No pytest, installation, real native parent, weights, cipher, entropy or NN.
The driver is not a security sandbox: reviewed test code has explicit fake seams.
"""
from __future__ import annotations

import argparse
import builtins
import ctypes
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import random
import secrets
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'synthetic-only; formal R0 compatibility NOT verified'


class Traps(importlib.abc.MetaPathFinder):
    """Each interpreter installs this BEFORE any project/NumPy import."""

    def __init__(self, *, check_only: bool = False, deny_numpy: bool = False) -> None:
        self.violations = 0
        self.check_only = check_only
        self.deny_numpy = deny_numpy

    def forbidden(self, *args: Any, **kwargs: Any) -> Any:
        self.violations += 1
        raise AssertionError('portable selfcheck forbidden real science/native operation')

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        top = fullname.split('.')[0]
        if (top in ('pytest', 'torch', 'tensorflow', 'keras', 'h5py', 'psutil', 'pynvml')
                or fullname == 'cipher_lab.ciphers'
                or fullname.startswith(('cipher_lab.ciphers.', 'cipher_lab.training.'))
                or (self.deny_numpy and top == 'numpy')
                or (self.check_only and (top == 'numpy' or fullname in (
                    'cipher_lab.artifacts', 'cipher_lab.execution')))):
            self.forbidden()
        return None

    def install(self) -> None:
        assert not any(n == 'cipher_lab' or n.startswith('cipher_lab.') for n in sys.modules)
        assert not any(n in sys.modules for n in ('pytest', 'numpy', 'torch', 'tensorflow'))
        sys.dont_write_bytecode = True
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        sys.meta_path.insert(0, self)
        for name in ('CDLL', 'WinDLL', 'OleDLL', 'PyDLL'):
            setattr(ctypes, name, self.forbidden)
        for name in ('cdll', 'windll', 'oledll', 'pydll'):
            setattr(ctypes, name, SimpleNamespace(LoadLibrary=self.forbidden))
        setattr(subprocess, 'Popen', self.forbidden)
        setattr(subprocess, 'run', self.forbidden)
        os.system = self.forbidden
        for name in ('fork', 'forkpty', 'posix_spawn', 'posix_spawnp', 'startfile'):
            if hasattr(os, name):
                setattr(os, name, self.forbidden)
        # NumPy may initialize its unused random module on import. All entropy
        # inputs here are synthetic zeros, then all random entrypoints are traps.
        os.urandom = lambda size: bytes(size)
        random._urandom = lambda size: bytes(size)  # type: ignore[attr-defined]
        secrets.randbits = lambda k: 0
        real_open = builtins.open
        real_io_open = io.open

        def checked_open(original: Any, file: Any, *args: Any, **kwargs: Any) -> Any:
            mode = args[0] if args else kwargs.get('mode', 'r')
            if (isinstance(file, (str, os.PathLike))
                    and str(file).lower().endswith(('.h5', '.hdf5')) and 'r' in mode):
                self.forbidden()
            return original(file, *args, **kwargs)

        builtins.open = lambda file, *a, **kw: checked_open(real_open, file, *a, **kw)
        io.open = lambda file, *a, **kw: checked_open(real_io_open, file, *a, **kw)
        try:
            import resource
        except ImportError:
            pass
        else:
            resource.getrusage = self.forbidden

    def finish_entropy(self) -> None:
        for module in (os, random, secrets):
            for name in ('urandom', '_urandom', 'SystemRandom', 'token_bytes', 'token_hex',
                         'token_urlsafe', 'randbits', 'randbelow', 'choice', 'seed', 'random',
                         'randint', 'randrange', 'getrandbits'):
                if hasattr(module, name):
                    setattr(module, name, self.forbidden)
        if 'numpy.random' in sys.modules:
            module = sys.modules['numpy.random']
            for name in ('default_rng', 'PCG64', 'seed', 'rand', 'randn', 'randint', 'random',
                         'random_sample', 'bytes', 'RandomState', 'Generator'):
                setattr(module, name, self.forbidden)


def load_entry(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True)
        stream.write('\n')


def policy(root: Path) -> Path:
    path = root / 'policy.json'
    write_json(path, dict(schema=2, wall_seconds=28800, cpu_seconds=1,
        deadline_utc='2099-01-01T00:00:00Z', real_blocks=542000000, null_words=0,
        rss_stop_bytes=32*2**30, output_bytes=4*2**30, optimizer_updates=0,
        forward_rows=1161000000))
    return path


def worker_check(root: Path, traps: Traps) -> int:
    traps.finish_entropy()
    bootstrap = load_entry('portable_main_source', ROOT / 'tools/speck-study/source.py')
    manifest = root / 'main.json'
    bootstrap.freeze_all(manifest, ROOT / 'src')
    counts = [len(json.loads(Path(str(manifest) + suffix).read_text())) for suffix in (
        '', '.provider.json', '.bootstrap.json', '.package.json')]
    assert counts == [36, 9, 3, 74]
    original = bootstrap.SourceOnly

    class GuardedSourceOnly(original):  # type: ignore[valid-type,misc]
        def exec_module(self, module: Any) -> None:
            traps.find_spec(module.__name__)
            super().exec_module(module)

    bootstrap.SourceOnly = GuardedSourceOnly
    assert bootstrap.main(['run', str(manifest), str(ROOT / 'src'), 'check',
        '--config', str(ROOT / 'configs/speck-study-paper.json'),
        '--policy', str(root / 'policy.json')]) == 0
    assert 'numpy' not in sys.modules and traps.violations == 0
    return 3


def worker_chain(root: Path, traps: Traps) -> int:
    import platform
    import runpy

    # Do NOT modify os.name or pretend this is a real Windows interpreter.
    # Require the driver itself to select the fake child before CLI dispatch.
    os.environ.pop('SPECK_STUDY_NATIVE', None)

    # uuid's Linux-only import branch asks platform.system once; give it an
    # explicit synthetic answer, without uname/ver. Subsequent claim still has
    # to use the driver's own synthetic identity (tested by a RED revert).
    platform.system = lambda: 'Windows'
    import uuid
    assert uuid.UUID(int=1).int == 1
    platform.system = traps.forbidden
    platform.machine = traps.forbidden
    weights = root / 'fake.h5'
    weights.write_bytes(b'FAKE NEVER LOAD: not weights')
    # These public pins are validated by the real LaunchSettings, not asset reads.
    namespace: dict[str, Any] = {}
    launch = ROOT / 'src/cipher_lab/speck_study_launch.py'
    # Read constants via AST: no early project import before source freeze.
    import ast
    for node in ast.parse(launch.read_bytes()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        'OFFICIAL_REVISION', 'OFFICIAL_WEIGHTS_SHA256'):
                    namespace[target.id] = ast.literal_eval(node.value)
    driver = ROOT / 'tests/speck_cli_child.py'
    sys.argv = [str(driver), str(ROOT / 'tools/speck-study/source.py'), str(ROOT / 'src'),
        str(root / 'chain-main.json'), 'run', '--config',
        str(ROOT / 'configs/speck-study-paper.json'), '--policy', str(root / 'policy.json'),
        '--backend', 'speck-study', '--root', str(root / 'out'), '--id', 'portable-fake',
        '--teacher-python', str(Path(sys.executable).resolve()), '--teacher-weights', str(weights),
        '--weights-revision', namespace['OFFICIAL_REVISION'],
        '--weights-sha256', namespace['OFFICIAL_WEIGHTS_SHA256'],
        '--gpu-uuid', 'GPU-12345678-1234-1234-1234-123456789abc',
        '--gpu-pci', '0000:01:00.0', '--source-manifest', str(root / 'chain-main.json')]
    # The existing driver has additional own traps and literal fake entropy;
    # its production SourceOnly assembly performs durable_model/read_saved.
    try:
        runpy.run_path(str(driver), run_name='__main__')
    except SystemExit as exc:
        assert exc.code == 0, 'fake source chain failed'
    else:
        raise AssertionError('driver must return an explicit exit code')
    saved = json.loads((root / 'out/data/portable-fake/run.json').read_text())
    assert saved['environment']['os'] == 'Windows'
    assert saved['environment']['architecture'] == 'AMD64'
    assert saved['summary']['scientific_validation'] is False
    assert (root / 'out/models/portable-fake/A-best.pt').is_file()
    assert (root / 'out/result/portable-fake/index.html').is_file()
    assert traps.violations == 0
    return 5


class FakeTF:
    """Graph construction recorder only: no TensorFlow module/version/inference."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.keras = SimpleNamespace(layers=self, regularizers=SimpleNamespace(l2=lambda x: x),
                                     models=SimpleNamespace(Model=lambda **kw: kw))

    def __getattr__(self, name: str) -> Any:
        def layer(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return object() if name == 'Input' else lambda value: object()
        return layer


class ManualByteArray:
    """Only the byte reshape contract used by production points_to_bits."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def reshape(self, rows: int, width: int) -> ManualByteArray:
        assert rows == -1 and width == 8 and len(self.payload) % 8 == 0
        return self


class ManualBitRow(list[int]):
    def tolist(self) -> list[int]:
        return list(self)


def manual_points_to_bits(convert: Any, payload: bytes) -> list[ManualBitRow]:
    """Scoped test-only NumPy API; never imports/initializes the real package.

    Intentionally tiny, NOT a replacement NumPy library. The production function
    still checks packed points before frombuffer and selects reshape/axis/dtype.
    A local fake module exists only for that synchronous call, then is removed.
    """
    assert not any(n == 'numpy' or n.startswith('numpy.') for n in sys.modules)
    fake = ModuleType('numpy')
    marker = object()

    def frombuffer(raw: bytes, dtype: Any) -> ManualByteArray:
        assert type(raw) is bytes and dtype is marker
        return ManualByteArray(raw)

    def unpackbits(array: ManualByteArray, axis: int) -> list[ManualBitRow]:
        assert axis == 1
        return [ManualBitRow((byte >> shift) & 1 for byte in array.payload[start:start+8]
                             for shift in range(7, -1, -1))
                for start in range(0, len(array.payload), 8)]

    fake.uint8 = marker  # type: ignore[attr-defined]
    fake.frombuffer = frombuffer  # type: ignore[attr-defined]
    fake.unpackbits = unpackbits  # type: ignore[attr-defined]
    sys.modules['numpy'] = fake
    try:
        result: list[ManualBitRow] = convert(payload)
        return result
    finally:
        assert sys.modules.pop('numpy') is fake


def provider_cases() -> int:
    from cipher_lab.speck_study import teacher_protocol as wire
    from cipher_lab.speck_study.teacher_network import make_official_teacher, points_to_bits
    from cipher_lab.speck_study.teacher_provider import serve

    tf = FakeTF()
    make_official_teacher(tf)
    regular = {'kernel_regularizer': .0001}
    conv = dict(regular, padding='same')
    expected = [
        ('Input', (), {'shape': (64,)}), ('Reshape', ((4, 16),), {}),
        ('Permute', ((2, 1),), {}),
        ('Conv1D', (32,), dict(conv, kernel_size=1)),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}),
        ('Conv1D', (32,), dict(conv, kernel_size=3)),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}),
        ('Conv1D', (32,), dict(conv, kernel_size=3)),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}), ('Add', (), {}),
        ('Flatten', (), {}), ('Dense', (64,), regular),
        ('BatchNormalization', (), {}), ('Activation', ('relu',), {}),
        ('Dense', (64,), regular), ('BatchNormalization', (), {}),
        ('Activation', ('relu',), {}), ('Dense', (1,), dict(regular, activation='sigmoid'))]
    assert tf.calls == expected
    point = 0x8001400280048008
    # Asymmetric multirow inputs catch row stride, endian and MSB order errors.
    points = [point, 0, (1 << 64) - 1, 1, 1 << 63]
    bits = manual_points_to_bits(points_to_bits, wire.pack_points(points))
    assert [row.tolist() for row in bits] == [[int(x) for x in f'{p:064b}'] for p in points]
    assert wire.unpack_reply(wire.pack_reply([0., .5, .75, 1.]), 4) == (
        (0., .5, .75, 1.), (0, 0, 1, 1))
    checks = 3

    def rejected(fn: Any, error: Any = wire.ProtocolError) -> None:
        nonlocal checks
        try:
            fn()
        except error:
            checks += 1
        else:
            raise AssertionError('malformed protocol accepted')

    def encode(seq: int = 1) -> bytes:
        output = io.BytesIO()
        wire.write_frame(output, wire.Frame(wire.REQUEST, seq, 1, bytes(8)),
                         time.monotonic() + 3)
        return output.getvalue()

    good = encode()
    for raw in (b'', good[:3], good[:-1], b'X' + good[1:], good[:-1] + b'X'):
        rejected(lambda: wire.read_frame(io.BytesIO(raw), time.monotonic() + 3))
    rejected(lambda: wire.read_frame(io.BytesIO(good), time.monotonic() + 3, expected_seq=2))
    rejected(lambda: wire.pack_points([0] * (wire.MAX_ROWS + 1)))
    rejected(lambda: wire.unpack_reply(struct.pack('>fB', .5, 1), 1))
    rejected(lambda: wire.unpack_reply(struct.pack('>fB', .75, 0), 1))
    rejected(lambda: wire.unpack_reply(struct.pack('>fB', float('nan'), 0), 1))
    rejected(lambda: wire.read_frame(io.BytesIO(good), time.monotonic() - 1), TimeoutError)
    oversized = wire.HEADER.pack(wire.MAGIC, wire.VERSION, wire.REQUEST, 1,
                                wire.MAX_ROWS + 1, 0xFFFFFFFF, bytes(32))
    rejected(lambda: wire.read_frame(io.BytesIO(oversized), time.monotonic() + 3))

    class Partial(io.BytesIO):
        def read(self, size: Any = -1) -> bytes:
            return super().read(3 if size is None or size < 0 else min(3, size))

        def write(self, data: Any) -> int:
            return super().write(data[:3])

    partial = Partial()
    wire.write_frame(partial, wire.Frame(wire.REQUEST, 1, 1, bytes(8)), time.monotonic() + 3)
    partial.seek(0)
    assert wire.read_frame(partial, time.monotonic() + 3).payload == bytes(8)
    checks += 1
    # Real serve loop, two sequential fake requests, then deliberate EOF. No NN.
    requests = io.BytesIO(encode(1) + encode(2))
    replies = io.BytesIO()
    calls: list[bytes] = []

    def predict(payload: bytes) -> list[float]:
        calls.append(payload)
        return [.5] if len(calls) == 1 else [.75]

    ready = {'fake_only': True, 'tensorflow': 'not imported'}
    rejected(lambda: serve(requests, replies, time.monotonic() + 3, ready, predict))
    assert len(calls) == 2
    replies.seek(0)
    wire.read_frame(replies, time.monotonic() + 3, expected_kind=wire.READY, expected_seq=0)
    for seq, label in ((1, 0), (2, 1)):
        frame = wire.read_frame(replies, time.monotonic() + 3,
                                expected_kind=wire.REPLY, expected_seq=seq)
        assert wire.unpack_reply(frame.payload, 1)[1] == (label,)
    # Wrong sequence is rejected BEFORE fake prediction; no resynchronisation.
    rejected(lambda: serve(io.BytesIO(encode(2)), io.BytesIO(), time.monotonic() + 3,
                           ready, lambda payload: (_ for _ in ()).throw(AssertionError())))
    return checks + 2


def worker_provider(root: Path, traps: Traps) -> int:
    bootstrap = load_entry('portable_provider_source', ROOT / 'tools/speck-study/freeze.py')
    manifest = {n: hashlib.sha256((ROOT / 'src' / n).read_bytes()).hexdigest()
                for n in bootstrap.CLOSURE}
    assert len(manifest) == 9
    path = root / 'provider.json'
    write_json(path, manifest)
    bad = dict(manifest)
    bad[next(iter(bad))] = '0' * 64
    try:
        bootstrap.checked_sources(ROOT / 'src', bad)
    except ValueError:
        pass
    else:
        raise AssertionError('bad source identity accepted')
    assert not any(n.startswith('cipher_lab') for n in sys.modules)
    original = bootstrap.SourceOnly
    result = [0]

    def fake_main(attestor: Any) -> int:
        assert callable(attestor)  # not called: would access actual CUDA/native
        traps.deny_numpy = True
        traps.finish_entropy()
        result[0] = provider_cases()
        return 0

    class GuardedSourceOnly(original):  # type: ignore[valid-type,misc]
        def exec_module(self, module: Any) -> None:
            traps.find_spec(module.__name__)
            super().exec_module(module)
            if module.__name__ == 'cipher_lab.speck_study.teacher_provider':
                module.main = fake_main

    bootstrap.SourceOnly = GuardedSourceOnly
    # Exercise the actual frozen provider entry, not an argparse usage exit2.
    assert bootstrap.teacher(path, ROOT / 'src', []) == 0
    assert result[0] == 20 and traps.violations == 0
    assert not any(n == 'numpy' or n.startswith('numpy.') for n in sys.modules)
    for name, module in tuple(sys.modules.items()):
        if name == 'cipher_lab' or name.startswith('cipher_lab.'):
            assert isinstance(module.__loader__, GuardedSourceOnly), name
    assert not any(n in sys.modules for n in ('pytest', 'tensorflow', 'torch', 'h5py'))
    return result[0] + 2


def metadata() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in ('numpy', 'tensorflow', 'keras', 'pytest'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'MISSING'
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('main', 'provider'))
    parser.add_argument('--rootowned', required=True, type=Path)
    parser.add_argument('--worker', choices=('check', 'chain', 'provider'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.rootowned.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('rootowned symlink forbidden')
    if args.worker:
        assert root.is_dir() and (root / 'owned.json').is_file()
        traps = Traps(check_only=args.worker == 'check', deny_numpy=args.worker != 'chain')
        traps.install()
        checks = {'check': worker_check, 'chain': worker_chain,
                  'provider': worker_provider}[args.worker](root, traps)
        assert traps.violations == 0
        write_json(root / (args.worker + '-result.json'),
                   {'checks': checks, 'guard_violations': traps.violations})
        return 0

    # Never adopt an existing root or delete evidence. Temporary/cache paths
    # remain inside the explicitly new root; no tempfile default locations.
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'owned.json', {'scope': SCOPE, 'mode': args.mode})
    for name in ('temp', 'cache'):
        (root / name).mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', TEMP=str(root / 'temp'),
                      TMP=str(root / 'temp'), TMPDIR=str(root / 'temp'),
                      XDG_CACHE_HOME=str(root / 'cache'),
                      PYTHONPYCACHEPREFIX=str(root / 'cache'))
    launch = subprocess.run  # sole reviewed process seam: our guarded workers
    traps = Traps()
    traps.install()
    traps.finish_entropy()
    actual = metadata()  # metadata only, never import TF or claim compatibility
    if args.mode == 'main':
        policy(root)
    workers = ('check', 'chain') if args.mode == 'main' else ('provider',)
    total = 0
    for worker in workers:
        # subprocess.run uses module-global Popen; supply the saved primitive
        # only within this reviewed launcher, then immediately restore trap.
        raise_if = subprocess.Popen
        setattr(subprocess, 'Popen', ORIGINAL_POPEN)
        try:
            child = launch([sys.executable, '-B', str(Path(__file__).resolve()),
                            '--mode', args.mode, '--rootowned', str(root), '--worker', worker],
                           cwd=ROOT, env={**os.environ, 'PYTHONPATH': ''},
                           capture_output=True, text=True, timeout=30)
        finally:
            setattr(subprocess, 'Popen', raise_if)
        (root / (worker + '.stdout.log')).write_text(child.stdout, encoding='utf8')
        (root / (worker + '.stderr.log')).write_text(child.stderr, encoding='utf8')
        if child.returncode != 0:
            raise AssertionError('guarded worker failed: ' + worker + '; see owned logs')
        record = json.loads((root / (worker + '-result.json')).read_text())
        assert record['guard_violations'] == 0
        total += record['checks']
    assert not list(root.rglob('*.pyc'))
    summary = dict(status='通过', mode=args.mode, checks=total, scope=SCOPE,
        actual_interpreter=sys.executable, actual_version=sys.version,
        actual_platform={'os_name': os.name, 'sys_platform': sys.platform},
        actual_metadata_only=actual,
        synthetic_environment_identity={'os': 'Windows', 'architecture': 'AMD64'}
            if args.mode == 'main' else None,
        counts={'real_blocks': 0, 'real_labels': 0, 'real_nn_forward': 0,
                'real_key_entropy': 0, 'real_native_calls': 0},
        guards={'installed': True, 'worker_count': len(workers), 'violations': traps.violations},
        limitation='仅合成工程验证；不验证实际TF版本兼容、Windows原生资源或正式R0')
    write_json(root / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


ORIGINAL_POPEN: Any = subprocess.Popen

if __name__ == '__main__':
    raise SystemExit(main())
