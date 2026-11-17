"""Task2-only fake graph and guarded executable IPC fixture.

Every exec installs its OWN guards; never assumes inherited monkeypatches.
"""
from __future__ import annotations

import builtins
import importlib.abc
import json
import os
import random
import secrets
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any


class FakeTF:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.gpu_calls: list[Any] = []
        self.keras = SimpleNamespace(layers=self, regularizers=SimpleNamespace(l2=lambda x: x),
                                     models=SimpleNamespace(Model=lambda **kw: kw))

    def __getattr__(self, name: str) -> Any:
        def layer(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            if name == 'Input':
                return object()
            return lambda value: object()
        return layer


def install_teacher_guards(patch: Any) -> None:
    """Trap real TF, H5 and scientific modules in THIS interpreter."""
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('Task2 forbids scientific execution')

    # NumPy's lazy random import can ask for entropy. Feed synthetic bytes first,
    # restore that temporary shim, then poison every actual randomness API.
    real_urandom, real_randbits = os.urandom, secrets.randbits
    patch.setattr(os, 'urandom', lambda size: bytes(size))
    patch.setattr(secrets, 'randbits', lambda bits: 0)
    import numpy as np
    importlib.import_module('numpy.random')
    patch.setattr(os, 'urandom', real_urandom)
    patch.setattr(secrets, 'randbits', real_randbits)
    for module in (os, random, secrets):
        for name in ('urandom', '_urandom', 'SystemRandom', 'token_bytes', 'token_hex',
                     'token_urlsafe', 'randbits', 'randbelow', 'choice'):
            if hasattr(module, name):
                patch.setattr(module, name, forbidden)
    for name in ('default_rng', 'seed', 'rand', 'randn', 'randint', 'random',
                 'random_sample', 'bytes', 'RandomState', 'Generator'):
        patch.setattr(np.random, name, forbidden)
    try:
        import resource
    except ImportError:
        pass
    else:
        patch.setattr(resource, 'getrusage', forbidden)
    # Do not import Torch/NN/ciphers merely to poison them; block new imports.
    # For a combined selected suite, already-loaded modules are trapped too.
    for module_name, loaded in tuple(sys.modules.items()):
        if module_name.startswith('cipher_lab.ciphers.'):
            for name in ('encrypt', 'encrypt_many', 'encrypt_batch', 'encrypt_scalar',
                         'encrypt_tensor', 'expand_key', 'key_schedule'):
                if hasattr(loaded, name):
                    patch.setattr(loaded, name, forbidden)
    torch = sys.modules.get('torch')
    if torch is not None:
        patch.setattr(torch.nn.Module, '__call__', forbidden)
        patch.setattr(torch.nn.Module, '_call_impl', forbidden)
        for cls in (torch.optim.Optimizer, torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
            patch.setattr(cls, '__init__', forbidden)
            patch.setattr(cls, 'step', forbidden)
        for name in ('init', '_lazy_init', 'is_available', 'device_count', 'set_device',
                     'current_device', 'get_device_properties', 'memory_allocated'):
            patch.setattr(torch.cuda, name, forbidden)
    real_open = builtins.open

    def safe_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)) and str(file).lower().endswith(('.h5', '.hdf5')):
            forbidden()
        return real_open(file, *args, **kwargs)

    patch.setattr(builtins, 'open', safe_open)

    class BlockScience(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if (fullname.split('.')[0] in ('tensorflow', 'keras', 'h5py', 'psutil',
                                          'pynvml', 'torch')
                    or fullname.startswith('cipher_lab.ciphers')
                    or fullname == 'cipher_lab.execution'):
                forbidden()
            return None

    patch.setattr(sys, 'meta_path', [BlockScience(), *sys.meta_path])
    original_import_module = importlib.import_module
    def guarded_import(name: str, package: Any = None) -> Any:
        BlockScience().find_spec(name)
        return original_import_module(name, package)
    patch.setattr(importlib, 'import_module', guarded_import)



def child() -> None:
    import pytest
    patch = pytest.MonkeyPatch()
    install_teacher_guards(patch)
    from cipher_lab_plugins.speck_study import teacher_protocol as wire
    from cipher_lab_plugins.speck_study.teacher_network import ALLOCATOR_MIB, MODEL_SPEC
    from cipher_lab_plugins.speck_study.teacher_provider import parser, serve
    args = parser().parse_args()
    mode = os.environ.get('SPECK_FAKE_MODE', 'normal')
    if hasattr(signal, 'SIGTERM') and mode in ('hang', 'partial', 'stderr-flood'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # Test-side evidence only. Explicitly not actual GPU proof.
    ready = {'python': '3.9.synthetic', 'tensorflow': 'synthetic-no-TF',
             'model': MODEL_SPEC, 'weights_sha256': args.sha256,
             'gpu': {'fake_only': True, 'pid': os.getpid(), 'gpu_uuid': args.gpu_uuid,
                     'pci_bus_id': args.gpu_pci}, 'allocator_mib': ALLOCATOR_MIB}
    if mode == 'bad-ready':
        ready['weights_sha256'] = '0' * 64
    if mode == 'early-exit':
        return
    if mode in ('partial-ready', 'hang-ready'):
        if mode == 'partial-ready':
            os.write(1, b'SPK')
        time.sleep(30)
        return
    if mode == 'stderr-flood':
        for _ in range(300):
            os.write(2, b'x' * 8192)
    if mode in ('partial', 'hang', 'bad-seq', 'bad-reply', 'wrong-count', 'extra',
                'no-read', 'stdout-flood', 'exit-request'):
        wire.write_frame(sys.stdout.buffer,
                         wire.Frame(wire.READY, 0, 0, json.dumps(ready).encode()), args.deadline)
        if mode == 'no-read':
            time.sleep(30)
            return
        request = wire.read_frame(sys.stdin.buffer, args.deadline)
        if mode == 'exit-request':
            return
        if mode == 'stdout-flood':
            for _ in range(300):
                os.write(1, b'x' * 8192)
        if mode == 'partial':
            os.write(1, b'SPK')
        elif mode in ('bad-seq', 'wrong-count', 'bad-reply', 'extra'):
            count = request.count + int(mode == 'wrong-count')
            reply = wire.pack_reply([.5] * count)
            if mode == 'bad-reply':
                reply = reply[:-1] + b'\x01'
            wire.write_frame(sys.stdout.buffer,
                             wire.Frame(wire.REPLY, request.seq + int(mode == 'bad-seq'),
                                        count, reply), args.deadline)
            if mode == 'extra':
                os.write(1, b'UNSOLICITED')
        time.sleep(30)
        return

    def predict(payload: bytes) -> list[float]:
        # Literal deterministic formula, no NN/cipher/random sampling.
        return [float(point & 3) / 4 for point in wire.unpack_points(payload)]

    serve(sys.stdin.buffer, sys.stdout.buffer, args.deadline, ready, predict)


if __name__ == '__main__':
    try:
        child()
    except BaseException:
        os._exit(2)
    os._exit(0)
