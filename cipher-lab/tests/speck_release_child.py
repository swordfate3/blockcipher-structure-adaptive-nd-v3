"""Standalone child: guard first, only inspect CLI assembly, no panels/science."""
import ctypes
import importlib.abc
import json
import os
import secrets
import subprocess
import sys
import types
from pathlib import Path


def forbidden(*args, **kwargs):
    raise AssertionError('candidate contract forbids science/probes/commands/entropy')


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in ('torch', 'numpy', 'tensorflow', 'h5py')
                or fullname == 'cipher_lab.ciphers' or fullname.startswith('cipher_lab.ciphers.')
                or fullname.startswith(('cipher_lab.training.', 'cipher_lab.data.'))):
            forbidden()
        return None


sys.meta_path.insert(0, Guard())
os.urandom = forbidden
secrets.token_bytes = forbidden
secrets.randbits = forbidden
subprocess.run = forbidden
subprocess.Popen = forbidden
os.system = forbidden
ctypes.CDLL = forbidden
ctypes.WinDLL = forbidden

from cipher_lab import artifacts, cli, demo, execution  # noqa: E402
from cipher_lab.config import Policy, load_experiment  # noqa: E402

mode = sys.argv[1]
result = {'mode': mode, 'collector_calls': 0, 'training_calls': 0}
policy = Policy(2, 28800, 1, '2099-01-01T00:00:00Z', 542000000, 0,
                32*2**30, 4*2**30, 0, 1161000000)


class Reached(Exception):
    pass


class Store:
    def write_snapshot(self, value):
        pass

    def finalize(self, value):
        result['terminal'] = value


store = Store()

def claim(*args, **kwargs):
    result['needs_models'] = kwargs['needs_models']
    return store


artifacts.RunStore.claim = claim
execution.Meter = lambda *args, **kwargs: types.SimpleNamespace(snapshot=lambda: {})
execution.production_rss_bytes = forbidden
cli._saved_report = lambda *args, **kwargs: 0

if mode == 'integral':
    module = types.ModuleType('cipher_lab.data.cubes')

    def collector(meter, oracle, private_random, chunk_points):
        assert oracle is demo.fake_encrypt_many
        assert type(private_random) is demo.DemoBytes
        assert chunk_points == 4
        result['collector_calls'] += 1
        raise Reached()

    module.CubeCollector = collector
    sys.modules[module.__name__] = module
    recipe = load_experiment(Path('configs/integral-example.json'))
    cli._run(recipe, policy, Path('/NEVER-CREATE'), 'sentinel', 'demo')
    assert result['collector_calls'] == 1, result
    assert result['needs_models'] is False
elif mode in ('prediction', 'shared-m'):
    module = types.ModuleType('cipher_lab.training.application')

    def prepare(recipe, meter, owner, backend):
        assert owner is store and backend == 'demo'
        result['training_calls'] += 1
        raise Reached()

    module.prepare_training = prepare
    sys.modules[module.__name__] = module
    recipe = load_experiment(Path('configs/'+mode+'-example.json'))
    cli._run(recipe, policy, Path('/NEVER-CREATE'), 'sentinel', 'demo')
    assert result['training_calls'] == 1, result
    assert result['needs_models'] is True
else:
    raise AssertionError(mode)
print(json.dumps(result, sort_keys=True))
