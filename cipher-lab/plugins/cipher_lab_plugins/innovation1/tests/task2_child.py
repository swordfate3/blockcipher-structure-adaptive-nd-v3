"""Test-only child safety boundary. Not imported by product code."""
from __future__ import annotations

import importlib.abc
import itertools
import json
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError('UNAUTHORIZED_SCIENCE_CHILD')


def main() -> None:
    mode, root = sys.argv[1], Path(sys.argv[2])
    if mode in ('help', 'check'):
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, *args: Any, **kwargs: Any) -> None:
                if fullname.split('.')[0] in ('torch', 'numpy', 'psutil', 'pynvml'):
                    forbidden()
        sys.meta_path.insert(0, Block())
        def audit(event: str, args: Any) -> None:
            if event in ('os.mkdir', 'subprocess.Popen', 'socket.__new__'):
                forbidden()
            if event == 'open' and isinstance(args[1], str) and any(
                    c in args[1] for c in ('w', 'a', '+', 'x')):
                forbidden()
        sys.addaudithook(audit)
        from cipher_lab.cli import main as cli
        try:
            code = cli(['--plugin', 'cipher_lab_plugins.innovation1.cli',
                        '--help' if mode == 'help' else 'check', *([] if mode == 'help' else
                        ['--config', 'configs/innovation1-fresh.json'])])
        except SystemExit as error:
            assert type(error.code) is int
            code = error.code
        assert code == 0
        assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    else:
        from checkpoint_child_guard import install_checkpoint_guard
        install_checkpoint_guard()
        import pytest
        patch = pytest.MonkeyPatch()
        for module, names in ((socket, ('socket', 'create_connection')),
                              (subprocess, ('Popen', 'run', 'check_output'))):
            for name in names:
                patch.setattr(module, name, forbidden)
        counter = itertools.count(1)
        patch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(counter)))
        import cipher_lab.artifacts as artifacts
        patch.setattr(artifacts, 'uuid4', lambda: uuid.UUID(int=next(counter)))
        from cipher_lab_plugins.innovation1 import resources
        patch.setattr(resources, 'production_rss_bytes', forbidden)
        patch.setattr(resources, 'gpu_sample', forbidden)
        from cipher_lab_plugins.innovation1 import device
        patch.setattr(device, 'actual_attestor', forbidden)
        import ctypes
        patch.setattr(ctypes, 'CDLL', forbidden)
        if hasattr(ctypes, 'WinDLL'):
            patch.setattr(ctypes, 'WinDLL', forbidden)
        from cipher_lab.cli import main as cli
        common = ['--plugin', 'cipher_lab_plugins.innovation1.cli']
        assert cli([*common, 'run', '--config', 'configs/innovation1-manual-demo.json',
                    '--root', str(root), '--id', 'manual', '--backend', 'demo',
                    '--device', 'cpu']) == 0
        saved: Any = artifacts.read_saved(root, 'manual')
        assert saved['status'] == 'SUCCEEDED' and saved['summary']['synthetic'] is True
        if mode == 'isolated-evaluate':
            # Remove already-imported generation closures so a warm module cache
            # cannot conceal a forbidden evaluator import. Keep actual saved bytes.
            forbidden_modules = ('cipher_lab_plugins.innovation1.data',
                                 'cipher_lab_plugins.innovation1.training',
                                 'cipher_lab_plugins.innovation1.synthetic',
                                 'cipher_lab.ciphers', 'cipher_lab.data', 'secrets')
            sys.modules.pop('secrets', None)
            for name in tuple(sys.modules):
                if name.startswith('cipher_lab_plugins.innovation1.') and name.rsplit(
                        '.', 1)[-1] in ('data', 'training', 'synthetic', 'dataset', 'scoring',
                                        'models', 'evaluate', 'manual_runtime'):
                    del sys.modules[name]
            class NoGenerators(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname: str, *args: Any, **kwargs: Any) -> None:
                    if any(fullname == name or fullname.startswith(name + '.')
                           for name in forbidden_modules):
                        forbidden()
            sys.meta_path.insert(0, NoGenerators())
            # Exercise the entire read-only closure, including the production
            # model adapter import (never construction/forward), not only demo.
            from importlib import import_module
            for name in ('batch', 'dataset', 'scoring', 'models', 'evaluate', 'manual_runtime'):
                import_module('cipher_lab_plugins.innovation1.' + name)
        cp, ds = 'model/M-best.pt', 'dataset/test-SIMECK.npz'
        assert cli([*common, 'evaluate', '--run-root', str(root), '--run-id', 'manual',
            '--run-sha256', saved['run_sha256'], '--checkpoint-ref', cp,
            '--checkpoint-sha256', saved['artifacts'][cp]['sha256'], '--dataset-ref', ds,
            '--dataset-sha256', saved['artifacts'][ds]['sha256'], '--root', str(root),
            '--id', 'explicit', '--backend', 'demo', '--device', 'cpu',
            '--max-forward-rows', '2', '--batch-groups', '2']) == 0
        if mode == 'isolated-evaluate':
            assert not any(name in sys.modules for name in forbidden_modules)
        evaluation: Any = artifacts.read_saved(root, 'explicit')
        assert evaluation['counts']['completed_real_blocks'] == 0
        assert evaluation['counts']['completed_optimizer_updates'] == 0
        assert evaluation['counts']['completed_forward_rows'] == 2
        for run_id in ('manual', 'explicit'):
            for filename in ('report.md', 'index.html', 'source.json'):
                assert (root / 'result' / run_id / filename).is_file()
        print('MANUAL_CLOSED_LOOP_OK', json.dumps(saved['counts'], sort_keys=True))
    print('CHILD_GUARD_OK')
