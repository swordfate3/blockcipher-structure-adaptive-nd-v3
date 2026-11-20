"""Fresh child check: audit writes, external execution and scientific imports."""
from __future__ import annotations

import importlib.abc
import os
import sys
from typing import Any


def forbidden() -> None:
    raise AssertionError('UNAUTHORIZED_SCIENCE_CHILD')


def main() -> None:
    mode = sys.argv[1]
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, *args: Any, **kwargs: Any) -> None:
            if (fullname.split('.')[0] in ('torch', 'numpy', 'psutil', 'pynvml')
                    or fullname.startswith('cipher_lab.ciphers')):
                forbidden()

    sys.meta_path.insert(0, Block())

    def audit(event: str, args: Any) -> None:
        if event in ('os.mkdir', 'subprocess.Popen', 'socket.__new__'):
            forbidden()
        if event == 'open' and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            forbidden()

    sys.addaudithook(audit)
    from cipher_lab.cli import main as cli

    args = ['--help'] if mode == 'help' else ['check', '--config', f'configs/{mode}.json']
    assert cli(['--plugin', 'cipher_lab_plugins.innovation1.cli', *args]) == 0
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print('CHILD_GUARD_OK')
