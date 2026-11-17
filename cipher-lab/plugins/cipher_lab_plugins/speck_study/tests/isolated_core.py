"""Hand-assembled core boundary for isolated plugin tests, NOT production integration.

Until the serial core owner removes old eager experimental imports, this fixture
supplies ONLY real core Policy validation and dispatch. No compatibility fallback
is installed into the product; existing full integration tests remain unchanged.
"""
from __future__ import annotations

import ast
from pathlib import Path

from cipher_lab_plugins.speck_study.tests.locations import CORE_ROOT


def write_boundary(core_root: Path) -> None:
    original = (CORE_ROOT/'cipher_lab/config.py').read_text()
    tree = ast.parse(original)
    keep = {'_integer','_positive_seconds','_deadline','_object','Policy',
            '_unique_object','_reject_constant','_finite_float','_load','load_policy'}
    parts = ['from __future__ import annotations\n',
             'import json, math, re\nfrom dataclasses import dataclass\n',
             'from datetime import datetime, timedelta\nfrom pathlib import Path\n',
             'from typing import cast\n_MAX_INPUT_BYTES = 65536\n']
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in keep:
            start = min([node.lineno]+[d.lineno for d in node.decorator_list])
            parts.append('\n'.join(original.splitlines()[start-1:node.end_lineno])+'\n')
    (core_root/'cipher_lab/config.py').write_text('\n'.join(parts))
    (core_root/'cipher_lab/cli.py').write_text('''"""Artificial dispatch-only test facade."""
from cipher_lab.plugin_cli import dispatch

def main(argv):
    result = dispatch(argv)
    assert result is not None, 'explicit plugin required in artificial facade'
    return result
''')
