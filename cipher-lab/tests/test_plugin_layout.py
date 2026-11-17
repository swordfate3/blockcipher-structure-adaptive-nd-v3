"""Global source layout contracts; no scientific imports or execution."""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OWNERS = (
    'differential_relations', 'residual', 'feature_probe', 'frontier',
    'fourier', 'speck_study', 'integral_search', 'innovation1',
)


def settings() -> dict[str, Any]:
    return tomllib.loads((ROOT / 'pyproject.toml').read_text())


def test_pytest_covers_all_registered_owner_paths() -> None:
    config = settings()['tool']['pytest']['ini_options']
    assert set(config['testpaths']) == {'tests', *(
        f'plugins/cipher_lab_plugins/{owner}/tests' for owner in OWNERS)}
    assert {'src', 'plugins', 'tests'} <= set(config['pythonpath'])


def test_mypy_checks_actual_source_tests_and_plugins() -> None:
    config = settings()['tool']['mypy']
    assert set(config['files']) == {'src', 'tests', 'plugins'}
    assert set(config['mypy_path']) == {'src', 'tests', 'plugins'}
    assert all((ROOT / part).is_dir() for part in config['files'])
    assert not config.get('ignore_missing_imports', False)


def test_core_wheel_does_not_package_or_discover_plugins() -> None:
    config = settings()['tool']['setuptools']['packages']['find']
    assert config['where'] == ['src']
    assert config['include'] == ['cipher_lab', 'cipher_lab.*']
    project = settings()['project']
    assert 'entry-points' not in project


def test_explicit_entrypoints_have_main_without_importing_them() -> None:
    for owner in OWNERS:
        path = ROOT / 'plugins/cipher_lab_plugins' / owner / 'cli.py'
        tree = ast.parse(path.read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        assert len(main.args.args) == 1
        assert isinstance(main.returns, ast.Name) and main.returns.id == 'int'


def test_default_collection_excludes_science_and_installs_test_only_poison() -> None:
    config = settings()['tool']['pytest']['ini_options']
    import shlex
    args = shlex.split(config['addopts'])
    assert args[args.index('-m') + 1] == 'not science'
    assert args[args.index('-p') + 1] == 'plugin_isolation'
    assert any(mark.startswith('science:') for mark in config['markers'])
    assert (ROOT / 'tests/plugin_isolation.py').is_file()


def test_core_has_no_static_plugin_import_or_plugin_registry() -> None:
    for path in (ROOT / 'src/cipher_lab').rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith('cipher_lab_plugins') for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('cipher_lab_plugins')
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'entry_points', 'iter_entry_points'}


def test_no_duplicate_owner_test_module_names() -> None:
    paths = [*ROOT.joinpath('tests').rglob('test_*.py'),
             *ROOT.joinpath('plugins').rglob('test_*.py')]
    names = [path.stem for path in paths]
    assert len(names) == len(set(names)), 'default pytest import mode requires unique test owners'
