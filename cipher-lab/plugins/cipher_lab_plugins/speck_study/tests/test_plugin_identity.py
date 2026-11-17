"""Source supplier and frozen dispatch: handwritten trees, never scientific work."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cipher_lab.artifacts import ArtifactError
from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe
from cipher_lab_plugins.speck_study.storage import StudyStore
from cipher_lab_plugins.speck_study.tests.test_plugin_roots import tree, wrapper


def test_saved_context_checks_fresh_same_supplier(monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab_plugins.speck_study.storage as storage

    recipe = SpeckStudyRecipe()
    source = {'cipher_lab_plugins/speck_study/codec.py': 'a'*64}
    owner = SimpleNamespace(current_source_identity=lambda: dict(source))
    store = StudyStore(owner, Path('/fake'), 'fake')  # type: ignore[arg-type]
    monkeypatch.setattr(storage, 'read_saved', lambda *args: {
        'recipe': recipe.to_dict(), 'source': dict(source),
        'summary': {'backend_identity': 'speck-study'}})
    store.verify_context(recipe, source, 'speck-study')
    owner.current_source_identity = lambda: {'cipher_lab_plugins/speck_study/codec.py': 'b'*64}
    with pytest.raises(ArtifactError, match='HASH_MISMATCH'):
        store.verify_context(recipe, source, 'speck-study')


def test_package_exact_core_own_runtime_initializers_not_other_plugin(tmp_path: Path) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    def put(root: Path, name: str) -> None:
        path = root/name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(b'# artificial only\n')
    put(core, 'cipher_lab/extra.py')
    put(plugins, 'cipher_lab_plugins/unrelated/secret.py')
    put(plugins, 'cipher_lab_plugins/speck_study/tests/not_source.py')
    put(plugins, 'cipher_lab_plugins/speck_study/runtime/freeze.py')
    package = module.package_snapshot(core, plugins)
    assert set(manifest) <= set(package)
    assert 'cipher_lab/extra.py' in package
    assert 'cipher_lab_plugins/speck_study/runtime/freeze.py' in package
    assert not any('unrelated' in key or '/tests/' in key for key in package)
    changed = plugins/'cipher_lab_plugins/speck_study/runtime/freeze.py'
    changed.write_bytes(b'# changed fake wrapper\n')
    assert module.package_snapshot(core, plugins) != package


def test_frozen_fake_cli_explicit_plugin_and_two_roots(tmp_path: Path) -> None:
    module = wrapper()
    core, plugins, manifest = tree(tmp_path, module.CLOSURE)
    # An artificial CLI proves forwarding, not core production integration.
    name = 'cipher_lab/cli.py'
    raw = b'''def main(argv):
    assert argv == ['run', '--plugin', 'cipher_lab_plugins.speck_study.cli', '--fake']
    from cipher_lab_plugins.speck_study import cli
    return cli.main(argv)
'''
    (core/name).write_bytes(raw)
    manifest[name] = hashlib.sha256(raw).hexdigest()
    name = 'cipher_lab_plugins/speck_study/cli.py'
    raw = b'''def main(argv):
    assert type(__loader__).__name__ == 'SourceOnly'
    print('fake dual-root child explicit-plugin verified')
    return 0
'''
    (plugins/name).write_bytes(raw)
    manifest[name] = hashlib.sha256(raw).hexdigest()
    destination = tmp_path/'manifest.json'
    destination.write_text(json.dumps(manifest))
    code = '''import sys,importlib.util
spec=importlib.util.spec_from_file_location('fake_entry',sys.argv[1])
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
raise SystemExit(entry.main(sys.argv[2:]))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, module.__file__, 'run',
                             str(destination), str(core), str(plugins), 'run', '--plugin',
                             'cipher_lab_plugins.speck_study.cli', '--fake'],
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'fake dual-root child explicit-plugin verified'
    assert not list(tmp_path.rglob('*.pyc'))


def test_package_refuses_symlink_directory_not_silently_omitted(tmp_path: Path) -> None:
    module = wrapper()
    core, plugins, _ = tree(tmp_path, module.CLOSURE)
    outside = tmp_path/'outside'
    outside.mkdir()
    (outside/'hidden.py').write_bytes(b'# fake only\n')
    (plugins/'cipher_lab_plugins/speck_study/linked').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        module.package_snapshot(core, plugins)


def test_isolated_frozen_fake_run_saved_codec_report(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Full production plugin behind artificial core seam; no integration claim."""
    import shutil

    from cipher_lab_plugins.speck_study.tests import test_speck_cli
    from cipher_lab_plugins.speck_study.tests.isolated_core import write_boundary
    from cipher_lab_plugins.speck_study.tests.locations import ROOT

    original_copytree = shutil.copytree
    def copytree(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_copytree(src, dst, *args, **kwargs)
        if Path(src) == ROOT/'src':
            write_boundary(Path(dst))
        return result
    monkeypatch.setattr(shutil, 'copytree', copytree)
    test_speck_cli.test_clean_source_fake_cli_check_run_read_saved_strict_codec(tmp_path)


def test_all_provider_source_bytes_py39_without_main_imports() -> None:
    import ast

    from cipher_lab_plugins.speck_study.tests.locations import CORE_ROOT, PLUGIN_ROOT
    module = wrapper('freeze')
    ast.parse(Path(module.__file__).read_bytes(), feature_version=(3,9))
    for name in module.CLOSURE:
        path = module.source_path(CORE_ROOT, PLUGIN_ROOT, name)
        parsed = ast.parse(path.read_bytes(), filename=str(path), feature_version=(3,9))
        for node in ast.walk(parsed):
            if isinstance(node, ast.ImportFrom):
                assert not any(part in (node.module or '').split('.')
                               for part in ('cli','config','execution','application'))
    root_init = ast.parse((PLUGIN_ROOT/'cipher_lab_plugins/__init__.py').read_bytes())
    assert all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
               and isinstance(node.value.value, str) for node in root_init.body)
    assert (PLUGIN_ROOT/'cipher_lab_plugins/speck_study/__init__.py').read_bytes().strip() == b''


def test_saved_report_unknown_method_cannot_be_relabelled(monkeypatch: pytest.MonkeyPatch,
                                                        capsys: Any) -> None:
    import cipher_lab.artifacts as artifacts
    from cipher_lab_plugins.speck_study import cli

    class Owner:
        def write_report_files(self, files: Any) -> None:
            raise AssertionError('unknown saved method rendered')
    monkeypatch.setattr(artifacts, 'RunStore', Owner)
    monkeypatch.setattr(artifacts, 'read_saved', lambda *args: {
        'summary': {'backend_identity':'speck-study', 'method':'other-method'},
        'recipe': {}, 'status':'SUCCEEDED'})
    code = cli._saved_report(Owner(), Path('/fake'), 'fake', 'speck-study')
    assert code == 6
    assert json.loads(capsys.readouterr().out)['reason'] == 'REPORT_METHOD_UNKNOWN'


def test_native_parent_four_manifests_share_package_supplier(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.speck_study import native_parent, sources
    from cipher_lab_plugins.speck_study.tests.locations import RUNTIME

    module = wrapper()
    provider = wrapper('freeze')
    core, plugins, _ = tree(tmp_path, tuple(set(module.CLOSURE) | set(provider.CLOSURE)))
    runtime = plugins/'cipher_lab_plugins/speck_study/runtime'
    for name in ('source.py','freeze.py','supervise.py'):
        (runtime/name).write_bytes((RUNTIME/name).read_bytes())
    manifest = tmp_path/'main.json'
    # freeze_all resolves its own verified local wrapper, while all source bytes
    # and the parent checked bootstrap paths belong to the hand-written tree.
    module.freeze_all(manifest,core,plugins)
    monkeypatch.setattr(sources,'source_identity',lambda: module.package_snapshot(core,plugins))
    frozen_loader = type('SourceOnly', (), {})()
    monkeypatch.setattr(native_parent,'__loader__',frozen_loader)
    settings = SimpleNamespace(source_manifest=manifest)
    files = native_parent.frozen_launch_files(settings,core,plugins)
    assert files['wrapper'] == runtime/'source.py'
    package = json.loads(files['package'].read_text())
    assert package == module.package_snapshot(core,plugins)
    assert 'cipher_lab_plugins/__init__.py' in package
    assert not any('/tests/' in key for key in package)
    # Keep package supplier equal while changing only independently reviewed
    # bootstrap identity: parent must reject before exec of those bytes.
    bootstrap = json.loads(files['bootstrap'].read_text())
    broken = dict(bootstrap, **{'freeze.py': '0'*64})
    files['bootstrap'].write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='bootstrap source mismatch'):
        native_parent.frozen_launch_files(settings,core,plugins)
    files['bootstrap'].write_text(json.dumps(bootstrap))
    (plugins/'cipher_lab_plugins/speck_study/codec.py').write_bytes(b'# fake changed\n')
    with pytest.raises(ValueError, match='package snapshot mismatch'):
        native_parent.frozen_launch_files(settings,core,plugins)
