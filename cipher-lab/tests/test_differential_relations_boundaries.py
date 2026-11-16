"""Architecture regression guards for the differential-relations plugin.

Every check here is AST/text inspection or one standard-library-only subprocess.
No torch import, no tensor forward, no cipher, no optimizer step, no real
RSS/GPU probe, no entropy draw, no SSH, no install and no Git operation.

Scope of this file is the already-frozen plugin modules (config, data, metrics,
checkpoint, resources, report) plus the pure core model they train. Checks whose
subject is still being written by their owner (``cli.py``, ``application.py``,
``training.py``, ``evaluate.py``) are present with exact file paths and names
starting with ``test_pending_``; they are executed only after those files are
final, never skipped and never silently dropped.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT / "src" / "cipher_lab"
PLUGIN = PROJECT / "plugins" / "cipher_lab_plugins" / "differential_relations"
OFFICIAL_CONFIG = PROJECT / "configs" / "differential-relations-simon6.json"
CORE_PLUGIN_CLI = CORE / "plugin_cli.py"

# Roots that a configuration/report/metric module must never reach.
SCIENCE = ("numpy", "torch", "cipher_lab", "cipher_lab_plugins")
STDLIB = set(sys.stdlib_module_names) | {"__future__"}
DYNAMIC = {"__import__", "importlib.import_module", "import_module", "exec", "eval",
           "importlib.reload", "runpy.run_module"}

# Files that still belong to their writing owner; not exercised in this round.
PENDING_FILES = ("application.py", "cli.py", "training.py", "evaluate.py")
CLI_MODULE = "cipher_lab_plugins.differential_relations.cli"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imported(tree: ast.AST) -> set[str]:
    """Absolute imports verbatim; relative ones keep their leading dots."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def _hits(names: set[str], roots: tuple[str, ...]) -> set[str]:
    """Exact module or dotted descendant; ``cipher_lab.training_config`` is unrelated."""
    return {name for name in names
            if any(name == root or name.startswith(root + ".") for root in roots)}


def _calls(tree: ast.AST) -> set[str]:
    return {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}


def _enclosing_function(tree: ast.AST, node: ast.AST) -> ast.AST | None:
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    parent = parents.get(node)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parent = parents.get(parent)
    return parent


def test_config_is_stdlib_only_and_reads_only_the_given_file() -> None:
    tree = _tree(PLUGIN / "config.py")
    names = _imported(tree)
    assert {name.split(".")[0] for name in names} <= STDLIB, names
    assert not _hits(names, SCIENCE)
    calls = _calls(tree)
    assert not calls & ({"open", "print", "input", "urandom", "os.urandom"} | DYNAMIC)
    # The single read is the caller-named config path inside load_config.
    readers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == "path.read_text"]
    assert len(readers) == 1
    owner = _enclosing_function(tree, readers[0])
    assert isinstance(owner, ast.FunctionDef) and owner.name == "load_config"
    assert not [call for call in calls if call.endswith(
        (".write_text", ".write_bytes", ".mkdir", ".unlink", ".read_bytes"))]


def test_report_is_stdlib_only_and_renders_from_its_argument_only() -> None:
    tree = _tree(PLUGIN / "report.py")
    names = _imported(tree)
    assert {name.split(".")[0] for name in names} <= STDLIB, names
    assert not _hits(names, SCIENCE)
    calls = _calls(tree)
    assert not calls & ({"open", "print", "input"} | DYNAMIC)
    assert not [call for call in calls if call.endswith(
        (".read_text", ".read_bytes", ".write_text", ".write_bytes",
         ".save", ".savez", ".load"))]
    render = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "render_report")
    assert isinstance(render.returns, ast.Name) and render.returns.id == "bytes"


MODEL_FILES = (
    "models/shared_m_relations.py",
    "models/shared_m/__init__.py",
    "models/shared_m/components.py",
    "models/shared_m/router.py",
    "models/shared_m/system.py",
    "models/shared_m/spec.py",
    "representations/neighborhoods.py",
    "representations/tasks.py",
)
MODEL_FORBIDDEN = ("cipher_lab.ciphers", "cipher_lab.data", "cipher_lab.training",
                   "cipher_lab_plugins", "secrets", "random", "subprocess",
                   "pathlib", "os", "json")


@pytest.mark.parametrize("relative", MODEL_FILES, ids=lambda value: value.split("/")[-1])
def test_model_modules_have_no_cipher_training_plugin_or_io_dependency(relative: str) -> None:
    tree = _tree(CORE / relative)
    names = _imported(tree)
    assert not _hits(names, MODEL_FORBIDDEN), relative
    calls = _calls(tree)
    assert not calls & ({"open", "print", "input"} | DYNAMIC)
    # Models receive public representations; they never read a key or encrypt.
    assert not [call for call in calls if "encrypt" in call.lower()]
    assert "secrets" not in names


def test_metrics_have_no_io_sampling_forward_or_storage() -> None:
    tree = _tree(PLUGIN / "metrics.py")
    names = _imported(tree)
    assert {name.split(".")[0] for name in names} <= STDLIB | {"numpy"}, names
    assert not _hits(names, ("torch", "cipher_lab", "cipher_lab_plugins", ".data",
                             ".checkpoint", ".training", ".application", ".cli",
                             ".evaluate", ".report", ".resources"))
    calls = _calls(tree)
    assert not calls & {"open", "print", "input"}
    assert not calls & {"default_rng", "np.save", "np.load", "np.savez",
                        "np.savez_compressed", "torch.save", "torch.load"}
    assert not [call for call in calls
                if ".random." in call or call.endswith((".permutation", "urandom"))]
    # Metrics consume concrete arrays; they never run a model or optimizer.
    assert not [call for call in calls
                if call.endswith((".forward", ".backward", ".step", ".zero_grad"))]


def test_data_has_no_model_training_cli_or_key_dependency() -> None:
    tree = _tree(PLUGIN / "data.py")
    names = _imported(tree)
    assert not _hits(names, ("torch", "cipher_lab", "cipher_lab_plugins", ".training",
                             ".application", ".cli", ".evaluate", ".metrics"))
    # Only the frozen recipe constants are needed at import time.
    module_level = _top_level_imports(tree)
    assert {name for name in module_level if name.startswith(".")} == {".config"}
    # The checkpoint decoder is reached lazily from the explicit load_dataset wrapper.
    nested = {node.module or "" for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node not in tree.body}
    assert nested == {"checkpoint"}
    lazy = [node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node not in tree.body]
    assert len(lazy) == 1 and lazy[0].level == 1
    owner = _enclosing_function(tree, lazy[0])
    assert isinstance(owner, ast.FunctionDef) and owner.name == "load_dataset"
    assert not [call for call in _calls(tree)
                if call.endswith((".forward", ".backward", ".step"))]


def test_checkpoint_and_saved_decode_have_no_cipher_generation_or_training_dependency() -> None:
    tree = _tree(PLUGIN / "checkpoint.py")
    names = _imported(tree)
    assert not _hits(names, ("cipher_lab", "cipher_lab_plugins", ".data", ".training",
                             ".application", ".cli", ".evaluate", ".metrics", ".report",
                             ".resources", "secrets", "random"))
    assert _hits(names, (".config",))  # identity constants come from the fixed recipe
    calls = _calls(tree)
    assert not [call for call in calls if call.endswith(
        (".savez", ".savez_compressed", "default_rng", ".permutation", "urandom",
         "RandomState", "relation_labels", "sample_orbits"))]
    assert "relation_labels" not in calls and "sample_orbits" not in calls
    assert not [call for call in calls if "encrypt" in call.lower()]
    functions = {node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"save_checkpoint", "load_checkpoint", "load_saved_dataset"} <= functions
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert {"Identity", "SavedDataset"} <= classes


def test_plugin_has_no_dynamic_import_and_core_dispatch_loads_exactly_once() -> None:
    modules = sorted(PLUGIN.glob("*.py"))
    assert {"config.py", "data.py", "metrics.py", "checkpoint.py", "resources.py",
            "report.py"} <= {path.name for path in modules}
    for path in modules:
        tree = _tree(path)
        assert not _calls(tree) & DYNAMIC, path.name
        assert "importlib" not in _imported(tree), path.name
    # The one dynamic boundary is the explicit core dispatch, not the plugin side.
    tree = _tree(CORE_PLUGIN_CLI)
    dynamic = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == "importlib.import_module"]
    assert len(dynamic) == 1
    owner = _enclosing_function(tree, dynamic[0])
    assert isinstance(owner, ast.FunctionDef) and owner.name == "dispatch"


def test_no_core_module_imports_the_plugin_package() -> None:
    tree = _tree(CORE_PLUGIN_CLI)
    assert {name.split(".")[0] for name in _imported(tree)} <= STDLIB
    offenders = [path.relative_to(PROJECT).as_posix()
                 for path in sorted(CORE.rglob("*.py"))
                 if path != CORE_PLUGIN_CLI
                 and _hits(_imported(_tree(path)), ("cipher_lab_plugins",))]
    assert offenders == []


def test_resources_owns_the_only_external_subprocess_and_nvidia_smi_default() -> None:
    owners = sorted(path.name for path in PLUGIN.glob("*.py")
                    if "subprocess" in _imported(_tree(path)))
    assert owners == ["resources.py"]
    tree = _tree(PLUGIN / "resources.py")
    assert not _hits(_imported(tree), ("torch", "numpy", "cipher_lab.ciphers",
                                        "cipher_lab.training", "cipher_lab.data"))
    assert not [call for call in _calls(tree)
                if call.endswith(("system", "popen", "spawn", "check_output"))]
    runs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "subprocess.run"]
    assert len(runs) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in runs[0].keywords}
    assert keywords.get("shell") == "False" and "timeout" in keywords
    owner = _enclosing_function(tree, runs[0])
    assert isinstance(owner, ast.FunctionDef) and owner.name == "_nvidia_smi_runner"
    # The default path is an explicitly built nvidia-smi argv, not a shell string.
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "args"
                           for target in node.targets)]
    assert len(assignments) == 1
    argv = assignments[0].value
    assert isinstance(argv, ast.List) and ast.unparse(argv.elts[0]) == "'nvidia-smi'"
    default = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)
               and ast.unparse(node.test) == "runner is None"]
    assert [ast.unparse(node.body) for node in default] == ["_nvidia_smi_runner"]


BLOCKED_MODULES = (
    "torch", "numpy", "secrets", "random",
    "cipher_lab.ciphers", "cipher_lab.execution", "cipher_lab.artifacts",
    "cipher_lab.training", "cipher_lab.data",
    "cipher_lab_plugins.differential_relations.data",
    "cipher_lab_plugins.differential_relations.metrics",
    "cipher_lab_plugins.differential_relations.checkpoint",
    "cipher_lab_plugins.differential_relations.training",
    "cipher_lab_plugins.differential_relations.application",
    "cipher_lab_plugins.differential_relations.evaluate",
    "cipher_lab_plugins.differential_relations.report",
    "cipher_lab_plugins.differential_relations.resources",
    CLI_MODULE,
)

# The only module a child may be exempted from blocking is the explicit entry
# module under test; science, resource, data, training, decoding and storage
# targets are never erasable. Erasing the entry module is not a relaxation of
# the scientific boundary: the CLI that runs is the same code being observed.
_ERASABLE = frozenset({CLI_MODULE})


def _blocked_modules(*, allowed: tuple[str, ...] = ()) -> tuple[str, ...]:
    """BLOCKED_MODULES minus exact allow-list names; every prefix stays blocked."""
    for name in allowed:
        assert name in _ERASABLE, f"cannot unblock non-entry module: {name}"
    return tuple(name for name in BLOCKED_MODULES if name not in allowed)


def _blocked_child(body: str, *, allowed: tuple[str, ...] = ()) -> str:
    """Child script that refuses science/probe imports before running ``body``."""
    preamble = "\n".join([
        "import importlib.abc",
        "import json",
        "import sys",
        "",
        f"blocked = {_blocked_modules(allowed=allowed)!r}",
        "",
        "class Blocker(importlib.abc.MetaPathFinder):",
        "    def find_spec(self, name, path=None, target=None):",
        "        if name in blocked:",
        '            raise ImportError("blocked science or probe import: " + name)',
        "        return None",
        "",
        "sys.meta_path.insert(0, Blocker())",
        "",
        body,
        "",
    ])
    return preamble


def _child_env() -> dict[str, str]:
    return dict(os.environ,
                PYTHONPATH=os.pathsep.join((str(PROJECT / "src"), str(PROJECT / "plugins"))),
                PYTHONDONTWRITEBYTECODE="1")


def test_config_alone_loads_the_full_official_json_with_science_imports_blocked(
        tmp_path: Path) -> None:
    expected = json.loads(OFFICIAL_CONFIG.read_text(encoding="utf-8"))
    assert expected['method'] == 'shared-m-differential-relations-v1'
    # This case loads config only, so even the CLI entry stays blocked here.
    assert CLI_MODULE in _blocked_modules()
    path = str(OFFICIAL_CONFIG)
    code = _blocked_child("\n".join([
        "from pathlib import Path",
        "from cipher_lab_plugins.differential_relations import config",
        f"print(json.dumps(config.load_config(Path({path!r})).to_dict(), sort_keys=True))",
    ]))
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                               env=_child_env(), capture_output=True, text=True,
                               timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == expected
    assert "nvidia-smi" not in completed.stdout and "RSS" not in completed.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", PENDING_FILES)
def test_pending_future_files_exist_at_exact_paths(name: str) -> None:
    # Exact paths only; no glob may stand in for a missing owner file.
    assert (PLUGIN / name).is_file(), f"missing plugin module: {name}"


def test_pending_cli_help_lists_check_run_and_evaluate() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "cipher_lab_plugins.differential_relations.cli", "--help"],
        cwd=PROJECT, env=_child_env(), capture_output=True, text=True,
        timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    for command in ("check", "run", "evaluate"):
        assert command in completed.stdout


def test_pending_cli_check_is_config_only_and_blocks_science(tmp_path: Path) -> None:
    # Regression: the child must import its own entry module, so that exact target
    # is not blocked; every science/resource/training/data/decoding/storage target
    # remains blocked and no other module may be erased (`_ERASABLE` guard).
    blocked = _blocked_modules(allowed=(CLI_MODULE,))
    assert CLI_MODULE not in blocked
    assert set(blocked) == set(BLOCKED_MODULES) - {CLI_MODULE}
    assert {
        "torch", "numpy", "secrets", "random", "cipher_lab.ciphers",
        "cipher_lab.execution", "cipher_lab.artifacts", "cipher_lab.training",
        "cipher_lab.data",
        "cipher_lab_plugins.differential_relations.data",
        "cipher_lab_plugins.differential_relations.metrics",
        "cipher_lab_plugins.differential_relations.checkpoint",
        "cipher_lab_plugins.differential_relations.training",
        "cipher_lab_plugins.differential_relations.application",
        "cipher_lab_plugins.differential_relations.evaluate",
        "cipher_lab_plugins.differential_relations.report",
        "cipher_lab_plugins.differential_relations.resources",
    } <= set(blocked)
    with pytest.raises(AssertionError):
        _blocked_modules(allowed=("numpy",))
    with pytest.raises(AssertionError):
        _blocked_modules(allowed=("cipher_lab_plugins.differential_relations.data",))
    path = str(OFFICIAL_CONFIG)
    code = _blocked_child("\n".join([
        "from cipher_lab_plugins.differential_relations import cli",
        f"raise SystemExit(cli.main(['check', '--config', {path!r}]))",
    ]), allowed=(CLI_MODULE,))
    assert repr(CLI_MODULE) not in code
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                               env=_child_env(), capture_output=True, text=True,
                               timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "CONFIG_OK" in completed.stdout
    assert list(tmp_path.iterdir()) == []
