"""Meaningful import/call boundaries, AST only (never comments/string prose)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "src/cipher_lab"


def imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def calls(tree: ast.AST) -> set[str]:
    return {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: p.name)
def test_no_old_package_or_historical_data_dependencies(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for name in imports(tree):
        assert not any(
            part in name.split(".")
            for part in ("cipher_experiments", "backups", "runs", "received")
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "__import__",
            "importlib.import_module",
            "exec",
            "eval",
        ):
            # The sole exception is the explicit trusted plugin dispatch call,
            # in that exact module/function, never another dynamic operation.
            parents = {child: parent for parent in ast.walk(tree)
                       for child in ast.iter_child_nodes(parent)}
            parent: ast.AST = node
            while parent in parents and not isinstance(parent, (ast.FunctionDef,
                                                               ast.AsyncFunctionDef)):
                parent = parents[parent]
            if (path == PACKAGE / "plugin_cli.py" and isinstance(parent, ast.FunctionDef)
                    and parent.name == "dispatch"
                    and ast.unparse(node.func) == "importlib.import_module"):
                continue
            pytest.fail(f"dynamic dependency evades boundaries: {path}")


@pytest.mark.parametrize(
    "relative,forbidden",
    [
        ("ciphers/simon.py", {"cipher_lab", "torch"}),
        ("integral/linear.py", {"cipher_lab", "numpy", "os", "pathlib", "random", "secrets"}),
        ("integral/observables.py", {"cipher_lab", "os", "pathlib", "random", "secrets"}),
        ("integral/report.py", {"cipher_lab", "os", "pathlib", "random", "secrets", "numpy"}),
        ("demo.py", {"cipher_lab.ciphers", "secrets", "random", "numpy"}),
    ],
)
def test_algorithm_and_report_import_boundaries(relative: str, forbidden: set[str]) -> None:
    tree = ast.parse((PACKAGE / relative).read_text())
    for name in imports(tree):
        assert not any(name == item or name.startswith(item + ".") for item in forbidden)
    if relative in ("integral/linear.py", "integral/observables.py", "integral/report.py"):
        assert not calls(tree) & {"open", "print", "input", "__import__"}
        assert not any(
            name.endswith((".read_text", ".read_bytes", ".write_text", ".write_bytes"))
            for name in calls(tree)
        )


def test_plugin_loader_has_exactly_one_stdlib_dynamic_import_boundary() -> None:
    path = PACKAGE / "plugin_cli.py"
    assert path.is_file(), "explicit plugin boundary is not implemented"
    tree = ast.parse(path.read_text())
    dynamic = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == "importlib.import_module"]
    assert len(dynamic) == 1
    dispatch = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    assert dynamic[0] in list(ast.walk(dispatch))
    for name in imports(tree):
        assert name.split(".")[0] in sys.stdlib_module_names | {"__future__"}
    # No alias can smuggle in a second loader that evades the exact call check.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.asname is None for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "importlib"
            assert node.level == 0


def test_production_wiring_is_explicit_and_claim_precedes_imports() -> None:
    tree = ast.parse((PACKAGE / "cli.py").read_text())
    runner = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    claim = next(
        node.lineno
        for node in ast.walk(runner)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "RunStore.claim"
    )
    production = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.ImportFrom)
        and node.module in ("secrets", "cipher_lab.ciphers.simon")
    ]
    assert {node.module for node in production} == {"secrets", "cipher_lab.ciphers.simon"}
    assert all(node.lineno > claim for node in production)
    assert {"token_bytes", "encrypt_many", "production_rss_bytes"} <= {
        node.id for node in ast.walk(runner) if isinstance(node, ast.Name)
    }
    # No runner-owned unit reservations/completions parallel to CubeCollector.
    assert not any(name.endswith((".reserve", ".complete")) for name in calls(runner))


def test_metadata_only_discovers_new_package() -> None:
    config = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find == {"where": ["src"], "include": ["cipher_lab", "cipher_lab.*"]}
    assert config["project"]["scripts"] == {"cipher-lab": "cipher_lab.cli:main"}


@pytest.mark.parametrize("directory", ["data", "models", "result"])
def test_artifact_ignore_rules_do_not_hide_package_source(directory: str) -> None:
    # Interpret only the three literal artifact-directory rules, independently of
    # Git/repository state. Do not create a repository or shell out from pytest.
    rules = [line.strip() for line in (PROJECT / ".gitignore").read_text().splitlines()
             if line.strip().strip("/") in {"data", "models", "result"}]
    assert len(rules) == 3

    def ignored(relative: str) -> bool:
        parents = relative.split("/")[:-1]
        return any(rule.endswith("/") and (
            parents[0] == rule.strip("/") if rule.startswith("/")
            else rule.strip("/") in parents
        ) for rule in rules)

    assert ignored(f"{directory}/sample/artifact.json")
    assert not ignored(f"src/cipher_lab/{directory}/__init__.py")
    assert not ignored(f"src/cipher_lab/{directory}/future.py")


def test_package_module_entrypoint_works_without_old_package(tmp_path: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-m", "cipher_lab", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "check" in completed.stdout and "run" in completed.stdout
    assert "evaluate" in completed.stdout and "{check,run,evaluate}" in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_evaluation_application_has_no_generation_training_or_private_rng_imports() -> None:
    tree = ast.parse((PACKAGE / "evaluation/application.py").read_text())
    forbidden = {"cipher_lab.training", "cipher_lab.ciphers", "cipher_lab.data.prediction",
                 "cipher_lab.data.differential", "cipher_lab.data.cubes", "secrets", "random"}
    for name in imports(tree):
        assert not any(name == item or name.startswith(item + ".") for item in forbidden)
    assert not any(name.endswith((".reserve", ".complete", ".write_model", ".write_dataset"))
                   for name in calls(tree))
