"""Explicit plugin boundary tests: synthetic modules only, never science."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from cipher_lab import cli


def dispatch(argv: Sequence[str]) -> int | None:
    # Import inside tests so the missing feature fails each behavioral test.
    from cipher_lab import plugin_cli

    return plugin_cli.dispatch(argv)


@pytest.mark.parametrize("argv", [[], ["--help"], ["check", "--plugin-extra", "x"]])
def test_no_plugin_does_not_import(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(name: str) -> ModuleType:
        pytest.fail("non-plugin path must not import dynamically")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    assert dispatch(argv) is None


@pytest.mark.parametrize("tail", [
    ["--plugin"], ["--plugin="], ["--plugin", ""], ["--plugin", "plain"],
    ["--plugin", ".relative"], ["--plugin", "x..y"], ["--plugin", "x.y:main"],
    ["--plugin", "x/y"], ["--plugin", "x-y.z"], ["--plugin", "x.2y"],
    ["--plugin", "x.y\n"], ["--plugin", "x.class"], ["--plugin", "x.λ"],
    ["--plugin", "--help"], ["--plugin=x.y", "--plugin=x.y"],
    ["--plugin", "x.y", "--plugin=x.y"],
])
def test_bad_plugin_is_rejected_before_import(
    tail: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(name: str) -> ModuleType:
        pytest.fail("invalid invocation must not import")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    assert dispatch(["check", *tail]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "cipher-lab: PLUGIN_PROTOCOL_ERROR\n"


@pytest.mark.parametrize("option", [["--plugin", "fake.entry"], ["--plugin=fake.entry"]])
@pytest.mark.parametrize("use_sys_argv", [False, True])
def test_cli_dispatches_before_core_parser_and_imports_once(
    option: list[str], use_sys_argv: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("fake.entry")
    imports: list[str] = []
    received: list[list[str]] = []

    def main(argv: Sequence[str]) -> int:
        received.append(list(argv))
        return 7

    module.main = main  # type: ignore[attr-defined]

    def load(name: str) -> ModuleType:
        imports.append(name)
        return module

    monkeypatch.setattr(importlib, "import_module", load)
    argv = ["check", *option, "--unknown-to-core", "a", "--config=x", "--help"]
    monkeypatch.setattr(sys, "argv", ["cipher-lab", *argv])
    assert cli.main(None if use_sys_argv else argv) == 7
    assert imports == ["fake.entry"]
    assert received == [["check", "--unknown-to-core", "a", "--config=x", "--help"]]


@pytest.mark.parametrize("code", [0, 2, 4, 255])
@pytest.mark.parametrize("exit_instead", [False, True])
def test_valid_return_or_systemexit_is_normalized(
    code: int, exit_instead: bool, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = ModuleType("fake.entry")

    def main(argv: Sequence[str]) -> int:
        if exit_instead:
            raise SystemExit(code)
        return code

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    assert dispatch(["run", "--plugin=fake.entry"]) == code
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("value", [None, True, False, "PRIVATE", 1.0, -1, 256])
@pytest.mark.parametrize("exit_instead", [False, True])
def test_bad_return_or_exit_is_protocol_error_without_private_detail(
    value: object, exit_instead: bool, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = ModuleType("fake.entry")

    def main(argv: Sequence[str]) -> object:
        if exit_instead:
            raise SystemExit(value)
        return value

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    assert dispatch(["run", "--plugin=fake.entry"]) == 2
    assert capsys.readouterr() == ("", "cipher-lab: PLUGIN_PROTOCOL_ERROR\n")


@pytest.mark.parametrize("main", [None, 42])
def test_missing_or_noncallable_main_is_protocol_error(
    main: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = ModuleType("fake.entry")
    if main is not None:
        module.main = main  # type: ignore[attr-defined]
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    assert dispatch(["check", "--plugin=fake.entry"]) == 2
    assert capsys.readouterr() == ("", "cipher-lab: PLUGIN_PROTOCOL_ERROR\n")


@pytest.mark.parametrize("missing,reason", [
    ("fake", "PLUGIN_NOT_FOUND"), ("fake.entry", "PLUGIN_NOT_FOUND"),
    ("other_dependency", "PLUGIN_LOAD_FAILED"),
    ("fake.entry.internal", "PLUGIN_LOAD_FAILED"), (None, "PLUGIN_LOAD_FAILED"),
])
def test_missing_module_name_distinguishes_target_from_internal_dependency(
    missing: str | None, reason: str, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def load(name: str) -> ModuleType:
        raise ModuleNotFoundError("PRIVATE filesystem detail", name=missing)

    monkeypatch.setattr(importlib, "import_module", load)
    assert dispatch(["run", "--plugin=fake.entry"]) == 2
    assert capsys.readouterr() == ("", f"cipher-lab: {reason}\n")


@pytest.mark.parametrize("phase", ["load", "execute", "lookup"])
@pytest.mark.parametrize("error", [RuntimeError, ImportError, SystemExit])
def test_exceptions_are_fixed_and_do_not_fall_back(
    phase: str, error: type[BaseException], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = ModuleType("fake.entry")

    def broken(*args: object) -> object:
        raise error("PRIVATE exception detail")

    if phase == "lookup":
        module.__getattr__ = broken  # type: ignore[method-assign]
    else:
        module.main = broken  # type: ignore[attr-defined]

    def load(name: str) -> ModuleType:
        if phase == "load":
            raise error("PRIVATE import detail")
        return module

    monkeypatch.setattr(importlib, "import_module", load)
    reason = {"load": "PLUGIN_LOAD_FAILED", "lookup": "PLUGIN_PROTOCOL_ERROR",
              "execute": ("PLUGIN_PROTOCOL_ERROR" if error is SystemExit
                          else "PLUGIN_EXECUTION_FAILED")}[phase]
    assert cli.main(["run", "--plugin=fake.entry"]) == (4 if reason.endswith("EXECUTION_FAILED")
                                                       else 2)
    assert capsys.readouterr() == ("", f"cipher-lab: {reason}\n")


def test_keyboard_interrupt_preserves_interrupt_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("fake.entry")

    def main(argv: Sequence[str]) -> int:
        raise KeyboardInterrupt

    module.main = main  # type: ignore[attr-defined]
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    with pytest.raises(KeyboardInterrupt):
        dispatch(["run", "--plugin=fake.entry"])


def test_actual_import_missing_target_and_internal_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    package = tmp_path / "synthetic_plugin_owned"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "broken.py").write_text("import synthetic_missing_owned_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert dispatch(["check", "--plugin=synthetic_plugin_owned.absent"]) == 2
        assert capsys.readouterr().err == "cipher-lab: PLUGIN_NOT_FOUND\n"
        assert dispatch(["check", "--plugin=synthetic_plugin_owned.broken"]) == 2
        assert capsys.readouterr().err == "cipher-lab: PLUGIN_LOAD_FAILED\n"
    finally:
        sys.modules.pop("synthetic_plugin_owned", None)
        sys.modules.pop("synthetic_plugin_owned.broken", None)


def test_default_core_help_remains_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "{check,run,evaluate}" in capsys.readouterr().out
