import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from cipher_lab_plugins.differential_relations import cli

CONFIG = Path("configs/differential-relations-simon6.json").resolve()
UUID = "GPU-12345678-1234-1234-1234-123456789ABC"


def test_check_help_zero_science_imports() -> None:
    code = """import sys
from cipher_lab_plugins.differential_relations.cli import main
assert main(['check','--config',sys.argv[1]]) == 0
assert not any(x.split('.')[0] in {'numpy','torch'} or '.ciphers' in x for x in sys.modules)
assert main(['--help']) == 0
"""
    p = subprocess.run(
        [sys.executable, "-c", code, str(CONFIG)], env=os.environ, capture_output=True, text=True
    )
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize(
    "extra",
    [["--device", "cpu"], ["--unused", "yes"], ["--gpu-uuid", "GPU-bad"], ["--gpu-pci", "N/A"]],
)
def test_invalid_arguments_stop_before_claim(tmp_path: Any, monkeypatch: Any, extra: Any) -> None:
    monkeypatch.setattr(cli, "_claim", lambda *a, **k: pytest.fail("claim before validation"))
    args = [
        "run",
        "--config",
        str(CONFIG),
        "--root",
        str(tmp_path / "out"),
        "--id",
        "new",
        "--device",
        "cuda:0",
        "--gpu-uuid",
        UUID,
        "--gpu-pci",
        "0000:01:00.0",
    ]
    # Replace option rather than rely on argparse last-value rules.
    if extra[0] in args:
        i = args.index(extra[0])
        args[i : i + 2] = extra
    else:
        args += extra
    assert cli.main(args) == 2


def test_old_python_never_claims(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "_python_version", lambda: (3, 10))
    monkeypatch.setattr(cli, "_claim", lambda *a, **k: pytest.fail("claim"))
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(CONFIG),
                "--root",
                str(tmp_path),
                "--id",
                "x",
                "--device",
                "cuda:0",
                "--gpu-uuid",
                UUID,
                "--gpu-pci",
                "0000:01:00.0",
            ]
        )
        == 2
    )


def test_clean_source_copy_unified_check_and_help(tmp_path: Path) -> None:
    import shutil

    source = Path.cwd()
    shutil.copytree(source / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        source / "plugins", tmp_path / "plugins", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copyfile(CONFIG, tmp_path / "config.json")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path / "src"), str(tmp_path / "plugins"))),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    code = """import builtins,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    assert name.split('.')[0] not in {'torch','numpy'} and '.ciphers' not in name, name
    return original(name,*args,**kwargs)
builtins.__import__=guard
from cipher_lab.cli import main
sys.argv=['cipher-lab',*sys.argv[1:]]
raise SystemExit(main())
"""
    for args in (["check", "--config", str(tmp_path / "config.json")], ["--help"]):
        command = [
            sys.executable,
            "-c",
            code,
            *args,
            "--plugin",
            "cipher_lab_plugins.differential_relations.cli",
        ]
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", ["check", "run"])
def test_abbreviated_options_rejected_before_claim(
    tmp_path: Any, monkeypatch: Any, command: str
) -> None:
    monkeypatch.setattr(cli, "_claim", lambda *a, **k: pytest.fail("claim for abbreviation"))
    if command == "check":
        args = ["check", "--conf", str(CONFIG)]
    else:
        args = [
            "run",
            "--config",
            str(CONFIG),
            "--root",
            str(tmp_path),
            "--id",
            "new",
            "--device",
            "cuda:0",
            "--gpu-u",
            UUID,
            "--gpu-pci",
            "0000:01:00.0",
        ]
    assert cli.main(args) == 2
