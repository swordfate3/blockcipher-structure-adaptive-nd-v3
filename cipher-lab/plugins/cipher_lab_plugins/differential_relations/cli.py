"""Explicit plugin entry; check/help are strictly standard-library only."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import load_config

_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_PCI = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def _python_version() -> tuple[int, int]:
    return sys.version_info[:2]


def safe_path(value: str | Path, *, file: bool = False) -> Path:
    path = Path(os.path.abspath(value))
    for p in (*reversed(path.parents), path):
        if p.is_symlink() or (p.exists() and getattr(p.stat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("UNSAFE_PATH")
    if file and not path.is_file():
        raise ValueError("MISSING_FILE")
    if not file and path.exists() and not path.is_dir():
        raise ValueError("INVALID_DIRECTORY")
    return path


def _claim(root: Path, run_id: str, limit: int) -> Any:
    from cipher_lab.artifacts import RunStore

    return RunStore.claim(
        root,
        run_id,
        output_bytes=limit,
        needs_models=True,
        model_slots=tuple(f"F{i}-best" for i in range(1, 6)),
    )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="Fixed SIMON differential-relations plugin"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", allow_abbrev=False)
    check.add_argument("--config", required=True)
    run = sub.add_parser("run", allow_abbrev=False)
    for option in ("config", "root", "id", "device", "gpu-uuid", "gpu-pci"):
        run.add_argument("--" + option, required=True)
    evaluation = sub.add_parser("evaluate", allow_abbrev=False)
    for option in ("run-record", "checkpoint", "dataset", "output"):
        evaluation.add_argument("--" + option, required=True)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as error:
        return error.code if type(error.code) is int else 2
    try:
        if args.command in ("check", "run"):
            recipe = load_config(safe_path(args.config, file=True))
        if args.command == "check":
            print("CONFIG_OK (no resource/scientific probe)")
            return 0
        if _python_version() < (3, 12):
            raise ValueError("PYTHON_REQUIRES_3_12")
        if args.command == "run":
            root = safe_path(args.root)
            if (
                _TOKEN.fullmatch(args.id) is None
                or args.device != "cuda:0"
                or _UUID.fullmatch(args.gpu_uuid) is None
                or _PCI.fullmatch(args.gpu_pci) is None
            ):
                raise ValueError("INVALID_RUN_ARGUMENTS")
            for kind in ("data", "models", "result"):
                target = safe_path(root / kind / args.id)
                if target.exists():
                    raise ValueError("OCCUPIED")
            store = _claim(root, args.id, recipe.artifact_bytes)
            from .application import execute

            return execute(
                store=store,
                recipe=recipe,
                root=root,
                experiment_id=args.id,
                device=args.device,
                gpu_uuid=args.gpu_uuid.upper(),
                gpu_pci=args.gpu_pci,
            )
        paths = {
            name: safe_path(getattr(args, name), file=True)
            for name in ("run_record", "checkpoint", "dataset")
        }
        output = safe_path(args.output)
        if output.exists():
            raise ValueError("OCCUPIED")
        from .evaluate import evaluate_saved

        evaluate_saved(
            run_record=paths["run_record"],
            checkpoint=paths["checkpoint"],
            dataset=paths["dataset"],
            output=output,
        )
        return 0
    except (ValueError, OSError):
        print("INVALID_INPUT", file=sys.stderr)
        return 2
    except Exception:
        # Never echo private exceptions/tracebacks from an execution boundary.
        print("EXECUTION_FAILED", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
