"""Run directories and metadata; never imports the training loop."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_identity(source_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, timeout=5, text=True).strip()
        status = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"],
                                         stderr=subprocess.DEVNULL, timeout=5, text=True)
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def source_identity() -> dict[str, str]:
    base = Path(__file__).parent
    return {path.relative_to(base).as_posix(): file_sha256(path)
            for path in sorted(base.rglob("*.py"))}


def append_progress(path: Path, record: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
