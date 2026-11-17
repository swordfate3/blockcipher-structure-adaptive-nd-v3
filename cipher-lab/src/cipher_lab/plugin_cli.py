"""Explicit trusted-module CLI delegation, with no discovery or scientific imports.

The selected module owns its arguments and any side effects. This is an error
presentation/protocol boundary, not a sandbox for untrusted Python code.
"""
from __future__ import annotations

import importlib
import keyword
import re
import sys
from collections.abc import Sequence

_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z", re.ASCII)


def _error(reason: str, code: int = 2) -> int:
    print(f"cipher-lab: {reason}", file=sys.stderr)
    return code


def dispatch(argv: Sequence[str]) -> int | None:
    """Remove one explicit --plugin option and invoke module.main exactly once.

    No option means no dynamic import and None (the caller's existing CLI path).
    Names must be absolute dotted ASCII Python identifiers. Errors never echo
    module names, arguments or underlying exceptions. Exit codes are strict
    integers in [0, 255], including integer SystemExit codes; bool is invalid.
    KeyboardInterrupt deliberately retains its interruption semantics.
    """
    remaining: list[str] = []
    selected: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--plugin":
            index += 1
            if index == len(argv):
                return _error("PLUGIN_PROTOCOL_ERROR")
            selected.append(argv[index])
        elif arg.startswith("--plugin="):
            selected.append(arg[len("--plugin="):])
        else:
            remaining.append(arg)
        index += 1
    if not selected:
        return None
    if (len(selected) != 1 or _MODULE.fullmatch(selected[0]) is None
            or any(keyword.iskeyword(part) for part in selected[0].split("."))):
        return _error("PLUGIN_PROTOCOL_ERROR")
    name = selected[0]
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        # import_module identifies the missing target or missing parent by name.
        # A missing dependency of the plugin is NOT a reason to fall back to core.
        if error.name is not None and (name == error.name or name.startswith(error.name + ".")):
            return _error("PLUGIN_NOT_FOUND")
        return _error("PLUGIN_LOAD_FAILED")
    except (Exception, SystemExit):
        return _error("PLUGIN_LOAD_FAILED")
    try:
        main = getattr(module, "main", None)
        if not callable(main):
            return _error("PLUGIN_PROTOCOL_ERROR")
    except (Exception, SystemExit):
        return _error("PLUGIN_PROTOCOL_ERROR")
    try:
        result: object = main(remaining)
    except SystemExit as exit_:
        result = exit_.code
    except Exception:
        return _error("PLUGIN_EXECUTION_FAILED", 4)
    if type(result) is not int or not 0 <= result <= 255:
        return _error("PLUGIN_PROTOCOL_ERROR")
    return result
