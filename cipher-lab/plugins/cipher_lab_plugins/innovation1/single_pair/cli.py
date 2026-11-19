"""Fixed Windows entry; help is stdlib-only and makes no claim or probes."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='cipher-lab evaluate --plugin cipher_lab_plugins.innovation1.single_pair.cli',
        allow_abbrev=False,
        description='Fixed SKINNY first-pair comparison of a sealed run (no training or retry).')
    parser.add_argument('command', choices=('evaluate',))
    parser.add_argument('--run-root', type=Path, required=True, help='root of the sealed input run')
    parser.add_argument('--root', type=Path, required=True, help=r'output root under G:\lxy')
    args = parser.parse_args(argv)
    if not sys.dont_write_bytecode:
        parser.error('bytecode writes must be disabled at interpreter startup (python -B)')
    if (sys.platform != 'win32' or not args.run_root.is_absolute()
            or not args.root.is_relative_to(Path(r'G:\lxy')) or '..' in args.root.parts):
        parser.error(r'Windows absolute --run-root and --root under G:\lxy required')

    from cipher_lab.execution import Stopped

    from .evaluate import run

    result, meter = run(args.run_root, args.root)
    result.update(command_resources_scope='PRE_STDOUT_SAMPLE' if meter else 'NO_SAMPLE',
                  exit_code_scope='REQUESTED_NOT_NATIVE')
    line = json.dumps(result, ensure_ascii=True, allow_nan=False)
    if meter is not None:
        # Charge serialization through a final cooperative check. Re-encode the
        # bounded result with that snapshot; stdout/flush and native exit are
        # deliberately left to the external wall/exit gate.
        try:
            meter.check()
        except Stopped as error:
            result.update(command_status='STOPPED', closeout_reason=error.reason, exit_code=3)
        result['command_resources'] = meter.snapshot()
        line = json.dumps(result, ensure_ascii=True, allow_nan=False)
    print(line, flush=True)
    return int(result['exit_code'])
