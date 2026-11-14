"""Parent resource-only launch seam; Task5 supplies the fixed CLI child argv.

Native CLI now assembles native_parent; launch() below is the preserved Task3
resource-only seam, not a second production entry or a remote orchestrator.
No SSH/Git, no RunStore or
cipher/model import. Actual deployment remains gated on Windows synthetic review.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cipher_lab.speck_study.resources import (
    ResourceStop,
    StudyResources,
    Supervisor,
    verify_sources,
)


class TrustedTeacherSpawn:
    """Popen-shaped Task2 callback: fixed source wrapper then Job suspended spawn.

    Parent checks frozen manifest/source identities before constructing this seam.
    Child compiles verified .py bytes and never imports project .pyc files.
    """
    def __init__(self, job: Any, manifest: Path, source_root: Path) -> None:
        self.job, self.manifest, self.source_root = job, manifest, source_root

    def __call__(self, args: list[str], **kwargs: Any) -> Any:
        if args[1:4] != ['-B', '-m', 'cipher_lab.speck_study.teacher_provider']:
            raise ValueError('untrusted teacher entry')
        wrapper = Path(__file__).with_name('freeze.py').resolve()
        command = [args[0], '-B', str(wrapper), '--teacher', str(self.manifest),
                   str(self.source_root)] + args[4:]
        return self.job.spawn(command, **kwargs)


def launch(resources: StudyResources, command: list[str], *,
           source_root: Path, source_manifest: dict[str, str],
           receipt: Path, terminal_verified: Callable[[], bool],
           request_closing: Callable[[], None], closing_acknowledged: Callable[[], bool],
           clock: Callable[[], float] = time.monotonic) -> Supervisor:
    """Blocking parent with independent sampler, owns only its Job and receipt.

    command must be Task5's frozen local Python -B -m cipher_lab.cli run entry.
    terminal_verified reads saved child terminal/identity; never manufactures it.
    Origin is handed to child BEFORE imports/remote model initialization. Child
    must consume it unchanged (Task5); no child-local origin reset is permitted.
    request_closing must send a bounded nonblocking IPC signal. Child single
    writer calls begin_closing before ACK, catches SCIENCE_DEADLINE for save-only
    teardown, and never uses science=False to admit work. ACK is distinct from
    terminal_verified; both are required after a closing request. Missing ACK
    cannot complete; close_deadline still hard-kills even an unresponsive child.
    """
    if (len(command) < 5 or command[1:5] != ['-B', '-m', 'cipher_lab.cli', 'run']):
        raise ValueError('expected frozen cipher_lab.cli run source entry')
    verify_sources(source_root, source_manifest)  # parent verifies before child import
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES=resources.binding.gpu_uuid,
               CUDA_DEVICE_ORDER='PCI_BUS_ID',
               SPECK_STUDY_ORIGIN=repr(resources.origin),
               SPECK_STUDY_SCIENCE_DEADLINE=repr(resources.science_deadline),
               SPECK_STUDY_CLOSE_DEADLINE=repr(resources.close_deadline))
    supervisor = Supervisor(resources, request_closing=request_closing,
                            closing_acknowledged=closing_acknowledged)
    process: Any = None
    last_clock = resources.origin

    def sample_clock() -> float:
        nonlocal last_clock
        now = clock()
        if (type(now) not in (int, float) or not math.isfinite(now)
                # Only this caller's ordered samples: a concurrent resource
                # sample may legitimately publish t2 after we already read t1.
                or now < last_clock):
            raise ResourceStop('CLOCK_UNAVAILABLE')
        last_clock = now
        return now

    try:
        supervisor.start()  # before Popen, teacher constructor or ready blocking
        process = resources.tree.spawn(command, env=env)
        while process.poll() is None:
            if supervisor.error:
                raise ResourceStop('CLEANUP_UNKNOWN')
            if sample_clock() >= resources.science_deadline:
                supervisor.tick()
            time.sleep(.05)
        supervisor.finish(receipt, child_exit=process.returncode,
                          child_terminal=terminal_verified())
        return supervisor
    finally:
        supervisor.stop_event.set()
        errors = supervisor.cleanup_tree()  # fallback CloseHandle immediately on TERM failure
        # Every cleanup leg is independent, including when fallback itself fails.
        if process is not None:
            timeout = .01
            try:
                timeout = max(.01, min(2., resources.close_deadline - sample_clock()))
            except BaseException as error:
                errors.append(error)
            try:
                process.wait(timeout=timeout)  # still attempt if timeout clock failed
            except BaseException as error:
                errors.append(error)
        if supervisor.thread:
            try:
                supervisor.thread.join(1.)
                if supervisor.thread.is_alive():
                    raise ResourceStop('CLEANUP_UNKNOWN')
            except BaseException as error:
                errors.append(error)
        try:
            resources.tree.dispose()  # retry orphan/handle cleanup, even after fallback
        except BaseException as error:
            errors.append(error)
        try:
            if sample_clock() >= resources.close_deadline:
                errors.append(ResourceStop('CLOSE_DEADLINE'))
        except BaseException as error:
            errors.append(error)
        if errors:
            supervisor.complete = False
            supervisor.error = ResourceStop('CLEANUP_UNKNOWN')
            raise supervisor.error from errors[0]


if __name__ == '__main__':
    raise SystemExit(
        'Use source.py run MANIFEST SRC run --backend speck-study; no standalone science')
