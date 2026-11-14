"""Windows parent lifecycle and independent resource view; never scientific owner.

CPU parent owns outer kill-on-close Job and separately opened identities. All
membership/slot changes share a lock with sampling. This is not an orchestrator.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cipher_lab.speck_study.gpu_telemetry import windows_monotonic
from cipher_lab.speck_study.resources import ResourceStop


class OwnedTree:
    def __init__(self, job: Any, pid: int, created: int, root: Path, *,
                 clock: Callable[[], float] = windows_monotonic) -> None:
        self.job, self.pid, self.created, self.root = job, pid, created, root
        self.clock = clock
        self.own_handle = job.api.open(pid)
        if job.api.times(self.own_handle)[0] != created:
            job.api.close_handle(self.own_handle)
            raise ResourceStop('PROCESS_UNVERIFIED')
        self.gpu: Any = None
        self.lock = threading.RLock()
        self.roles: dict[int,str] = {}
        self.expected_provider = False
        self.provider_reaped = False
        self.provider_deadline = 0.

    def path(self, role: str) -> Path:
        if role not in ('parent','main','provider'):
            raise ResourceStop('PROCESS_UNVERIFIED')
        return self.root/(role+'-telemetry.json')

    def register(self, pid: int, created: int, role: str) -> None:
        with self.lock:
            if role in self.roles.values() or pid in self.roles:
                raise ResourceStop('PROCESS_UNVERIFIED')
            if role == 'parent' and (pid,created) != (self.pid,self.created):
                raise ResourceStop('PROCESS_UNVERIFIED')
            self.roles[pid] = role
            self.gpu.register(pid,created,self.path(role),first_frame_deadline=self.clock()+5.)
            if role != 'parent':
                self.gpu.scientific_pids.add(pid)

    def creation_time(self, pid: int) -> int:
        handle = self.own_handle if pid == self.pid else self.job.handles[pid][1]
        created = self.job.api.times(handle)[0]
        expected = self.created if pid == self.pid else self.job.handles[pid][0]
        if created != expected:
            raise ResourceStop('PROCESS_UNVERIFIED')
        return int(created)

    def sample(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.expected_provider and self.clock() >= self.provider_deadline:
                raise ResourceStop('PROCESS_UNVERIFIED')
            rows = self.job.sample()
            # A predeclared provider may be visible in the Job before its child
            # has sent the registration RPC. Independently adopt while suspended.
            unknown = [r for r in rows if r['pid'] not in self.roles]
            if unknown:
                if (not self.expected_provider or len(unknown) != 1
                        or self.clock() >= self.provider_deadline):
                    raise ResourceStop('PROCESS_UNVERIFIED')
                row = unknown[0]
                self.register(row['pid'],row['created'],'provider')
                self.expected_provider = False
            active = {r['pid'] for r in rows}
            for pid in list(self.roles):
                if pid != self.pid and pid not in active:
                    self.creation_time(pid)  # retained handle identity, including exit/reuse
                    if not self.job.api.exited(self.job.handles[pid][1]):
                        raise ResourceStop('PROCESS_UNVERIFIED')
                    last = self.gpu.seen.get(pid)
                    if last is None or not 0 <= self.clock()-last[1] <= 1.5:
                        raise ResourceStop('GPU_UNVERIFIED')
                    # Remove role first, so unregister's member callback cannot
                    # recursively retire the same slot. Handle stays retained.
                    del self.roles[pid]
                    self.gpu.unregister(pid)
            if self.creation_time(self.pid) != self.created:
                raise ResourceStop('PROCESS_UNVERIFIED')
            _, cpu = self.job.api.times(self.own_handle)
            return list(rows)+[dict(pid=self.pid,created=self.created,cpu=cpu,
                                    rss=self.job.api.rss(self.own_handle))]

    def cpu_seconds(self) -> float:
        own = self.job.api.times(self.own_handle)[1]
        value = self.job.cpu_seconds()+own
        if not math.isfinite(value) or value < 0:
            raise ResourceStop('CPU_UNAVAILABLE')
        return float(value)

    def close(self) -> None:
        self.job.close()

    def empty(self) -> bool:
        return bool(self.job.empty())  # parent is deliberately NOT inside Job

    def dispose(self) -> None:
        # Supervisor fallback may run before native_parent's final all-member
        # reap. Kill the Job, retain every independent process identity handle.
        self.job.dispose(release_processes=False)

    def close_own_handle(self) -> None:
        self.job.api.close_handle(self.own_handle)


class LockedGpu:
    """Lifecycle serialization, not a lock held across native identity resolution."""
    requires_exact_members = True
    def __init__(self, tree: OwnedTree, gpu: Any) -> None:
        self.tree, self.gpu = tree,gpu

    @property
    def ready(self) -> bool:
        roles = set(self.tree.roles.values())
        return bool(self.gpu.ready and 'main' in roles
                    and ('provider' in roles or self.tree.provider_reaped))

    def processes(self, binding: Any) -> dict[int,int]:
        with self.tree.lock:
            self.tree.sample()  # retire/adopt before exact slot comparison
            return dict(self.gpu.processes(binding))

    def reconcile_rows(self, rows: list[dict[str, Any]], members: set[int]
                       ) -> list[dict[str, Any]]:
        """Keep the exact GPU collection lifecycle; no second OS snapshot.

        Old RSS/CPU rows have already been checked/charged conservatively.
        Only independently confirmed retirements can be absent from usage.
        """
        identities = {row['pid']: row['created'] for row in rows}
        if len(identities) != len(rows) or not members <= identities.keys():
            raise ResourceStop('PROCESS_UNVERIFIED')
        for pid, created in identities.items():
            if pid not in members:
                if (self.gpu.retired_identities.get(pid) != created
                        or self.tree.creation_time(pid) != created
                        or not self.tree.job.api.exited(self.tree.job.handles[pid][1])):
                    raise ResourceStop('PROCESS_UNVERIFIED')
        return [row for row in rows if row['pid'] in members]

    def verify(self, evidence: dict[str,Any], pid: int, uuid: str, bus: str) -> bool:
        return bool(self.gpu.verify(evidence,pid,uuid,bus))
