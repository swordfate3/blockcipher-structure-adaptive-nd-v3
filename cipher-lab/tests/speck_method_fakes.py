"""Task4-only deterministic fakes; no child processes or scientific execution."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import pytest
from speck_study_fakes import forbidden, install_guards
from speck_study_fakes import np as np

from cipher_lab.execution import Stopped


@pytest.fixture(autouse=True)
def science_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    install_guards(monkeypatch)
    monkeypatch.setattr(np.random, "PCG64", forbidden)
    import cipher_lab.execution as execution

    monkeypatch.setattr(execution, "production_rss_bytes", forbidden)
    import cipher_lab.platform_resources.gpu as gpu

    monkeypatch.setattr(gpu.GpuBinding, "from_snapshot", forbidden)


class Resources:
    def __init__(self) -> None:
        self.closing = False
        self.acked = False
        self.requested = False
        self.r0_calls = 0

    def check(self, *, science: bool = True) -> None:
        if science and (self.closing or self.requested):
            raise Stopped("WALL_LIMIT")

    def begin_closing(self) -> None:
        self.closing = True

    def acknowledge_closing(self) -> None:
        assert self.closing
        self.acked = True

    def formal_r0(self, data: Any) -> bool:
        self.r0_calls += 1
        return True


class Meter:
    def __init__(self, resources: Resources) -> None:
        self.resources = resources
        self.meter = self
        self.counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def check(self) -> None:
        self.resources.check()

    def reserve(self, kind: str, rows: int) -> None:
        assert type(rows) is int and rows >= 0
        self.check()
        self.counts[kind][0] += rows

    def complete(self, kind: str, rows: int) -> None:
        self.check()
        assert self.counts[kind][1] + rows <= self.counts[kind][0]
        self.counts[kind][1] += rows

    def snapshot(self) -> dict[str, object]:
        return {
            f"{status}_{k}": v[i]
            for k, v in self.counts.items()
            for i, status in enumerate(("attempted", "completed"))
        }


class Sampler:
    """Counter-shaped fake stream; independent stage and lane, batching invariant."""

    def __call__(self, seed: int, domain: str, offset: int, rows: int, lanes: int) -> Any:
        base = int.from_bytes(hashlib.sha256(f"{seed}:{domain}".encode()).digest()[:8], "big")
        return np.array(
            [
                [(base + 6364136223846793005 * (i * lanes + j)) % 2**64 for j in range(lanes)]
                for i in range(offset, offset + rows)
            ],
            dtype=np.uint64,
        )


class Cipher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, points: Any, keys: Any, rounds: int) -> Any:
        self.calls.append((len(points), rounds))
        assert keys.shape in ((4,), (len(points), 4))
        return points.copy()  # explicitly not encryption


class Private:
    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, size: int) -> bytes:
        self.sizes.append(size)
        return bytes((i + len(self.sizes)) % 256 for i in range(size))


class Teacher:
    def __init__(self, resources: Resources) -> None:
        self.resources = resources
        self.calls = 0
        self.closed = False
        self.bad: float | None = None
        self.fail_close = False

    def predict(self, points: list[int]) -> tuple[Any, Any]:
        assert not self.closed
        self.calls += len(points)
        probs = tuple(
            (self.bad if self.bad is not None else (0.75 if x & 1 else 0.25)) for x in points
        )
        return probs, tuple(int(p > 0.5) for p in probs)

    def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("private error must not escape")
        self.closed = True  # fake contract: returned close confirms reaped


class Store:
    """Narrow owner-shaped fake; public binaries only, immutable registration."""

    def __init__(self) -> None:
        self.raw: dict[str, bytes] = {}
        self.saved: dict[str, Any] = {}
        self.events: list[Any] = []
        self.finalizations = 0
        self.fail_read = False

    def source_snapshot(self) -> dict[str, str]:
        return {"cipher_lab/fake.py": "a" * 64}

    def write_snapshot(self, value: Any) -> None:
        self.saved.update(value)

    def append_event(self, value: Any) -> None:
        self.events.append(value)

    def write_payload(self, name: str, value: Any) -> str:
        self.events.append((name, value))
        return f"payload/{name}.json"

    def write_dataset(self, name: str, content: bytes) -> str:
        ref = f"dataset/{name}.npz"
        assert ref not in self.raw
        self.raw[ref] = content
        return ref

    def write_model(self, name: str, content: bytes) -> str:
        ref = f"model/{name}.pt"
        assert ref not in self.raw
        self.raw[ref] = content
        return ref

    def read_model(self, ref: str, expected_sha256: str, max_bytes: int) -> bytes:
        from cipher_lab.artifacts import ArtifactError

        raw = self.raw[ref]
        if (
            self.fail_read
            or len(raw) > max_bytes
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise ArtifactError("HASH_MISMATCH")
        return raw

    def read_dataset(self, ref: str, digest: str, size: int) -> bytes:
        return self.read_model(ref, digest, size)

    def verify_context(self, recipe: Any, source: Any, backend: str) -> None:
        from cipher_lab.artifacts import ArtifactError

        if (
            self.saved.get("recipe") != recipe.to_dict()
            or self.saved.get("summary", {}).get("backend_identity") != backend
            or source != self.source_snapshot()
        ):
            raise ArtifactError("HASH_MISMATCH")

    def finalize(self, value: Any) -> None:
        self.finalizations += 1
        self.saved.update(value)
