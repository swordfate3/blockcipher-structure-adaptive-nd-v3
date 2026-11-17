"""Study-only secret owner, release gates and immutable public R1 chunks.

Cipher seam: cipher(uint32[n], uint16[n,4] or uint16[4], rounds)->uint32[n].
Private seam receives a byte count only; public sampler never receives a key.
No raw key/plaintext/state is serialized. B consumers receive PT and target sign,
not ciphertext. Saved R1 chunks are public and retained for saved-only audits.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cipher_lab.artifacts import ArtifactError
from cipher_lab.execution import Stopped
from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe
from cipher_lab_plugins.speck_study.parity import array_rows
from cipher_lab_plugins.speck_study.teacher_protocol import pack_reply, unpack_reply


class PublicSampler:
    """Random-access PCG64 raw words: offset counts rows, never compute batches.

    Each row consumes exactly `lanes` uint64 words; no rejection/bit-cache calls.
    Independent SHA256 stage/depth domain; replacement sampling, not unseen.
    """

    def __call__(self, seed: int, domain: str, offset: int, rows: int, lanes: int) -> Any:
        digest = hashlib.sha256(f"sha256-stage-depth-v1:{seed}:{domain}".encode("ascii")).digest()
        engine = np.random.PCG64(int.from_bytes(digest, "big"))
        engine.advance(offset * lanes)
        return engine.random_raw(rows * lanes).reshape(rows, lanes)


@dataclass(frozen=True, slots=True)
class ClassificationChunk:
    points: NDArray[np.uint64]
    labels: NDArray[np.uint8]
    probabilities: NDArray[np.float32]
    hardlabels: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ChunkRef:
    ref: str
    sha256: str
    rows: int
    size: int


def _readonly(array: Any) -> Any:
    # bytes-backed array cannot be made writeable again (not merely flag=False).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class StudyData:
    def __init__(
        self,
        recipe: SpeckStudyRecipe,
        cipher: Any,
        private_random: Callable[[int], bytes],
        public_sampler: Any,
        meter: Any,
        store: Any,
    ) -> None:
        self.recipe, self.cipher, self.private_random = recipe, cipher, private_random
        self.sampler, self.meter, self.store = public_sampler, meter, store
        self.refs: list[ChunkRef] = []
        self.used: set[str] = set()
        self.closed = False
        self._classification_complete = False
        self._prediction_released = False
        self._prediction_attempted = False
        self._key: Any = None
        self._b_access = "none"
        self._r0: dict[str, int] = {"reference": 0, "batch": 0}

    def _check(self) -> None:
        if self.closed:
            raise Stopped("WALL_LIMIT")
        self.meter.check()

    def counts(self) -> dict[str, Any]:
        result = dict(self.meter.meter.snapshot())
        for kind, values in self.meter.counts.items():
            if kind in ("scoring", "parity"):
                result[f"attempted_{kind}"] = values[0]
                result[f"completed_{kind}"] = values[1]
        return result

    def close(self) -> None:
        self.closed = True
        self._key = None

    def _claim(self, domain: str) -> None:
        self._check()
        if domain in self.used:
            raise Stopped("DOMAIN_REUSED")
        self.used.add(domain)  # attempt consumes domain even if later interrupted

    def _private(self, size: int) -> bytes:
        self._check()
        try:
            raw = self.private_random(size)
            if type(raw) is not bytes or len(raw) != size:
                raise ValueError
            return raw
        except Exception:
            raise Stopped("RANDOM_UNAVAILABLE") from None

    def _encrypt(self, points: Any, keys: Any, rounds: int) -> Any:
        self._check()
        n = array_rows(points, np.dtype(np.uint32))
        self.meter.meter.reserve("real_blocks", n)
        try:
            out = self.cipher(points, keys, rounds)
        except Exception:
            raise Stopped("ORACLE_FAILED") from None
        try:
            if array_rows(out, np.dtype(np.uint32)) != n:
                raise ValueError
        except Exception:
            raise Stopped("ORACLE_RESPONSE") from None
        self._check()
        self.meter.meter.complete("real_blocks", n)
        return out

    def public_r0(self, points: Any, keys: Any, rounds: int, *, implementation: str) -> Any:
        if implementation not in self._r0:
            raise ValueError("invalid R0 implementation")
        n = array_rows(points, np.dtype(np.uint32))
        if (
            self.recipe.synthetic
            or self._r0[implementation] + n > 256
            or sum(self._r0.values()) + n > self.recipe.r0_max_blocks
        ):
            raise ValueError("R0 block cap")
        self._r0[implementation] += n
        # Task5 cipher adapter selects independently audited reference/batch path.
        selector = getattr(self.cipher, "public_r0", None)
        if selector is None:
            raise ValueError("explicit R0 adapter required")
        self._check()
        self.meter.meter.reserve("real_blocks", n)
        try:
            out = selector(points, keys, rounds, implementation=implementation)
        except Exception:
            raise Stopped("ORACLE_FAILED") from None
        if array_rows(out, np.dtype(np.uint32)) != n:
            raise Stopped("ORACLE_RESPONSE")
        self._check()
        self.meter.meter.complete("real_blocks", n)
        return out

    def _samples(
        self, stage: str, size: int, lanes: int, chunk_rows: int, depth: int = 0
    ) -> Iterator[Any]:
        domain = f"{stage}:{depth}"
        self._claim(domain)
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= self.recipe.chunk_rows:
            raise ValueError("invalid chunk rows")
        seed_name = {
            "Rclassification": "r_classification",
            "Rsearch": "r_search",
            "Rcoef": "r_coef",
            "Ragreement": "r_agreement",
            "Bsearch": "b_search",
            "Bcoef": "b_coef",
            "Bdev": "b_dev",
            "Btest": "b_test",
        }[stage]
        seed = getattr(self.recipe, f"public_{seed_name}_seed")
        for start in range(0, size, chunk_rows):
            self._check()
            n = min(chunk_rows, size - start)
            raw = self.sampler(seed, domain, start, n, lanes)
            if (
                type(raw) is not np.ndarray
                or raw.dtype != np.dtype(np.uint64)
                or raw.shape != (n, lanes)
            ):
                raise ValueError("invalid public sampler response")
            yield raw

    def teacher_signs(self, teacher: Any, points: Any) -> tuple[Any, Any, Any]:
        self._check()
        if self._prediction_attempted:
            raise ValueError("R teacher forbidden during B")
        n = array_rows(points, np.dtype(np.uint64))
        self.meter.reserve("teacher", n)
        try:
            probs, labels = teacher.predict(points.tolist())
            if len(probs) != n or len(labels) != n:
                raise ValueError
            # Validate probabilities *and* provided hardlabels, no silently repaired reply.
            p, y = unpack_reply(pack_reply(probs), n)
            if any(type(a) is not int or a != b for a, b in zip(labels, y)):
                raise ValueError
        except (Stopped, ArtifactError):
            raise
        except Exception:
            raise Stopped("ORACLE_RESPONSE") from None
        self._check()
        self.meter.complete("teacher", n)
        bits = np.array(y, dtype=np.uint8)
        return (1 - 2 * bits.astype(np.int8)), np.array(p, dtype=np.float32), bits

    def make_classification(
        self, teacher: Any, *, chunk_rows: int = 50_000
    ) -> Iterator[ClassificationChunk]:
        for raw in self._samples(
            "Rclassification", self.recipe.r_classification_size, 3, chunk_rows
        ):
            n = len(raw)
            labels = (raw[:, 0] & np.uint64(1)).astype(np.uint8)
            p0 = raw[:, 1].astype(np.uint32)
            p1 = np.where(
                labels == 1, p0 ^ np.uint32(0x00400000), raw[:, 2].astype(np.uint32)
            ).astype(np.uint32)
            keys = np.frombuffer(self._private(8 * n), dtype=">u2").astype(np.uint16).reshape(n, 4)
            c0 = self._encrypt(p0, keys, self.recipe.rounds)
            c1 = self._encrypt(p1, keys, self.recipe.rounds)
            del keys, p0, p1
            points = (c0.astype(np.uint64) << np.uint64(32)) | c1.astype(np.uint64)
            _, probs, hard = self.teacher_signs(teacher, points)
            # Fixed public format: header + count + packed pair + Y + float32 p + hardlabel.
            encoded = (
                b"SPKR1\x01"
                + n.to_bytes(4, "big")
                + points.astype(">u8").tobytes()
                + labels.tobytes()
                + probs.astype(">f4").tobytes()
                + hard.tobytes()
            )
            ref = self.store.write_dataset(f"R1-{len(self.refs):06d}", encoded)
            self.refs.append(ChunkRef(ref, hashlib.sha256(encoded).hexdigest(), n, len(encoded)))
            self.store.write_payload(f"R1-ref-{len(self.refs):06d}", self.manifest())
            yield ClassificationChunk(*(_readonly(x) for x in (points, labels, probs, hard)))
        self._classification_complete = True
        self.store.write_payload("R1-manifest", self.manifest())

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "complete": self._classification_complete,
            "rows": sum(r.rows for r in self.refs),
            "format": "SPKR1-v1-public-only",
            "chunks": [
                dict(ref=r.ref, sha256=r.sha256, rows=r.rows, bytes=r.size) for r in self.refs
            ],
        }

    def classification_chunks(self) -> Iterator[ClassificationChunk]:
        if not self._classification_complete:
            raise ValueError("classification incomplete")
        for ref in self.refs:
            self._check()
            raw = self.store.read_dataset(ref.ref, ref.sha256, ref.size)
            if (
                len(raw) != 10 + ref.rows * 14
                or hashlib.sha256(raw).hexdigest() != ref.sha256
                or raw[:6] != b"SPKR1\x01"
                or int.from_bytes(raw[6:10], "big") != ref.rows
            ):
                raise ArtifactError("HASH_MISMATCH")
            n = ref.rows
            points = np.frombuffer(raw, dtype=">u8", count=n, offset=10).astype(np.uint64)
            labels = np.frombuffer(raw, dtype=np.uint8, count=n, offset=10 + 8 * n)
            probs = np.frombuffer(raw, dtype=">f4", count=n, offset=10 + 9 * n).astype(np.float32)
            hard = np.frombuffer(raw, dtype=np.uint8, count=n, offset=10 + 13 * n)
            yield ClassificationChunk(*(_readonly(x) for x in (points, labels, probs, hard)))

    def release_prediction(self, *, reproduction_matched: bool, teacher_reaped: bool) -> None:
        self._check()
        if (
            reproduction_matched is not True
            or teacher_reaped is not True
            or self._prediction_released
        ):
            raise ValueError("R gate and reaped teacher required")
        self._prediction_released = True

    def begin_prediction(self) -> None:
        self._check()
        if not self._prediction_released or self._prediction_attempted:
            raise ValueError("prediction key unavailable")
        self._prediction_attempted = True
        self._key = np.frombuffer(self._private(8), dtype=">u2").astype(np.uint16)
        self._b_access = "search"

    def _b_signs(self, points: Any) -> Any:
        if self._key is None:
            raise ValueError("prediction not started")
        plain = (points >> np.uint64(32)).astype(np.uint32)
        cipher = self._encrypt(plain, self._key, self.recipe.rounds)
        bit = ((cipher >> np.uint32(31)) & np.uint32(1)).astype(np.int8)
        return (1 - 2 * bit).astype(np.int8)

    def paired_chunks(self, stage: str, depth: int, *, chunk_rows: int = 50_000) -> Iterator[Any]:
        if stage not in ("Rsearch", "Bsearch"):
            raise ValueError("invalid search stage")
        bits = self.recipe.r_bits if stage == "Rsearch" else self.recipe.b_bits
        if not self.recipe.initial_depth < depth <= bits:
            raise ValueError("invalid depth")
        if stage == "Bsearch" and self._b_access != "search":
            raise ValueError("Bsearch not released")
        for raw in self._samples(stage, self.recipe.pairs_per_layer, 2, chunk_rows, depth):
            mask = np.uint64((2**64 - 1) ^ ((1 << (64 - bits)) - 1))
            prefix = np.uint64((2**64 - 1) ^ ((1 << (64 - depth)) - 1))
            points = raw[:, 0] & mask
            paired = (raw[:, 1] & prefix) | (points & ~prefix)
            yield points, paired

    def release_panel(self, stage: str) -> None:
        expected = {"Bcoef": "search", "Bdev": "coef", "Btest": "dev"}
        if stage not in expected or self._b_access != expected[stage]:
            raise ValueError("invalid panel release")
        self._check()
        self._b_access = stage[1:].lower()

    def panel_chunks(self, stage: str, *, chunk_rows: int = 50_000) -> Iterator[Any]:
        sizes = {
            "Rcoef": self.recipe.r_coef_size,
            "Ragreement": self.recipe.r_agreement_size,
            "Bcoef": self.recipe.b_coef_size,
            "Bdev": self.recipe.b_dev_size,
            "Btest": self.recipe.b_test_size,
        }
        if stage not in sizes or (stage.startswith("B") and self._b_access != stage[1:].lower()):
            raise ValueError("panel not released")
        bits = self.recipe.r_bits if stage.startswith("R") else self.recipe.b_bits
        for raw in self._samples(stage, sizes[stage], 1, chunk_rows):
            yield raw[:, 0] & np.uint64((2**64 - 1) ^ ((1 << (64 - bits)) - 1))

    def signs(self, stage: str, points: Any, teacher: Any) -> Any:
        if stage.startswith("R"):
            return self.teacher_signs(teacher, points)[0]
        expected = {"Bsearch": "search", "Bcoef": "coef", "Bdev": "dev", "Btest": "test"}
        if stage not in expected or self._b_access != expected[stage]:
            raise ValueError("invalid or unreleased label domain")
        n = array_rows(points, np.dtype(np.uint64))
        if n <= 0 or np.any(points & np.uint64(2**32 - 1)):
            raise ValueError("B plaintext outside high32 domain")
        return self._b_signs(points)
