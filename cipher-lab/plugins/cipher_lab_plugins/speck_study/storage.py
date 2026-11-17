"""Study-only retained RunStore adapter: registered binary reads and context.

Task5 passes the already claimed owner plus its independently known root/run_id.
No ownership acquisition, no RunStore private fields, no fallback/new finalizer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cipher_lab.artifacts import ArtifactError, RunStore, read_binary, read_saved
from cipher_lab_plugins.speck_study.codec import MAX_MODEL_BYTES, StudyPredictor, decode_predictor
from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe


def source_digest(source: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class StudyStore:
    """Only this run's live original owner is retained. Dataset/model are opaque."""

    def __init__(self, owner: RunStore, root: Path, run_id: str) -> None:
        self.owner, self.root, self.run_id = owner, root, run_id

    def __getattr__(self, name: str) -> Any:
        # Explicit forwarding allowlist, not a generic registry or reopened writer.
        if name not in {
            "source_snapshot",
            "write_snapshot",
            "append_event",
            "write_payload",
            "write_dataset",
            "write_model",
            "read_model",
            "finalize",
        }:
            raise AttributeError(name)
        return getattr(self.owner, name)

    def read_dataset(self, ref: str, digest: str, size: int) -> bytes:
        return read_binary(self.root, self.run_id, ref, digest, size)

    def verify_context(
        self, recipe: SpeckStudyRecipe, source: dict[str, str], backend: str
    ) -> None:
        saved = read_saved(self.root, self.run_id)
        summary = saved["summary"]
        if (
            self.owner.current_source_identity() != source
            or saved["recipe"] != recipe.to_dict()
            or saved["source"] != source
            or not isinstance(summary, dict)
            or summary.get("backend_identity") != backend
        ):
            raise ArtifactError("HASH_MISMATCH")


def durable_model(
    store: Any,
    model: StudyPredictor,
    *,
    expected_kind: str,
    expected_recipe: SpeckStudyRecipe,
    expected_source: dict[str, str],
    expected_backend: str,
) -> StudyPredictor:
    """External identity is constructed from caller's frozen context, never model bytes."""
    from cipher_lab_plugins.speck_study.codec import encode_predictor

    if expected_kind not in ("R", "B") or model.kind != expected_kind:
        raise ValueError("external kind mismatch")
    identity: dict[str, object] = dict(
        kind=expected_kind,
        task="teacher-reproduction" if expected_kind == "R" else "absolute-bit-prediction",
        source_sha256=source_digest(expected_source),
        method=expected_recipe.method,
        synthetic=expected_recipe.synthetic,
    )
    raw = encode_predictor(model)
    ref = store.write_model("A-best" if expected_kind == "R" else "B-best", raw)
    digest = hashlib.sha256(raw).hexdigest()
    store.verify_context(expected_recipe, expected_source, expected_backend)
    saved = store.read_model(ref, digest, MAX_MODEL_BYTES)
    # RunStore.read_model checks canonical registration AND actual bytes. Repeat raw
    # digest here for injected adapters too; do not trust an adapter to supply fallback.
    if hashlib.sha256(saved).hexdigest() != digest:
        raise ArtifactError("HASH_MISMATCH")
    return decode_predictor(saved, expected_identity=identity, expected_recipe=expected_recipe)
