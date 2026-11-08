"""Controlled opaque bytes only: no NPZ/tensor decoding or scientific execution."""
import builtins
import json
import os
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

import cipher_lab.artifacts as artifacts
from cipher_lab.artifacts import ArtifactError, RunStore, read_saved


def claim(root: Path, *, models: bool = True, budget: int = 1_000_000) -> RunStore:
    return RunStore.claim(root, "binary", output_bytes=budget, needs_models=models)


def record_path(root: Path) -> Path:
    return root / "data/binary/run.json"


def metadata(root: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(record_path(root).read_bytes()))


def read(root: Path, ref: str, content: bytes, limit: int = 1_000_000) -> bytes:
    return artifacts.read_binary(root, "binary", ref, sha256(content).hexdigest(), limit)


def test_schema2_refs_are_dispatched_and_only_json_is_decoded(tmp_path: Path) -> None:
    store = claim(tmp_path)
    payload = store.write_payload("metrics", {"value": 3})
    dataset = store.write_dataset("test", b"opaque\x00NPZ\xff")
    model = store.write_model("best", b"opaque\x00weights\xff")
    assert (payload, dataset, model) == (
        "payload/metrics.json", "dataset/test.npz", "model/best.pt")
    saved = read_saved(tmp_path, "binary")
    assert saved["schema"] == 2
    assert saved["payloads"] == {payload: {"value": 3}}
    refs = cast(dict[str, dict[str, object]], saved["artifacts"])
    for ref, kind, path in [
        (payload, "payload", tmp_path / "data/binary/payload/metrics.json"),
        (dataset, "dataset", tmp_path / "data/binary/dataset/test.npz"),
        (model, "model", tmp_path / "models/binary/best.pt"),
    ]:
        content = path.read_bytes()
        assert refs[ref] == {"kind": kind, "bytes": len(content),
                             "sha256": sha256(content).hexdigest()}
        if kind != "payload":
            assert read(tmp_path, ref, content, len(content)) == content
    assert not (tmp_path / "data/binary/model").exists()
    assert not list((tmp_path / "result/binary").iterdir())


def test_dataset_and_nonbest_model_are_immutable_and_best_is_explicit(tmp_path: Path) -> None:
    store = claim(tmp_path)
    dataset = store.write_dataset("test", b"first")
    model = store.write_model("best", b"old")
    store.write_model("named", b"first")
    operations: list[Callable[[], object]] = [lambda: store.write_dataset("test", b"second"),
                      lambda: store.write_model("best", b"second"),
                      lambda: store.write_model("named", b"second", replace=True)]
    for operation in operations:
        with pytest.raises(ArtifactError, match="^OCCUPIED$"):
            operation()
    assert read(tmp_path, dataset, b"first") == b"first"
    assert read(tmp_path, model, b"old") == b"old"
    assert store.write_model("best", b"new", replace=True) == model
    assert read(tmp_path, model, b"new") == b"new"
    with pytest.raises(ArtifactError, match="^HASH_MISMATCH$"):
        read(tmp_path, model, b"old")


def test_model_requires_the_original_explicit_models_owner(tmp_path: Path) -> None:
    store = claim(tmp_path, models=False)
    # A later directory does not grant the writer a capability.
    (tmp_path / "models/binary").mkdir(parents=True)
    with pytest.raises(ArtifactError, match="^OWNERSHIP_LOST$"):
        store.write_model("best", b"weights")
    assert list((tmp_path / "models/binary").iterdir()) == []
    assert metadata(tmp_path)["artifacts"] == {}


@pytest.mark.parametrize("kind", ["dataset", "model"])
@pytest.mark.parametrize("name", ["../bad", "/bad", "a/b", "a\\b", "a.pt", ".x", "", "x" * 65])
def test_binary_names_cannot_escape(tmp_path: Path, kind: str, name: str) -> None:
    store = claim(tmp_path)
    with pytest.raises(ArtifactError, match="^INVALID_NAME$"):
        getattr(store, "write_" + kind)(name, b"bytes")
    assert metadata(tmp_path)["artifacts"] == {}


@pytest.mark.parametrize("content", [None, "text", bytearray(b"bytes"), memoryview(b"bytes"), True])
@pytest.mark.parametrize("kind", ["dataset", "model"])
def test_binary_content_requires_bytes(tmp_path: Path, kind: str, content: object) -> None:
    store = claim(tmp_path)
    with pytest.raises(ArtifactError, match="^INVALID_CONTENT$"):
        getattr(store, "write_" + kind)("best", content)
    assert metadata(tmp_path)["artifacts"] == {}


@pytest.mark.parametrize("replace", [1, "true", None])
def test_replace_requires_a_boolean(tmp_path: Path, replace: object) -> None:
    store = claim(tmp_path)
    with pytest.raises(ArtifactError, match="^INVALID_STATE$"):
        store.write_model("best", b"x", replace=cast(bool, replace))


@pytest.mark.parametrize("ref", ["../outside", "/etc/passwd", "models/binary/best.pt",
                                  "model/../best.pt", "model/a/b.pt", "model/best.npz",
                                  "dataset/test.pt", "payload/a.json", "model/a\\b.pt"])
def test_read_binary_rejects_nonbinary_or_arbitrary_references(tmp_path: Path, ref: str) -> None:
    store = claim(tmp_path)
    store.write_payload("a", {})
    with pytest.raises(ArtifactError, match="^INVALID_REFERENCE$"):
        read(tmp_path, ref, b"x")


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_read_binary_rejects_invalid_limits(tmp_path: Path, limit: int) -> None:
    store = claim(tmp_path)
    ref = store.write_dataset("test", b"data")
    with pytest.raises(ArtifactError, match="^INVALID_BUDGET$"):
        read(tmp_path, ref, b"data", limit)


def test_read_binary_explicit_sha_metadata_size_and_empty_limit(tmp_path: Path) -> None:
    store = claim(tmp_path)
    ref = store.write_dataset("test", b"data")
    empty = store.write_dataset("empty", b"")
    assert read(tmp_path, empty, b"", 0) == b""
    for limit in [0, 3]:
        with pytest.raises(ArtifactError, match="^SIZE_LIMIT$"):
            read(tmp_path, ref, b"data", limit)
    with pytest.raises(ArtifactError, match="^INVALID_REFERENCE$"):
        read(tmp_path, "dataset/absent.npz", b"data")
    with pytest.raises(ArtifactError, match="^INVALID_SHA256$"):
        artifacts.read_binary(tmp_path, "binary", ref, "bad", 4)
    for bad_id in ["../binary", "other"]:
        with pytest.raises(ArtifactError):
            artifacts.read_binary(tmp_path, bad_id, ref, sha256(b"data").hexdigest(), 4)
    (tmp_path / "data/binary/dataset/test.npz").write_bytes(b"oops")
    readers: list[Callable[[], object]] = [lambda: read(tmp_path, ref, b"data"),
                                          lambda: read_saved(tmp_path, "binary")]
    for reader in readers:
        with pytest.raises(ArtifactError, match="^HASH_MISMATCH$"):
            reader()


@pytest.mark.parametrize("mutation", ["kind", "extra", "missing-kind", "size", "sha", "schema1"])
def test_run_artifact_metadata_is_fixed_schema2(tmp_path: Path, mutation: str) -> None:
    store = claim(tmp_path)
    ref = store.write_dataset("test", b"data")
    value = metadata(tmp_path)
    meta = cast(dict[str, dict[str, object]], value["artifacts"])[ref]
    if mutation == "kind":
        meta["kind"] = "model"
    elif mutation == "extra":
        meta["path"] = "outside"
    elif mutation == "missing-kind":
        del meta["kind"]
    elif mutation == "size":
        meta["bytes"] = True
    elif mutation == "sha":
        meta["sha256"] = "BAD"
    else:
        value["schema"] = 1
    record_path(tmp_path).write_text(json.dumps(value))
    readers: list[Callable[[], object]] = [
        lambda: read(tmp_path, ref, b"data"), lambda: read_saved(tmp_path, "binary")]
    for reader in readers:
        with pytest.raises(ArtifactError, match="^INVALID_SCHEMA$"):
            reader()


@pytest.mark.parametrize("kind", ["dataset", "model"])
@pytest.mark.parametrize("location", ["container", "file"])
def test_binary_symlink_and_foreign_occupancy_never_overwrite(
    tmp_path: Path, kind: str, location: str,
) -> None:
    store = claim(tmp_path)
    parent = tmp_path / ("data/binary/dataset" if kind == "dataset" else "models/binary")
    outside = tmp_path / "outside"
    outside.mkdir()
    evidence = outside / "original"
    evidence.write_bytes(b"foreign")
    if location == "container":
        if parent.exists():
            parent.rmdir()
        parent.symlink_to(outside, target_is_directory=True)
    else:
        parent.mkdir(exist_ok=True)
        (parent / ("best.npz" if kind == "dataset" else "best.pt")).symlink_to(evidence)
    with pytest.raises(ArtifactError, match="^UNSAFE_PATH$"):
        getattr(store, "write_" + kind)("best", b"new")
    assert evidence.read_bytes() == b"foreign"
    assert list(outside.iterdir()) == [evidence]


@pytest.mark.parametrize("kind", ["dataset", "model"])
def test_unowned_existing_binary_is_preserved(tmp_path: Path, kind: str) -> None:
    store = claim(tmp_path)
    parent = tmp_path / ("data/binary/dataset" if kind == "dataset" else "models/binary")
    parent.mkdir(exist_ok=True)
    target = parent / ("best.npz" if kind == "dataset" else "best.pt")
    target.write_bytes(b"foreign")
    with pytest.raises(ArtifactError):
        getattr(store, "write_" + kind)("best", b"new")
    assert target.read_bytes() == b"foreign"


@pytest.mark.parametrize("kind", ["dataset", "model"])
def test_readers_reject_symlinked_binary(tmp_path: Path, kind: str) -> None:
    store = claim(tmp_path)
    ref = getattr(store, "write_" + kind)("best", b"data")
    target = tmp_path / ("data/binary/dataset/best.npz" if kind == "dataset"
                         else "models/binary/best.pt")
    outside = tmp_path / "outside"
    outside.write_bytes(b"data")
    target.unlink()
    target.symlink_to(outside)
    readers: list[Callable[[], object]] = [
        lambda: read(tmp_path, ref, b"data"), lambda: read_saved(tmp_path, "binary")]
    for reader in readers:
        with pytest.raises(ArtifactError, match="^UNSAFE_PATH$"):
            reader()
    assert outside.read_bytes() == b"data"


def test_report_reader_streams_binary_without_read_bytes_or_scientific_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path, budget=10_000_000)
    content = b"opaque\xff" * 300_000
    ref = store.write_dataset("test", content)
    store.write_model("best", content)
    store.finalize({"status": "SUCCEEDED"})
    original_read = Path.read_bytes
    original_import = builtins.__import__

    def guarded_read(path: Path) -> bytes:
        assert path.suffix not in {".pt", ".npz"}, "binary must stream, never read_bytes"
        return original_read(path)

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        assert name.split(".")[0] not in {"torch", "numpy"}
        return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    saved = read_saved(tmp_path, "binary")
    assert saved["payloads"] == {}
    assert read(tmp_path, ref, content, len(content)) == content


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "STOPPED", "ALARM"])
def test_terminal_binary_is_immutable_but_readable(tmp_path: Path, status: str) -> None:
    store = claim(tmp_path)
    ref = store.write_model("best", b"weights")
    store.finalize({"status": status})
    before = record_path(tmp_path).read_bytes()
    operations: list[Callable[[], object]] = [
        lambda: store.write_model("best", b"new", replace=True),
                      lambda: store.write_dataset("late", b"new")]
    for operation in operations:
        with pytest.raises(ArtifactError, match="^FINALIZED$"):
            operation()
    assert read(tmp_path, ref, b"weights") == b"weights"
    assert read_saved(tmp_path, "binary")["status"] == status
    assert record_path(tmp_path).read_bytes() == before


@pytest.mark.parametrize("failure", ["model-rename", "metadata-once", "metadata-always", "partial"])
def test_best_replace_failures_never_pair_new_bytes_with_successful_old_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    store = claim(tmp_path)
    identity = {"backend_identity": "ciphers", "scientific_validation": False}
    store.write_snapshot({"summary": identity})
    ref = store.write_model("best", b"old-complete")
    original = os.replace
    write = os.write
    failed = False

    def broken(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal failed
        target = Path(dst)
        if ((failure == "model-rename" and target.name == "best.pt")
                or (failure.startswith("metadata") and target.name == "run.json"
                    and (not failed or failure == "metadata-always"))):
            failed = True
            raise PermissionError("PRIVATE media detail")
        original(src, dst)

    def partial(fd: int, content: bytes) -> int:
        nonlocal failed
        if not failed:
            failed = True
            write(fd, content[:2])
            raise OSError("PRIVATE partial write")
        return write(fd, content)

    monkeypatch.setattr(os, "replace", broken)
    if failure == "partial":
        monkeypatch.setattr(os, "write", partial)
    with pytest.raises(ArtifactError, match="^WRITE_FAILED$"):
        store.write_model("best", b"new-complete", replace=True)
    assert not list(tmp_path.rglob("*.tmp"))
    assert b"PRIVATE" not in record_path(tmp_path).read_bytes()
    with pytest.raises(ArtifactError, match="^FINALIZED$"):
        store.write_model("best", b"late", replace=True)
    if failure == "metadata-always":
        assert (tmp_path / "models/binary/best.pt").read_bytes() == b"new-complete"
        readers: list[Callable[[], object]] = [lambda: read_saved(tmp_path, "binary"),
                                              lambda: read(tmp_path, ref, b"old-complete")]
        for reader in readers:
            with pytest.raises(ArtifactError, match="^HASH_MISMATCH$"):
                reader()
    else:
        saved = read_saved(tmp_path, "binary")
        assert saved["status"] == "FAILED"
        assert saved["summary"] == {**identity, "reason": "WRITE_FAILED"}
        content = b"new-complete" if failure == "metadata-once" else b"old-complete"
        assert read(tmp_path, ref, content) == content
        assert saved["source"] == store.source_snapshot()
        assert saved["environment"] == metadata(tmp_path)["environment"]


def test_best_peak_budget_includes_old_model_temp_and_new_metadata(tmp_path: Path) -> None:
    store = claim(tmp_path)
    store.write_model("best", b"o" * 100)
    current = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    # A final replacement fits, but old+new+metadata temporary peak does not.
    store._limit = current + 10_000
    with pytest.raises(ArtifactError, match="^OUTPUT_LIMIT$"):
        store.write_model("best", b"n" * 10_000, replace=True)
    assert (tmp_path / "models/binary/best.pt").read_bytes() == b"o" * 100
    assert read_saved(tmp_path, "binary")["status"] == "FAILED"
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("kind", ["dataset", "model"])
def test_binary_and_metadata_preflight_share_all_existing_files_budget(
    tmp_path: Path, kind: str,
) -> None:
    store = claim(tmp_path)
    store.append_event({"kind": "progress", "summary": {"public": "e" * 1000}})
    store.write_payload("metrics", {"public": "p" * 1000})
    current = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    store._limit = current + 10_000
    with pytest.raises(ArtifactError, match="^OUTPUT_LIMIT$"):
        getattr(store, "write_" + kind)("best", b"x" * 9900)
    assert set(cast(dict[str, object], metadata(tmp_path)["artifacts"])) == {"payload/metrics.json"}
    assert not list(tmp_path.rglob("*.npz"))
    assert not list(tmp_path.rglob("*.pt"))


def test_read_binary_validates_actual_size_against_metadata(tmp_path: Path) -> None:
    store = claim(tmp_path)
    ref = store.write_dataset("test", b"data")
    (tmp_path / "data/binary/dataset/test.npz").write_bytes(b"data-extra")
    with pytest.raises(ArtifactError, match="^HASH_MISMATCH$"):
        read(tmp_path, ref, b"data")


def test_binary_read_is_read_only_even_if_returned_metadata_is_mutated(tmp_path: Path) -> None:
    store = claim(tmp_path)
    ref = store.write_dataset("test", b"data")
    before = record_path(tmp_path).read_bytes()
    saved = read_saved(tmp_path, "binary")
    cast(dict[str, object], saved["artifacts"]).clear()
    assert read(tmp_path, ref, b"data") == b"data"
    assert record_path(tmp_path).read_bytes() == before
    with pytest.raises(ArtifactError, match="^OCCUPIED$"):
        claim(tmp_path)


def test_failed_temp_creation_does_not_unlink_a_foreign_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path)
    foreign = tmp_path / "models/binary/.best.pt-fixed.tmp"
    foreign.write_bytes(b"foreign evidence")

    class Fixed:
        hex = "fixed"

    monkeypatch.setattr(artifacts, "uuid4", Fixed)
    with pytest.raises(ArtifactError, match="^WRITE_FAILED$"):
        store.write_model("best", b"data")
    assert foreign.read_bytes() == b"foreign evidence"


def test_read_binary_does_not_decode_unrelated_json_or_binary(tmp_path: Path) -> None:
    store = claim(tmp_path)
    ref = store.write_dataset("test", b"data")
    store.write_payload("unrelated", {})
    (tmp_path / "data/binary/payload/unrelated.json").unlink()
    assert read(tmp_path, ref, b"data") == b"data"


def test_binary_report_reads_and_owner_writes_only_three_report_files(tmp_path: Path) -> None:
    from cipher_lab.integral.report import render_report

    store = claim(tmp_path)
    store.write_dataset("test", b"opaque-npz")
    store.write_model("best", b"opaque-checkpoint")
    store.finalize({"status": "SUCCEEDED"})
    store.write_report_files(render_report(read_saved(tmp_path, "binary")))
    assert {p.name for p in (tmp_path / "result/binary").iterdir()} == {
        "report.md", "index.html", "source.json"}
