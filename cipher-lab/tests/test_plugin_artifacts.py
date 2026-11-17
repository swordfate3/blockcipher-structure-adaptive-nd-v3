"""Generic storage extensions use opaque fake bytes and pytest-owned paths."""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

import cipher_lab.artifacts as artifacts
from cipher_lab.artifacts import ArtifactError, RunStore, read_saved


@pytest.fixture(autouse=True)
def fake_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifacts, "source_identity", lambda: {"cipher_lab/fake.py": "0" * 64})
    monkeypatch.setattr(artifacts, "_environment_identity", lambda: {
        "python_implementation": "CPython", "python_version": "3.12.0",
        "os": "Synthetic", "architecture": "fake",
    })


def test_five_declared_slots_can_be_replaced_and_read_from_disk(tmp_path: Path) -> None:
    names = ("first-best", "second-best", "third-best", "fourth-best", "fifth-best")
    store = RunStore.claim(tmp_path, "slots", output_bytes=100_000, needs_models=True,
                           model_slots=(*names, names[0]))
    for round_ in range(3):
        for name in names:
            content = f"fake {name} revision {round_}".encode()
            ref = store.write_model(name, content, replace=True)
            assert ref == f"model/{name}.pt"
            assert store.read_model(ref, sha256(content).hexdigest(), 1000) == content
    assert len(list((tmp_path / "models/slots").iterdir())) == 5
    store.finalize({"status": "SUCCEEDED"})
    assert len(cast(dict[str, object], read_saved(tmp_path, "slots")["artifacts"])) == 5


@pytest.mark.parametrize("name", ["best", "A-best", "B-best", "C-best"])
def test_legacy_best_permissions_unchanged(tmp_path: Path, name: str) -> None:
    store = RunStore.claim(tmp_path, "old", output_bytes=100_000, needs_models=True)
    ref = store.write_model(name, b"fake original", replace=True)
    store.write_model(name, b"fake updated", replace=True)
    assert store.read_model(ref, sha256(b"fake updated").hexdigest(), 100) == b"fake updated"


def test_undeclared_slot_stays_immutable_and_permission_is_owner_local(tmp_path: Path) -> None:
    permitted = RunStore.claim(tmp_path, "permitted", output_bytes=100_000, needs_models=True,
                               model_slots=("custom-best",))
    denied = RunStore.claim(tmp_path, "denied", output_bytes=100_000, needs_models=True)
    permitted.write_model("custom-best", b"fake", replace=True)
    denied.write_model("custom-best", b"first")
    for replace in (False, True):
        with pytest.raises(ArtifactError, match="^OCCUPIED$"):
            denied.write_model("custom-best", b"second", replace=replace)
    assert (tmp_path / "models/denied/custom-best.pt").read_bytes() == b"first"
    with pytest.raises(ArtifactError, match="^OCCUPIED$"):
        permitted.write_model("undeclared", b"fake", replace=True)


@pytest.mark.parametrize("slots", [
    ("",), ("../x",), ("x/y",), ("x.y",), ("x\n",), ("x" * 65,),
    (True,), ["okay"], "okay", None,
])
def test_slots_reject_invalid_tokens_or_container_before_io(
    tmp_path: Path, slots: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> dict[str, str]:
        pytest.fail("invalid slots must fail before source scan or claim")

    monkeypatch.setattr(artifacts, "source_identity", forbidden)
    with pytest.raises(ArtifactError, match="^INVALID_NAME$"):
        RunStore.claim(tmp_path, "invalid", output_bytes=100_000, needs_models=True,
                       model_slots=cast(tuple[str, ...], slots))
    assert list(tmp_path.iterdir()) == []


def test_slots_do_not_implicitly_claim_models_directory(tmp_path: Path) -> None:
    store = RunStore.claim(tmp_path, "no-models", output_bytes=100_000,
                           model_slots=("custom-best",))
    assert not (tmp_path / "models").exists()
    with pytest.raises(ArtifactError, match="^OWNERSHIP_LOST$"):
        store.write_model("custom-best", b"fake", replace=True)


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "STOPPED", "ALARM"])
def test_html_requires_original_finalized_owner_and_preserves_frozen_record(
    tmp_path: Path, status: str,
) -> None:
    store = RunStore.claim(tmp_path, "html", output_bytes=100_000)
    with pytest.raises(ArtifactError, match="^NOT_FINALIZED$"):
        store.write_html_report(b"<p>too early</p>")
    store.finalize({"status": status})
    record = tmp_path / "data/html/run.json"
    frozen = record.read_bytes()
    for content in (b"<p>fake saved output</p>", b"<p>fake re-render</p>"):
        store.write_html_report(content)
        assert record.read_bytes() == frozen
        assert (tmp_path / "result/html/report.html").read_bytes() == content
        assert read_saved(tmp_path, "html")["status"] == status
    assert [p.name for p in (tmp_path / "result/html").iterdir()] == ["report.html"]
    with pytest.raises(ArtifactError, match="^OCCUPIED$"):
        RunStore.claim(tmp_path, "html", output_bytes=100_000)
    other = RunStore.claim(tmp_path, "other", output_bytes=100_000)
    with pytest.raises(ArtifactError, match="^NOT_FINALIZED$"):
        other.write_html_report(b"fake")


@pytest.mark.parametrize("content", [None, "html", bytearray(b"html"), True])
def test_html_accepts_only_bytes(tmp_path: Path, content: object) -> None:
    store = RunStore.claim(tmp_path, "html", output_bytes=100_000)
    store.finalize({"status": "SUCCEEDED"})
    with pytest.raises(ArtifactError, match="^INVALID_REPORT$"):
        store.write_html_report(cast(bytes, content))
    assert list((tmp_path / "result/html").iterdir()) == []


@pytest.mark.parametrize("mutation", ["run", "missing-run", "run-symlink", "foreign-html",
                                      "changed-html", "missing-html", "html-symlink"])
def test_html_rejects_run_tampering_and_target_ownership_loss(
    tmp_path: Path, mutation: str,
) -> None:
    store = RunStore.claim(tmp_path, "html", output_bytes=100_000)
    store.finalize({"status": "STOPPED"})
    run = tmp_path / "data/html/run.json"
    target = tmp_path / "result/html/report.html"
    external = tmp_path / "outside"
    external.write_bytes(b"external fake bytes")
    if mutation in ("changed-html", "missing-html"):
        store.write_html_report(b"previous")
    if mutation == "run":
        run.write_bytes(b"{}")
    elif mutation == "missing-run":
        run.unlink()
    elif mutation == "run-symlink":
        run.unlink()
        run.symlink_to(external)
    elif mutation == "foreign-html":
        target.write_bytes(b"foreign")
    elif mutation == "changed-html":
        target.write_bytes(b"changed")
    elif mutation == "missing-html":
        target.unlink()
    else:
        target.symlink_to(external)
    before = run.read_bytes() if run.exists() else None
    with pytest.raises(ArtifactError, match="^(OWNERSHIP_LOST|UNSAFE_PATH)$"):
        store.write_html_report(b"replacement")
    assert (run.read_bytes() if run.exists() else None) == before
    assert external.read_bytes() == b"external fake bytes"


def test_html_initial_and_replacement_peak_budget(tmp_path: Path) -> None:
    probe = RunStore.claim(tmp_path / "probe", "html", output_bytes=100_000)
    probe.finalize({"status": "SUCCEEDED"})
    size = (tmp_path / "probe/data/html/run.json").stat().st_size
    content = b"h" * 10_000
    # Enough to finalize and write once; exact final size is not replacement peak.
    store = RunStore.claim(tmp_path / "exact", "html", output_bytes=size + len(content))
    store.finalize({"status": "SUCCEEDED"})
    store.write_html_report(content)
    frozen = (tmp_path / "exact/data/html/run.json").read_bytes()
    with pytest.raises(ArtifactError, match="^OUTPUT_LIMIT$"):
        store.write_html_report(content)
    assert (tmp_path / "exact/result/html/report.html").read_bytes() == content
    assert (tmp_path / "exact/data/html/run.json").read_bytes() == frozen
    assert not list((tmp_path / "exact/result/html").glob("*.tmp"))
    small = RunStore.claim(tmp_path / "small", "html", output_bytes=size + len(content) - 1)
    small.finalize({"status": "SUCCEEDED"})
    with pytest.raises(ArtifactError, match="^OUTPUT_LIMIT$"):
        small.write_html_report(content)
    assert list((tmp_path / "small/result/html").iterdir()) == []


def test_html_atomic_failure_does_not_change_scientific_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore.claim(tmp_path, "html", output_bytes=100_000)
    store.finalize({"status": "SUCCEEDED"})
    store.write_html_report(b"previous fake report")
    frozen = (tmp_path / "data/html/run.json").read_bytes()

    def broken(src: object, dst: object) -> None:
        raise OSError("PRIVATE media detail")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(ArtifactError, match="^WRITE_FAILED$") as caught:
        store.write_html_report(b"replacement")
    assert caught.value.__cause__ is None
    assert (tmp_path / "data/html/run.json").read_bytes() == frozen
    assert (tmp_path / "result/html/report.html").read_bytes() == b"previous fake report"
    assert not list((tmp_path / "result/html").glob("*.tmp"))


def test_three_file_report_api_still_works_alongside_single_html(tmp_path: Path) -> None:
    store = RunStore.claim(tmp_path, "legacy", output_bytes=100_000)
    store.finalize({"status": "STOPPED"})
    saved = read_saved(tmp_path, "legacy")
    files = {"report.md": b"fake markdown", "index.html": b"fake legacy html",
             "source.json": json.dumps({
                 "schema": 1, "run_id": "legacy", "run_ref": saved["run_ref"],
                 "run_sha256": saved["run_sha256"],
             }).encode()}
    store.write_report_files(files)
    store.write_html_report(b"fake single html")
    store.write_report_files(files)
    for name, content in files.items():
        assert (tmp_path / "result/legacy" / name).read_bytes() == content
    assert (tmp_path / "result/legacy/report.html").read_bytes() == b"fake single html"


@pytest.mark.parametrize("operation", ["atomic", "append"])
def test_low_level_writes_explicitly_request_binary_mode_and_preserve_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    # Emulate the Windows flag on Linux while retaining real temporary-file I/O.
    # The fake CRT translates LF only when the requested mode omits O_BINARY.
    binary_flag = 1 << 29
    original_open, original_write = os.open, os.write
    opened: list[int] = []
    modes: dict[int, int] = {}

    def open_(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        fd = original_open(path, flags & ~binary_flag, mode)
        opened.append(flags)
        modes[fd] = flags
        return fd

    def write_(fd: int, content: bytes) -> int:
        if not modes[fd] & binary_flag:
            original_write(fd, content.replace(b"\n", b"\r\n"))
            return len(content)
        return original_write(fd, content)

    store = RunStore.claim(tmp_path, "binary", output_bytes=100_000, needs_models=True)
    monkeypatch.setattr(os, "O_BINARY", binary_flag, raising=False)
    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "write", write_)
    if operation == "atomic":
        content = b"fake npz/pt\x00\n\r\n\x1a\xff"
        ref = store.write_model("best", content, replace=True)
        assert (tmp_path / "models/binary/best.pt").read_bytes() == content
        assert store.read_model(ref, sha256(content).hexdigest(), 100) == content
        store.write_dataset("fake", content)
        assert (tmp_path / "data/binary/dataset/fake.npz").read_bytes() == content
        store.finalize({"status": "SUCCEEDED"})
        store.write_html_report(content)
        assert (tmp_path / "result/binary/report.html").read_bytes() == content
    else:
        store.append_event({"kind": "fake", "summary": {"line": "a\nb"}})
        store.append_event({"kind": "fake"})
        raw = (tmp_path / "data/binary/events.jsonl").read_bytes()
        assert raw.count(b"\n") == 2
        assert b"\r" not in raw
        assert [json.loads(line)["sequence"] for line in raw.splitlines()] == [1, 2]
    assert opened and all(flags & binary_flag for flags in opened)
