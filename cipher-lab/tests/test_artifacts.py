"""Storage tests use public synthetic JSON and real pytest temporary directories only."""
import json
import os
import platform
from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

import cipher_lab.artifacts as artifacts
from cipher_lab.artifacts import ArtifactError, RunStore, read_saved, source_identity


def claim(root: Path, budget: int = 1_000_000) -> RunStore:
    return RunStore.claim(root, "sample-1", output_bytes=budget)


def run_path(root: Path) -> Path:
    return root / "data/sample-1/run.json"


def disk_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def render_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Synthetic pure renderer only; Task5 implements the actual report renderer."""
    return {
        "report.md": b"synthetic public report",
        "index.html": b"<p>synthetic public report</p>",
        "source.json": json.dumps({
            "schema": 1, "run_id": saved["run_id"], "run_ref": saved["run_ref"],
            "run_sha256": saved["run_sha256"],
        }).encode(),
    }


def report_files(root: Path) -> dict[str, bytes]:
    return render_report(read_saved(root, "sample-1"))


@pytest.mark.parametrize("run_id", ["", ".", "..", "a.b", "a/b", "../x", "/tmp/x", "a\\b",
                                   "a b", "a\n", "x" * 65])
def test_id_rejects_traversal_before_mkdir(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ArtifactError):
        RunStore.claim(tmp_path, run_id, output_bytes=100_000)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", ["RUNNING", "SUCCEEDED", "FAILED", "STOPPED", "ALARM",
                                   "storage-failure"])
def test_environment_is_claim_only_public_identity_retained_through_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    expected = {"python_implementation": "CPython", "python_version": "3.12.15",
                "os": "Linux", "architecture": "x86_64"}
    monkeypatch.setenv("CIPHER_LAB_ENV_CANARY", "PRIVATE-environment-canary")
    probes = {"python_implementation": "python_implementation", "python_version": "python_version",
              "system": "os", "machine": "architecture"}
    for probe, field in probes.items():
        # A one-element source fails if claim probes a field more than once.
        values = iter([expected[field]])
        monkeypatch.setattr(platform, probe, lambda values=values: next(values))
    store = claim(tmp_path, 30_000)
    assert read_saved(tmp_path, "sample-1")["environment"] == expected

    def forbidden() -> str:
        pytest.fail("saved identity/report must not probe the current environment")

    for probe in probes:
        monkeypatch.setattr(platform, probe, forbidden)
    store.write_snapshot({"summary": {"synthetic": True}})
    store.append_event({"kind": "progress"})
    store.write_payload("public", {"synthetic": True})
    if status == "storage-failure":
        with pytest.raises(ArtifactError, match="^OUTPUT_LIMIT$"):
            store.write_payload("too-large", "x" * 30_000)
    elif status != "RUNNING":
        store.finalize({"status": status})
    saved = read_saved(tmp_path, "sample-1")
    assert saved["environment"] == expected
    assert saved["status"] == ("FAILED" if status == "storage-failure" else status)
    if status != "RUNNING":
        # Exercise the actual pure renderer and writer, without any science.
        from cipher_lab.integral.report import render_report as actual_render_report

        store.write_report_files(actual_render_report(saved))
        assert read_saved(tmp_path, "sample-1")["environment"] == expected
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"PRIVATE-environment-canary" not in path.read_bytes()


@pytest.mark.parametrize("probe", ["python_implementation", "python_version", "system", "machine"])
@pytest.mark.parametrize("failure", ["exception", "empty", "type", "unsafe", "long"])
def test_environment_probe_failure_is_closed_before_claim_creates_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: str, failure: str,
) -> None:
    def broken() -> object:
        if failure == "exception":
            raise RuntimeError("PRIVATE-probe-failure")
        return {"empty": "", "type": True, "unsafe": "/private/path", "long": "x" * 65}[failure]

    monkeypatch.setattr(platform, probe, broken)
    with pytest.raises(ArtifactError, match="^ENVIRONMENT_UNAVAILABLE$") as caught:
        claim(tmp_path / "output")
    assert caught.value.__cause__ is None
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["missing", "not-mapping", "missing-key", "extra-key",
                                      "type", "empty", "version", "unsafe", "long"])
def test_read_saved_strictly_validates_environment(tmp_path: Path, mutation: str) -> None:
    claim(tmp_path)
    value = json.loads(run_path(tmp_path).read_bytes())
    # Explicit valid schema fixture: no dependency on the machine reading it.
    environment = {"python_implementation": "CPython", "python_version": "3.12.15",
                   "os": "Linux", "architecture": "x86_64"}
    value["environment"] = environment
    if mutation == "missing":
        del value["environment"]
    elif mutation == "not-mapping":
        value["environment"] = []
    elif mutation == "missing-key":
        del environment["os"]
    elif mutation == "extra-key":
        environment["hostname"] = "not-allowed"
    elif mutation == "type":
        value["environment"] = {**environment, "architecture": True}
    elif mutation == "empty":
        environment["os"] = ""
    elif mutation == "version":
        environment["python_version"] = "3.x"
    elif mutation == "unsafe":
        environment["architecture"] = "../private"
    elif mutation == "long":
        environment["python_implementation"] = "x" * 65
    run_path(tmp_path).write_text(json.dumps(value))
    with pytest.raises(ArtifactError, match="^INVALID_SCHEMA$"):
        read_saved(tmp_path, "sample-1")


def test_read_saved_accepts_other_saved_environment_without_reprobing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim(tmp_path)
    value = json.loads(run_path(tmp_path).read_bytes())
    environment = {"python_implementation": "PyPy", "python_version": "3.13.1",
                   "os": "Windows", "architecture": "ARM64"}
    value["environment"] = environment
    run_path(tmp_path).write_text(json.dumps(value))

    def forbidden() -> str:
        pytest.fail("reader must validate saved values, not compare against this machine")

    for probe in ("python_implementation", "python_version", "system", "machine"):
        monkeypatch.setattr(platform, probe, forbidden)
    assert read_saved(tmp_path, "sample-1")["environment"] == environment


def test_environment_bytes_are_included_in_claim_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"cipher_lab/fake.py": "0" * 64}
    monkeypatch.setattr(artifacts, "source_identity", lambda: identity.copy())
    # The old, environment-less record would fit this exact budget.
    old_record = {"schema": 2, "run_id": "sample-1", "source": identity, "status": "RUNNING",
                  "recipe": {}, "policy": {}, "counts": {}, "summary": {}, "artifacts": {}}
    budget = len((json.dumps(old_record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    with pytest.raises(ArtifactError, match="^OUTPUT_LIMIT$"):
        claim(tmp_path / "output", budget)
    assert not (tmp_path / "output").exists()


def test_claim_snapshot_source_and_default_no_models(tmp_path: Path) -> None:
    store = claim(tmp_path)
    assert (tmp_path / "result/sample-1").is_dir()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "data/sample-1/payload").exists()
    store.write_snapshot({"recipe": {"synthetic": True}, "policy": {"output_bytes": 1_000_000},
                          "counts": {"attempted": 0}, "summary": {"note": "public"}})
    saved = read_saved(tmp_path, "sample-1")
    assert saved["schema"] == 2
    assert saved["status"] == "RUNNING"
    assert saved["recipe"] == {"synthetic": True}
    assert saved["run_sha256"] == sha256(run_path(tmp_path).read_bytes()).hexdigest()
    identity = source_identity()
    assert "cipher_lab/artifacts.py" in identity
    assert all(k.startswith("cipher_lab/") and k.endswith(".py") for k in identity)
    assert saved["source"] == identity
    with pytest.raises(ArtifactError):
        claim(tmp_path)


@pytest.mark.parametrize("status", ["RUNNING", "SUCCEEDED", "FAILED", "STOPPED", "ALARM"])
def test_source_snapshot_is_detached_claim_identity_without_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    identity = {"cipher_lab/fake.py": "0" * 64}
    monkeypatch.setattr(artifacts, "source_identity", lambda: identity.copy())
    store = claim(tmp_path)
    if status != "RUNNING":
        store.finalize({"status": status})
    before = run_path(tmp_path).read_bytes()
    identity["cipher_lab/fake.py"] = "1" * 64

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("source_snapshot must not inspect source or perform I/O")

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "source_identity", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(Path, "write_bytes", forbidden)
        snapshot = store.source_snapshot()
        assert snapshot == {"cipher_lab/fake.py": "0" * 64}
        snapshot["cipher_lab/fake.py"] = "2" * 64
        snapshot["cipher_lab/added.py"] = "3" * 64
        assert store.source_snapshot() == {"cipher_lab/fake.py": "0" * 64}
    assert run_path(tmp_path).read_bytes() == before
    saved = read_saved(tmp_path, "sample-1")
    assert saved["source"] == {"cipher_lab/fake.py": "0" * 64}
    assert saved["status"] == status


@pytest.mark.parametrize("entrypoint", ["source_identity", "claim"])
@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_source_scan_failure_rejects_partial_identity_and_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str,
    error_type: type[OSError],
) -> None:
    package = tmp_path / "synthetic/cipher_lab"
    blocked = package / "nested"
    blocked.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"# public synthetic package\n")
    (blocked / "child.py").write_bytes(b"# public synthetic child\n")
    monkeypatch.setattr(artifacts, "__file__", str(package / "__init__.py"))
    assert source_identity() == {
        "cipher_lab/__init__.py": sha256(b"# public synthetic package\n").hexdigest(),
        "cipher_lab/nested/child.py": sha256(b"# public synthetic child\n").hexdigest(),
    }
    original = os.scandir

    def scan(path: str | os.PathLike[str]) -> Iterator[os.DirEntry[str]]:
        if Path(path) == blocked:
            raise error_type("PRIVATE scanning failure")
        return original(path)

    root = tmp_path / "output"
    monkeypatch.setattr(os, "scandir", scan)
    with pytest.raises(ArtifactError, match="^SOURCE_UNAVAILABLE$") as caught:
        if entrypoint == "source_identity":
            source_identity()
        else:
            claim(root)
    assert caught.value.__cause__ is None
    assert not root.exists()


@pytest.mark.parametrize("location", ["data", "result", "models"])
def test_occupied_directories_preserved(tmp_path: Path, location: str) -> None:
    occupied = tmp_path / location / "sample-1"
    occupied.mkdir(parents=True)
    (occupied / "old").write_bytes(b"untouched")
    with pytest.raises(ArtifactError):
        RunStore.claim(tmp_path, "sample-1", output_bytes=100_000, needs_models=True)
    assert (occupied / "old").read_bytes() == b"untouched"
    assert sorted(p for p in tmp_path.rglob("*") if p.is_file()) == [occupied / "old"]
    for other in {"data", "result", "models"} - {location}:
        assert not (tmp_path / other / "sample-1").exists()


@pytest.mark.parametrize("location", ["data", "result", "data/sample-1", "result/sample-1"])
def test_claim_rejects_symlink_locations(tmp_path: Path, location: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "root" / location
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError):
        claim(tmp_path / "root")
    assert list(outside.iterdir()) == []
    assert link.is_symlink()


def test_root_symlink_parent_and_dotdot_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(ArtifactError):
        claim(tmp_path / "alias/new")
    with pytest.raises(ArtifactError):
        claim(target / "../new")
    assert list(target.iterdir()) == []


def test_mid_claim_mkdir_failure_only_removes_owned_empty_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/other").write_bytes(b"existing")
    original = Path.mkdir

    def broken(path: Path, *args: object, **kwargs: object) -> None:
        if path == tmp_path / "result/sample-1":
            raise OSError("private detail")
        original(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", broken)
    with pytest.raises(ArtifactError, match="CLAIM_FAILED") as caught:
        claim(tmp_path)
    assert "private" not in str(caught.value)
    assert (tmp_path / "data/other").read_bytes() == b"existing"
    assert not (tmp_path / "data/sample-1").exists()
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_or_zero_budget_fails_closed(tmp_path: Path, budget: int) -> None:
    with pytest.raises(ArtifactError):
        claim(tmp_path, budget)
    assert list(tmp_path.iterdir()) == []


def test_metadata_charged_and_atomic_peak_not_only_final_size(tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    probe_store = claim(probe)
    initial = disk_bytes(probe)
    assert initial > 0
    exact = tmp_path / "exact"
    store = claim(exact, initial)
    old = run_path(exact).read_bytes()
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_snapshot({"summary": {}})
    assert run_path(exact).read_bytes() == old
    with pytest.raises(ArtifactError):
        store.append_event({"kind": "progress"})
    assert disk_bytes(exact) == initial
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        claim(tmp_path / "too-small", initial - 1)
    assert not (tmp_path / "too-small/data/sample-1").exists()
    # A same-size replacement fits the *final* size, but not old + temporary bytes.
    probe_store.write_snapshot({"summary": {"note": "ok"}})


def test_events_are_small_append_only_records(tmp_path: Path) -> None:
    store = claim(tmp_path)
    before = run_path(tmp_path).read_bytes()
    store.append_event({"kind": "progress", "counts": {"completed": 2}})
    store.append_event({"kind": "progress", "summary": {"cell": 1}})
    lines = (tmp_path / "data/sample-1/events.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["sequence"] for e in events] == [1, 2]
    assert all(e["schema"] == 2 and "recipe" not in e and "source" not in e for e in events)
    assert run_path(tmp_path).read_bytes() == before
    with pytest.raises(ArtifactError):
        store.append_event({"kind": "progress", "recipe": {}})


@pytest.mark.parametrize("bad", [{"key": "private"}, {"source": {}}, {"environment": {}},
                                  {"schema": 99}, {"status": "SUCCEEDED"},
                                  {"summary": RuntimeError("secret")},
                                  {"summary": {"x": float("nan")}}, {"counts": []}])
def test_fixed_snapshot_fields_and_json_boundary(tmp_path: Path, bad: Mapping[str, object]) -> None:
    store = claim(tmp_path)
    before = run_path(tmp_path).read_bytes()
    with pytest.raises(ArtifactError):
        store.write_snapshot(bad)
    assert run_path(tmp_path).read_bytes() == before
    assert b"private" not in before and b"secret" not in before


def test_payload_ref_hash_content_and_freeze(tmp_path: Path) -> None:
    store = claim(tmp_path)
    ref = store.write_payload("basis", {"rows": [0, 3], "width": 2})
    assert ref == "payload/basis.json"
    raw = (tmp_path / "data/sample-1" / ref).read_bytes()
    saved = read_saved(tmp_path, "sample-1")
    assert saved["artifacts"] == {
        ref: {"kind": "payload", "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}}
    assert saved["payloads"] == {ref: {"rows": [0, 3], "width": 2}}
    with pytest.raises(ArtifactError):
        store.write_payload("basis", {})
    assert (tmp_path / "data/sample-1" / ref).read_bytes() == raw


@pytest.mark.parametrize("name", ["../bad", "/bad", "a/b", "a\\b", ".hidden", "x.txt", "x..json"])
def test_payload_names_reject_escape(tmp_path: Path, name: str) -> None:
    store = claim(tmp_path)
    with pytest.raises(ArtifactError):
        store.write_payload(name, {})
    assert not (tmp_path / "data/sample-1/payload").exists()


def test_payload_and_snapshot_symlink_targets_rejected(tmp_path: Path) -> None:
    store = claim(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "data/sample-1/payload").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError):
        store.write_payload("basis", {})
    assert list(outside.iterdir()) == []
    raw = run_path(tmp_path).read_bytes()
    run_path(tmp_path).unlink()
    external = tmp_path / "external.json"
    external.write_bytes(raw)
    run_path(tmp_path).symlink_to(external)
    with pytest.raises(ArtifactError):
        store.write_snapshot({"summary": {"overwrite": True}})
    assert external.read_bytes() == raw


def test_atomic_replace_failure_preserves_old_and_hides_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path)
    before = run_path(tmp_path).read_bytes()

    def broken(src: object, dst: object) -> None:
        raise OSError("SECRET raw failure")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(ArtifactError, match="WRITE_FAILED") as caught:
        store.write_snapshot({"summary": {"next": True}})
    assert caught.value.__cause__ is None
    assert "SECRET" not in str(caught.value)
    assert run_path(tmp_path).read_bytes() == before
    assert not list((tmp_path / "data/sample-1").glob("*.tmp"))


def test_partial_temp_write_failure_never_deletes_existing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path)
    before = run_path(tmp_path).read_bytes()
    original = os.write

    def broken(fd: int, data: bytes) -> int:
        original(fd, data[:3])
        raise OSError("SECRET storage detail")

    monkeypatch.setattr(os, "write", broken)
    with pytest.raises(ArtifactError, match="WRITE_FAILED"):
        store.write_snapshot({"summary": {"next": True}})
    assert run_path(tmp_path).read_bytes() == before
    assert not list((tmp_path / "data/sample-1").glob("*.tmp"))


def test_payload_survives_later_metadata_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path)
    original = os.replace
    failed = False

    def broken(src: object, dst: object) -> None:
        nonlocal failed
        if Path(str(dst)).name == "run.json" and not failed:
            failed = True
            raise OSError("SECRET")
        original(cast(Path, src), cast(Path, dst))

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(ArtifactError, match="WRITE_FAILED"):
        store.write_payload("basis", {"rows": [1]})
    assert (tmp_path / "data/sample-1/payload/basis.json").is_file()
    saved = read_saved(tmp_path, "sample-1")
    assert saved["status"] == "FAILED"
    assert saved["summary"] == {"reason": "WRITE_FAILED"}
    assert "payload/basis.json" in cast(dict[str, object], saved["artifacts"])
    assert b"SECRET" not in run_path(tmp_path).read_bytes()


def test_finalize_freezes_all_data_but_allows_budgeted_reports(tmp_path: Path) -> None:
    store = claim(tmp_path)
    store.finalize({"status": "SUCCEEDED", "summary": {"synthetic": True}})
    frozen = run_path(tmp_path).read_bytes()
    operations: list[Callable[[], object]] = [
        lambda: store.write_snapshot({}), lambda: store.append_event({"kind": "x"}),
        lambda: store.write_payload("late", {}), lambda: store.finalize({"status": "FAILED"}),
    ]
    for operation in operations:
        with pytest.raises(ArtifactError, match="FINALIZED"):
            operation()
    saved = read_saved(tmp_path, "sample-1")
    saved_before = json.dumps(saved, sort_keys=True)
    size_before = disk_bytes(tmp_path)
    files = render_report(saved)  # no destination/owner: rendering only returns bytes
    assert json.dumps(saved, sort_keys=True) == saved_before
    assert disk_bytes(tmp_path) == size_before
    assert list((tmp_path / "result/sample-1").iterdir()) == []
    assert set(files) == {"report.md", "index.html", "source.json"}
    assert all(type(content) is bytes for content in files.values())
    assert json.loads(files["source.json"]) == {
        "schema": 1, "run_id": "sample-1", "run_ref": "data/sample-1/run.json",
        "run_sha256": sha256(frozen).hexdigest(),
    }
    store.write_report_files(files)  # the original finalized owner is the only writer
    assert run_path(tmp_path).read_bytes() == frozen
    for name, content in files.items():
        assert (tmp_path / "result/sample-1" / name).read_bytes() == content
    # Controlled re-render is allowed, always bound to the unchanged run identity.
    store.write_report_files(files)
    assert read_saved(tmp_path, "sample-1")["status"] == "SUCCEEDED"


def test_reports_reject_early_unknown_names_and_wrong_identity(tmp_path: Path) -> None:
    store = claim(tmp_path)
    with pytest.raises(ArtifactError):
        store.write_report_files(report_files(tmp_path))
    store.finalize({"status": "STOPPED"})
    files = report_files(tmp_path)
    for wrong in [{**files, "../bad": b"x"}, {"index.html": b"x"},
                  {**files, "source.json": b"{}"}]:
        with pytest.raises(ArtifactError):
            store.write_report_files(wrong)
    assert list((tmp_path / "result/sample-1").iterdir()) == []


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "STOPPED", "ALARM"])
def test_report_budget_failure_does_not_change_scientific_terminal(
    tmp_path: Path, status: str,
) -> None:
    store = claim(tmp_path, 50_000)
    store.finalize({"status": status})
    before = run_path(tmp_path).read_bytes()
    saved = read_saved(tmp_path, "sample-1")
    files = render_report(saved)
    files["index.html"] = b"x" * 50_000
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_report_files(files)
    assert run_path(tmp_path).read_bytes() == before
    assert read_saved(tmp_path, "sample-1")["status"] == status
    assert list((tmp_path / "result/sample-1").iterdir()) == []
    assert disk_bytes(tmp_path) <= 50_000


def test_events_and_payload_share_budget(tmp_path: Path) -> None:
    store = claim(tmp_path, 30_000)
    store.append_event({"kind": "progress", "summary": {"text": "a" * 9000}})
    store.write_payload("one", {"text": "b" * 9000})
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_payload("two", {"text": "c" * 20_000})
    assert disk_bytes(tmp_path) <= 30_000
    assert (tmp_path / "data/sample-1/payload/one.json").exists()
    assert not (tmp_path / "data/sample-1/payload/two.json").exists()


@pytest.mark.parametrize("mutation", ["schema", "source", "summary", "escape", "hash", "size",
                                      "extra", "duplicate", "nan"])
def test_read_saved_rejects_bad_structure_and_references(tmp_path: Path, mutation: str) -> None:
    store = claim(tmp_path)
    ref = store.write_payload("basis", {"rows": [1]})
    value = json.loads(run_path(tmp_path).read_bytes())
    if mutation == "schema":
        value["schema"] = True
    elif mutation == "source":
        value["source"] = {"../old.py": "0" * 64}
    elif mutation == "summary":
        value["summary"] = []
    elif mutation == "escape":
        value["artifacts"]["../external.json"] = value["artifacts"].pop(ref)
    elif mutation == "hash":
        value["artifacts"][ref]["sha256"] = "0" * 64
    elif mutation == "size":
        value["artifacts"][ref]["bytes"] = True
    elif mutation == "extra":
        value["private"] = "not public"
    if mutation == "duplicate":
        raw = b'{"schema":2,"schema":2}'
    elif mutation == "nan":
        raw = json.dumps(value).replace('"summary": {}', '"summary": {"x": NaN}').encode()
    else:
        raw = json.dumps(value).encode()
    run_path(tmp_path).write_bytes(raw)
    with pytest.raises(ArtifactError):
        read_saved(tmp_path, "sample-1")


def test_reader_rejects_symlink_payload_and_corrupt_json_even_with_valid_hash(
    tmp_path: Path,
) -> None:
    store = claim(tmp_path)
    ref = store.write_payload("basis", {})
    target = tmp_path / "data/sample-1" / ref
    target.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"{}")
    target.symlink_to(outside)
    with pytest.raises(ArtifactError):
        read_saved(tmp_path, "sample-1")
    target.unlink()
    target.write_bytes(b"not json")
    value = json.loads(run_path(tmp_path).read_bytes())
    value["artifacts"][ref] = {
        "kind": "payload", "sha256": sha256(b"not json").hexdigest(), "bytes": 8}
    run_path(tmp_path).write_text(json.dumps(value))
    with pytest.raises(ArtifactError):
        read_saved(tmp_path, "sample-1")


@pytest.mark.parametrize("status", [[], {}, True, "UNKNOWN", "RUNNING", None])
def test_finalize_invalid_status_is_public_and_does_not_mutate(
    tmp_path: Path, status: object,
) -> None:
    store = claim(tmp_path)
    before = run_path(tmp_path).read_bytes()
    with pytest.raises(ArtifactError, match="INVALID_STATE"):
        store.finalize({"status": status})
    assert run_path(tmp_path).read_bytes() == before


def test_read_saved_rejects_oversized_integer_json(tmp_path: Path) -> None:
    claim(tmp_path)
    run_path(tmp_path).write_bytes(b'{"x":' + b"9" * 5000 + b"}")
    with pytest.raises(ArtifactError, match="INVALID_JSON"):
        read_saved(tmp_path, "sample-1")


def test_report_replace_failure_preserves_terminal_and_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path)
    store.finalize({"status": "SUCCEEDED"})
    files = report_files(tmp_path)
    store.write_report_files(files)
    frozen = run_path(tmp_path).read_bytes()

    def broken(src: object, dst: object) -> None:
        raise OSError("PRIVATE MEDIA FAILURE")

    monkeypatch.setattr(os, "replace", broken)
    changed = {**files, "index.html": b"new report"}
    with pytest.raises(ArtifactError, match="WRITE_FAILED"):
        store.write_report_files(changed)
    assert run_path(tmp_path).read_bytes() == frozen
    assert (tmp_path / "result/sample-1/index.html").read_bytes() == files["index.html"]
    assert not list((tmp_path / "result/sample-1").glob("*.tmp"))


def test_report_symlink_and_run_tampering_rejected(tmp_path: Path) -> None:
    store = claim(tmp_path)
    store.finalize({"status": "SUCCEEDED"})
    files = report_files(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"old")
    link = tmp_path / "result/sample-1/index.html"
    link.symlink_to(outside)
    with pytest.raises(ArtifactError):
        store.write_report_files(files)
    assert outside.read_bytes() == b"old"
    link.unlink()
    run_path(tmp_path).write_bytes(b"{}")
    with pytest.raises(ArtifactError):
        store.write_report_files(files)
    assert list((tmp_path / "result/sample-1").iterdir()) == []


def test_report_preflight_accounts_for_all_files(tmp_path: Path) -> None:
    store = claim(tmp_path, 25_000)
    store.finalize({"status": "SUCCEEDED"})
    files = report_files(tmp_path)
    files["index.html"] = b"h" * 12_000
    files["report.md"] = b"m" * 12_000
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_report_files(files)
    assert list((tmp_path / "result/sample-1").iterdir()) == []


def test_event_partial_failure_preserves_evidence_and_attempts_fixed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path)
    original = os.write
    failed = False

    def broken(fd: int, content: bytes) -> int:
        nonlocal failed
        if not failed:
            failed = True
            original(fd, content[:5])
            raise OSError("PRIVATE")
        return original(fd, content)

    monkeypatch.setattr(os, "write", broken)
    with pytest.raises(ArtifactError, match="WRITE_FAILED"):
        store.append_event({"kind": "progress"})
    assert (tmp_path / "data/sample-1/events.jsonl").stat().st_size == 5
    assert read_saved(tmp_path, "sample-1")["status"] == "FAILED"
    assert b"PRIVATE" not in run_path(tmp_path).read_bytes()


def test_unknown_file_is_counted_but_never_overwritten(tmp_path: Path) -> None:
    store = claim(tmp_path, 30_000)
    foreign = tmp_path / "result/sample-1/foreign"
    foreign.write_bytes(b"x" * 30_000)
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.append_event({"kind": "progress"})
    assert foreign.read_bytes() == b"x" * 30_000


def test_models_created_only_when_explicitly_requested(tmp_path: Path) -> None:
    RunStore.claim(tmp_path, "training-stub", output_bytes=30_000, needs_models=True)
    assert (tmp_path / "models/training-stub").is_dir()
    assert list((tmp_path / "models/training-stub").iterdir()) == []


def test_report_replacement_peak_includes_existing_report(tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    probe_store = claim(probe)
    probe_store.finalize({"status": "SUCCEEDED"})
    files = report_files(probe)
    files["index.html"] = b"h" * 6000
    files["report.md"] = b"m" * 6000
    final_size = disk_bytes(probe) + sum(map(len, files.values()))
    root = tmp_path / "exact"
    store = claim(root, final_size)
    store.finalize({"status": "SUCCEEDED"})
    store.write_report_files(files)
    assert disk_bytes(root) == final_size
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_report_files(files)
    for name, content in files.items():
        assert (root / "result/sample-1" / name).read_bytes() == content


def test_full_reference_metadata_reserved_before_payload_bytes(tmp_path: Path) -> None:
    # A large payload fits alone, but its reference metadata replacement does not.
    probe = tmp_path / "probe"
    claim(probe)
    initial_size = disk_bytes(probe)
    root = tmp_path / "limited"
    store = claim(root, initial_size + 10_000)
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_payload("basis", {"data": "x" * 9900})
    assert not (root / "data/sample-1/payload/basis.json").exists()
    assert disk_bytes(root) <= initial_size + 10_000


def test_short_writes_are_completed_without_truncating_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = claim(tmp_path)
    original = os.write

    def short_write(fd: int, content: bytes) -> int:
        return original(fd, content[:11])

    monkeypatch.setattr(os, "write", short_write)
    store.write_snapshot({"summary": {"public": "x" * 200}})
    store.append_event({"kind": "progress"})
    assert read_saved(tmp_path, "sample-1")["summary"] == {"public": "x" * 200}
    event = json.loads((tmp_path / "data/sample-1/events.jsonl").read_bytes())
    assert event["kind"] == "progress"


@pytest.mark.parametrize("backend", ["demo", "simon", "UNVERIFIED"])
def test_storage_failure_retains_only_committed_minimal_public_identity(
    tmp_path: Path, backend: str,
) -> None:
    store = claim(tmp_path, 30_000)
    identity = {"backend_identity": backend, "scientific_validation": False}
    store.write_snapshot({"summary": {**identity, "cells": [{"outcome": "PENDING"}],
                                      "padding": "x" * 15_000}})
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_snapshot({"summary": {"backend_identity": "not-committed",
                                          "scientific_validation": True, "x": "y" * 30_000}})
    saved = read_saved(tmp_path, "sample-1")
    assert saved["status"] == "FAILED"
    assert saved["summary"] == {**identity, "reason": "OUTPUT_LIMIT"}
    before = run_path(tmp_path).read_bytes()
    with pytest.raises(ArtifactError, match="FINALIZED"):
        store.finalize({"status": "SUCCEEDED"})
    store.write_report_files(report_files(tmp_path))
    assert run_path(tmp_path).read_bytes() == before
    assert disk_bytes(tmp_path) <= 30_000


@pytest.mark.parametrize("mode", ["opted-in", "default", "over-cap", "reserved-key"])
def test_failure_summary_opt_in_copies_only_one_bounded_committed_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    store = claim(tmp_path)
    evidence = "x" * 150_000 if mode == "over-cap" else {"rows": [1, 2]}
    if mode != "default":
        store._failure_evidence_key = "backend_identity" if mode == "reserved-key" else "evidence"
    store.write_snapshot({"counts": {"items": 2}, "summary": {
        "evidence": evidence, "backend_identity": "unrecognized", "unselected": "omit",
    }})
    atomic = store._atomic

    def fail_proposed_record(target: Path, content: bytes) -> None:
        if target == run_path(tmp_path) and json.loads(content)["status"] == "RUNNING":
            raise OSError("injected write failure")
        atomic(target, content)

    monkeypatch.setattr(store, "_atomic", fail_proposed_record)
    with pytest.raises(ArtifactError, match="^WRITE_FAILED$"):
        store.write_snapshot({"counts": {"items": 3}, "summary": {"evidence": {"rows": [3]}}})
    saved = read_saved(tmp_path, "sample-1")
    assert saved["status"] == "FAILED"
    assert saved["counts"] == {"items": 2}
    expected: dict[str, object] = {"reason": "WRITE_FAILED"}
    if mode == "opted-in":
        expected["evidence"] = {"rows": [1, 2]}
    assert saved["summary"] == expected


@pytest.mark.parametrize("summary,retained", [
    ({}, {}),
    ({"backend_identity": "PRIVATE", "scientific_validation": True}, {}),
    ({"backend_identity": ["demo"], "scientific_validation": 0}, {}),
    ({"backend_identity": "demo", "scientific_validation": "false"},
     {"backend_identity": "demo"}),
    ({"scientific_validation": False}, {"scientific_validation": False}),
])
def test_failure_identity_is_value_restricted_not_arbitrary_summary_copy(
    tmp_path: Path, summary: dict[str, object], retained: dict[str, object],
) -> None:
    store = claim(tmp_path, 30_000)
    store.write_snapshot({"summary": summary})
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_payload("too-large", "x" * 30_000)
    assert read_saved(tmp_path, "sample-1")["summary"] == {**retained, "reason": "OUTPUT_LIMIT"}


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "STOPPED", "ALARM"])
def test_report_overflow_keeps_terminal_identity_and_refuses_later_finalization(
    tmp_path: Path, status: str,
) -> None:
    store = claim(tmp_path, 30_000)
    summary = {"backend_identity": "demo", "scientific_validation": False, "note": "public"}
    store.finalize({"status": status, "summary": summary})
    before = run_path(tmp_path).read_bytes()
    files = report_files(tmp_path)
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_report_files({**files, "index.html": b"x" * 30_000})
    with pytest.raises(ArtifactError, match="FINALIZED"):
        store.finalize({"status": "FAILED"})
    assert run_path(tmp_path).read_bytes() == before
    saved = read_saved(tmp_path, "sample-1")
    assert saved["status"] == status and saved["summary"] == summary


def test_failure_without_space_for_terminal_keeps_running_identity_as_unknown(
    tmp_path: Path,
) -> None:
    identity = {"backend_identity": "demo", "scientific_validation": False}
    probe = tmp_path / "probe"
    probe_store = claim(probe)
    initial_size = disk_bytes(probe)
    probe_store.write_snapshot({"summary": identity})
    identified_size = disk_bytes(probe)
    root = tmp_path / "limited"
    # Exactly enough for initial -> identity atomic replacement, not another
    # replacement plus its FAILED reason. No bypass of temporary peak accounting.
    limit = initial_size + identified_size
    store = claim(root, limit)
    store.write_snapshot({"summary": identity})
    before = run_path(root).read_bytes()
    with pytest.raises(ArtifactError, match="OUTPUT_LIMIT"):
        store.write_payload("oversized", "x" * limit)
    saved = read_saved(root, "sample-1")
    assert saved["status"] == "RUNNING"  # reader/CLI interpret missing terminal as UNKNOWN
    assert saved["summary"] == identity
    assert run_path(root).read_bytes() == before
    assert disk_bytes(root) <= limit
    with pytest.raises(ArtifactError, match="FINALIZED"):
        store.finalize({"status": "FAILED"})
