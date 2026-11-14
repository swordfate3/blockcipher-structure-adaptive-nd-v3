"""Technical storage for explicitly public caller data (run schema 2).

Snapshot updates accept only recipe/policy/counts/summary JSON mappings. The caller
must construct these public summaries: this module cannot detect secrets in JSON.
Payloads are immutable JSON; datasets and models are opaque bytes. References
are category-bound, never arbitrary paths. Only best models may be replaced.
Events accept kind/counts/summary, never a complete snapshot.
Terminal states are SUCCEEDED, FAILED, STOPPED, ALARM. Missing terminal = UNKNOWN.

Report handoff: retain the original claimed store, finalize it, then read_saved().
Pass that saved mapping to a pure render_report(saved) -> dict[str, bytes] renderer
(no destination or disk I/O), then pass its three files to that same store's
write_report_files(). The renderer uses the reader's run_ref/run_sha256, not a new
source scan. This module supplies storage, not the renderer. No report writer can
be reopened in another process; report failures never alter the scientific terminal.

Only a single process and a trusted, non-concurrently-mutated root are supported.
Symlink/ownership checks are defensive, NOT an OS sandbox or a TOCTOU guarantee.
The byte limit covers file contents (including metadata and temporary copies), not
filesystem allocation overhead. No resume, concurrent writers or crash durability
transaction across files is promised. Failed writes retain committed evidence.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import stat
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import cast
from uuid import uuid4

_SCHEMA = 2
# Independent three-file report provenance format; not a run-schema fallback.
_REPORT_SCHEMA = 1
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "STOPPED", "ALARM"})
_FIELDS = frozenset({"schema", "run_id", "source", "environment", "status", "recipe", "policy",
                     "counts", "summary", "artifacts"})
_ENVIRONMENT_FIELDS = frozenset({"python_implementation", "python_version", "os", "architecture"})
_PYTHON_VERSION = re.compile(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\Z", re.ASCII)
_PUBLIC = frozenset({"recipe", "policy", "counts", "summary"})
_REPORTS = frozenset({"report.md", "index.html", "source.json"})


class ArtifactError(ValueError):
    """Fixed public reason only; underlying filesystem exceptions are suppressed."""


def _token(value: object) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ArtifactError("INVALID_NAME")
    return value


def _path(path: Path) -> Path:
    """Check the lexical parent chain without resolving through any symlink."""
    if ".." in path.parts:
        raise ArtifactError("UNSAFE_PATH")
    absolute = path.absolute()
    for item in [*reversed(absolute.parents), absolute]:
        try:
            mode = item.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ArtifactError("UNSAFE_PATH")
        if item != absolute and not stat.S_ISDIR(mode):
            raise ArtifactError("UNSAFE_PATH")
        if item == absolute and not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ArtifactError("UNSAFE_PATH")
    return absolute


def _json_value(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArtifactError("INVALID_JSON")
            _json_value(child)
        return
    elif isinstance(value, (list, tuple)):
        for child in value:
            _json_value(child)
        return
    raise ArtifactError("INVALID_JSON")


def _encode(value: object) -> bytes:
    try:
        _json_value(value)
        return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError, RecursionError):
        raise ArtifactError("INVALID_JSON") from None


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ArtifactError("INVALID_JSON")
        value[key] = child
    return value


def _constant(value: str) -> object:
    raise ArtifactError("INVALID_JSON")


def _decode(raw: bytes) -> object:
    try:
        value: object = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
        _json_value(value)
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ArtifactError("INVALID_JSON") from None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ArtifactError("INVALID_SCHEMA")
    return cast(dict[str, object], value)


def _public(value: Mapping[str, object], *, terminal: bool = False) -> dict[str, object]:
    allowed = _PUBLIC | ({"status"} if terminal else set())
    if not set(value) <= allowed:
        raise ArtifactError("INVALID_SCHEMA")
    result = _mapping(_decode(_encode(dict(value))))
    for key in _PUBLIC & result.keys():
        _mapping(result[key])
    status = result.get("status")
    if terminal and (not isinstance(status, str) or status not in _TERMINAL):
        raise ArtifactError("INVALID_STATE")
    return result


def _ref(value: object) -> str:
    if isinstance(value, str):
        for kind, extension in (("payload", ".json"), ("dataset", ".npz"), ("model", ".pt")):
            prefix = kind + "/"
            if value.startswith(prefix) and value.endswith(extension):
                stem = value[len(prefix):-len(extension)]
                if _TOKEN.fullmatch(stem) is not None:
                    return value
    raise ArtifactError("INVALID_REFERENCE")


def _location(root: Path, run_id: str, ref: str) -> Path:
    """Explicit category dispatch; caller-supplied paths are never accepted."""
    _ref(ref)
    if ref.startswith("payload/"):
        return _path(root / "data" / run_id / "payload" / ref.split("/")[1])
    if ref.startswith("dataset/"):
        return _path(root / "data" / run_id / "dataset" / ref.split("/")[1])
    if ref.startswith("model/"):
        return _path(root / "models" / run_id / ref.split("/")[1])
    raise ArtifactError("INVALID_REFERENCE")


def _stream(path: Path, metadata: dict[str, object], *, collect: bool = False) -> bytes:
    """Bounded chunks; verify metadata size and SHA before returning any bytes."""
    size = cast(int, metadata["bytes"])
    if not path.is_file() or path.stat().st_size != size:
        raise ArtifactError("HASH_MISMATCH")
    digest = sha256()
    total = 0
    content = bytearray()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            total += len(chunk)
            if total > size:
                raise ArtifactError("HASH_MISMATCH")
            digest.update(chunk)
            if collect:
                content.extend(chunk)
    if total != size or digest.hexdigest() != metadata["sha256"]:
        raise ArtifactError("HASH_MISMATCH")
    return bytes(content)


def _source(value: object) -> None:
    source = _mapping(value)
    if not source:
        raise ArtifactError("INVALID_SCHEMA")
    for name, digest in source.items():
        parts = name.split("/")
        if (len(parts) < 2 or parts[0] != "cipher_lab" or not name.endswith(".py")
                or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts[1:-1])
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", parts[-1])
                or not isinstance(digest, str) or _SHA.fullmatch(digest) is None):
            raise ArtifactError("INVALID_SCHEMA")


def _environment(value: object) -> None:
    """Fixed public identity only; bounded ASCII tokens and major.minor.micro."""
    environment = _mapping(value)
    if set(environment) != _ENVIRONMENT_FIELDS:
        raise ArtifactError("INVALID_SCHEMA")
    for name, text in environment.items():
        pattern = _PYTHON_VERSION if name == "python_version" else _TOKEN
        if type(text) is not str or pattern.fullmatch(text) is None:
            raise ArtifactError("INVALID_SCHEMA")


def _environment_identity() -> dict[str, str]:
    """Probe once per claim; never collect platform(), uname(), node(), or env."""
    try:
        result = {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "os": platform.system(),
            "architecture": platform.machine(),
        }
        _environment(result)
        return result
    except Exception:
        # A probe may raise more than OSError; no private detail or fallback.
        raise ArtifactError("ENVIRONMENT_UNAVAILABLE") from None


def source_identity() -> dict[str, str]:
    """SHA256 of only this package's .py files; scan/read errors fail closed."""
    def scan_failed(error: OSError) -> None:
        raise ArtifactError("SOURCE_UNAVAILABLE") from None

    try:
        package = _path(Path(__file__).absolute().parent)
        result: dict[str, str] = {}
        for directory, dirs, files in os.walk(package, onerror=scan_failed, followlinks=False):
            for name in dirs:
                _path(Path(directory) / name)
            for name in sorted(files):
                if name.endswith(".py"):
                    path = _path(Path(directory) / name)
                    key = "cipher_lab/" + path.relative_to(package).as_posix()
                    result[key] = sha256(path.read_bytes()).hexdigest()
        _source(result)
        return dict(sorted(result.items()))
    except OSError:
        raise ArtifactError("SOURCE_UNAVAILABLE") from None


def _validate(record: object, run_id: str) -> dict[str, object]:
    value = _mapping(record)
    if (set(value) != _FIELDS or type(value["schema"]) is not int or value["schema"] != _SCHEMA
            or value["run_id"] != run_id or not isinstance(value["status"], str)
            or value["status"] not in _TERMINAL | {"RUNNING"}):
        raise ArtifactError("INVALID_SCHEMA")
    _source(value["source"])
    _environment(value["environment"])
    for name in _PUBLIC:
        _mapping(value[name])
    artifacts = _mapping(value["artifacts"])
    for ref, metadata in artifacts.items():
        _ref(ref)
        meta = _mapping(metadata)
        digest = meta.get("sha256")
        size = meta.get("bytes")
        if (set(meta) != {"kind", "sha256", "bytes"} or meta["kind"] != ref.split("/")[0]
                or not isinstance(digest, str)
                or _SHA.fullmatch(digest) is None or type(size) is not int or size < 0):
            raise ArtifactError("INVALID_SCHEMA")
    return value


def read_saved(root: Path, run_id: str) -> dict[str, object]:
    """Validate saved public structure/references/SHA; do not recompute science.

    Returns the run fields plus run_sha256, run_ref and decoded payloads. These
    derived reader fields are not written back into the sole canonical run.json.
    For reports, finalize the retained owner first, then give this mapping to a
    pure renderer. The SHA identifies the saved run.json bytes, not current code.
    Only JSON payloads are decoded. Dataset/model files are size/SHA checked in
    bounded chunks, never loaded into memory or decoded by this report reader.
    This reader neither acquires nor restores write ownership.
    """
    _token(run_id)
    try:
        base = _path(root) / "data" / run_id
        path = _path(base / "run.json")
        raw = path.read_bytes()
        value = _validate(_decode(raw), run_id)
        payloads: dict[str, object] = {}
        for ref, metadata in _mapping(value["artifacts"]).items():
            target = _location(_path(root), run_id, ref)
            meta = _mapping(metadata)
            if meta["kind"] == "payload":
                payloads[ref] = _decode(_stream(target, meta, collect=True))
            else:
                _stream(target, meta)
        return {**value, "run_sha256": sha256(raw).hexdigest(),
                "run_ref": f"data/{run_id}/run.json", "payloads": payloads}
    except OSError:
        raise ArtifactError("READ_FAILED") from None


def read_binary(root: Path, run_id: str, ref: str, expected_sha256: str,
                max_bytes: int) -> bytes:
    """Read one explicitly requested dataset/model, bound to canonical run metadata.

    Does not decode content or acquire ownership. max_bytes is inclusive and may
    be zero for an empty artifact. No unrelated payloads/binaries are read.
    """
    _token(run_id)
    _ref(ref)
    if not ref.startswith(("dataset/", "model/")):
        raise ArtifactError("INVALID_REFERENCE")
    if type(expected_sha256) is not str or _SHA.fullmatch(expected_sha256) is None:
        raise ArtifactError("INVALID_SHA256")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ArtifactError("INVALID_BUDGET")
    try:
        root = _path(root)
        record = _validate(_decode(_path(root / "data" / run_id / "run.json").read_bytes()),
                           run_id)
        refs = _mapping(record["artifacts"])
        if ref not in refs:
            raise ArtifactError("INVALID_REFERENCE")
        meta = _mapping(refs[ref])
        if meta["sha256"] != expected_sha256:
            raise ArtifactError("HASH_MISMATCH")
        if cast(int, meta["bytes"]) > max_bytes:
            raise ArtifactError("SIZE_LIMIT")
        return _stream(_location(root, run_id, ref), meta, collect=True)
    except OSError:
        raise ArtifactError("READ_FAILED") from None


@dataclass
class RunStore:
    """Claim-only writer; use claim(), never construct manually.

    A storage failure seals this writer; it tries a fixed FAILED record if possible.
    If the medium/remaining budget prevents that, the last complete record survives
    (RUNNING means UNKNOWN, not success). Reports never change scientific status.
    """

    _root: Path
    _run_id: str
    _limit: int
    _record: dict[str, object]
    _dirs: set[Path] = field(default_factory=set)
    _files: dict[Path, str] = field(default_factory=dict)
    _sequence: int = 0
    _sealed: bool = False
    _frozen_sha: str | None = None

    @classmethod
    def claim(cls, root: Path, run_id: str, *, output_bytes: int,
              needs_models: bool = False) -> RunStore:
        _token(run_id)
        if type(output_bytes) is not int or output_bytes <= 0 or type(needs_models) is not bool:
            raise ArtifactError("INVALID_BUDGET")
        created: list[Path] = []
        store: RunStore | None = None
        try:
            root = _path(root)
            identity = source_identity()
            record: dict[str, object] = {
                "schema": _SCHEMA, "run_id": run_id, "source": identity,
                "environment": _environment_identity(), "status": "RUNNING",
                "recipe": {}, "policy": {}, "counts": {}, "summary": {}, "artifacts": {},
            }
            if len(_encode(record)) > output_bytes:
                raise ArtifactError("OUTPUT_LIMIT")
            locations = [root / kind / run_id for kind in
                         (["data", "result", "models"] if needs_models else ["data", "result"])]
            for path in locations:
                if _path(path).exists():
                    raise ArtifactError("OCCUPIED")
            # Root and missing container directories are tracked individually for rollback.
            for path in locations:
                missing = [p for p in [*reversed(path.parents), path] if not p.exists()]
                for item in missing:
                    _path(item)
                    item.mkdir()  # no exist_ok: no accidental adoption of a competing claim
                    created.append(item)
            store = cls(root, run_id, output_bytes, record, set(locations))
            store._atomic(store._data / "run.json", _encode(record))
            return store
        except (OSError, ArtifactError) as error:
            # Never recursive deletion: even owned directories may now contain evidence.
            for path in reversed(created):
                try:
                    _path(path).rmdir()
                except (OSError, ArtifactError):
                    pass
            reason = str(error) if isinstance(error, ArtifactError) else "CLAIM_FAILED"
            raise ArtifactError(reason) from None

    @property
    def _data(self) -> Path:
        return self._root / "data" / self._run_id

    @property
    def _result(self) -> Path:
        return self._root / "result" / self._run_id

    def _total(self) -> int:
        total = 0
        for directory in self._dirs:
            _path(directory)
            if not directory.is_dir():
                raise ArtifactError("OWNERSHIP_LOST")
            for child in directory.iterdir():
                _path(child)
                if child.is_dir():
                    if child not in self._dirs:
                        raise ArtifactError("OCCUPIED")
                elif child.is_file():
                    total += child.stat().st_size
        return total

    def _space(self, additional: int) -> None:
        if self._total() + additional > self._limit:
            raise ArtifactError("OUTPUT_LIMIT")

    def _target(self, target: Path) -> None:
        _path(target)
        if target.parent not in self._dirs:
            raise ArtifactError("OWNERSHIP_LOST")
        if target.exists():
            expected = self._files.get(target)
            if (expected is None or not target.is_file()
                    or not self._matches(target, expected)):
                raise ArtifactError("OWNERSHIP_LOST")
        elif target in self._files:
            raise ArtifactError("OWNERSHIP_LOST")

    @staticmethod
    def _matches(target: Path, expected: str) -> bool:
        try:
            _stream(target, {"sha256": expected, "bytes": target.stat().st_size})
            return True
        except ArtifactError:
            return False

    @staticmethod
    def _write_all(fd: int, content: bytes) -> None:
        while content:
            written = os.write(fd, content)
            if written <= 0:
                raise OSError("short write")
            content = content[written:]
        os.fsync(fd)

    def _atomic(self, target: Path, content: bytes) -> None:
        self._target(target)
        self._space(len(content))  # old file + full temporary, including all metadata
        temporary = target.with_name(f".{target.name}-{uuid4().hex}.tmp")
        made = False
        fd: int | None = None
        try:
            _path(temporary)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            made = True
            self._write_all(fd, content)
            os.close(fd)
            fd = None
            self._target(target)
            os.replace(temporary, target)
            made = False
            self._files[target] = sha256(content).hexdigest()
        finally:
            if fd is not None:
                os.close(fd)
            if made:
                # Only the exact temporary this operation successfully created.
                try:
                    _path(temporary).unlink()
                except (OSError, ArtifactError):
                    pass

    def _running(self) -> None:
        if self._sealed:
            raise ArtifactError("FINALIZED")

    def _failed(self, reason: str) -> None:
        self._sealed = True
        # Preserve only committed, bounded public identity, never the proposed
        # update or a possibly huge/incomplete scientific summary. Storage does
        # not certify science: only the explicit False validation flag is valid.
        previous = _mapping(self._record["summary"])
        summary: dict[str, object] = {"reason": reason}
        backend = previous.get("backend_identity")
        if isinstance(backend, str) and backend in ("demo", "simon", "ciphers", "UNVERIFIED"):
            summary["backend_identity"] = backend
        if previous.get("scientific_validation") is False:
            summary["scientific_validation"] = False
        failed = {**self._record, "status": "FAILED", "summary": summary}
        try:
            self._atomic(self._data / "run.json", _encode(failed))
            self._record = failed
            self._frozen_sha = self._files[self._data / "run.json"]
        except (OSError, ArtifactError):
            pass

    def _save(self, value: dict[str, object]) -> None:
        try:
            self._atomic(self._data / "run.json", _encode(value))
            self._record = value
        except (OSError, ArtifactError) as error:
            reason = str(error) if isinstance(error, ArtifactError) else "WRITE_FAILED"
            self._failed(reason)
            raise ArtifactError(reason) from None

    def source_snapshot(self) -> dict[str, str]:
        """Return a detached claim-time identity, even after finalization.

        This read-only accessor performs no I/O or fresh source scan; mutating
        the returned deep copy cannot change the owner's canonical record.
        """
        return deepcopy(cast(dict[str, str], self._record["source"]))

    def write_snapshot(self, value: Mapping[str, object]) -> None:
        self._running()
        update = _public(value)
        self._save({**self._record, **update})

    def append_event(self, value: Mapping[str, object]) -> None:
        self._running()
        if (not set(value) <= {"kind", "counts", "summary"} or "kind" not in value):
            raise ArtifactError("INVALID_SCHEMA")
        _token(value["kind"])
        for key in {"counts", "summary"} & value.keys():
            if not isinstance(value[key], Mapping):
                raise ArtifactError("INVALID_SCHEMA")
        content = _encode({**value, "schema": _SCHEMA, "sequence": self._sequence + 1})
        target = self._data / "events.jsonl"
        fd: int | None = None
        try:
            self._target(target)
            self._space(len(content))
            flags = os.O_WRONLY | os.O_APPEND
            if not target.exists():
                flags |= os.O_CREAT | os.O_EXCL
            fd = os.open(target, flags, 0o600)
            self._write_all(fd, content)
            self._files[target] = sha256(target.read_bytes()).hexdigest()
            self._sequence += 1
        except (OSError, ArtifactError) as error:
            # Partial event bytes remain as evidence; no truncation or rewind.
            reason = str(error) if isinstance(error, ArtifactError) else "WRITE_FAILED"
            self._failed(reason)
            raise ArtifactError(reason) from None
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def write_payload(self, name: str, value: object) -> str:
        self._running()
        ref = f"payload/{_token(name)}.json"
        content = _encode(value)
        return self._write_artifact(ref, content)

    def write_dataset(self, name: str, content: bytes) -> str:
        """Write one immutable opaque dataset; NPZ validation belongs to data."""
        self._running()
        ref = f"dataset/{_token(name)}.npz"
        if type(content) is not bytes:
            raise ArtifactError("INVALID_CONTENT")
        return self._write_artifact(ref, content)

    def write_model(self, name: str, content: bytes, *, replace: bool = False) -> str:
        """Write opaque checkpoint bytes; only best permits explicit replacement."""
        self._running()
        ref = f"model/{_token(name)}.pt"
        if type(content) is not bytes:
            raise ArtifactError("INVALID_CONTENT")
        if type(replace) is not bool:
            raise ArtifactError("INVALID_STATE")
        if replace and name not in ("best", "A-best", "B-best"):
            raise ArtifactError("OCCUPIED")
        if self._root / "models" / self._run_id not in self._dirs:
            raise ArtifactError("OWNERSHIP_LOST")
        return self._write_artifact(ref, content, replace=replace)

    def read_model(self, ref: str, expected_sha256: str, max_bytes: int) -> bytes:
        """Read a committed model through this live owner, never a memory fallback.

        Canonical metadata must still belong to this owner. The bounded reader
        checks registration, path/symlinks, declared size and SHA before decoding.
        A failed read seals this writer just like a failed artifact write.
        """
        self._running()
        try:
            _ref(ref)
            if not ref.startswith("model/"):
                raise ArtifactError("INVALID_REFERENCE")
            self._target(self._data / "run.json")
            return read_binary(self._root, self._run_id, ref, expected_sha256, max_bytes)
        except (OSError, ArtifactError) as error:
            reason = str(error) if isinstance(error, ArtifactError) else "READ_FAILED"
            self._failed(reason)
            raise ArtifactError(reason) from None

    def _write_artifact(self, ref: str, content: bytes, *, replace: bool = False) -> str:
        artifacts = dict(_mapping(self._record["artifacts"]))
        if ref in artifacts and not replace:
            raise ArtifactError("OCCUPIED")
        kind = ref.split("/")[0]
        artifacts[ref] = {"kind": kind, "sha256": sha256(content).hexdigest(),
                          "bytes": len(content)}
        updated = {**self._record, "artifacts": artifacts}
        try:
            # Existing model + new temporary bytes + new run metadata temporary
            # are all charged. No cross-file transaction/rollback is promised.
            target = _location(self._root, self._run_id, ref)
            self._space(len(content) + len(_encode(updated)))
            directory = target.parent
            if directory not in self._dirs:
                directory.mkdir()  # exclusive: never adopt someone else's directory
                self._dirs.add(directory)
            self._atomic(target, content)
            # The file has committed: failure records must reference its NEW hash.
            # If metadata also fails persistently, the old on-disk ref will fail
            # reader verification, never silently identify new bytes as old best.
            self._record = updated
            self._save(updated)
            return ref
        except (OSError, ArtifactError) as error:
            reason = str(error) if isinstance(error, ArtifactError) else "WRITE_FAILED"
            if not self._sealed:
                self._failed(reason)
            raise ArtifactError(reason) from None

    def finalize(self, value: Mapping[str, object]) -> None:
        self._running()
        update = _public(value, terminal=True)
        self._save({**self._record, **update})
        self._sealed = True
        self._frozen_sha = self._files[self._data / "run.json"]

    def write_report_files(self, files: Mapping[str, bytes]) -> None:
        """Write exactly report.md/index.html/source.json against the frozen run.

        After finalize -> read_saved -> pure render_report(saved), give this
        original owner the returned three bytes values. Rendering has no destination
        and performs no I/O; this method is the sole report disk-write boundary.
        source.json is schema 1 with exactly run_id/run_ref/run_sha256 taken from
        that saved mapping. Preflight budgets all three replacements' temporary
        peaks. A media failure may leave a partial report; it never rewrites
        run.json. Re-render is allowed only via
        this same owning store; no cross-process reopen/resume is provided.
        """
        if not self._sealed or self._frozen_sha is None:
            raise ArtifactError("NOT_FINALIZED")
        if set(files) != _REPORTS or any(type(v) is not bytes for v in files.values()):
            raise ArtifactError("INVALID_REPORT")
        source = _decode(files["source.json"])
        expected: dict[str, object] = {
            "schema": _REPORT_SCHEMA, "run_id": self._run_id,
            "run_ref": f"data/{self._run_id}/run.json", "run_sha256": self._frozen_sha,
        }
        if source != expected or type(_mapping(source).get("schema")) is not int:
            raise ArtifactError("INVALID_REPORT")
        try:
            self._target(self._data / "run.json")
            if self._files[self._data / "run.json"] != self._frozen_sha:
                raise ArtifactError("HASH_MISMATCH")
            total = self._total()
            for name in sorted(files):
                target = self._result / name
                self._target(target)
                if total + len(files[name]) > self._limit:
                    raise ArtifactError("OUTPUT_LIMIT")
                total += len(files[name]) - (target.stat().st_size if target.exists() else 0)
            for name in sorted(files):
                self._atomic(self._result / name, files[name])
        except OSError:
            raise ArtifactError("WRITE_FAILED") from None
