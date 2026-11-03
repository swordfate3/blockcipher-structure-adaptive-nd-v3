"""One-shot prediction capture and strict, bounded, original-order evidence files.

No model loading, extra forward, label generation, or automatic statistics.
Input x SHA is declared identity; saved uint8 bytes and original float32 y bytes
have deliberately separate hashes. A manifest is published last, never replaced.
"""
from __future__ import annotations

# Persistence is independent of model/checkpoint loading and label generation.
import hashlib
import io
import json
import os
import re
import stat
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


class PredictionCapture:
    """Own ordered CPU copies without evaluating models or writing artifacts."""

    def __init__(self, expected_rows: int) -> None:
        if type(expected_rows) is not int:
            raise TypeError("expected_rows must be a built-in int")
        if expected_rows <= 0:
            raise ValueError("expected_rows must be positive")
        self.expected_rows = expected_rows
        self._rows = 0
        self._finished = False
        self._logits = np.empty(expected_rows, dtype=np.float32)
        self._labels = np.empty(expected_rows, dtype=np.uint8)

    def append(self, logits: Tensor, targets: Tensor) -> None:
        if self._finished:
            raise RuntimeError("capture already finished")
        for name, tensor in (("logits", logits), ("targets", targets)):
            if not isinstance(tensor, Tensor) or tensor.dtype != torch.float32:
                raise TypeError(f"{name} must be a float32 tensor")
            if (tensor.layout != torch.strided or tensor.ndim != 2
                    or tensor.shape[1] != 1 or tensor.shape[0] == 0):
                raise ValueError(f"{name} must have nonempty shape [batch,1]")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite")
        if targets.shape != logits.shape:
            raise ValueError("logits and targets must have identical shapes")
        if not bool(((targets == 0) | (targets == 1)).all()):
            raise ValueError("targets must be binary")
        end = self._rows + len(logits)
        if end > self.expected_rows:
            raise ValueError("capture row overflow")
        # Assignment copies even when detach().cpu() shares the caller's storage.
        self._logits[self._rows:end] = logits.detach().cpu().numpy()[:, 0]
        self._labels[self._rows:end] = targets.detach().cpu().numpy()[:, 0]
        self._rows = end

    def finish(self) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        if self._finished:
            raise RuntimeError("capture already finished")
        if self._rows != self.expected_rows:
            raise ValueError("capture is underfilled")
        self._finished = True
        return self._logits, self._labels


_MAX_FILE = 32 * 1024 * 1024
_MAX_EXPANDED = 16 * 1024 * 1024
_MAX_MANIFEST = 256 * 1024
_SCHEMA = "cipher-experiments-predictions-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(value: Any) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError("identity must be lowercase SHA256")


def _constant(bit: Any, probability: Any) -> None:
    if type(bit) is not int or bit not in (0, 1):
        raise ValueError("majority bit must be built-in int 0 or 1")
    if type(probability) is not float or not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("probability must be built-in finite float strictly between 0 and 1")


def _array(value: Any, dtype: np.dtype[Any], rows: int) -> None:
    if not isinstance(value, np.ndarray) or value.dtype.str != dtype.str:
        raise ValueError("array has wrong dtype")
    if value.shape != (rows,) or not np.isfinite(value).all():
        raise ValueError("array must be finite and shape [rows]")
    if dtype == np.dtype(np.uint8) and not ((value == 0) | (value == 1)).all():
        raise ValueError("bits must be binary")


@dataclass(frozen=True)
class PredictionEvidence:
    """Ordered, validated evidence; x identity is declared, not proved by logits."""

    logits: NDArray[np.float32]
    labels: NDArray[np.uint8]
    x_sha256: str
    y_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    split: str
    rows: int
    reference_bits: NDArray[np.uint8] | None = None
    reference_majority: int = 0
    reference_probability: float = .5
    train_majority: int = 0
    train_probability: float = .5

    def __post_init__(self) -> None:
        if type(self.rows) is not int or not 0 < self.rows <= 1_000_000:
            raise ValueError("rows must be built-in int in 1..1000000")
        if type(self.split) is not str or self.split not in ("train", "dev", "test"):
            raise ValueError("unknown split")
        for value in (self.x_sha256, self.y_sha256, self.config_sha256, self.checkpoint_sha256):
            _identity(value)
        _constant(self.reference_majority, self.reference_probability)
        _constant(self.train_majority, self.train_probability)
        _array(self.logits, np.dtype(np.float32), self.rows)
        _array(self.labels, np.dtype(np.uint8), self.rows)
        if self.reference_bits is not None:
            _array(self.reference_bits, np.dtype(np.uint8), self.rows)
        if _sha(self.labels.astype(np.float32).reshape(-1, 1).tobytes()) != self.y_sha256:
            raise ValueError("original float32 [N,1] y identity mismatch")
        for name in ("logits", "labels", "reference_bits"):
            value = getattr(self, name)
            if value is not None:
                owned = value.copy(order="C")
                owned.flags.writeable = False
                object.__setattr__(self, name, owned)


def _safe_components(path: Path) -> None:
    absolute = path.absolute()
    if ".." in path.parts:
        raise ValueError("parent traversal forbidden")
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise ValueError("symlink path forbidden")


def _bounded_file(path: Path, limit: int) -> bytes:
    _safe_components(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("not a bounded regular file")
    # NONBLOCK also prevents a raced-in FIFO from hanging before fstat; NOFOLLOW
    # protects the final component on platforms that provide it.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("not a bounded regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file exceeds size limit")
    return data


def _publish(path: Path, data: bytes) -> None:
    # Hard-link publication is atomic and fails if ANY target already exists.
    # Temp resides on the same filesystem; failed attempt residue is retained.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evidence-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)
    temp.unlink()


def _array_info(value: NDArray[Any]) -> dict[str, Any]:
    return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _sha(value.tobytes())}


def write_prediction_evidence(
    output: Path, *, split: str, logits: NDArray[np.float32], labels: NDArray[np.uint8],
    x_sha256: str, y_sha256: str, config_sha256: str, checkpoint_sha256: str,
    reference_bits: NDArray[np.uint8] | None = None, reference_majority: int = 0,
    reference_probability: float = .5, train_majority: int = 0, train_probability: float = .5,
) -> dict[str, Any]:
    """Write one split, manifest last, returning only its relative path and byte SHA."""
    evidence = PredictionEvidence(
        logits, labels, x_sha256, y_sha256, config_sha256, checkpoint_sha256,
        split, len(labels), reference_bits, reference_majority, reference_probability,
        train_majority, train_probability)
    groups: dict[str, dict[str, NDArray[Any]]] = {
        "predictions": {"logits": evidence.logits, "labels": evidence.labels}}
    if evidence.reference_bits is not None:
        groups["reference"] = {"bits": evidence.reference_bits}
    if sum(a.nbytes for group in groups.values() for a in group.values()) > _MAX_EXPANDED:
        raise ValueError("expanded array limit exceeded")
    encoded: dict[str, bytes] = {}
    files: dict[str, Any] = {}
    for group, arrays in groups.items():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, array in arrays.items():
                with archive.open(f"{name}.npy", "w") as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
        data = buffer.getvalue()
        if len(data) > _MAX_FILE:
            raise ValueError("NPZ size limit exceeded")
        filename = f"{split}-{group}.npz"
        encoded[filename] = data
        files[group] = {"path": filename, "size_bytes": len(data), "sha256": _sha(data),
                        "arrays": {name: _array_info(a) for name, a in arrays.items()}}
    manifest = {"schema": _SCHEMA, "split": split, "rows": evidence.rows,
                "x_sha256": x_sha256, "y_sha256": y_sha256, "config_sha256": config_sha256,
                "checkpoint_sha256": checkpoint_sha256, "files": files,
                "train_constant": {"majority": train_majority, "probability": train_probability},
                "reference": ({"majority": reference_majority, "probability": reference_probability}
                              if reference_bits is not None else None)}
    manifest_data = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    manifest_name = f"{split}-predictions.json"
    directory = output / "predictions"
    _safe_components(directory)
    directory.mkdir(exist_ok=True)
    # Preflight all names before publishing anything; link() also enforces no replace on races.
    for filename in (*encoded, manifest_name):
        if os.path.lexists(directory / filename):
            raise FileExistsError(directory / filename)
    for filename, data in encoded.items():
        _publish(directory / filename, data)
    _publish(directory / manifest_name, manifest_data)
    return {"manifest": f"predictions/{manifest_name}", "sha256": _sha(manifest_data)}


def _keys(value: Any, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError("unexpected schema fields")


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class _BoundedZipMember:
    """Read the compressed span directly: ZipExtFile may hide excess expansion.

    Input is already bounded by _MAX_FILE. Every DEFLATE call has an explicit
    positive output cap; never flush (which can emit unbounded output). Actual
    EOF, byte count and CRC are checked independently of central file_size.
    """

    def __init__(self, data: bytes, info: zipfile.ZipInfo) -> None:
        offset = info.header_offset
        if offset < 0 or offset + 30 > len(data):
            raise ValueError("invalid ZIP local offset")
        name_size, extra_size = struct.unpack_from("<HH", data, offset + 26)
        start = offset + 30 + name_size + extra_size
        end = start + info.compress_size
        if end > len(data):
            raise ValueError("truncated ZIP compressed span")
        self.pending = data[start:end]
        self.decoder = (zlib.decompressobj(-zlib.MAX_WBITS)
                        if info.compress_type == zipfile.ZIP_DEFLATED else None)
        self.size = 0
        self.crc = 0
        self.info = info

    def read(self, limit: int) -> bytes:
        if limit <= 0:
            raise ValueError("ZIP reads require a positive bound")
        if self.decoder is None:
            result, self.pending = self.pending[:limit], self.pending[limit:]
        else:
            output = bytearray()
            while len(output) < limit and not self.decoder.eof:
                chunk = self.decoder.decompress(self.pending, limit - len(output))
                self.pending = self.decoder.unconsumed_tail
                output.extend(chunk)
                if not chunk and not self.pending:
                    break
            result = bytes(output)
        self.size += len(result)
        self.crc = zlib.crc32(result, self.crc)
        return result

    def finish(self) -> None:
        if (self.pending or (self.decoder is not None and (
                not self.decoder.eof or self.decoder.unused_data))
                or self.size != self.info.file_size or self.crc != self.info.CRC):
            raise ValueError("ZIP actual size/CRC/end mismatch")


def _read_npy_header(member: _BoundedZipMember) -> tuple[Any, Any, Any]:
    # NumPy checks max_header_size only AFTER reading header_length bytes.
    # Read fixed-size fields ourselves, reject oversized lengths, and only then
    # give the public NumPy parser an isolated, <=4096-byte header.
    version = np.lib.format.read_magic(io.BytesIO(member.read(8)))
    if version not in ((1, 0), (2, 0)):
        raise ValueError("unsupported NPY version")
    length_size = 2 if version == (1, 0) else 4
    length_bytes = member.read(length_size)
    if len(length_bytes) != length_size:
        raise ValueError("truncated NPY header length")
    length = int.from_bytes(length_bytes, "little")
    if length > 4096:
        raise ValueError("NPY header exceeds 4096 bytes")
    header = member.read(length) if length else b""
    if len(header) != length:
        raise ValueError("truncated NPY header")
    stream = io.BytesIO(length_bytes + header)
    parser = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    return parser(stream, max_header_size=4096)


def _read_npz(data: bytes, arrays: dict[str, Any], rows: int) -> dict[str, NDArray[Any]]:
    """Validate ZIP + NPY headers and exact byte bounds BEFORE NumPy materialization."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) != len(arrays) or {i.filename for i in infos} != {
                f"{name}.npy" for name in arrays}:
            raise ValueError("duplicate, extra or missing ZIP members")
        for info in infos:
            name = info.filename[:-4]
            meta = arrays[name]
            dtype = np.dtype(meta["dtype"])
            expected_bytes = rows * dtype.itemsize
            if (info.flag_bits & 1 or info.is_dir() or info.file_size > expected_bytes + 4096
                    or info.file_size < expected_bytes or info.compress_type not in (
                        zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError("unsafe ZIP expansion/member")
            # Let zipfile validate local filename, flags and overlapping spans,
            # but do NOT read through ZipExtFile's file_size-truncated decoder.
            with archive.open(info):
                pass
            member = _BoundedZipMember(data, info)
            shape, fortran, stored_dtype = _read_npy_header(member)
            if (shape != (rows,) or fortran or stored_dtype.str != dtype.str
                    or stored_dtype.hasobject or stored_dtype.fields is not None
                    or info.file_size != member.size + expected_bytes):
                raise ValueError("invalid NPY header/expanded size")
            payload = member.read(expected_bytes + 1)
            if len(payload) != expected_bytes or _sha(payload) != meta["sha256"]:
                raise ValueError("array bytes SHA/size mismatch")
            member.finish()
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        return {name: loaded[name] for name in arrays}


def load_prediction_evidence(manifest_path: Path) -> PredictionEvidence:
    """Validate saved artifacts only; never load a model or regenerate labels."""
    try:
        return _load_prediction_evidence(manifest_path)
    except (OSError, KeyError, TypeError, zipfile.BadZipFile, EOFError,
            UnicodeError, zlib.error) as error:
        raise ValueError(f"invalid prediction evidence: {error}") from error


def _load_prediction_evidence(manifest_path: Path) -> PredictionEvidence:
    manifest = json.loads(_bounded_file(manifest_path, _MAX_MANIFEST),
                          object_pairs_hook=_json_pairs)
    _keys(manifest, {"schema", "split", "rows", "x_sha256", "y_sha256", "config_sha256",
                     "checkpoint_sha256", "files", "train_constant", "reference"})
    if type(manifest["schema"]) is not str or manifest["schema"] != _SCHEMA:
        raise ValueError("unknown evidence schema")
    split, rows = manifest["split"], manifest["rows"]
    if type(split) is not str or split not in ("train", "dev", "test"):
        raise ValueError("unknown split")
    if manifest_path.name != f"{split}-predictions.json":
        raise ValueError("split/manifest name mismatch")
    if type(rows) is not int or not 0 < rows <= 1_000_000:
        raise ValueError("invalid rows")
    for field in ("x_sha256", "y_sha256", "config_sha256", "checkpoint_sha256"):
        _identity(manifest[field])
    constant = manifest["train_constant"]
    _keys(constant, {"majority", "probability"})
    _constant(constant["majority"], constant["probability"])
    reference = manifest["reference"]
    groups: dict[str, dict[str, np.dtype[Any]]] = {
        "predictions": {"logits": np.dtype(np.float32), "labels": np.dtype(np.uint8)}}
    if reference is not None:
        _keys(reference, {"majority", "probability"})
        _constant(reference["majority"], reference["probability"])
        groups["reference"] = {"bits": np.dtype(np.uint8)}
    _keys(manifest["files"], set(groups))
    if sum(rows * dtype.itemsize for group in groups.values()
           for dtype in group.values()) > _MAX_EXPANDED:
        raise ValueError("expanded array limit exceeded")
    values: dict[str, NDArray[Any]] = {}
    for group, expected in groups.items():
        file = manifest["files"][group]
        _keys(file, {"path", "size_bytes", "sha256", "arrays"})
        if type(file["path"]) is not str or file["path"] != f"{split}-{group}.npz":
            raise ValueError("unsafe/unexpected file path")
        if type(file["size_bytes"]) is not int or not 0 < file["size_bytes"] <= _MAX_FILE:
            raise ValueError("invalid file size")
        _identity(file["sha256"])
        _keys(file["arrays"], set(expected))
        for name, dtype in expected.items():
            meta = file["arrays"][name]
            _keys(meta, {"shape", "dtype", "sha256"})
            if (type(meta["shape"]) is not list or len(meta["shape"]) != 1
                    or type(meta["shape"][0]) is not int or meta["shape"] != [rows]
                    or type(meta["dtype"]) is not str or meta["dtype"] != dtype.str):
                raise ValueError("invalid array schema")
            _identity(meta["sha256"])
        data = _bounded_file(manifest_path.parent / file["path"], _MAX_FILE)
        if len(data) != file["size_bytes"] or _sha(data) != file["sha256"]:
            raise ValueError("file size/SHA mismatch")
        values.update(_read_npz(data, file["arrays"], rows))
    return PredictionEvidence(
        logits=values["logits"], labels=values["labels"], split=split, rows=rows,
        x_sha256=manifest["x_sha256"], y_sha256=manifest["y_sha256"],
        config_sha256=manifest["config_sha256"], checkpoint_sha256=manifest["checkpoint_sha256"],
        reference_bits=values.get("bits"),
        reference_majority=reference["majority"] if reference else 0,
        reference_probability=reference["probability"] if reference else .5,
        train_majority=constant["majority"], train_probability=constant["probability"])


def checked_bce_delta(
    logits: NDArray[np.float32], labels: NDArray[np.uint8], original_bce: float,
) -> float:
    """Compare stable FP64 evidence BCE with legacy FP32, never replace the metric."""
    _array(logits, np.dtype(np.float32), len(labels))
    _array(labels, np.dtype(np.uint8), len(labels))
    z = logits.astype(np.float64)
    precise = float(np.mean(np.logaddexp(0., z) - labels.astype(np.float64) * z))
    delta = precise - original_bce
    if not np.isfinite(delta) or abs(delta) > 1e-5:
        raise ValueError(f"FP64 evidence BCE discrepancy exceeds 1e-5: {delta}")
    return delta
