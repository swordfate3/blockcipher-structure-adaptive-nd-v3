"""Artificial file-only evidence contracts; never encrypt or evaluate a model."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_experiments.prediction_evidence import (
    load_prediction_evidence,
    write_prediction_evidence,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_args(**changes: Any) -> dict[str, Any]:
    y = np.array([0, 1, 1, 0], dtype=np.uint8)
    values = dict(split="test", logits=np.array([0., -2., 3., .25], dtype=np.float32),
                  labels=y, x_sha256="a" * 64,
                  y_sha256=sha(y.astype(np.float32).reshape(-1, 1).tobytes()),
                  config_sha256="c" * 64, checkpoint_sha256="d" * 64,
                  train_majority=1, train_probability=.6)
    values.update(changes)
    return values


def write(tmp_path: Path, **changes: Any) -> tuple[Path, dict[str, Any]]:
    index = write_prediction_evidence(tmp_path, **fixture_args(**changes))
    return tmp_path / index["manifest"], index


def read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def save_json(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("split", ["train", "dev", "test"])
@pytest.mark.parametrize("reference", [False, True])
def test_roundtrip_exact_arrays_separate_reference_and_hashes(
    tmp_path: Path, split: str, reference: bool,
) -> None:
    ref = np.array([1, 0, 0, 1], dtype=np.uint8)
    path, index = write(tmp_path, split=split,
                        **({"reference_bits": ref, "reference_majority": 0,
                            "reference_probability": .5} if reference else {}))
    assert index == {"manifest": f"predictions/{split}-predictions.json",
                     "sha256": sha(path.read_bytes())}
    m = read_json(path)
    assert m["rows"] == 4 and m["split"] == split
    assert m["y_sha256"] != m["files"]["predictions"]["arrays"]["labels"]["sha256"]
    evidence = load_prediction_evidence(path)
    for name in ("logits", "labels"):
        np.testing.assert_array_equal(getattr(evidence, name), fixture_args()[name])
        entry = m["files"]["predictions"]["arrays"][name]
        assert entry == {"dtype": getattr(evidence, name).dtype.str, "shape": [4],
                         "sha256": sha(getattr(evidence, name).tobytes())}
    assert (evidence.train_majority, evidence.train_probability) == (1, .6)
    for field in ("x_sha256", "y_sha256", "config_sha256", "checkpoint_sha256"):
        assert getattr(evidence, field) == fixture_args()[field]
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        assert set(arrays.files) == {"logits", "labels"}
    assert (evidence.reference_bits is not None) == reference
    if reference:
        np.testing.assert_array_equal(evidence.reference_bits, ref)
        assert (evidence.reference_majority, evidence.reference_probability) == (0, .5)
        with np.load(path.parent / f"{split}-reference.npz", allow_pickle=False) as arrays:
            assert arrays.files == ["bits"] and arrays["bits"].dtype == np.uint8


@pytest.mark.parametrize("changes", [
    {"split": "TEST"}, {"x_sha256": "A" * 64}, {"config_sha256": "no"},
    {"y_sha256": "e" * 64}, {"labels": np.array([0, 2, 1, 0], dtype=np.uint8)},
    {"labels": np.array([0, 1, 1, 0], dtype=np.float32)},
    {"logits": np.zeros(4, dtype=np.float64)}, {"logits": np.zeros((4, 1), dtype=np.float32)},
    {"logits": np.array([0, 1, np.nan, 2], dtype=np.float32)},
    {"logits": np.array([0, 1, np.inf, 2], dtype=np.float32)},
    {"reference_bits": np.zeros(3, dtype=np.uint8)},
    {"reference_bits": np.array([0, 1, 2, 0], dtype=np.uint8)},
    {"train_majority": True}, {"reference_majority": 2},
    {"train_probability": 0.}, {"reference_probability": float("nan")},
    {"train_probability": np.float64(.5)}, {"reference_probability": True},
])
def test_writer_validates_before_directory_creation(
    tmp_path: Path, changes: dict[str, Any],
) -> None:
    with pytest.raises((ValueError, TypeError)):
        write_prediction_evidence(tmp_path, **fixture_args(**changes))
    assert not (tmp_path / "predictions").exists()


def test_never_overwrite_or_follow_symlink(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    with pytest.raises(FileExistsError):
        write(tmp_path)
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
    link = tmp_path / "linked"
    link.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        load_prediction_evidence(link / path.name)
    path.unlink()
    path.symlink_to(path.parent / "test-predictions.npz")
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


@pytest.mark.parametrize("mutation", ["extra", "boolrows", "badrows", "split", "identity",
                                      "array_sha", "array_dtype", "array_shape", "path",
                                      "absolute", "filesize", "unknown_reference", "nan"])
def test_rejects_strict_manifest_mutations(tmp_path: Path, mutation: str) -> None:
    path, _ = write(tmp_path)
    m = read_json(path)
    f = m["files"]["predictions"]
    if mutation == "extra":
        m["secret"] = 1
    elif mutation == "boolrows":
        m["rows"] = True
    elif mutation == "badrows":
        m["rows"] = 1_000_001
    elif mutation == "split":
        m["split"] = "dev"
    elif mutation == "identity":
        m["y_sha256"] = "e" * 64
    elif mutation.startswith("array_"):
        f["arrays"]["labels"][mutation[6:]] = {
            "sha": "f" * 64, "dtype": "<f4", "shape": [2, 2]}[mutation[6:]]
        if mutation == "array_sha":
            f["arrays"]["labels"]["sha256"] = f["arrays"]["labels"].pop("sha")
    elif mutation == "path":
        f["path"] = "../test-predictions.npz"
    elif mutation == "absolute":
        f["path"] = str(path.with_suffix(".npz"))
    elif mutation == "filesize":
        f["size_bytes"] += 1
    elif mutation == "unknown_reference":
        m["reference"] = {"majority": 0, "probability": .5}
    else:
        m["train_constant"]["probability"] = float("nan")
    save_json(path, m)
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


def replace_npz(path: Path, data: bytes) -> None:
    npz = path.with_suffix(".npz")
    npz.write_bytes(data)
    m = read_json(path)
    m["files"]["predictions"].update(size_bytes=len(data), sha256=sha(data))
    save_json(path, m)


def npy_bytes(value: Any) -> bytes:
    out = io.BytesIO()
    np.save(out, value, allow_pickle=True)
    return out.getvalue()


@pytest.mark.parametrize("kind", ["duplicate", "extra", "traversal", "object", "dtype",
                                  "shape", "fortran", "nan", "bomb", "trailing", "truncated"])
def test_hostile_zip_headers_rejected_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    path, _ = write(tmp_path)
    args = fixture_args()
    z = npy_bytes(args["logits"])
    if kind == "object":
        z = npy_bytes(np.array([object()] * 4))
    elif kind == "dtype":
        z = npy_bytes(np.zeros(4, dtype=np.float64))
    elif kind in ("shape", "fortran", "bomb"):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            "descr": "<f4", "fortran_order": kind == "fortran",
            "shape": (4, 1) if kind == "shape" else ((10**12,) if kind == "bomb" else (4,))})
        z = header.getvalue() + b"\0" * 16
    elif kind == "nan":
        z = npy_bytes(np.array([0., 1., np.nan, 2.], dtype=np.float32))
    elif kind == "trailing":
        z += b"extra"
    elif kind == "truncated":
        z = z[:-1]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("logits.npy", z)
        archive.writestr("labels.npy", npy_bytes(args["labels"]))
        if kind in ("duplicate", "extra", "traversal"):
            with pytest.warns(UserWarning) if kind == "duplicate" else nullcontext():
                archive.writestr({"duplicate": "labels.npy", "extra": "extra.npy",
                                  "traversal": "../other.npy"}[kind], npy_bytes(args["labels"]))
    replace_npz(path, out.getvalue())
    if kind != "nan":
        def no_load(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("must reject header/zip before np.load")
        monkeypatch.setattr(np, "load", no_load)
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


def test_file_sha_checked_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = write(tmp_path)
    with path.with_suffix(".npz").open("ab") as f:
        f.write(b"corrupt")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("late SHA check"))
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


def test_manifest_size_and_duplicate_json_keys(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    path.write_bytes(b" " * (256 * 1024 + 1))
    with pytest.raises(ValueError):
        load_prediction_evidence(path)
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


def test_bce_discrepancy_is_checked_without_changing_original_metrics() -> None:
    from cipher_experiments.prediction_evidence import checked_bce_delta

    z = np.array([100., -100., 0., -2.], dtype=np.float32)
    y = np.array([1, 0, 1, 1], dtype=np.uint8)
    precise = float(np.mean(np.logaddexp(0., z.astype(np.float64)) - y * z.astype(np.float64)))
    assert checked_bce_delta(z, y, precise + 1e-6) == pytest.approx(-1e-6)
    with pytest.raises(ValueError, match="BCE"):
        checked_bce_delta(z, y, precise + 2e-5)
    with pytest.raises(ValueError, match="BCE"):
        checked_bce_delta(z, y, float("nan"))


def test_nonregular_file_rejected_before_opening_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    path, _ = write(tmp_path)
    archive = path.with_suffix(".npz")
    archive.unlink()
    os.mkfifo(archive)
    real_open = Path.open

    def guarded_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == archive:
            pytest.fail("must reject FIFO without opening (would block)")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


@pytest.mark.parametrize("kind", ["declared_size", "actual_size", "bool_shape", "bits_dtype",
                                  "reference_sha", "reference_symlink", "empty", "huge_rows"])
def test_additional_limits_and_reference_identity(tmp_path: Path, kind: str) -> None:
    if kind in ("empty", "huge_rows"):
        n = 0 if kind == "empty" else 1_000_001
        y = np.zeros(n, dtype=np.uint8)
        with pytest.raises(ValueError):
            write(tmp_path, labels=y, logits=np.zeros(n, dtype=np.float32),
                  y_sha256=sha(y.astype(np.float32).tobytes()))
        assert not (tmp_path / "predictions").exists()
        return
    path, _ = write(tmp_path, reference_bits=np.array([1, 0, 1, 0], dtype=np.uint8))
    m = read_json(path)
    prediction = m["files"]["predictions"]
    reference = m["files"]["reference"]
    if kind == "declared_size":
        prediction["size_bytes"] = 32 * 1024 * 1024 + 1
    elif kind == "actual_size":
        with path.with_suffix(".npz").open("r+b") as stream:
            stream.truncate(32 * 1024 * 1024 + 1)
    elif kind == "bool_shape":
        prediction["arrays"]["logits"]["shape"] = [True]
    elif kind == "bits_dtype":
        reference["arrays"]["bits"]["dtype"] = "|i1"
    elif kind == "reference_sha":
        reference["sha256"] = "0" * 64
    else:
        refpath = path.parent / reference["path"]
        refpath.unlink()
        refpath.symlink_to(path.with_suffix(".npz"))
    save_json(path, m)
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


def test_failed_publication_never_creates_manifest_or_removes_unknown_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    real_link = os.link
    calls = 0

    def fail_second(source: Any, target: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("artificial failure")
        real_link(source, target)

    monkeypatch.setattr(os, "link", fail_second)
    with pytest.raises(OSError, match="artificial"):
        write(tmp_path, reference_bits=np.zeros(4, dtype=np.uint8))
    directory = tmp_path / "predictions"
    assert (directory / "test-predictions.npz").exists()
    assert not (directory / "test-predictions.json").exists()
    assert len(list(directory.glob(".evidence-*"))) == 1
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        write(tmp_path)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("kind", ["nan", "inf", "nonbinary", "order"])
def test_matching_container_and_array_hashes_do_not_bypass_value_or_y_identity(
    tmp_path: Path, kind: str,
) -> None:
    path, _ = write(tmp_path)
    m = read_json(path)
    args = fixture_args()
    arrays = {"logits": args["logits"], "labels": args["labels"]}
    if kind in ("nan", "inf"):
        arrays["logits"][2] = np.nan if kind == "nan" else np.inf
        changed = "logits"
    elif kind == "nonbinary":
        arrays["labels"][2] = 2
        changed = "labels"
    else:
        arrays["labels"] = arrays["labels"][[1, 0, 2, 3]]
        changed = "labels"
    m["files"]["predictions"]["arrays"][changed]["sha256"] = sha(arrays[changed].tobytes())
    save_json(path, m)
    out = io.BytesIO()
    np.savez(out, **arrays)
    replace_npz(path, out.getvalue())
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


def test_writer_copies_noncontiguous_arrays_in_logical_order(tmp_path: Path) -> None:
    z = np.arange(8, dtype=np.float32)[::2]
    y = np.array([0, 9, 1, 9, 1, 9, 0, 9], dtype=np.uint8)[::2]
    ref = np.array([1, 9, 0, 9, 1, 9, 0, 9], dtype=np.uint8)[::2]
    path, _ = write(tmp_path, logits=z, labels=y, reference_bits=ref)
    saved = load_prediction_evidence(path)
    np.testing.assert_array_equal(saved.logits, z)
    np.testing.assert_array_equal(saved.labels, y)
    np.testing.assert_array_equal(saved.reference_bits, ref)
    z[:] = -9
    y[:] = 0
    ref[:] = 0
    assert saved.logits.tolist() == [0., 2., 4., 6.]
    assert saved.labels.tolist() == [0, 1, 1, 0]
    assert saved.reference_bits is not None
    assert saved.reference_bits.tolist() == [1, 0, 1, 0]


def test_numpy20_savez_signature_does_not_add_allow_pickle_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NumPy 2.0/2.1 treats every keyword as an array, including allow_pickle.
    def legacy_savez(file: Any, *args: Any, **kwds: Any) -> None:
        with zipfile.ZipFile(file, "w") as archive:
            for name, value in kwds.items():
                archive.writestr(f"{name}.npy", npy_bytes(np.asanyarray(value)))

    monkeypatch.setattr(np, "savez", legacy_savez)
    path, _ = write(tmp_path, reference_bits=np.array([1, 0, 1, 0], dtype=np.uint8))
    for filename, expected in (("test-predictions.npz", {"logits.npy", "labels.npy"}),
                               ("test-reference.npz", {"bits.npy"})):
        with zipfile.ZipFile(path.parent / filename) as archive:
            assert set(archive.namelist()) == expected
    saved = load_prediction_evidence(path)
    np.testing.assert_array_equal(saved.logits, fixture_args()["logits"])
    np.testing.assert_array_equal(saved.reference_bits, [1, 0, 1, 0])


def zip_with_logits(z: bytes, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=compression) as archive:
        archive.writestr("logits.npy", z)
        archive.writestr("labels.npy", npy_bytes(fixture_args()["labels"]))
    return out.getvalue()


def lie_about_logits_size(data: bytes, size: int) -> bytes:
    # Patch local and central uncompressed lengths; compressed bytes stay small.
    import struct

    changed = bytearray(data)
    struct.pack_into("<I", changed, 22, size)
    central = changed.index(b"PK\x01\x02")
    struct.pack_into("<I", changed, central + 24, size)
    return bytes(changed)


@pytest.mark.parametrize("version", [1, 2])
def test_oversized_npy_header_rejected_before_large_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int,
) -> None:
    import struct

    path, _ = write(tmp_path)
    z = b"\x93NUMPY" + bytes([version, 0])
    z += struct.pack("<H" if version == 1 else "<I", 65535 if version == 1 else 2**31)
    data = bytearray(lie_about_logits_size(zip_with_logits(z + b" " * 512), 144))
    import zlib
    central = data.index(b"PK\x01\x02")
    struct.pack_into("<I", data, central + 16, zlib.crc32((z + b" " * 512)[:144]))
    replace_npz(path, bytes(data))
    original = zipfile.ZipExtFile.read

    def guarded_read(self: Any, n: int = -1) -> bytes:
        assert 0 <= n <= 4096, f"unbounded NPY read request: {n}"
        return original(self, n)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", guarded_read)
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


@pytest.mark.parametrize("kind", ["header", "payload", "trailing", "crc", "truncated_deflate"])
def test_actual_deflate_is_bounded_even_when_zip_size_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    import zlib

    path, _ = write(tmp_path)
    valid = npy_bytes(fixture_args()["logits"])
    if kind == "header":
        z = b"\x93NUMPY\x02\x00\x00\x00\x00\x80" + b" " * 8192
    elif kind == "payload":
        z = valid + b"\0" * 8192
    else:
        z = valid
    data = zip_with_logits(z)
    if kind in ("header", "payload"):
        changed = bytearray(lie_about_logits_size(data, len(valid)))
        import struct
        central = changed.index(b"PK\x01\x02")
        struct.pack_into("<I", changed, central + 16, zlib.crc32(z[:len(valid)]))
        data = bytes(changed)
    else:
        changed = bytearray(data)
        central = changed.index(b"PK\x01\x02")
        if kind == "crc":
            changed[central + 16] ^= 1
        elif kind == "trailing":
            # Include one byte beyond the DEFLATE end in the compressed span.
            import struct
            compressed = struct.unpack_from("<I", changed, central + 20)[0]
            struct.pack_into("<I", changed, central + 20, compressed + 1)
        else:
            import struct
            compressed = struct.unpack_from("<I", changed, central + 20)[0]
            struct.pack_into("<I", changed, central + 20, compressed - 1)
        data = bytes(changed)
    replace_npz(path, data)
    real_factory = zlib.decompressobj
    requests: list[int] = []

    class GuardedDecompressor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = real_factory(*args, **kwargs)

        def decompress(self, data: bytes, max_length: int = 0) -> bytes:
            requests.append(max_length)
            # Tiny fixture: no valid stage needs >4096 output bytes.
            assert 0 < max_length <= 4096, f"unbounded decompress request: {max_length}"
            return self.inner.decompress(data, max_length)

        def flush(self, *args: Any) -> bytes:
            pytest.fail("flush may emit unbounded output")

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

    monkeypatch.setattr(zlib, "decompressobj", GuardedDecompressor)
    with pytest.raises(ValueError):
        load_prediction_evidence(path)
    if kind in ("header", "payload", "crc", "truncated_deflate"):
        assert requests


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
@pytest.mark.parametrize("version", [1, 2])
def test_safe_zip_compression_and_npy_versions_still_load(
    tmp_path: Path, compression: int, version: int,
) -> None:
    path, _ = write(tmp_path)
    out = io.BytesIO()
    np.lib.format.write_array(out, fixture_args()["logits"], version=(version, 0),
                              allow_pickle=False)
    replace_npz(path, zip_with_logits(out.getvalue(), compression))
    np.testing.assert_array_equal(load_prediction_evidence(path).logits, fixture_args()["logits"])


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_actual_payload_tail_cannot_hide_behind_declared_size_and_prefix_crc(
    tmp_path: Path, compression: int,
) -> None:
    import struct
    import zlib

    path, _ = write(tmp_path)
    valid = npy_bytes(fixture_args()["logits"])
    changed = bytearray(lie_about_logits_size(
        zip_with_logits(valid + b"hidden tail" * 16, compression), len(valid)))
    central = changed.index(b"PK\x01\x02")
    # Both ZIP declarations and CRC describe only the valid prefix.
    struct.pack_into("<I", changed, central + 16, zlib.crc32(valid))
    struct.pack_into("<I", changed, 14, zlib.crc32(valid))
    replace_npz(path, bytes(changed))
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


@pytest.mark.parametrize("header", [
    b"\x93NUMPY\x03\x00" + b"\0" * 16,
    b"\x93NUMPY\x01\x01" + b"\0" * 16,
    b"\x93NUMPY\x02\x00\x10",
    b"\x93NUMPY\x02\x00\x20\x00\x00\x00short",
])
def test_unsupported_or_truncated_npy_prefix_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, header: bytes,
) -> None:
    path, _ = write(tmp_path)
    replace_npz(path, zip_with_logits(header))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("late header rejection"))
    with pytest.raises(ValueError):
        load_prediction_evidence(path)


def test_bytes_after_deflate_end_rejected_without_zip_overlap(tmp_path: Path) -> None:
    import struct

    path, _ = write(tmp_path)
    # Insert junk inside the declared compressed span, shifting all subsequent
    # offsets consistently so ZIP overlap checks cannot substitute for EOF checks.
    data = bytearray(zip_with_logits(npy_bytes(fixture_args()["logits"])))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        first, second = archive.infolist()
    name_size, extra_size = struct.unpack_from("<HH", data, 26)
    end = 30 + name_size + extra_size + first.compress_size
    data[end:end] = b"junk"
    first_central = data.index(b"PK\x01\x02")
    second_central = data.index(b"PK\x01\x02", first_central + 4)
    eocd = data.index(b"PK\x05\x06")
    struct.pack_into("<I", data, 18, first.compress_size + 4)
    struct.pack_into("<I", data, first_central + 20, first.compress_size + 4)
    struct.pack_into("<I", data, second_central + 42, second.header_offset + 4)
    struct.pack_into("<I", data, eocd + 16, first_central)
    replace_npz(path, bytes(data))
    with pytest.raises(ValueError, match="ZIP actual size/CRC/end mismatch"):
        load_prediction_evidence(path)
