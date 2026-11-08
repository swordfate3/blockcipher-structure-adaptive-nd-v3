"""Artificial CPU tensors only; hostile NPZs never invoke collection/ciphers."""
from __future__ import annotations

import io
import json
import struct
import zipfile
import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from cipher_lab.data.bundles import (
    DatasetBundle,
    PredictionTask,
    decode_bundle,
    encode_bundle,
)
from cipher_lab.training_config import TaskSpec


def prediction() -> DatasetBundle:
    return DatasetBundle("bit-prediction-v1", (PredictionTask("SIMON", 13, 0, "cipher"),),
                         {"SIMON": (torch.zeros(2, 64), torch.tensor([[0.], [1.]]))})


def shared() -> DatasetBundle:
    tasks = tuple(TaskSpec(c, r, d) for c, r, d in (
        ("SIMECK", 16, 64), ("PRESENT", 7, 0xD00000),
        ("SIMON", 13, 64), ("SKINNY", 8, 0x2000)))
    return DatasetBundle("shared-m-four-task-v1", tasks,
                         {t.cipher: (torch.zeros(2, 4, 2, 64), torch.ones(2, 1)) for t in tasks})


def entries(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def archive(parts: Mapping[str, bytes], *, compressed: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=(zipfile.ZIP_DEFLATED if compressed
                                                else zipfile.ZIP_STORED)) as target:
        for name, value in parts.items():
            target.writestr(name, value)
    return output.getvalue()


def npy(value: np.ndarray[Any, Any]) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=True)
    return output.getvalue()


@pytest.mark.parametrize("factory", [prediction, shared])
def test_roundtrip_and_uint8_no_pickle(factory: object) -> None:
    original = prediction() if factory is prediction else shared()
    content = encode_bundle(original)
    decoded = decode_bundle(content)
    assert decoded.method == original.method
    assert decoded.task_declarations == original.task_declarations
    assert set(decoded.panels) == set(original.panels)
    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
        for name in saved.files:
            value = saved[name]
            assert value.dtype == np.uint8 and value.flags.c_contiguous
    for name, (x, y) in decoded.panels.items():
        assert x.dtype == y.dtype == torch.float32
        assert torch.equal(x, original.panels[name][0])
        assert torch.equal(y, original.panels[name][1])
    assert b"private" not in content and b"seed" not in content and b"key" not in content


@pytest.mark.parametrize("change", ["rounds", "target_bit", "label_mode"])
def test_prediction_identity_not_just_cipher(change: str) -> None:
    fields: dict[str, object] = {"cipher": "SIMON", "rounds": 13,
                                 "target_bit": 0, "label_mode": "cipher"}
    fields[change] = {"rounds": 14, "target_bit": 63, "label_mode": "null"}[change]
    # Explicit constructors avoid loose runtime dictionaries in the public typed API.
    task = PredictionTask(str(fields["cipher"]), int(str(fields["rounds"])),
                          int(str(fields["target_bit"])), str(fields["label_mode"]))
    bundle = prediction()
    other = DatasetBundle(bundle.method, (task,), bundle.panels)
    assert decode_bundle(encode_bundle(other)).task_declarations != bundle.task_declarations


@pytest.mark.parametrize("field,value", [("cipher", "PRESENT"), ("rounds", True),
    ("rounds", 45), ("target_bit", -1), ("target_bit", 64), ("label_mode", "secret")])
def test_invalid_prediction_metadata(field: str, value: object) -> None:
    raw = {"schema": 1, "method": "bit-prediction-v1", "task_declarations": [
        {"cipher": "SIMON", "rounds": 13, "target_bit": 0, "label_mode": "cipher"}]}
    raw["task_declarations"][0][field] = value  # type: ignore[index]
    parts = entries(encode_bundle(prediction()))
    parts["metadata.npy"] = npy(np.frombuffer(json.dumps(raw).encode(), dtype=np.uint8))
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(archive(parts))


@pytest.mark.parametrize("bad", [
    torch.zeros(2, 63), torch.zeros(2, 64, dtype=torch.int64),
    torch.full((2, 64), float("nan")), torch.full((2, 64), 2.),
    torch.zeros(64, 2).t(), torch.zeros(0, 64),
])
def test_encode_rejects_invalid_tensor(bad: Tensor) -> None:
    with pytest.raises(ValueError, match="bundle"):
        encode_bundle(DatasetBundle("bit-prediction-v1", prediction().task_declarations,
                                    {"SIMON": (bad, torch.zeros(2, 1))}))


@pytest.mark.parametrize("method,tasks,panels", [
    ("unknown", prediction().task_declarations, prediction().panels),
    ("bit-prediction-v1", (TaskSpec("SIMON", 13, 64),), prediction().panels),
    ("bit-prediction-v1", prediction().task_declarations, {}),
    ("shared-m-four-task-v1", shared().task_declarations[::-1], shared().panels),
])
def test_encode_rejects_wrong_identity(method: str, tasks: tuple[TaskSpec | PredictionTask, ...],
                                      panels: Mapping[str, tuple[Tensor, Tensor]]) -> None:
    with pytest.raises(ValueError, match="bundle"):
        encode_bundle(DatasetBundle(method, tasks, panels))


@pytest.mark.parametrize("attack", ["extra", "path", "duplicate", "object", "fortran",
    "float", "shape", "binary", "overflow", "header", "bomb", "entries", "truncated",
    "metadata_extra", "metadata_duplicate", "metadata_nan", "metadata_utf8", "metadata_shape",
    "schema_bool", "task_extra", "panel_rows", "central_count", "encrypted"])
def test_decode_hostile_npz(attack: str) -> None:
    parts = entries(encode_bundle(prediction()))
    if attack in ("extra", "path"):
        parts["extra.npy" if attack == "extra" else "../p0_x.npy"] = npy(np.zeros(1))
    elif attack == "duplicate":
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as target:
            for name, value in parts.items():
                target.writestr(name, value)
            with pytest.warns(UserWarning):
                target.writestr("metadata.npy", parts["metadata.npy"])
        with pytest.raises(ValueError, match="bundle"):
            decode_bundle(output.getvalue())
        return
    elif attack in ("object", "fortran", "float", "shape", "binary", "panel_rows"):
        values = {"object": np.array([object()], dtype=object),
                  "fortran": np.asfortranarray(np.zeros((2, 64), dtype=np.uint8)),
                  "float": np.zeros((2, 64)), "shape": np.zeros((2, 63), dtype=np.uint8),
                  "binary": np.full((2, 64), 2, dtype=np.uint8),
                  "panel_rows": np.zeros((3, 64), dtype=np.uint8)}
        parts["p0_x.npy"] = npy(values[attack])
    elif attack in ("overflow", "header", "bomb"):
        header = ("{'descr': '|u1', 'fortran_order': False, 'shape': "
                  + ("(2, 999999999999999999999999999999999), }" if attack == "overflow"
                     else "(268435457,), }"))
        raw = header.encode() + b"\n"
        parts["p0_x.npy"] = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(raw)) + raw
        if attack == "header":
            parts["p0_x.npy"] = b"\x93NUMPY\x02\x00" + struct.pack("<I", 2**32-1)
    elif attack == "entries":
        parts.update({f"extra{i}.npy": b"" for i in range(17)})
    elif attack.startswith("metadata") or attack in ("schema_bool", "task_extra"):
        raw_meta = np.load(io.BytesIO(parts["metadata.npy"]), allow_pickle=False).tobytes()
        meta = json.loads(raw_meta)
        if attack == "metadata_extra":
            meta["private_seed"] = 5
        elif attack == "schema_bool":
            meta["schema"] = True
        elif attack == "task_extra":
            meta["task_declarations"][0]["key"] = 0
        data = json.dumps(meta).encode()
        if attack == "metadata_duplicate":
            data = data[:-1] + b',"schema":1}'
        elif attack == "metadata_nan":
            data = data.replace(b'"schema": 1', b'"schema": NaN')
        elif attack == "metadata_utf8":
            data = b"\xff"
        array: np.ndarray[Any, np.dtype[np.uint8]] = np.frombuffer(data, dtype=np.uint8)
        if attack == "metadata_shape":
            array = array.reshape(1, -1)
        parts["metadata.npy"] = npy(array)
    content = archive(parts, compressed=True)
    if attack == "truncated":
        content = content[:-5]
    elif attack == "central_count":
        content = content[:-12] + struct.pack("<H", 65535) + content[-10:]
    elif attack == "encrypted":
        altered = bytearray(content)
        altered[6] |= 1
        content = bytes(altered)
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(content)


def test_compressed_valid_and_input_size_precheck(monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab.data.bundles as module
    content = archive(entries(encode_bundle(prediction())), compressed=True)
    assert decode_bundle(content).method == "bit-prediction-v1"
    monkeypatch.setattr(module, "MAX_BYTES", 32)
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(content)
    with pytest.raises(ValueError, match="bundle"):
        encode_bundle(prediction())


def test_all_headers_preflight_before_tensor_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    parts = entries(encode_bundle(shared()))
    parts["p3_y.npy"] = npy(np.zeros((2, 2), dtype=np.uint8))
    allocations: list[int] = []
    original = torch.from_numpy
    def track(value: np.ndarray[Any, Any]) -> Tensor:
        allocations.append(1)
        return original(value)
    monkeypatch.setattr(torch, "from_numpy", track)
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(archive(parts))
    assert allocations == []


def test_local_header_sizes_must_agree_with_central_directory() -> None:
    content = bytearray(encode_bundle(prediction()))
    # Forge the first local header's declared uncompressed size (not central).
    struct.pack_into("<I", content, 22, 1)
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(bytes(content))


def test_zip_declared_size_bomb_rejected_without_decompression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = bytearray(encode_bundle(prediction()))
    central = content.index(b"PK\x01\x02")
    struct.pack_into("<I", content, central + 24, 256 * 1024 * 1024 + 1)
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("oversized ZIP must be rejected before opening any member")
    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(bytes(content))


def test_total_uncompressed_size_cap_and_metadata_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab.data.bundles as module
    content = archive(entries(encode_bundle(shared())), compressed=True)
    expanded = sum(len(value) for value in entries(content).values())
    assert len(content) < expanded
    monkeypatch.setattr(module, "MAX_BYTES", expanded - 1)
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(content)
    monkeypatch.setattr(module, "MAX_BYTES", 256 * 1024 * 1024)
    parts = entries(encode_bundle(prediction()))
    parts["metadata.npy"] = npy(np.zeros(8193, dtype=np.uint8))
    with pytest.raises(ValueError, match="bundle"):
        decode_bundle(archive(parts))


def forged_last_member(parts: Mapping[str, bytes], compressed: bytes, *,
                       deflated: bool = True) -> bytes:
    """Keep both headers and CRC consistent with the visible NPY prefix."""
    original = archive(parts, compressed=deflated)
    with zipfile.ZipFile(io.BytesIO(original)) as source:
        info = source.infolist()[-1]
        central = source.start_dir
    local = info.header_offset
    name_size, extra_size = struct.unpack_from("<HH", original, local + 26)
    payload = local + 30 + name_size + extra_size
    result = bytearray(original[:payload] + compressed + original[central:])
    delta = len(compressed) - info.compress_size
    struct.pack_into("<I", result, local + 18, len(compressed))
    position = central + delta
    while result[position:position + 4] == b"PK\x01\x02":
        name, extra, note = struct.unpack_from("<3H", result, position + 28)
        if result[position + 46:position + 46 + name] == info.filename.encode():
            struct.pack_into("<I", result, position + 20, len(compressed))
            break
        position += 46 + name + extra + note
    else:
        raise AssertionError("missing last member in central directory")
    struct.pack_into("<I", result, len(result) - 6, central + delta)
    return bytes(result)


def deflate(raw: bytes, *, terminated: bool = True) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(raw) + compressor.flush(
        zlib.Z_FINISH if terminated else zlib.Z_SYNC_FLUSH)


@pytest.mark.parametrize("attack", ["stored_tail", "hidden_tail", "missing_eof",
                                   "compressed_tail", "late_compressed_tail",
                                   "concatenated_stream"])
def test_real_member_stream_checked_before_tensor_allocation(
    attack: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = entries(encode_bundle(prediction()))
    prefix = parts["p0_y.npy"]
    if attack == "stored_tail":
        raw = prefix + b"hidden"
    elif attack == "hidden_tail":
        raw = deflate(prefix + b"hidden")
    elif attack == "missing_eof":
        # All declared output is present, but DEFLATE has no final block.
        raw = deflate(prefix, terminated=False)
    elif attack == "late_compressed_tail":
        # EOF must also reject bytes beyond the current bounded input chunk.
        raw = deflate(prefix) + b"\x00" * 70000
    else:
        raw = deflate(prefix) + (b"hidden" if attack == "compressed_tail"
                                 else deflate(b"hidden"))
    content = forged_last_member(parts, raw, deflated=attack != "stored_tail")
    # Establish the exploit: ZipExtFile returns exactly the valid prefix and
    # accepts its CRC even though the physical member stream is malformed.
    assert entries(content) == parts
    allocations: list[int] = []
    original = torch.from_numpy
    def track(value: np.ndarray[Any, Any]) -> Tensor:
        allocations.append(1)
        return original(value)
    monkeypatch.setattr(torch, "from_numpy", track)
    with pytest.raises(ValueError, match="^invalid bundle$"):
        decode_bundle(content)
    assert allocations == []


def test_forged_small_size_cannot_expand_past_real_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cipher_lab.data.bundles as module
    parts = entries(encode_bundle(prediction()))
    content = forged_last_member(parts, deflate(parts["p0_y.npy"] + b"\x00" * 20000))
    # The file and sum of declared sizes fit; the real expansion does not.
    assert len(content) < 4096
    assert sum(map(len, parts.values())) < 4096
    assert entries(content) == parts
    monkeypatch.setattr(module, "MAX_BYTES", 4096)
    with pytest.raises(ValueError, match="^invalid bundle$"):
        decode_bundle(content)


@pytest.mark.parametrize("rows", [1022, 2048])
@pytest.mark.parametrize("level", [0, 6])
def test_valid_deflate_crosses_bounded_input_and_output_chunks(rows: int, level: int) -> None:
    # NPY header is 128 bytes: 1022*64 + 128 is exactly one 64KiB chunk.
    bundle = DatasetBundle("bit-prediction-v1", prediction().task_declarations,
                           {"SIMON": (torch.zeros(rows, 64), torch.ones(rows, 1))})
    parts = entries(encode_bundle(bundle))
    # Exercise multi-chunk input and highly compressed multi-chunk output.
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=level) as target:
        for name, value in parts.items():
            target.writestr(name, value)
    decoded = decode_bundle(output.getvalue())
    assert torch.equal(decoded.panels["SIMON"][0], bundle.panels["SIMON"][0])
    assert torch.equal(decoded.panels["SIMON"][1], bundle.panels["SIMON"][1])


def test_valid_deflate_drains_internal_match_after_input_is_consumed() -> None:
    bundle = DatasetBundle("bit-prediction-v1", prediction().task_declarations,
                           {"SIMON": (torch.zeros(1023, 64), torch.ones(1023, 1))})
    parts = entries(encode_bundle(bundle))
    raw = parts.pop("p0_x.npy")
    assert len(raw) == 65600  # 128-byte header plus 1023*64 uint8 values.
    assert raw[-259:] == b"\x00" * 259

    # Hand-build one final fixed-Huffman block: 65342 literals, then a
    # length-258/distance-1 match and EOB. Only the 0x93 NPY magic byte has a
    # nine-bit literal code. At the 64KiB output boundary zlib has consumed
    # ALL input, but still holds 64 output bytes and EOB internally.
    bits = [1, 1, 0]  # BFINAL=1, BTYPE=01 (least-significant bit first).
    def symbol(code: int, width: int) -> None:
        bits.extend((code >> shift) & 1 for shift in range(width - 1, -1, -1))
    for value in raw[:-258]:
        symbol(value + 0x30 if value < 144 else value + 0x100,
               8 if value < 144 else 9)
    symbol(0xC5, 8)  # Fixed code 285: length 258, no extra bits.
    symbol(0, 5)  # Distance 1, no extra bits.
    symbol(0, 7)  # Fixed code 256: EOB.
    assert len(bits) == 65345 * 8
    compressed = bytes(sum(bit << shift for shift, bit in enumerate(bits[start:start + 8]))
                       for start in range(0, len(bits), 8))
    inflater = zlib.decompressobj(-15)
    assert inflater.decompress(compressed, 65536) == raw[:65536]
    assert not inflater.eof
    assert inflater.unconsumed_tail == inflater.unused_data == b""
    assert inflater.decompress(b"", 65536) == raw[65536:]
    assert inflater.eof
    assert inflater.unconsumed_tail == inflater.unused_data == b""

    # Reuse the ZIP helper to insert these real bytes with correct local and
    # central sizes/CRC; moving x last leaves all earlier offsets unchanged.
    parts["p0_x.npy"] = raw
    content = forged_last_member(parts, compressed)
    assert entries(content) == parts
    decoded = decode_bundle(content)
    assert decoded.task_declarations == bundle.task_declarations
    assert torch.equal(decoded.panels["SIMON"][0], bundle.panels["SIMON"][0])
    assert torch.equal(decoded.panels["SIMON"][1], bundle.panels["SIMON"][1])


@pytest.mark.parametrize("damage", ["truncated_output", "bad_crc"])
def test_real_member_rejects_truncation_and_crc_before_allocation(
    damage: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = entries(encode_bundle(prediction()))
    prefix = parts["p0_y.npy"]
    raw = prefix[:-1] if damage == "truncated_output" else prefix[:-1] + b"\x00"
    content = forged_last_member(parts, deflate(raw))
    allocations: list[int] = []
    original = torch.from_numpy
    def track(value: np.ndarray[Any, Any]) -> Tensor:
        allocations.append(1)
        return original(value)
    monkeypatch.setattr(torch, "from_numpy", track)
    with pytest.raises(ValueError, match="^invalid bundle$"):
        decode_bundle(content)
    assert allocations == []


@pytest.mark.parametrize("rows", [1, 3])
def test_shared_rejects_nonbalanced_group_counts(rows: int) -> None:
    bundle = shared()
    panels = dict(bundle.panels)
    panels["SKINNY"] = (torch.zeros(rows, 4, 2, 64), torch.zeros(rows, 1))
    with pytest.raises(ValueError, match="bundle"):
        encode_bundle(DatasetBundle(bundle.method, bundle.task_declarations, panels))
