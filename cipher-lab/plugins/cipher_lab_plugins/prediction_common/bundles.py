"""Public TEST tensors and bounded, non-pickle NPZ serialization; no collection.

NPZ members are C-layout uint8 NPY arrays. Metadata is bounded UTF8 JSON,
never an arbitrary Python object. Public cards are NOT accepted from files;
the future representation owner must reconstruct them from these declarations.
Tensor outputs are CPU float32; inputs to the encoder must already be detached,
contiguous binary CPU float32. This module does not write paths or import ciphers.
"""
from __future__ import annotations

import ast
import io
import json
import math
import struct
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import IO, cast

import numpy as np
import torch
from torch import Tensor

from cipher_lab.data.bundles import PredictionTask as PredictionTask
from cipher_lab.training_config import TASK_FAMILIES, TaskSpec
from cipher_lab_plugins.feature_probe.tasks import FeatureProbeTask as FeatureProbeTask
from cipher_lab_plugins.frontier.config import parse_frontier_identity
from cipher_lab_plugins.frontier.tasks import FrontierTask as FrontierTask
from cipher_lab_plugins.residual.tasks import ResidualPredictionTask as ResidualPredictionTask

MAX_BYTES = 256 * 1024 * 1024
_MAX_METADATA = 8192
_MAX_HEADER = 4096










TaskDeclaration = (TaskSpec | PredictionTask | ResidualPredictionTask
                   | FeatureProbeTask | FrontierTask)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    method: str
    task_declarations: tuple[TaskDeclaration, ...]
    panels: Mapping[str, tuple[Tensor, Tensor]]
    references: Mapping[str, Tensor] = field(default_factory=dict)


def _identity(method: str, tasks: tuple[TaskDeclaration, ...]) -> tuple[str, ...]:
    if type(tasks) is not tuple:
        raise ValueError
    if method == "bit-public-feature-frontier-v1":
        if len(tasks) != 1 or type(tasks[0]) is not FrontierTask:
            raise ValueError
        tasks[0].__post_init__()
        return ("SIMON",)
    if method == "bit-public-feature-probe-v1":
        if len(tasks) != 1 or type(tasks[0]) is not FeatureProbeTask:
            raise ValueError
        tasks[0].__post_init__()
        return ("SIMON",)
    if method == "bit-residual-paired-v1":
        if len(tasks) != 1 or type(tasks[0]) is not ResidualPredictionTask:
            raise ValueError
        tasks[0].__post_init__()
        return ("SIMON",)
    if method == "bit-prediction-v1":
        if len(tasks) != 1 or type(tasks[0]) is not PredictionTask:
            raise ValueError
        return ("SIMON",)
    if method == "shared-m-four-task-v1":
        if (len(tasks) != 4 or any(type(task) is not TaskSpec for task in tasks)
                or tuple(task.cipher for task in tasks) != tuple(f.cipher for f in TASK_FAMILIES)):
            raise ValueError
        return tuple(task.cipher for task in tasks)
    raise ValueError


def _shape(shape: tuple[int, ...], method: str, *, label: bool = False) -> None:
    prediction = method in ("bit-prediction-v1", "bit-residual-paired-v1",
                            "bit-public-feature-probe-v1", "bit-public-feature-frontier-v1")
    limit = 2000000 if prediction else 65536
    minimum = 1 if prediction else 2
    suffix = (1,) if label else ((64,) if prediction else (4, 2, 64))
    if (not shape or not minimum <= shape[0] <= limit or shape[1:] != suffix
            or (method == "shared-m-four-task-v1" and shape[0] % 2)):
        raise ValueError


def _tensor(value: Tensor, method: str, *, label: bool = False) -> None:
    if (not isinstance(value, Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.layout != torch.strided
            or value.requires_grad or not value.is_contiguous()):
        raise ValueError
    _shape(tuple(value.shape), method, label=label)
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError


def encode_bundle(bundle: DatasetBundle) -> bytes:
    """Return bounded bytes for RunStore; never persist private material."""
    try:
        names = _identity(bundle.method, bundle.task_declarations)
        if set(bundle.panels) != set(names):
            raise ValueError
        residual = bundle.method in ("bit-residual-paired-v1", "bit-public-feature-probe-v1",
                                     "bit-public-feature-frontier-v1")
        if set(bundle.references) != (set(names) if residual else set()):
            raise ValueError
        schema = (4 if bundle.method == "bit-public-feature-frontier-v1" else
                  3 if bundle.method == "bit-public-feature-probe-v1" else 2 if residual else 1)
        metadata = json.dumps({"schema": schema, "method": bundle.method,
                               "task_declarations": [t.to_dict()
                                                     for t in bundle.task_declarations]},
                              allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(metadata) > _MAX_METADATA:
            raise ValueError
        # Check payload budget before tensor conversions/serialization; NPY/ZIP
        # overhead is bounded again below. Only one uint8 conversion at a time.
        total = len(metadata)
        for name in names:
            x, y = bundle.panels[name]
            _tensor(x, bundle.method)
            _tensor(y, bundle.method, label=True)
            if x.shape[0] != y.shape[0]:
                raise ValueError
            total += x.numel() + y.numel()
            if residual:
                h = bundle.references[name]
                _tensor(h, bundle.method, label=True)
                if h.shape != y.shape:
                    raise ValueError
                total += h.numel()
        if total + (len(names) * (3 if residual else 2) + 1) * _MAX_HEADER > MAX_BYTES:
            raise ValueError
        output = io.BytesIO()
        expanded = 0
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            def add(name: str, array: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]) -> None:
                nonlocal expanded
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                content = buffer.getvalue()
                expanded += len(content)
                if expanded > MAX_BYTES:
                    raise ValueError
                archive.writestr(name, content)
            add("metadata.npy", np.frombuffer(metadata, dtype=np.uint8))
            for index, name in enumerate(names):
                for axis, tensor in zip(("x", "y"), bundle.panels[name], strict=True):
                    add(f"p{index}_{axis}.npy", tensor.to(torch.uint8).numpy())
                if residual:
                    add(f"p{index}_h.npy", bundle.references[name].to(torch.uint8).numpy())
        content = output.getvalue()
        if len(content) > MAX_BYTES:
            raise ValueError
        return content
    except Exception:
        raise ValueError("invalid bundle") from None


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError
    return cast(dict[str, object], value)


def _metadata(raw: bytes) -> tuple[str, tuple[TaskDeclaration, ...]]:
    def bad_constant(_: str) -> object:
        raise ValueError
    node = _object(json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                              parse_constant=bad_constant),
                   {"schema", "method", "task_declarations"})
    method = node["method"]
    if (type(node["schema"]) is not int
            or node["schema"] != (4 if method == "bit-public-feature-frontier-v1" else
                                  3 if method == "bit-public-feature-probe-v1" else
                                  2 if method == "bit-residual-paired-v1" else 1)):
        raise ValueError
    declarations = node["task_declarations"]
    if type(method) is not str or type(declarations) is not list:
        raise ValueError
    tasks: list[TaskDeclaration] = []
    for value in declarations:
        if method == "bit-public-feature-frontier-v1":
            task = _object(value, {"cipher", "rounds", "target_bit", "label_mode", "reference",
                                   "identity"})
            tasks.append(FrontierTask(cast(str, task['cipher']), cast(int, task['rounds']),
                cast(int, task['target_bit']), cast(str, task['label_mode']),
                cast(str, task['reference']), parse_frontier_identity(task['identity'])))
        elif method == "bit-public-feature-probe-v1":
            task = _object(value, {"cipher", "rounds", "target_bit", "label_mode", "reference",
                                   "representation"})
            tasks.append(FeatureProbeTask(cast(str, task['cipher']), cast(int, task['rounds']),
                cast(int, task['target_bit']), cast(str, task['label_mode']),
                cast(str, task['reference']), cast(str, task['representation'])))
        elif method == "bit-residual-paired-v1":
            task = _object(value, {"cipher", "rounds", "target_bit", "label_mode", "reference"})
            tasks.append(ResidualPredictionTask(
                cast(str, task["cipher"]), cast(int, task["rounds"]),
                cast(int, task["target_bit"]), cast(str, task["label_mode"]),
                cast(str, task["reference"])))
        elif method == "bit-prediction-v1":
            task = _object(value, {"cipher", "rounds", "target_bit", "label_mode"})
            tasks.append(PredictionTask(cast(str, task["cipher"]), cast(int, task["rounds"]),
                                        cast(int, task["target_bit"]),
                                        cast(str, task["label_mode"])))
        elif method == "shared-m-four-task-v1":
            task = _object(value, {"cipher", "rounds", "difference"})
            tasks.append(TaskSpec(cast(str, task["cipher"]), cast(int, task["rounds"]),
                                  cast(int, task["difference"])))
        else:
            raise ValueError
    result = tuple(tasks)
    _identity(method, result)
    return method, result


def _zip_preflight(content: bytes) -> None:
    # Fixed EOCD with no comment, spanning or ZIP64: cap entries BEFORE ZipFile
    # materializes its central directory. Reject trailing/ambiguous archives.
    if type(content) is not bytes or not 22 <= len(content) <= MAX_BYTES:
        raise ValueError
    signature, disk, central_disk, count_disk, count, size, offset, comment = struct.unpack(
        "<4s4H2IH", content[-22:])
    if (signature != b"PK\x05\x06" or disk or central_disk or count != count_disk
            or not 1 <= count <= 16 or comment or offset + size != len(content) - 22):
        raise ValueError
    position = offset
    for _ in range(count):
        if content[position:position + 4] != b"PK\x01\x02" or position + 46 > len(content)-22:
            raise ValueError
        name, extra, note = struct.unpack_from("<3H", content, position + 28)
        position += 46 + name + extra + note
        if position > len(content) - 22:
            raise ValueError
    if position != len(content) - 22:
        raise ValueError


def _member_preflight(content: bytes, info: zipfile.ZipInfo, payload: int,
                      budget: int) -> int:
    """Verify the physical stream, not ZipExtFile's declared-size/CRC prefix.

    Discard fixed-size output chunks; never flush into an unbounded allocation.
    One byte past the remaining declaration/budget suffices to reject a bomb.
    """
    chunk_size = 64 * 1024
    end = payload + info.compress_size
    actual = 0
    crc = 0
    if info.compress_type == zipfile.ZIP_STORED:
        if info.compress_size != info.file_size or info.file_size > budget:
            raise ValueError
        for position in range(payload, end, chunk_size):
            raw = memoryview(content)[position:min(position + chunk_size, end)]
            actual += len(raw)
            crc = zlib.crc32(raw, crc)
    else:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        position = payload
        pending = b""
        while True:
            if not pending:
                following = min(position + chunk_size, end)
                pending = content[position:following]
                position = following
            limit = min(chunk_size, info.file_size - actual + 1, budget - actual + 1)
            submitted = len(pending)
            output = inflater.decompress(pending, limit)
            pending = inflater.unconsumed_tail
            actual += len(output)
            if actual > info.file_size or actual > budget:
                raise ValueError
            crc = zlib.crc32(output, crc)
            if inflater.eof:
                # This also rejects concatenated streams and compressed bytes
                # after EOF, whether in this chunk or not yet submitted.
                if inflater.unused_data or pending or position != end:
                    raise ValueError
                break
            # Input can be exhausted while zlib still holds a pending match.
            # Drain with the same output bound; reject a stalled/truncated stream.
            if not output and len(pending) == submitted:
                raise ValueError
    if actual != info.file_size or crc != info.CRC:
        raise ValueError
    return actual


def _header(stream: IO[bytes], member_size: int) -> tuple[tuple[int, ...], int]:
    prefix = stream.read(8)
    if prefix[:6] != b"\x93NUMPY" or prefix[6:] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise ValueError
    width = 2 if prefix[6] == 1 else 4
    raw_size = stream.read(width)
    if len(raw_size) != width:
        raise ValueError
    size = int.from_bytes(raw_size, "little")
    if not 1 <= size <= _MAX_HEADER or 8 + width + size > member_size:
        raise ValueError
    raw_header = stream.read(size)
    if len(raw_header) != size or not raw_header.endswith(b"\n"):
        raise ValueError
    text = raw_header.decode("utf-8" if prefix[6] == 3 else "latin1")
    syntax = ast.parse(text.strip(), mode="eval")
    if not isinstance(syntax.body, ast.Dict) or len(syntax.body.keys) != 3:
        raise ValueError
    header = _object(ast.literal_eval(syntax), {"descr", "fortran_order", "shape"})
    shape = header["shape"]
    if (header["descr"] != "|u1" or header["fortran_order"] is not False
            or type(shape) is not tuple or not 1 <= len(shape) <= 4
            or any(type(item) is not int or not 1 <= item <= MAX_BYTES for item in shape)):
        raise ValueError
    shape = cast(tuple[int, ...], shape)
    amount = math.prod(shape)
    if amount > MAX_BYTES or 8 + width + size + amount != member_size:
        raise ValueError
    return shape, amount


def decode_bundle(content: bytes) -> DatasetBundle:
    """Preflight ZIP and NPY headers before allocating each bounded payload.

    No np.load/pickle/torch.load, filesystem extraction, card deserialization or
    data provider imports. Errors deliberately omit file-controlled contents.
    """
    try:
        _zip_preflight(content)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (len(names) != len(set(names)) or len(infos) > 16
                    or sum(info.file_size for info in infos) > MAX_BYTES):
                raise ValueError
            allowed = {"metadata.npy"} | {
                f"p{i}_{axis}.npy" for i in range(4) for axis in ("x", "y", "h")
            }
            ranges: list[tuple[int, int]] = []
            members: list[tuple[zipfile.ZipInfo, int]] = []
            for info in infos:
                if (info.filename not in allowed or info.flag_bits & ~8
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or info.file_size < 10 or info.compress_size < 1):
                    raise ValueError
                start = info.header_offset
                if start < 0 or content[start:start+4] != b"PK\x03\x04":
                    raise ValueError
                flags, compression = struct.unpack_from("<HH", content, start + 6)
                name_size, extra_size = struct.unpack_from("<HH", content, start + 26)
                payload = start + 30 + name_size + extra_size
                if (flags != info.flag_bits or compression != info.compress_type
                        or content[start+30:start+30+name_size] != info.filename.encode("ascii")
                        or payload + info.compress_size > archive.start_dir):
                    raise ValueError
                crc, compressed, expanded = struct.unpack_from("<III", content, start + 14)
                if not flags & 8 and (crc != info.CRC or compressed != info.compress_size
                                      or expanded != info.file_size):
                    raise ValueError
                ranges.append((start, payload + info.compress_size))
                members.append((info, payload))
            ranges.sort()
            if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
                raise ValueError
            # Validate ALL real streams before even the first tensor allocation.
            # ZipExtFile alone hides extra output behind a forged file_size and
            # computes CRC only over that prefix; its apparent EOF is not proof.
            expanded_total = 0
            for info, payload in members:
                expanded_total += _member_preflight(content, info, payload,
                                                    MAX_BYTES - expanded_total)
            with archive.open("metadata.npy") as stream:
                shape, amount = _header(stream, archive.getinfo("metadata.npy").file_size)
                if len(shape) != 1 or amount > _MAX_METADATA:
                    raise ValueError
                method, tasks = _metadata(stream.read(amount))
                if stream.read(1):
                    raise ValueError
            expected_names = _identity(method, tasks)
            axes = (("x", "y", "h") if method in ("bit-residual-paired-v1",
                    "bit-public-feature-probe-v1", "bit-public-feature-frontier-v1")
                    else ("x", "y"))
            if set(names) != {"metadata.npy"} | {
                f"p{i}_{axis}.npy" for i in range(len(tasks)) for axis in axes
            }:
                raise ValueError
            # Preflight EVERY array header/shape before any tensor allocation,
            # including row agreement. A malicious late member cannot make us
            # allocate earlier large arrays first.
            for index in range(len(tasks)):
                rows: list[int] = []
                for axis in axes:
                    member = f"p{index}_{axis}.npy"
                    with archive.open(member) as stream:
                        shape, _amount = _header(stream, archive.getinfo(member).file_size)
                        _shape(shape, method, label=axis != "x")
                        rows.append(shape[0])
                if len(set(rows)) != 1:
                    raise ValueError
            panels: dict[str, tuple[Tensor, Tensor]] = {}
            references: dict[str, Tensor] = {}
            for index, name in enumerate(expected_names):
                tensors: list[Tensor] = []
                for axis in axes:
                    member = f"p{index}_{axis}.npy"
                    with archive.open(member) as stream:
                        shape, amount = _header(stream, archive.getinfo(member).file_size)
                        _shape(shape, method, label=axis != "x")
                        raw = stream.read(amount)
                        if len(raw) != amount or stream.read(1):
                            raise ValueError
                        array = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
                        if not bool((array <= 1).all()):
                            raise ValueError
                        tensors.append(torch.from_numpy(array.copy()).to(torch.float32))
                x, y = tensors[:2]
                if len(tensors) == 3:
                    references[name] = tensors[2]
                if x.shape[0] != y.shape[0]:
                    raise ValueError
                panels[name] = (x, y)
        return DatasetBundle(method, tasks, panels, references)
    except Exception:
        raise ValueError("invalid bundle") from None

