"""Bounded public-only CPU checkpoints, not optimizer/resume or file persistence.

ZIP and pickle storage declarations are inspected BEFORE torch.load. The bounded
unpickler only builds inert metadata/tensor descriptors, never imports globals.
It rejects a large declared storage even if the raw archive is tiny. Actual tensor
loading remains weights_only=True, followed by exact state and wiring validation.
"""
from __future__ import annotations

import hashlib
import io
import math
import pickle
import pickletools
import re
import struct
import zipfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import torch
from torch import Tensor

from cipher_lab.data.bundles import PredictionTask
from cipher_lab.models import BitMLP, SharedM
from cipher_lab.training_config import MLPSpec, SharedMSpec, TaskSpec
from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec
from cipher_lab_plugins.feature_probe.model import FeatureBitMLP
from cipher_lab_plugins.feature_probe.representation import FEATURE_ORDER
from cipher_lab_plugins.feature_probe.tasks import FeatureProbeTask
from cipher_lab_plugins.frontier.config import FrontierMLPSpec
from cipher_lab_plugins.frontier.model import FrontierBitMLP
from cipher_lab_plugins.frontier.tasks import FrontierTask
from cipher_lab_plugins.prediction_common.bundles import TaskDeclaration
from cipher_lab_plugins.prediction_common.metrics import (
    Model,
    PublicModelSpec,
    TrainBaseline,
    validate_baseline,
    validate_model,
)
from cipher_lab_plugins.residual.tasks import ResidualPredictionTask

MAX_BYTES = 256 * 1024 * 1024
_MAX_ENTRIES = 2048
_MAX_PICKLE = 1024 * 1024
_MAX_TUPLE_WORK = 100000
_SHA = re.compile('[0-9a-f]{64}')


@dataclass(frozen=True, slots=True)
class Selection:
    epoch: int
    updates: int
    selected_dev_bce: float

    def __post_init__(self) -> None:
        if (type(self.epoch) is not int or not 1 <= self.epoch <= 100000
                or type(self.updates) is not int or not 1 <= self.updates <= 2**63-1
                or type(self.selected_dev_bce) not in (float, int)
                or not math.isfinite(self.selected_dev_bce) or self.selected_dev_bce < 0):
            raise ValueError('invalid checkpoint selection')


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Model
    public_model_spec: PublicModelSpec
    task_declarations: tuple[TaskDeclaration, ...]
    selection: Selection
    baseline: Mapping[str, TrainBaseline]
    recipe_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'baseline', MappingProxyType(dict(self.baseline)))


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError
    return cast(dict[str, object], value)


def _sha(value: object) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError
    return value


def _family(spec: PublicModelSpec) -> str:
    if type(spec.model) is FrontierMLPSpec:
        return 'frontier-bit-mlp-v1'
    if type(spec.model) is FeatureMLPSpec:
        return 'feature-bit-mlp-v1'
    return 'bit-mlp-v1' if type(spec.model) is MLPSpec else 'shared-m-v1'


def _wiring(spec: PublicModelSpec) -> dict[str, object]:
    if type(spec.model) in (MLPSpec, FeatureMLPSpec, FrontierMLPSpec):
        wiring: dict[str, object] = {'input_bits': 64, 'output_logits': 1, 'hidden_layers': 2,
                                     'activation': 'relu', 'bit_order': 'msb0'}
        if spec.method == 'bit-public-feature-frontier-v1':
            task = cast(FrontierTask, spec.task_declarations[0])
            wiring.update(arm_id=spec.arm_id, output_mode=spec.output_mode,
                representation=spec.input_representation,
                network_input_width=(spec.model.identity.dimension
                    if type(spec.model) is FrontierMLPSpec else 64),
                plan_identity=task.identity.to_dict())
        if spec.method == 'bit-public-feature-probe-v1':
            wiring.update(arm_id=spec.arm_id, output_mode=spec.output_mode,
                representation=spec.input_representation,
                network_input_width=13 if type(spec.model) is FeatureMLPSpec else 64,
                feature_order=list(FEATURE_ORDER) if type(spec.model) is FeatureMLPSpec else [])
        if spec.method == 'bit-residual-paired-v1':
            wiring['output_mode'] = spec.output_mode
        return wiring
    return {'input_shape': [4, 2, 64], 'output_logits': 1, 'public_width': 13,
            'stems': ['local', 'word', 'nibble'], 'route': 'learned-stable-top2',
            'view': 'single', 'relation': True, 'readout': 'pair',
            'representation': 'public-backstep-rounds-v1'}


def _new(spec: PublicModelSpec, *, meta: bool = False) -> Model:
    # Reinitialization must not advance the caller's CPU RNG. No CUDA access.
    with torch.random.fork_rng(devices=[]), torch.device('meta' if meta else 'cpu'):
        if type(spec.model) is MLPSpec:
            return BitMLP(spec.model)
        if type(spec.model) is FeatureMLPSpec:
            return FeatureBitMLP(spec.model)
        if type(spec.model) is FrontierMLPSpec:
            return FrontierBitMLP(spec.model)
        return SharedM(cast(SharedMSpec, spec.model))


def _metadata(node: object, plan: object = None
              ) -> tuple[PublicModelSpec, Selection, dict[str, TrainBaseline], str]:
    if plan is not None:
        from cipher_lab_plugins.frontier.checkpoint import frontier_metadata
        from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan
        spec, selected, frontier_baseline, sha = frontier_metadata(
            node, cast(PublicFeaturePlan, plan))
        return spec, cast(Selection, selected), frontier_baseline, sha
    obj = _object(node, {'schema', 'method', 'model_family', 'model_spec', 'wiring',
                         'task_declarations', 'selection', 'baseline', 'recipe_sha', 'state'})
    probe = obj['method'] == 'bit-public-feature-probe-v1'
    residual = obj['method'] == 'bit-residual-paired-v1'
    if type(obj['schema']) is not int or obj['schema'] != (3 if probe else 2 if residual else 1):
        raise ValueError
    raw = obj['model_spec']
    if obj['model_family'] == 'bit-mlp-v1':
        model_node = _object(raw, {'hidden_width'})
        model: MLPSpec | FeatureMLPSpec | SharedMSpec = MLPSpec(
            cast(int, model_node['hidden_width']))
    elif obj['model_family'] == 'feature-bit-mlp-v1':
        model_node = _object(raw, {'hidden_width', 'representation'})
        model = FeatureMLPSpec(cast(int, model_node['hidden_width']),
                               cast(str, model_node['representation']))
    elif obj['model_family'] == 'shared-m-v1':
        model_node = _object(raw, {'stem_width', 'channels', 'norm_groups', 'suffix_kernels',
                                   'head_hidden', 'router_hidden'})
        kernels = model_node['suffix_kernels']
        if type(kernels) is not list:
            raise ValueError
        model = SharedMSpec(cast(int, model_node['stem_width']), cast(int, model_node['channels']),
                            cast(int, model_node['norm_groups']), tuple(cast(list[int], kernels)),
                            cast(int, model_node['head_hidden']),
                            cast(int, model_node['router_hidden']))
    else:
        raise ValueError
    raw_tasks = obj['task_declarations']
    if type(raw_tasks) is not list or not 1 <= len(raw_tasks) <= 4:
        raise ValueError
    tasks: list[TaskDeclaration] = []
    for value in raw_tasks:
        if probe:
            task = _object(value, {'cipher', 'rounds', 'target_bit', 'label_mode', 'reference',
                                   'representation'})
            tasks.append(FeatureProbeTask(cast(str, task['cipher']), cast(int, task['rounds']),
                cast(int, task['target_bit']), cast(str, task['label_mode']),
                cast(str, task['reference']), cast(str, task['representation'])))
        elif type(model) is MLPSpec and residual:
            task = _object(value, {'cipher', 'rounds', 'target_bit', 'label_mode', 'reference'})
            tasks.append(ResidualPredictionTask(
                cast(str, task['cipher']), cast(int, task['rounds']),
                cast(int, task['target_bit']), cast(str, task['label_mode']),
                cast(str, task['reference'])))
        elif type(model) is MLPSpec:
            task = _object(value, {'cipher', 'rounds', 'target_bit', 'label_mode'})
            tasks.append(PredictionTask(cast(str, task['cipher']), cast(int, task['rounds']),
                                        cast(int, task['target_bit']),
                                        cast(str, task['label_mode'])))
        else:
            task = _object(value, {'cipher', 'rounds', 'difference'})
            tasks.append(TaskSpec(cast(str, task['cipher']), cast(int, task['rounds']),
                                  cast(int, task['difference'])))
    output_mode = "direct"
    if residual:
        wiring = _object(obj['wiring'], {'input_bits', 'output_logits', 'hidden_layers',
                                         'activation', 'bit_order', 'output_mode'})
        output_mode = cast(str, wiring['output_mode'])
    arm_id = input_representation = None
    if probe:
        wiring = _object(obj['wiring'], {'input_bits', 'output_logits', 'hidden_layers',
            'activation', 'bit_order', 'output_mode', 'arm_id', 'representation',
            'network_input_width', 'feature_order'})
        output_mode = cast(str, wiring['output_mode'])
        arm_id = cast(str, wiring['arm_id'])
        input_representation = cast(str, wiring['representation'])
    spec = PublicModelSpec(model, tuple(tasks), output_mode, arm_id, input_representation)
    if obj['method'] != spec.method or obj['wiring'] != _wiring(spec):
        raise ValueError
    # Equality alone admits True == 1. Require the exact public primitive types too.
    if not _same_public(obj['wiring'], _wiring(spec)):
        raise ValueError
    selected = _object(obj['selection'], {'epoch', 'updates', 'selected_dev_bce'})
    selection = Selection(cast(int, selected['epoch']), cast(int, selected['updates']),
                          cast(float, selected['selected_dev_bce']))
    counts = _object(obj['baseline'], set(spec.names))
    baseline: dict[str, TrainBaseline] = {}
    for name, value in counts.items():
        count = _object(value, {'rows', 'ones'})
        baseline[name] = TrainBaseline(cast(int, count['rows']), cast(int, count['ones']))
    validate_baseline(spec, baseline)
    return spec, selection, baseline, _sha(obj['recipe_sha'])


def _same_public(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        left, right = cast(dict[str, object], a), cast(dict[str, object], b)
        return set(left) == set(right) and all(_same_public(left[k], right[k]) for k in left)
    if type(a) is list:
        left_list, right_list = cast(list[object], a), cast(list[object], b)
        return len(left_list) == len(right_list) and all(
            _same_public(x, y) for x, y in zip(left_list, right_list, strict=True))
    return bool(a == b)


def _validate_state(state: object, expected: Mapping[str, Tensor], model: Model) -> None:
    values = _object(state, set(expected))
    fixed = set(dict(model.named_buffers()))
    for name, template in expected.items():
        value = values[name]
        if (type(value) is not Tensor or value.device.type != 'cpu'
                or value.layout != torch.strided or value.dtype != template.dtype
                or value.shape != template.shape or not value.is_contiguous()
                or value.storage_offset() != 0 or value.requires_grad
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()
                or not bool(torch.isfinite(value).all())):
            raise ValueError
        if name in fixed and not torch.equal(value, template):
            raise ValueError


def encode_checkpoint(model: Model, public_model_spec: PublicModelSpec,
                      task_declarations: tuple[TaskDeclaration, ...], selection: Selection,
                      baseline: Mapping[str, TrainBaseline], recipe_sha: str) -> bytes:
    """Serialize detached CPU copies only; caller owns RunStore persistence."""
    try:
        public_model_spec.__post_init__()
        # Frontier requires its independent plan member and dedicated encoder.
        # Never accidentally produce a schema1 archive for a schema4 identity.
        if public_model_spec.method == 'bit-public-feature-frontier-v1':
            raise ValueError
        if task_declarations != public_model_spec.task_declarations:
            raise ValueError
        if type(selection) is not Selection:
            raise ValueError
        selection.__post_init__()
        _sha(recipe_sha)
        validate_baseline(public_model_spec, baseline)
        validate_model(model, public_model_spec)
        # Meta construction bounds even legal but too-large specifications before allocation.
        expected_meta = _new(public_model_spec, meta=True).state_dict()
        if sum(t.numel() * t.element_size() for t in expected_meta.values()) > MAX_BYTES:
            raise ValueError
        reference = _new(public_model_spec)
        _validate_state(dict(model.state_dict()), reference.state_dict(), reference)
        state = {k: v.detach().to(device='cpu', copy=True).contiguous()
                 for k, v in model.state_dict().items()}
        node: dict[str, object] = {
            'schema': (3 if public_model_spec.method == 'bit-public-feature-probe-v1' else
                       2 if public_model_spec.method == 'bit-residual-paired-v1' else 1),
            'method': public_model_spec.method,
            'model_family': _family(public_model_spec),
            'model_spec': public_model_spec.model.to_dict(), 'wiring': _wiring(public_model_spec),
            'task_declarations': [t.to_dict() for t in task_declarations],
            'selection': {'epoch': selection.epoch, 'updates': selection.updates,
                          'selected_dev_bce': selection.selected_dev_bce},
            'baseline': {name: {'rows': value.rows, 'ones': value.ones}
                         for name, value in baseline.items()},
            'recipe_sha': recipe_sha, 'state': state}
        output = io.BytesIO()
        torch.save(node, output, pickle_protocol=2)
        content = output.getvalue()
        if len(content) > MAX_BYTES:
            raise ValueError
        return content
    except Exception:
        raise ValueError('invalid checkpoint') from None


@dataclass(frozen=True)
class _Storage:
    key: str
    dtype: torch.dtype
    count: int


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass(frozen=True)
class _StorageType:
    dtype: torch.dtype


class _Inspect(pickle._Unpickler):
    """Protocol-2 descriptors, with container checks BEFORE dict hashing.

    Use the Python dispatch deliberately: C Unpickler does not expose SETITEM
    hooks. This is not an external-checkpoint compatibility unpickler.
    """

    # Runtime attributes of the stdlib Python unpickler, absent from its stubs.
    stack: list[object]
    append: Callable[[object], None]
    pop_mark: Callable[[], list[object]]
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, content: bytes, sizes: Mapping[str, int]) -> None:
        super().__init__(io.BytesIO(content))
        self.sizes = sizes
        self.used: set[str] = set()
        # Integer identity keys NEVER hash a tuple. Keep the tuples alive so IDs
        # cannot be reused when the pickle overwrites a memo entry.
        self.tuple_costs: dict[int, tuple[tuple[object, ...], int]] = {}
        self.tuple_work = 0

    def _setitems(self, items: list[object]) -> None:
        if (type(self.stack[-1]) is not dict or len(items) % 2
                or any(type(items[i]) is not str for i in range(0, len(items), 2))):
            raise ValueError
        target = cast(dict[str, object], self.stack[-1])
        # Check the whole batch first, before even the first key is hashed.
        for i in range(0, len(items), 2):
            target[cast(str, items[i])] = items[i+1]

    def _load_setitem(self) -> None:
        if len(self.stack) < 3:
            raise ValueError
        items = self.stack[-2:]
        del self.stack[-2:]
        self._setitems(items)

    def _load_setitems(self) -> None:
        self._setitems(self.pop_mark())

    def _push_tuple(self, items: list[object]) -> None:
        # Cost is the number of nodes in the expanded tuple tree. Memo aliases
        # are charged at their full stored cost, without walking or hashing them.
        # Non-tuples are leaves (mutable containers are unhashable). Bound both
        # one tuple and cumulative construction/comparison work, not just depth.
        cost = 1
        for item in items:
            cost += self.tuple_costs[id(item)][1] if type(item) is tuple and item else 1
            if cost > _MAX_TUPLE_WORK:
                raise ValueError
        self.tuple_work += cost
        if self.tuple_work > _MAX_TUPLE_WORK:
            raise ValueError
        value = tuple(items)
        self.tuple_costs[id(value)] = (value, cost)
        self.append(value)

    def _load_tuple(self) -> None:
        self._push_tuple(self.pop_mark())

    def _load_empty_tuple(self) -> None:
        self._push_tuple([])

    def _small_tuple(self, arity: int) -> None:
        if len(self.stack) < arity:
            raise ValueError
        items = self.stack[-arity:]
        del self.stack[-arity:]
        self._push_tuple(items)

    def _load_tuple1(self) -> None:
        self._small_tuple(1)

    def _load_tuple2(self) -> None:
        self._small_tuple(2)

    def _load_tuple3(self) -> None:
        self._small_tuple(3)

    for _opcode, _handler in ((pickle.SETITEM, _load_setitem), (pickle.SETITEMS, _load_setitems),
                              (pickle.EMPTY_TUPLE, _load_empty_tuple), (pickle.TUPLE, _load_tuple),
                              (pickle.TUPLE1, _load_tuple1),
                              (pickle.TUPLE2, _load_tuple2), (pickle.TUPLE3, _load_tuple3)):
        dispatch[_opcode[0]] = cast(Callable[[pickle.Unpickler], None], _handler)
    del _opcode, _handler

    def find_class(self, module: str, name: str) -> object:
        if module == 'torch' and name in ('FloatStorage', 'LongStorage'):
            return _StorageType(torch.float32 if name == 'FloatStorage' else torch.int64)
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return self._tensor
        if (module, name) == ('collections', 'OrderedDict'):
            return self._empty_dict
        raise ValueError

    @staticmethod
    def _empty_dict() -> dict[str, object]:
        # Only zero-argument hooks are emitted by this format. Never expose dict's
        # copying constructor: memo references could expand a tiny pickle repeatedly.
        return {}

    def persistent_load(self, pid: object) -> _Storage:
        if type(pid) is not tuple or len(pid) != 5:
            raise ValueError
        kind, storage_type, key, location, count = pid
        if (kind != 'storage' or type(storage_type) is not _StorageType
                or type(key) is not str or not key.isascii() or not key.isdecimal()
                or location != 'cpu' or type(count) is not int or count <= 0):
            raise ValueError
        size = count * (4 if storage_type.dtype == torch.float32 else 8)
        if size > MAX_BYTES or self.sizes.get(key) != size or key in self.used:
            raise ValueError
        self.used.add(key)
        return _Storage(key, storage_type.dtype, count)

    @staticmethod
    def _tensor(storage: _Storage, offset: object, shape: object, stride: object,
                requires_grad: object, hooks: object) -> _Tensor:
        if (type(storage) is not _Storage or type(offset) is not int or offset != 0
                or type(shape) is not tuple or not 1 <= len(shape) <= 4
                or any(type(n) is not int or not 1 <= n <= MAX_BYTES for n in shape)
                or type(stride) is not tuple or len(stride) != len(shape)
                or requires_grad is not False or type(hooks) is not dict or hooks):
            raise ValueError
        expected_stride: list[int] = []
        elements = 1
        for dimension in reversed(shape):
            expected_stride.insert(0, elements)
            elements *= dimension
        if (elements != storage.count or not _same_public(stride, tuple(expected_stride))
                or any(type(n) is not int for n in stride)):
            raise ValueError
        return _Tensor(storage, shape, stride)


def _pickle_preflight(raw: bytes) -> None:
    # Protocol 2 only; bound memo indices before metadata inspection.
    # BUILD/object/new-style reducers, extension caches, sets and byte buffers
    # are unnecessary for this schema. Bytes/opcodes bound physical containers;
    # _Inspect separately enforces string dict keys and expanded tuple work.
    # Reducers only create empty dicts or fixed-size, shape-checked descriptors,
    # never copy memoized containers. Recursive/aliased metadata fails validation.
    allowed = {'PROTO', 'EMPTY_DICT', 'MARK', 'BINPUT', 'LONG_BINPUT', 'BINGET',
               'LONG_BINGET', 'BINUNICODE', 'BININT', 'BININT1', 'BININT2',
               'LONG1', 'BINFLOAT', 'EMPTY_LIST', 'APPEND', 'APPENDS', 'SETITEM',
               'SETITEMS', 'GLOBAL', 'TUPLE', 'TUPLE1', 'TUPLE2', 'TUPLE3',
               'EMPTY_TUPLE', 'REDUCE', 'NEWFALSE', 'NEWTRUE', 'BINPERSID', 'STOP'}
    count = 0
    stop = -1
    for opcode, arg, position in pickletools.genops(raw):
        count += 1
        if count > 100000 or opcode.name not in allowed:
            raise ValueError
        if opcode.name == 'PROTO' and (position != 0 or arg != 2):
            raise ValueError
        if opcode.name in ('BINPUT', 'LONG_BINPUT', 'BINGET', 'LONG_BINGET'):
            if type(arg) is not int or not 0 <= arg < 100000:
                raise ValueError
        if opcode.name == 'STOP':
            if position is None:
                raise ValueError
            stop = position
    if raw[:2] != b'\x80\x02' or stop != len(raw)-1:
        raise ValueError


def _preflight(content: bytes) -> tuple[PublicModelSpec, Selection, dict[str, TrainBaseline], str]:
    # torch.save ZIP64 local extras are supported; central directory itself must
    # remain ordinary bounded ZIP, with no comments/trailing or duplicate entries.
    if len(content) < 22:
        raise ValueError
    sig, disk, cd_disk, disk_count, count, size, offset, comment = struct.unpack(
        '<4s4H2IH', content[-22:])
    if (sig != b'PK\x05\x06' or disk or cd_disk or disk_count != count
            or not 1 <= count <= _MAX_ENTRIES or comment or offset + size > len(content)-22):
        raise ValueError
    # ZIP64 can override the ordinary EOCD fields in ZipFile. torch emits a
    # fixed 56-byte ZIP64 EOCD + 20-byte locator; require exact agreement before
    # permitting the library to choose its directory count/offset/size.
    tail = content[offset+size:-22]
    if tail:
        if len(tail) != 76:
            raise ValueError
        (zsig, record_size, _, _, zdisk, zcd_disk, zcount_disk, zcount,
         zsize, zoffset) = struct.unpack('<4sQ2H2I4Q', tail[:56])
        lsig, ldisk, loffset, disks = struct.unpack('<4sIQI', tail[56:])
        if (zsig != b'PK\x06\x06' or record_size != 44 or zdisk or zcd_disk
                or (zcount_disk, zcount, zsize, zoffset) != (count, count, size, offset)
                or lsig != b'PK\x06\x07' or ldisk or loffset != offset+size or disks != 1):
            raise ValueError
    # Check the directory count physically before ZipFile allocates ZipInfo objects.
    position = offset
    for _ in range(count):
        if content[position:position+4] != b'PK\x01\x02' or position+46 > offset+size:
            raise ValueError
        name_len, extra_len, note_len = struct.unpack_from('<3H', content, position+28)
        position += 46 + name_len + extra_len + note_len
    if position != offset + size:
        raise ValueError
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != count or len(set(names)) != count:
            raise ValueError
        frontier = 'archive/public-plan.json' in names
        # schema4 uses canonical flat ZIP, no ZIP64 gaps, extras or descriptors.
        if frontier and (tail or names[0].split('/')[0] != 'archive'):
            raise ValueError
        if frontier:
            # ZipInfo.filename is sanitized (notably truncated at NUL). Never
            # whitelist/register a normalized prefix of a different raw name.
            # With flags=0 below, ASCII orig_filename is the exact central bytes.
            for info in infos:
                if (info.orig_filename != info.filename
                        or not info.orig_filename.isascii()):
                    raise ValueError
        root = names[0].split('/')[0]
        if not re.fullmatch('[A-Za-z0-9_-]{1,80}', root):
            raise ValueError
        sizes: dict[str, int] = {}
        total = 0
        allowed = {'data.pkl', 'byteorder', 'version', '.data/serialization_id',
                   '.format_version', '.storage_alignment'}
        previous_end = 0
        for info in infos:
            header = info.header_offset
            if (header < previous_end or header + 30 > offset
                    or (frontier and header != previous_end)):
                raise ValueError
            fields = struct.unpack_from('<4s5H3I2H', content, header)
            signature, _, flags, compression, _, _, crc, packed, expanded, nl, el = fields
            data_start = header + 30 + nl + el
            data_end = data_start + info.file_size
            if (signature != b'PK\x03\x04' or flags != info.flag_bits
                    or flags & ~0x808 or compression != zipfile.ZIP_STORED
                    or data_end > offset
                    or content[header+30:header+30+nl].decode('ascii') != info.filename):
                raise ValueError
            if frontier and (flags or info.extra or info.comment or el
                    or content[header+30:header+30+nl] != info.orig_filename.encode('ascii')):
                raise ValueError
            if flags & 8:
                if (packed not in (0, info.file_size) or expanded not in (0, info.file_size)
                        or crc not in (0, info.CRC) or data_end+16 > offset
                        or struct.unpack_from('<4s3I', content, data_end)
                        != (b'PK\x07\x08', info.CRC, info.file_size, info.file_size)):
                    raise ValueError
                previous_end = data_end + 16
            else:
                if (packed, expanded, crc) != (info.file_size, info.file_size, info.CRC):
                    raise ValueError
                previous_end = data_end
            # Only torch's alignment padding is needed in this bounded schema.
            extra_position = header + 30 + nl
            while extra_position < data_start:
                if extra_position + 4 > data_start:
                    raise ValueError
                tag, length = struct.unpack_from('<2H', content, extra_position)
                extra_position += 4 + length
                if tag != 0x4246 or extra_position > data_start:
                    raise ValueError
            if (not info.filename.startswith(root+'/') or info.flag_bits & 1
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.file_size != info.compress_size or info.file_size > MAX_BYTES):
                raise ValueError
            relative = info.filename[len(root)+1:]
            if relative.startswith('data/') and re.fullmatch('data/[0-9]+', relative):
                sizes[relative[5:]] = info.file_size
            elif frontier and relative == 'public-plan.json':
                from cipher_lab_plugins.frontier.simon_frontier import MAX_PLAN_BYTES
                if not 0 < info.file_size <= MAX_PLAN_BYTES:
                    raise ValueError
            elif relative not in allowed or info.file_size > (
                    _MAX_PICKLE if relative == 'data.pkl' else 128):
                raise ValueError
            total += info.file_size
            if total > MAX_BYTES:
                raise ValueError
        plan = None
        if frontier:
            if previous_end != offset:
                raise ValueError
            # Validate every physical stored byte/CRC before plan or pickle parsing.
            for info in infos:
                header = info.header_offset
                nl, el = struct.unpack_from('<2H', content, header+26)
                start = header + 30 + nl + el
                if zlib.crc32(memoryview(content)[start:start+info.file_size]) != info.CRC:
                    raise ValueError
            from cipher_lab_plugins.frontier.simon_frontier import decode_plan
            plan = decode_plan(archive.read('archive/public-plan.json'))
        if root+'/data.pkl' not in names or archive.read(root+'/byteorder') != b'little':
            raise ValueError
        raw = archive.read(root+'/data.pkl')
        _pickle_preflight(raw)
        inspector = _Inspect(raw, sizes)
        node = inspector.load()
        spec, selection, baseline, recipe_sha = _metadata(node, plan)
        if inspector.used != set(sizes):
            raise ValueError
        expected = _new(spec, meta=True).state_dict()
        if sum(t.numel()*t.element_size() for t in expected.values()) > MAX_BYTES:
            raise ValueError
        values = _object(node['state'], set(expected))
        for name, template in expected.items():
            value = values[name]
            if (type(value) is not _Tensor or value.shape != tuple(template.shape)
                    or value.storage.dtype != template.dtype):
                raise ValueError
        return spec, selection, baseline, recipe_sha


def load_checkpoint(content: bytes, expected_sha: str) -> LoadedModel:
    """SHA-before-load, bounded declarations, strict CPU weights_only loading."""
    try:
        if (type(content) is not bytes or len(content) > MAX_BYTES
                or hashlib.sha256(content).hexdigest() != _sha(expected_sha)):
            raise ValueError
        spec, selection, baseline, recipe_sha = _preflight(content)
        node = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
        plan = None
        if spec.method == 'bit-public-feature-frontier-v1':
            from cipher_lab_plugins.frontier.simon_frontier import decode_plan
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                plan = decode_plan(archive.read('archive/public-plan.json'))
        loaded_metadata = _metadata(node, plan)
        if loaded_metadata != (spec, selection, baseline, recipe_sha):
            raise ValueError
        model = _new(spec)
        _validate_state(node['state'], model.state_dict(), model)
        model.load_state_dict(node['state'], strict=True)
        model.eval()
        return LoadedModel(model, spec, spec.task_declarations, selection, baseline, recipe_sha)
    except Exception:
        raise ValueError('invalid checkpoint') from None
