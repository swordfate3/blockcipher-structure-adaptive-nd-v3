"""Strict public checkpoints, artificial CPU tensors, no optimizer or collection."""
from __future__ import annotations

import hashlib
import io
import pickle
import struct
import zipfile
from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import Tensor

from cipher_lab.data.bundles import PredictionTask
from cipher_lab.evaluation import (
    LoadedModel,
    PublicModelSpec,
    Selection,
    TrainBaseline,
    encode_checkpoint,
    load_checkpoint,
)
from cipher_lab.models import BitMLP, SharedM
from cipher_lab.training_config import TASK_FAMILIES, MLPSpec, SharedMSpec, TaskSpec

torch.set_num_threads(1)
TASK = PredictionTask('SIMON', 3, 7, 'cipher')
SPEC = MLPSpec(3)
RECIPE_SHA = 'a'*64


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def encoded() -> bytes:
    return encode_checkpoint(BitMLP(SPEC), PublicModelSpec(SPEC, (TASK,)), (TASK,),
                             Selection(1, 2, .4), {'SIMON': TrainBaseline(4, 3)}, RECIPE_SHA)


def unpack(content: bytes) -> dict[str, object]:
    return cast(dict[str, object], torch.load(io.BytesIO(content), weights_only=True))


def pack(node: dict[str, object]) -> bytes:
    stream = io.BytesIO()
    torch.save(node, stream)
    return stream.getvalue()


def test_prediction_roundtrip_exact_interface_cpu_rng_and_logits() -> None:
    model = BitMLP(SPEC)
    model.eval()
    before_state = {k: v.clone() for k, v in model.state_dict().items()}
    content = encode_checkpoint(model, PublicModelSpec(SPEC, (TASK,)), (TASK,),
                                Selection(1, 2, .4), {'SIMON': TrainBaseline(4, 3)}, RECIPE_SHA)
    rng = torch.get_rng_state().clone()
    loaded = load_checkpoint(content, sha(content))
    assert isinstance(loaded, LoadedModel)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded.public_model_spec == PublicModelSpec(SPEC, (TASK,))
    assert loaded.task_declarations == (TASK,)
    assert loaded.selection == Selection(1, 2, .4)
    assert loaded.baseline == {'SIMON': TrainBaseline(4, 3)}
    assert loaded.recipe_sha == RECIPE_SHA
    assert not loaded.model.training and not model.training
    x = torch.zeros(2, 64)
    assert torch.equal(model(x), loaded.model(x))
    for key, value in model.state_dict().items():
        assert value.device.type == 'cpu' and torch.equal(value, before_state[key])


def test_shared_m_roundtrip_and_wiring_identity() -> None:
    spec = SharedMSpec(2, 2, 1, (1,), 2, 2)
    tasks = tuple(TaskSpec(f.cipher, 1, 1) for f in TASK_FAMILIES)
    model = SharedM(spec).eval()
    content = encode_checkpoint(model, PublicModelSpec(spec, tasks), tasks,
                                Selection(1, 4, .7),
                                {t.cipher: TrainBaseline(2, 1) for t in tasks}, RECIPE_SHA)
    loaded = load_checkpoint(content, sha(content))
    from cipher_lab.representations.tasks import public_features
    x = torch.zeros(2, 4, 2, 64)
    public = torch.tensor(public_features('SKINNY'))
    assert torch.equal(model(x, public, rounds=1), loaded.model(x, public, rounds=1))
    node = unpack(content)
    state = cast(dict[str, Tensor], node['state'])
    key = next(k for k in state if k.endswith('indices'))
    state[key] = state[key].roll(1, 0)
    bad = pack(node)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(bad, sha(bad))


@pytest.mark.parametrize('mutate', [
    lambda n: n.update(schema=0),
    lambda n: n.update(schema=True),
    lambda n: n.update(keyseed='SECRET-CANARY'),
    lambda n: n.update(optimizer={}),
    lambda n: n.update(private_rng='SECRET-CANARY'),
    lambda n: n.update(model_family='old-mlp'),
    lambda n: cast(dict[str, object], n['model_spec']).update(hidden_width=4),
    lambda n: cast(dict[str, object], n['model_spec']).update(seed=7),
    lambda n: cast(dict[str, object], n['wiring']).update(input_bits=63),
    lambda n: cast(dict[str, object], n['selection']).update(epoch=0),
    lambda n: cast(dict[str, object], n['selection']).update(updates=True),
    lambda n: cast(dict[str, object], n['selection']).update(selected_dev_bce=float('nan')),
    lambda n: cast(dict[str, object], n['baseline']).update(DEV={'rows': 4, 'ones': 2}),
    lambda n: n.update(recipe_sha='not-a-sha'),
    lambda n: n.update(task_declarations=[]),
    lambda n: cast(list[dict[str, object]], n['task_declarations'])[0].update(target_bit=64),
    lambda n: cast(list[dict[str, object]], n['task_declarations'])[0].update(rounds=45),
    lambda n: cast(list[dict[str, object]], n['task_declarations'])[0].update(label_mode='other'),
    lambda n: cast(list[dict[str, object]], n['task_declarations'])[0].update(difference=1),
])
def test_metadata_whitelist(mutate: Callable[[dict[str, object]], None]) -> None:
    node = unpack(encoded())
    mutate(node)
    content = pack(node)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


@pytest.mark.parametrize('kind', ['extra', 'missing', 'shape', 'dtype', 'sparse', 'nan', 'layout'])
def test_state_strictness(kind: str) -> None:
    node = unpack(encoded())
    state = cast(dict[str, Tensor], node['state'])
    key = 'layers.0.weight'
    if kind == 'extra':
        state['secret'] = torch.zeros(1)
    elif kind == 'missing':
        del state[key]
    elif kind == 'shape':
        state[key] = torch.zeros(4, 64)
    elif kind == 'dtype':
        state[key] = state[key].double()
    elif kind == 'sparse':
        state[key] = state[key].to_sparse()
    elif kind == 'nan':
        state[key][0, 0] = float('nan')
    else:
        state[key] = state[key].t().contiguous().t()
    content = pack(node)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


def test_sha_checked_before_torch_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('must not call torch.load')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(encoded(), '0'*64)


def test_fixed_error_no_pickle_details() -> None:
    content = b'private pickle error SECRET-CANARY'
    with pytest.raises(ValueError) as caught:
        load_checkpoint(content, sha(content))
    assert str(caught.value) == 'invalid checkpoint'
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def rewrite_pickle(content: bytes, transform: Callable[[bytes], bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as source, zipfile.ZipFile(stream, 'w') as dest:
        for info in source.infolist():
            raw = source.read(info)
            dest.writestr(info.filename,
                          transform(raw) if info.filename.endswith('data.pkl') else raw)
    return stream.getvalue()


@pytest.mark.parametrize('operation', [b's', b'u'])
@pytest.mark.parametrize('tuple_opcode', [b'\x86', b't'])
def test_memo_tuple_key_rejected_before_schema_and_torchload(
        monkeypatch: pytest.MonkeyPatch, operation: bytes, tuple_opcode: bytes) -> None:
    import cipher_lab.evaluation.checkpoints as checkpoints
    # Tiny depth only: t0=(0,), then t1=(t0,t0), t2=(t1,t1).
    # A MARK belongs to TUPLE but not TUPLE2. Memoize via a surrounding list
    # because POP is deliberately outside this checkpoint's opcode whitelist.
    raw = b'\x80\x02]K\x00\x85q\x00a'
    for index in (1, 2):
        mark = b'(' if tuple_opcode == b't' else b''
        raw += (mark + b'h' + bytes([index-1]) + b'h' + bytes([index-1])
                + tuple_opcode + b'q' + bytes([index]) + b'a')
    raw += b'}' + (b'(' if operation == b'u' else b'') + b'h\x02K\x00' + operation + b'.'
    content = rewrite_pickle(encoded(), lambda _: raw)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('tuple key reached schema or torch.load instead of pre-hash rejection')

    monkeypatch.setattr(checkpoints, '_metadata', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


@pytest.mark.parametrize('opcode,arity', [(b't', 2), (b'\x85', 1), (b'\x86', 2),
                                        (b'\x87', 3)])
def test_tuple_semantic_budget_counts_memo_references_before_schema(
        monkeypatch: pytest.MonkeyPatch, opcode: bytes, arity: int) -> None:
    import cipher_lab.evaluation.checkpoints as checkpoints
    # Lower only the semantic budget for a tiny three-level fixture; never
    # construct the high-depth CPU attack. Both nested and cumulative work count.
    monkeypatch.setattr(checkpoints, '_MAX_TUPLE_WORK', 12, raising=False)
    raw = b'\x80\x02]K\x00\x85q\x00a'
    for index in (1, 2, 3):
        raw += ((b'(' if opcode == b't' else b'')
                + (b'h' + bytes([index-1])) * arity
                + opcode + b'q' + bytes([index]) + b'a')
    content = rewrite_pickle(encoded(), lambda _: raw + b'.')

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('memoized tuple expansion escaped its semantic budget')

    monkeypatch.setattr(checkpoints, '_metadata', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


def test_empty_tuple_construction_also_uses_semantic_budget(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab.evaluation.checkpoints as checkpoints
    monkeypatch.setattr(checkpoints, '_MAX_TUPLE_WORK', 2)
    content = rewrite_pickle(encoded(), lambda _: b'\x80\x02])a)a)a.')

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('EMPTY_TUPLE bypassed cumulative tuple construction budget')

    monkeypatch.setattr(checkpoints, '_metadata', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


@pytest.mark.parametrize('base_key', [b'K\x01', b'X\x01\x00\x00\x00x'])
@pytest.mark.parametrize('argument', [b'h\x01', b'}', b']'])
def test_ordered_dict_reducer_rejects_first_argument_before_expansion(
        monkeypatch: pytest.MonkeyPatch, argument: bytes, base_key: bytes) -> None:
    import cipher_lab.evaluation.checkpoints as checkpoints
    # Keep the original {1: 1} payload, rejected before even the reducer lookup;
    # {'x': 1} reaches the reducer so its first-argument guard remains covered.
    # Memo 1 is the tiny base dict. The mapping case reuses it in both
    # reducers: accepting mapping construction would copy it repeatedly.
    # Only two copies are encoded, and the guard fails on the first acceptance.
    raw = (b'\x80\x02]}q\x01' + base_key + b'K\x01saccollections\nOrderedDict\nq\x00'
           + argument + b'\x85Rah\x00' + argument + b'\x85Ra.')
    content = rewrite_pickle(encoded(), lambda _: raw)
    original = checkpoints._Inspect.find_class
    attempts: list[int] = []

    def observed(self: checkpoints._Inspect, module: str, name: str) -> object:
        reducer = cast(Callable[..., object], original(self, module, name))
        def guarded(*args: object) -> object:
            attempts.append(len(args))
            reducer(*args)
            pytest.fail('first nonzero-argument OrderedDict reducer was accepted')
        return guarded

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('nonzero-argument reducer must fail before schema or torch.load')

    monkeypatch.setattr(checkpoints._Inspect, 'find_class', observed)
    monkeypatch.setattr(checkpoints, '_metadata', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))
    assert attempts == ([] if base_key == b'K\x01' else [1])


def test_declared_giant_storage_preflight_before_torchload(monkeypatch: pytest.MonkeyPatch) -> None:
    # FloatStorage of layers.0.weight normally has 192 elements (BININT1 192).
    # Replace its persistent size by 2**31-1 without allocating that storage.
    def giant(raw: bytes) -> bytes:
        needle = b'X\x03\x00\x00\x00cpu'
        start = raw.index(needle) + len(needle)
        position = raw.index(b'K\xc0', start)
        return raw[:position] + b'J' + struct.pack('<i', 2**31-1) + raw[position+2:]
    content = rewrite_pickle(encoded(), giant)
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('unsafe storage must be rejected before torch.load')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


def test_zip_entry_limits_before_torchload(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for index in range(2050):
            archive.writestr(f'archive/data/{index}', b'')
    content = stream.getvalue()
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('oversized container must not load')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError):
        load_checkpoint(content, sha(content))


def test_encode_rejects_identity_mismatch() -> None:
    with pytest.raises(ValueError):
        encode_checkpoint(BitMLP(SPEC), PublicModelSpec(SPEC, (TASK,)),
                          (PredictionTask('SIMON', 4, 7, 'cipher'),), Selection(1, 1, .5),
                          {'SIMON': TrainBaseline(2, 1)}, RECIPE_SHA)


@pytest.mark.parametrize('epoch,updates,bce', [(True, 1, .5), (0, 1, .5), (1, 0, .5),
                                            (1, 1, -.1), (1, 1, float('inf'))])
def test_invalid_selection(epoch: int, updates: int, bce: float) -> None:
    with pytest.raises(ValueError):
        Selection(epoch, updates, bce)


def test_nonzip_and_executable_pickle_rejected_without_side_effect() -> None:
    content = pickle.dumps({'not': 'checkpoint'})
    with pytest.raises(ValueError):
        load_checkpoint(content, sha(content))


def test_raw_file_limit_precedes_torch_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab.evaluation.checkpoints as checkpoints
    content = encoded()
    monkeypatch.setattr(checkpoints, 'MAX_BYTES', len(content)-1)
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('oversized file must not load')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


def test_giant_pickle_memo_precedes_unpickling(monkeypatch: pytest.MonkeyPatch) -> None:
    import cipher_lab.evaluation.checkpoints as checkpoints
    content = rewrite_pickle(encoded(), lambda raw: raw[:3] + b'r' + struct.pack('<I', 2**31-1)
                             + raw[5:])  # initial EMPTY_DICT BINPUT 0 -> LONG_BINPUT giant
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('giant memo must be rejected before unpickling')
    monkeypatch.setattr(checkpoints._Inspect, 'load', forbidden)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        load_checkpoint(content, sha(content))


@pytest.mark.parametrize('kind', ['order', 'round', 'difference', 'unbalanced', 'family'])
def test_shared_task_and_baseline_strictness(kind: str) -> None:
    spec = SharedMSpec(2, 2, 1, (1,), 2, 2)
    tasks = tuple(TaskSpec(f.cipher, 1, 1) for f in TASK_FAMILIES)
    content = encode_checkpoint(SharedM(spec), PublicModelSpec(spec, tasks), tasks,
                                Selection(1, 4, .7),
                                {t.cipher: TrainBaseline(2, 1) for t in tasks}, RECIPE_SHA)
    node = unpack(content)
    declarations = cast(list[dict[str, object]], node['task_declarations'])
    if kind == 'order':
        declarations.reverse()
    elif kind == 'round':
        declarations[3]['rounds'] = 33
    elif kind == 'difference':
        declarations[0]['difference'] = 0
    elif kind == 'unbalanced':
        cast(dict[str, object], node['baseline'])['SIMON'] = {'rows': 2, 'ones': 0}
    else:
        node['model_family'] = 'bit-mlp-v1'
    bad = pack(node)
    with pytest.raises(ValueError):
        load_checkpoint(bad, sha(bad))


def test_local_container_size_disagreement_precedes_torchload(
        monkeypatch: pytest.MonkeyPatch) -> None:
    raw = bytearray(encoded())
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = next(i for i in archive.infolist() if i.filename.endswith('data/0'))
    struct.pack_into('<I', raw, info.header_offset+22, 0x7fffffff)
    content = bytes(raw)
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('local size disagreement must not reach torch.load')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError):
        load_checkpoint(content, sha(content))


def test_import_and_load_do_not_import_collection_modules() -> None:
    import os
    import subprocess
    import sys
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith('cipher_lab.ciphers') or
            fullname.startswith('cipher_lab.training.') or
            fullname in ('cipher_lab.data.prediction', 'cipher_lab.data.differential',
                         'cipher_lab.science')):
            raise AssertionError('collection import forbidden')
sys.meta_path.insert(0, Block())
from cipher_lab.evaluation import *
from cipher_lab.models import BitMLP
from cipher_lab.training_config import MLPSpec
from cipher_lab.data.bundles import PredictionTask
import hashlib
s = MLPSpec(1)
t = (PredictionTask('SIMON', 1, 0, 'null'),)
b = encode_checkpoint(BitMLP(s), PublicModelSpec(s, t), t, Selection(1, 1, .7),
                      {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
load_checkpoint(b, hashlib.sha256(b).hexdigest())
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            timeout=30, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr


def test_zip64_footer_cannot_override_prechecked_directory_limits(
        monkeypatch: pytest.MonkeyPatch) -> None:
    raw = bytearray(encoded())
    marker = raw.rfind(b'PK\x06\x06')
    assert marker >= 0  # torch.save writes its ZIP64 footer even for small files
    struct.pack_into('<Q', raw, marker+24, 10000000)  # disk entries
    struct.pack_into('<Q', raw, marker+32, 10000000)  # total entries
    content = bytes(raw)
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('ZIP64 disagreement must be checked before ZipFile')
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    with pytest.raises(ValueError):
        load_checkpoint(content, sha(content))
