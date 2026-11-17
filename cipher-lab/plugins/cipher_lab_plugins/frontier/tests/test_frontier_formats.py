from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import Any, cast

import pytest
import torch

from cipher_lab_plugins.frontier.tests.frontier_fakes import plan, poison

__all__ = ['poison']


def task() -> Any:
    from cipher_lab_plugins.frontier.tasks import FrontierTask
    p = plan()
    return FrontierTask('SIMON', 4, 0, 'cipher', 'simon64-zero-master-v1', p.identity)


def bundle() -> Any:
    from cipher_lab_plugins.prediction_common.bundles import DatasetBundle
    return DatasetBundle('bit-public-feature-frontier-v1', (task(),),
        {'SIMON': (torch.zeros(2, 64), torch.tensor([[0.], [1.]]))},
        {'SIMON': torch.tensor([[1.], [0.]])})


def spec(arm: str, rounds: int = 4) -> Any:
    from cipher_lab.training_config import MLPSpec
    from cipher_lab_plugins.frontier.config import FrontierMLPSpec
    from cipher_lab_plugins.frontier.tasks import FrontierTask
    from cipher_lab_plugins.prediction_common.metrics import PublicModelSpec
    p = plan(rounds)
    t = FrontierTask('SIMON', rounds, 0, 'cipher', 'simon64-zero-master-v1', p.identity)
    model = FrontierMLPSpec(2, p.identity, p) if arm == 'C' else MLPSpec(2)
    return PublicModelSpec(model, (t,), 'direct' if arm == 'A' else 'residual', arm,
                           p.representation if arm == 'C' else 'raw64-msb0-v1')


def content(arm: str, rounds: int = 4) -> bytes:
    from cipher_lab_plugins.frontier.checkpoint import encode_frontier_checkpoint
    from cipher_lab_plugins.prediction_common.checkpoints import Selection
    from cipher_lab_plugins.prediction_common.metrics import TrainBaseline
    from cipher_lab_plugins.prediction_common.steps import initialized_model
    identity = spec(arm, rounds)
    return encode_frontier_checkpoint(initialized_model(identity, 31), identity,
        identity.task_declarations, Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)},
        'a'*64, plan=plan(rounds))


def parts(raw: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def archive(items: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, value in items.items():
            z.writestr(name, value)
    return out.getvalue()


@pytest.mark.parametrize('rounds', [4, 5])
@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_self_contained_checkpoint(arm: str, rounds: int) -> None:
    from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint
    raw = content(arm, rounds)
    members = parts(raw)
    from cipher_lab_plugins.frontier.simon_frontier import encode_plan
    assert members['archive/public-plan.json'] == encode_plan(plan(rounds))
    assert len(members['archive/data.pkl']) < 1024*1024
    loaded = load_checkpoint(raw, hashlib.sha256(raw).hexdigest())
    assert loaded.public_model_spec == spec(arm, rounds)
    layers = cast(torch.nn.Sequential, loaded.model.layers)
    assert layers[0].weight.shape == (2, plan(rounds).dimension if arm == 'C' else 64)


def test_compact_bundle_roundtrip() -> None:
    from cipher_lab_plugins.prediction_common.bundles import decode_bundle, encode_bundle
    raw = encode_bundle(bundle())
    loaded = decode_bundle(raw)
    assert loaded.task_declarations == (task(),)
    assert loaded.panels['SIMON'][0].shape == (2, 64)
    assert torch.equal(loaded.references['SIMON'], bundle().references['SIMON'])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.getinfo('metadata.npy').file_size < 8192


@pytest.mark.parametrize('damage', ['tail', 'plan', 'mask', 'identity', 'duplicate',
    'hidden', 'path', 'crc', 'nested', 'missing', 'compressed'])
def test_preflight_before_model_or_torch_load(damage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.prediction_common import checkpoints as checkpoints
    raw = content('C')
    items = parts(raw)
    if damage == 'tail':
        raw += b'tail'
    elif damage == 'crc':
        raw = raw.replace(b'public-plan.json', b'public-plan.json', 1)
        pos = raw.index(items['archive/public-plan.json'])
        raw = raw[:pos] + b'!' + raw[pos+1:]
    elif damage == 'compressed':
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name, value in items.items():
                z.writestr(name, value)
        raw = out.getvalue()
    else:
        if damage in ('plan', 'mask', 'identity'):
            value = json.loads(items['archive/public-plan.json'])
            if damage == 'plan':
                value['features'][0] = [999999]
            elif damage == 'mask':
                value['leaves'][0] = [2**64]
            else:
                value['identity']['dimension'] = True
            items['archive/public-plan.json'] = json.dumps(value).encode()
        elif damage == 'hidden':
            items['archive/.hidden'] = b'x'
        elif damage == 'path':
            items['../public-plan.json'] = items.pop('archive/public-plan.json')
        elif damage == 'nested':
            items['archive/payload.zip'] = raw
        elif damage == 'missing':
            del items['archive/public-plan.json']
        raw = archive(items)
        if damage == 'duplicate':
            out = io.BytesIO(raw)
            with zipfile.ZipFile(out, 'a') as z:
                z.writestr('archive/public-plan.json', items['archive/public-plan.json'])
            raw = out.getvalue()
    calls: list[str] = []
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append('called')
        raise AssertionError('unsafe model/load before preflight')
    monkeypatch.setattr(checkpoints, '_new', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(raw, hashlib.sha256(raw).hexdigest())
    assert calls == []


@pytest.mark.parametrize('field', ['schema', 'wiring', 'identity', 'recipe', 'shape'])
def test_metadata_rejects_before_load(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint
    raw = content('C')
    node = torch.load(io.BytesIO(raw), weights_only=True)
    if field == 'schema':
        node['schema'] = 3
    elif field == 'wiring':
        node['wiring']['network_input_width'] = 64
    elif field == 'identity':
        node['task_declarations'][0]['identity']['dimension'] = True
    elif field == 'recipe':
        node['recipe_sha'] = 'A'*64
    else:
        node['state']['layers.0.weight'] = torch.zeros(2, 64)
    out = io.BytesIO()
    torch.save(node, out, pickle_protocol=2)
    items = parts(out.getvalue())
    items['archive/public-plan.json'] = parts(raw)['archive/public-plan.json']
    corrupt = archive(items)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('unsafe load')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError):
        load_checkpoint(corrupt, hashlib.sha256(corrupt).hexdigest())


def test_mutated_spec_and_plan_are_revalidated() -> None:
    from cipher_lab_plugins.frontier.config import FrontierMLPSpec
    from cipher_lab_plugins.prediction_common.metrics import PublicModelSpec
    p = plan()
    object.__setattr__(p, 'features', ((9999,),))
    with pytest.raises(ValueError):
        FrontierMLPSpec(2, plan().identity, p)
    s = spec('C')
    object.__setattr__(s.model.identity, 'dimension', True)
    with pytest.raises(ValueError):
        PublicModelSpec(s.model, s.task_declarations, s.output_mode, s.arm_id,
                        s.input_representation)


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_explicit_evaluate_artificial_absolute_logits(arm: str, tmp_path: Any) -> None:
    from test_training_steps import make_meter

    from cipher_lab.artifacts import RunStore
    from cipher_lab_plugins.prediction_common.bundles import encode_bundle
    from cipher_lab_plugins.prediction_common.evaluation import run_evaluation
    checkpoint, dataset = content(arm), encode_bundle(bundle())
    cp, dp = tmp_path/'m.pt', tmp_path/'d.npz'
    cp.write_bytes(checkpoint)
    dp.write_bytes(dataset)
    store = RunStore.claim(tmp_path, 'eval', output_bytes=10000000)
    meter = make_meter(rows=2)
    result = run_evaluation(cp, hashlib.sha256(checkpoint).hexdigest(), dp,
                            hashlib.sha256(dataset).hexdigest(), meter, store)
    assert result['status'] == 'SUCCEEDED'
    assert cast(dict[str, Any], result['summary'])['arm_id'] == arm
    assert meter.snapshot()['completed_forward_rows'] == 2  # fake ledger only


@pytest.mark.parametrize('damage', ['overlap', 'gap', 'pickle-limit', 'plan-limit', 'local-size'])
def test_physical_limits_before_adapter(damage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import struct

    from cipher_lab_plugins.prediction_common import checkpoints as checkpoints
    raw = content('C')
    items = parts(raw)
    if damage == 'pickle-limit':
        items['archive/data.pkl'] = b'x'*(1024*1024+1)
        raw = archive(items)
    elif damage == 'plan-limit':
        items['archive/public-plan.json'] = b'x'*(16*1024*1024+1)
        raw = archive(items)
    else:
        changed = bytearray(raw)
        offset = struct.unpack_from('<I', raw, len(raw)-6)[0]
        if damage == 'overlap':
            # second central entry points into the first local record
            n, e, c = struct.unpack_from('<3H', raw, offset+28)
            second = offset + 46+n+e+c
            struct.pack_into('<I', changed, second+42, 0)
        elif damage == 'gap':
            struct.pack_into('<I', changed, offset+42, 1)
        else:
            struct.pack_into('<I', changed, 18, 1)
        raw = bytes(changed)
    calls: list[str] = []
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append('unsafe')
        raise AssertionError('preflight was too late')
    monkeypatch.setattr(checkpoints, '_new', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(raw, hashlib.sha256(raw).hexdigest())
    assert not calls


@pytest.mark.parametrize('change', ['dimension', 'sha', 'bit', 'round', 'extra', 'schema'])
def test_frontier_bundle_identity_before_allocation(change: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from test_bundles import archive as npz_archive
    from test_bundles import entries, npy

    from cipher_lab_plugins.prediction_common.bundles import decode_bundle, encode_bundle
    members = entries(encode_bundle(bundle()))
    metadata = json.loads(np.load(io.BytesIO(members['metadata.npy'])).tobytes())
    target = metadata['task_declarations'][0]
    if change == 'dimension':
        target['identity']['dimension'] = True
    elif change == 'sha':
        target['identity']['plan_sha256'] = 'A'*64
    elif change == 'bit':
        target['target_bit'] = 1
    elif change == 'round':
        target['rounds'] = 5
    elif change == 'extra':
        target['key'] = 0
    else:
        metadata['schema'] = 3
    members['metadata.npy'] = npy(np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    allocations: list[int] = []
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        allocations.append(1)
        raise AssertionError('invalid identity allocated tensor')
    monkeypatch.setattr(torch, 'from_numpy', forbidden)
    with pytest.raises(ValueError):
        decode_bundle(npz_archive(members))
    assert not allocations


def test_legacy_encoder_does_not_emit_undecodable_schema1_for_frontier() -> None:
    from cipher_lab_plugins.prediction_common.checkpoints import Selection, encode_checkpoint
    from cipher_lab_plugins.prediction_common.metrics import TrainBaseline
    from cipher_lab_plugins.prediction_common.steps import initialized_model
    identity = spec('A')
    with pytest.raises(ValueError):
        encode_checkpoint(initialized_model(identity, 31), identity, identity.task_declarations,
            Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)}, 'a'*64)


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_storage_central_nul_name_rejected_before_adapter_or_load(arm: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Change only central storage name/name length and directory size, not local bytes."""
    import struct

    from cipher_lab_plugins.prediction_common import checkpoints as checkpoints
    raw = content(arm)
    eocd = len(raw)-22
    directory_size, directory_offset = struct.unpack_from('<2I', raw, eocd+12)
    position = directory_offset
    while True:
        assert raw[position:position+4] == b'PK\x01\x02'
        name_len, extra_len, note_len = struct.unpack_from('<3H', raw, position+28)
        name = raw[position+46:position+46+name_len]
        if name == b'archive/data/0':
            break
        position += 46+name_len+extra_len+note_len
        assert position < eocd
    suffix = b'\x00suffix'
    insertion = position+46+name_len
    changed = bytearray(raw[:insertion]+suffix+raw[insertion:])
    struct.pack_into('<H', changed, position+28, name_len+len(suffix))
    struct.pack_into('<I', changed, eocd+len(suffix)+12, directory_size+len(suffix))
    corrupt = bytes(changed)
    # Confirm the exact stdlib ambiguity and unchanged entire local payload area.
    assert corrupt[:directory_offset] == raw[:directory_offset]
    with zipfile.ZipFile(io.BytesIO(corrupt)) as z:
        info = z.getinfo('archive/data/0')
        assert info.filename == 'archive/data/0'
        assert info.orig_filename == 'archive/data/0\x00suffix'
    calls: list[str] = []
    def new(*args: Any, **kwargs: Any) -> Any:
        calls.append('_new')
        raise AssertionError('adapter construction must not precede ZIP name rejection')
    def load(*args: Any, **kwargs: Any) -> Any:
        calls.append('torch.load')
        raise AssertionError('weights_only loading must not precede ZIP name rejection')
    monkeypatch.setattr(checkpoints, '_new', new)
    monkeypatch.setattr(torch, 'load', load)
    with pytest.raises(ValueError, match='^invalid checkpoint$'):
        checkpoints.load_checkpoint(corrupt, hashlib.sha256(corrupt).hexdigest())
    assert calls == []
