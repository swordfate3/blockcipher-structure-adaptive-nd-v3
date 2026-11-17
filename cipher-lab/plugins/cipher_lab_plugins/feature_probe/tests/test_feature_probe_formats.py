from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from cipher_lab_plugins.feature_probe.tests.feature_probe_fakes import poison

__all__ = ['poison']


def task() -> Any:
    from cipher_lab_plugins.feature_probe.tasks import FeatureProbeTask
    return FeatureProbeTask('SIMON', 3, 0, 'cipher', 'simon64-zero-master-v1',
                            'simon-r3-residual13-v1')


def bundle() -> Any:
    from cipher_lab_plugins.prediction_common.bundles import DatasetBundle
    return DatasetBundle('bit-public-feature-probe-v1', (task(),),
        {'SIMON': (torch.zeros(2, 64), torch.tensor([[0.], [1.]]))},
        {'SIMON': torch.tensor([[1.], [0.]])})


def spec(arm: str) -> Any:
    from cipher_lab.training_config import MLPSpec
    from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec
    from cipher_lab_plugins.prediction_common.metrics import PublicModelSpec
    model = FeatureMLPSpec(2, 'simon-r3-residual13-v1') if arm == 'C' else MLPSpec(2)
    return PublicModelSpec(model, (task(),), 'direct' if arm == 'A' else 'residual', arm,
                           'simon-r3-residual13-v1' if arm == 'C' else 'raw64-msb0-v1')


def content(arm: str) -> bytes:
    from cipher_lab_plugins.prediction_common.checkpoints import Selection, encode_checkpoint
    from cipher_lab_plugins.prediction_common.metrics import TrainBaseline
    from cipher_lab_plugins.prediction_common.steps import initialized_model
    identity = spec(arm)
    return encode_checkpoint(initialized_model(identity, 31), identity, (task(),),
                             Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)}, 'a'*64)


def test_bundle_roundtrip_raw64() -> None:
    from cipher_lab_plugins.prediction_common.bundles import decode_bundle, encode_bundle
    encoded = encode_bundle(bundle())
    loaded = decode_bundle(encoded)
    assert loaded.method == 'bit-public-feature-probe-v1'
    assert loaded.task_declarations == (task(),)
    assert loaded.panels['SIMON'][0].shape == (2, 64)
    assert torch.equal(loaded.references['SIMON'], bundle().references['SIMON'])
    with pytest.raises(ValueError):
        encode_bundle(replace(bundle(), panels={'SIMON': (torch.zeros(2, 13),
                                                          torch.zeros(2, 1))}))


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_checkpoint_roundtrip(arm: str) -> None:
    from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint
    encoded = content(arm)
    node = torch.load(io.BytesIO(encoded), weights_only=True)
    assert node['schema'] == 3
    loaded = load_checkpoint(encoded, hashlib.sha256(encoded).hexdigest())
    assert loaded.public_model_spec == spec(arm)
    layers = cast(torch.nn.Sequential, loaded.model.layers)
    assert layers[0].weight.shape == (2, 13 if arm == 'C' else 64)


@pytest.mark.parametrize('field,value', [('arm_id', 'B'), ('output_mode', 'direct'),
    ('representation', 'wrong'), ('network_input_width', 64), ('feature_order', ['1'])])
def test_checkpoint_bad_wiring(field: str, value: object) -> None:
    from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint
    node = torch.load(io.BytesIO(content('C')), weights_only=True)
    node['wiring'][field] = value
    buffer = io.BytesIO()
    torch.save(node, buffer, pickle_protocol=2)
    encoded = buffer.getvalue()
    with pytest.raises(ValueError):
        load_checkpoint(encoded, hashlib.sha256(encoded).hexdigest())


@pytest.mark.parametrize('change', ['bit', 'shape', 'sha', 'schema'])
def test_checkpoint_other_corruption(change: str) -> None:
    from cipher_lab_plugins.prediction_common.checkpoints import load_checkpoint
    node = torch.load(io.BytesIO(content('C')), weights_only=True)
    if change == 'bit':
        node['task_declarations'][0]['target_bit'] = 1
    elif change == 'shape':
        node['state']['layers.0.weight'] = torch.zeros(2, 64)
    elif change == 'schema':
        node['schema'] = 2
    buffer = io.BytesIO()
    torch.save(node, buffer, pickle_protocol=2)
    encoded = buffer.getvalue()
    with pytest.raises(ValueError):
        load_checkpoint(encoded, 'b'*64 if change == 'sha'
                        else hashlib.sha256(encoded).hexdigest())


@pytest.mark.parametrize('arm,field,value', [('A', 'output_mode', 'residual'),
    ('B', 'arm_id', 'C'), ('C', 'input_representation', 'raw64-msb0-v1')])
def test_spec_rejects_wrong_binding(arm: str, field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(spec(arm), **{field: value})


@pytest.mark.parametrize('schema', [1, 2])
def test_legacy_schema_roundtrip_unchanged(schema: int) -> None:
    from cipher_lab.training_config import MLPSpec
    from cipher_lab_plugins.prediction_common.bundles import (
        DatasetBundle,
        PredictionTask,
        decode_bundle,
        encode_bundle,
    )
    from cipher_lab_plugins.prediction_common.checkpoints import (
        Selection,
        encode_checkpoint,
        load_checkpoint,
    )
    from cipher_lab_plugins.prediction_common.metrics import PublicModelSpec, TrainBaseline
    from cipher_lab_plugins.prediction_common.steps import initialized_model
    from cipher_lab_plugins.residual.tasks import ResidualPredictionTask
    old_task = (PredictionTask('SIMON', 6, 2, 'cipher') if schema == 1 else
                ResidualPredictionTask('SIMON', 6, 2, 'cipher', 'simon64-zero-master-v1'))
    old_spec = PublicModelSpec(MLPSpec(2), (old_task,), 'direct' if schema == 1 else 'residual')
    old_bundle = DatasetBundle(old_spec.method, (old_task,), bundle().panels,
                               {} if schema == 1 else bundle().references)
    loaded_bundle = decode_bundle(encode_bundle(old_bundle))
    assert loaded_bundle.task_declarations == (old_task,)
    encoded = encode_checkpoint(initialized_model(old_spec, 31), old_spec, (old_task,),
        Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
    node = torch.load(io.BytesIO(encoded), weights_only=True)
    assert node['schema'] == schema
    assert 'arm_id' not in node['wiring'] and 'feature_order' not in node['wiring']
    assert ('output_mode' in node['wiring']) == (schema == 2)
    loaded = load_checkpoint(encoded, hashlib.sha256(encoded).hexdigest())
    assert loaded.public_model_spec == old_spec


@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_saved_evaluate_fake_absolute_logits(arm: str, tmp_path: Any) -> None:
    from test_training_steps import make_meter

    from cipher_lab.artifacts import RunStore
    from cipher_lab_plugins.prediction_common.bundles import encode_bundle
    from cipher_lab_plugins.prediction_common.checkpoints import Selection, encode_checkpoint
    from cipher_lab_plugins.prediction_common.evaluation import run_evaluation
    from cipher_lab_plugins.prediction_common.metrics import TrainBaseline
    from cipher_lab_plugins.prediction_common.steps import initialized_model
    identity = spec(arm)
    model = initialized_model(identity, 31)
    with torch.no_grad():
        model.state_dict()['layers.4.bias'].fill_(2.)
    checkpoint = encode_checkpoint(model, identity, identity.task_declarations,
        Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
    dataset = encode_bundle(bundle())
    path, data_path = tmp_path/'model.pt', tmp_path/'data.npz'
    path.write_bytes(checkpoint)
    data_path.write_bytes(dataset)
    meter = make_meter(rows=2)
    store = RunStore.claim(tmp_path, 'eval', output_bytes=10000000)
    result = run_evaluation(path, hashlib.sha256(checkpoint).hexdigest(), data_path,
                            hashlib.sha256(dataset).hexdigest(), meter, store)
    assert result['status'] == 'SUCCEEDED'
    summary = cast(dict[str, Any], result['summary'])
    assert summary['arm_id'] == arm
    assert summary['metrics']['tasks']['SIMON']['acc'] == (.5 if arm == 'A' else 1.)
    assert meter.snapshot()['completed_forward_rows'] == 2  # artificial outputs only
    assert meter.snapshot()['completed_real_blocks'] == 0


@pytest.mark.parametrize('change', ['schema', 'representation', 'bit', 'secret', 'h_missing',
                                    'h_rows', 'x13'])
def test_bundle_probe_identity_preflight(change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import numpy as np
    from test_bundles import archive, entries, npy

    from cipher_lab_plugins.prediction_common.bundles import decode_bundle, encode_bundle
    parts = entries(encode_bundle(bundle()))
    if change == 'h_missing':
        del parts['p0_h.npy']
    elif change == 'h_rows':
        parts['p0_h.npy'] = npy(np.zeros((3, 1), dtype=np.uint8))
    elif change == 'x13':
        parts['p0_x.npy'] = npy(np.zeros((2, 13), dtype=np.uint8))
    else:
        metadata = json.loads(np.load(io.BytesIO(parts['metadata.npy'])).tobytes())
        if change == 'schema':
            metadata['schema'] = 2
        else:
            key = {'representation': 'representation', 'bit': 'target_bit',
                   'secret': 'key_seed'}[change]
            metadata['task_declarations'][0][key] = 'invalid' if change == 'representation' else 1
        parts['metadata.npy'] = npy(np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    allocations: list[int] = []
    original = torch.from_numpy

    def track(value: Any) -> Any:
        allocations.append(1)
        return original(value)
    monkeypatch.setattr(torch, 'from_numpy', track)
    with pytest.raises(ValueError):
        decode_bundle(archive(parts))
    assert allocations == []


@pytest.mark.parametrize('fault', ['none', 'limit', 'sha', 'unregistered', 'dataset', 'symlink',
                                   'metadata', 'deleted'])
def test_owner_model_read_is_bound_and_bounded(tmp_path: Any, fault: str) -> None:
    import json

    from cipher_lab.artifacts import ArtifactError, RunStore
    store = RunStore.claim(tmp_path, 'read', output_bytes=10000000, needs_models=True)
    raw = b'artificial checkpoint bytes'
    ref = store.write_model('C-e1', raw)
    sha = hashlib.sha256(raw).hexdigest()
    limit = len(raw)
    if fault == 'limit':
        limit -= 1
    elif fault == 'sha':
        sha = 'b'*64
    elif fault == 'unregistered':
        ref = 'model/unregistered.pt'
    elif fault == 'dataset':
        ref = store.write_dataset('test', raw)
    elif fault == 'symlink':
        target = tmp_path/'models/read/C-e1.pt'
        target.unlink()
        other = tmp_path/'other.pt'
        other.write_bytes(raw)
        target.symlink_to(other)
    elif fault == 'deleted':
        (tmp_path/'models/read/C-e1.pt').unlink()
    elif fault == 'metadata':
        path = tmp_path/'data/read/run.json'
        node = json.loads(path.read_text())
        del node['artifacts'][ref]
        path.write_text(json.dumps(node))
    if fault == 'none':
        assert store.read_model(ref, sha, limit) == raw
    else:
        with pytest.raises(ArtifactError):
            store.read_model(ref, sha, limit)
        # Failed owner reads seal the writer; callers must not finalize it again.
        with pytest.raises(ArtifactError, match='FINALIZED'):
            store.finalize({'status': 'FAILED'})
