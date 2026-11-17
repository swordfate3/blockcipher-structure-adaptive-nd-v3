"""Bounded public residual formats and strict arm/reference identity."""
from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_bundles import archive, entries, npy
from test_training_steps import make_meter

from cipher_lab.artifacts import RunStore
from cipher_lab.models import BitMLP
from cipher_lab.training_config import MLPSpec
from cipher_lab_plugins.prediction_common import bundles as bundles
from cipher_lab_plugins.prediction_common.checkpoints import (
    Selection,
    encode_checkpoint,
    load_checkpoint,
)
from cipher_lab_plugins.prediction_common.evaluation import run_evaluation
from cipher_lab_plugins.prediction_common.metrics import PublicModelSpec, TrainBaseline
from cipher_lab_plugins.residual.tests.test_residual_config import poison

__all__ = ['poison']


def task() -> bundles.ResidualPredictionTask:
    return bundles.ResidualPredictionTask('SIMON', 6, 0, 'cipher', 'simon64-zero-master-v1')


def bundle() -> bundles.DatasetBundle:
    return bundles.DatasetBundle('bit-residual-paired-v1', (task(),),
        {'SIMON': (torch.zeros(2, 64), torch.tensor([[0.], [1.]]))},
        {'SIMON': torch.tensor([[1.], [0.]])})


def test_bundle_requires_h_and_distinct_schema() -> None:
    value = bundle()
    content = bundles.encode_bundle(value)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        metadata = json.loads(np.load(io.BytesIO(archive.read('metadata.npy'))).tobytes())
        assert metadata['schema'] == 2
        assert set(archive.namelist()) == {'metadata.npy', 'p0_x.npy', 'p0_y.npy', 'p0_h.npy'}
    for references in ({}, {'SIMON': torch.zeros(1, 1)}, {'SIMON': torch.ones(2, 2)},
                       {'SIMON': torch.full((2, 1), 2.)}):
        with pytest.raises(ValueError):
            bundles.encode_bundle(replace(value, references=references))
    old = bundles.DatasetBundle('bit-prediction-v1',
        (bundles.PredictionTask('SIMON', 6, 0, 'cipher'),), value.panels, value.references)
    with pytest.raises(ValueError):
        bundles.encode_bundle(old)


@pytest.mark.parametrize('mode', ['direct', 'residual'])
def test_checkpoint_binds_arm_and_reference(mode: str) -> None:
    spec = PublicModelSpec(MLPSpec(2), (task(),), output_mode=mode)
    content = encode_checkpoint(BitMLP(MLPSpec(2)), spec, spec.task_declarations,
                                Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
    loaded = load_checkpoint(content, hashlib.sha256(content).hexdigest())
    assert loaded.public_model_spec == spec
    with pytest.raises(ValueError):
        load_checkpoint(content, 'b'*64)
    node = torch.load(io.BytesIO(content), weights_only=True)
    assert node['schema'] == 2 and node['wiring']['output_mode'] == mode
    node['wiring']['output_mode'] = 'invalid'
    output = io.BytesIO()
    torch.save(node, output, pickle_protocol=2)
    tampered = output.getvalue()
    with pytest.raises(ValueError):
        load_checkpoint(tampered, hashlib.sha256(tampered).hexdigest())


def test_old_checkpoint_wiring_unchanged() -> None:
    spec = PublicModelSpec(MLPSpec(2), (bundles.PredictionTask('SIMON', 3, 0, 'null'),))
    content = encode_checkpoint(BitMLP(MLPSpec(2)), spec, spec.task_declarations,
                                Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
    node = torch.load(io.BytesIO(content), weights_only=True)
    assert node['schema'] == 1 and 'output_mode' not in node['wiring']
    with pytest.raises(ValueError):
        replace(spec, output_mode='residual')


@pytest.mark.parametrize('attack', ['missing_h', 'h_shape', 'h_rows', 'h_dtype', 'h_binary',
    'schema', 'reference', 'secret', 'expanded'])
def test_hostile_h_metadata_preflight_before_tensor_allocation(attack: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    parts = entries(bundles.encode_bundle(bundle()))
    if attack == 'missing_h':
        del parts['p0_h.npy']
    elif attack in ('h_shape', 'h_rows', 'h_dtype', 'h_binary'):
        values: dict[str, np.ndarray[Any, Any]] = {
                  'h_shape': np.zeros((2, 2), dtype=np.uint8),
                  'h_rows': np.zeros((3, 1), dtype=np.uint8),
                  'h_dtype': np.zeros((2, 1)),
                  'h_binary': np.full((2, 1), 2, dtype=np.uint8)}
        parts['p0_h.npy'] = npy(values[attack])
    elif attack == 'expanded':
        header = b"{'descr': '|u1', 'fortran_order': False, 'shape': (268435457,), }\n"
        parts['p0_h.npy'] = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header
    else:
        metadata = json.loads(np.load(io.BytesIO(parts['metadata.npy'])).tobytes())
        if attack == 'schema':
            metadata['schema'] = 1
        elif attack == 'reference':
            metadata['task_declarations'][0]['reference'] = 'zero-round-keys'
        else:
            metadata['task_declarations'][0]['private_seed'] = 7
        parts['metadata.npy'] = npy(np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    allocations = []
    original = torch.from_numpy
    def track(value: Any) -> torch.Tensor:
        allocations.append(1)
        return original(value)
    monkeypatch.setattr(torch, 'from_numpy', track)
    with pytest.raises(ValueError):
        bundles.decode_bundle(archive(parts, compressed=True))
    if attack != 'h_binary':  # binary payload is checked while converting bounded arrays
        assert not allocations


def test_explicit_evaluate_saved_reference_no_cipher_and_reports_mode(tmp_path: Path) -> None:
    spec = PublicModelSpec(MLPSpec(2), (task(),), 'residual')
    model = BitMLP(MLPSpec(2))
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
        cast(torch.Tensor, model.layers[4].bias).fill_(2.)
    checkpoint = encode_checkpoint(model, spec, spec.task_declarations, Selection(1, 1, .7),
                                   {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
    dataset = bundles.encode_bundle(bundle())
    model_path, data_path = tmp_path/'explicit.pt', tmp_path/'explicit.npz'
    model_path.write_bytes(checkpoint)
    data_path.write_bytes(dataset)
    meter = make_meter(rows=2)
    store = RunStore.claim(tmp_path, 'eval', output_bytes=10000000)
    result = run_evaluation(model_path, hashlib.sha256(checkpoint).hexdigest(), data_path,
                            hashlib.sha256(dataset).hexdigest(), meter, store)
    assert result['status'] == 'SUCCEEDED'
    summary = cast(dict[str, Any], result['summary'])
    assert summary['output_mode'] == 'residual'
    assert summary['metrics']['tasks']['SIMON']['acc'] == 1.
    assert meter.snapshot()['completed_forward_rows'] == 2  # artificial tensor output
    assert meter.snapshot()['completed_real_blocks'] == 0
