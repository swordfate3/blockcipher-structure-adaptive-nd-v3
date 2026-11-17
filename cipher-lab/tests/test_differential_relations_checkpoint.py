from dataclasses import replace
from pathlib import Path

import pytest
import torch
from cipher_lab_plugins.differential_relations.checkpoint import (
    Identity,
    load_checkpoint,
    save_checkpoint,
)
from torch import nn


def identity() -> Identity:
    return Identity(experiment_id='exp-a', fit_id='F3', train_hash='a' * 64,
                    dev_hash='b' * 64, target_train_hash='c' * 64,
                    target_dev_hash='d' * 64, target_version='real-v1')


def test_roundtrip_without_forward(tmp_path: Path) -> None:
    model = nn.Linear(2, 4)
    other = nn.Linear(2, 4)
    path = tmp_path / 'model.pt'
    digest = save_checkpoint(path, model, identity())
    assert load_checkpoint(path, other, identity(), expected_sha256=digest) == digest
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        load_checkpoint(path, other, replace(identity(), experiment_id='exp-b'))
    with pytest.raises(ValueError):
        load_checkpoint(path, other, identity(), expected_sha256='0' * 64)


@pytest.mark.parametrize('mutation', ['extra', 'schema', 'identity', 'keys', 'shape', 'dtype',
                                     'nan'])
def test_strict_load_rejects_without_mutating(tmp_path: Path, mutation: str) -> None:
    model = nn.Linear(2, 4)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, identity())
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if mutation == 'extra':
        payload['optimizer'] = {}
    elif mutation == 'schema':
        payload['schema'] = True
    elif mutation == 'identity':
        payload['identity']['extra'] = 'x'
    elif mutation == 'keys':
        payload['state_dict']['extra'] = torch.zeros(1)
    elif mutation == 'shape':
        payload['state_dict']['weight'] = torch.zeros(4, 3)
    elif mutation == 'dtype':
        payload['state_dict']['weight'] = payload['state_dict']['weight'].double()
    else:
        payload['state_dict']['weight'][0, 0] = float('nan')
    torch.save(payload, path)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        load_checkpoint(path, model, identity())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_invalid_identity_and_nonfinite_save(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        replace(identity(), fit_id='F5')
    model = nn.Linear(2, 4)
    with torch.no_grad():
        model.weight[0, 0] = float('inf')
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / 'bad.pt', model, identity())


@pytest.mark.parametrize('field', ['train_hash', 'dev_hash', 'target_train_hash',
                                  'target_dev_hash'])
def test_each_actual_content_hash_is_identity_bound(tmp_path: Path, field: str) -> None:
    model = nn.Linear(2, 4)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, identity())
    changed = replace(identity(), **{field: 'e' * 64})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        load_checkpoint(path, model, changed)


def test_safe_load_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(2, 4)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, identity())
    original = torch.load
    seen: list[dict[str, object]] = []
    def spy(*args: object, **kwargs: object) -> object:
        seen.append(kwargs)
        return original(*args, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(torch, 'load', spy)
    load_checkpoint(path, model, identity())
    assert seen == [{'weights_only': True, 'map_location': 'cpu'}]


def test_source_replaced_during_load_rejected_before_state_apply(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(2, 4)
    path = tmp_path / 'model.pt'
    digest = save_checkpoint(path, model, identity())
    original = torch.load
    payload = original(path, weights_only=True, map_location='cpu')
    payload['state_dict']['weight'] = payload['state_dict']['weight'] + 1
    before = {k: v.clone() for k, v in model.state_dict().items()}
    def swapping_load(*args: object, **kwargs: object) -> object:
        torch.save(payload, path)
        return original(*args, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(torch, 'load', swapping_load)
    with pytest.raises(ValueError):
        load_checkpoint(path, model, identity(), expected_sha256=digest)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_saved_only_decoder_without_data_import(tmp_path: Path) -> None:
    import subprocess
    import sys

    from cipher_lab_plugins.differential_relations.data import (
        DataSession,
        SplitSizes,
        make_null_targets,
        sample_orbits,
        save_dataset,
    )

    class Meter:
        def reserve(self, kind: str, amount: int) -> None:
            pass
        def complete(self, kind: str, amount: int) -> None:
            pass
    sess = DataSession('exp-a', sample_orbits(SplitSizes(4, 8, 4, 4)), key=None,
                       oracle=lambda b, k: b, meter=Meter())
    p = sess.prepare()
    path = tmp_path / 'saved.npz'
    save_dataset(path, p, null=make_null_targets(p.train_labels, p.dev_labels))
    probe = '''
import sys
from pathlib import Path
from cipher_lab_plugins.differential_relations.checkpoint import load_saved_dataset
saved = load_saved_dataset(Path(sys.argv[1]), experiment_id='exp-a')
assert saved.train.shape == (8,)
assert saved.train_labels.shape == (8,4)
assert saved.null_train is not None
assert 'cipher_lab_plugins.differential_relations.data' not in sys.modules
assert not any(n.startswith('cipher_lab.ciphers') for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', probe, str(path)], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mutation', ['content', 'metadata', 'dtype', 'extra'])
def test_saved_only_decoder_strict_rejections(tmp_path: Path, mutation: str) -> None:
    import json

    import numpy as np
    from cipher_lab_plugins.differential_relations.checkpoint import load_saved_dataset
    from cipher_lab_plugins.differential_relations.data import (
        DataSession,
        SplitSizes,
        sample_orbits,
        save_dataset,
    )

    class Meter:
        def reserve(self, kind: str, amount: int) -> None:
            pass
        def complete(self, kind: str, amount: int) -> None:
            pass
    p = DataSession('exp-a', sample_orbits(SplitSizes(4, 8, 4, 4)), key=None,
                    oracle=lambda b, k: b, meter=Meter()).prepare()
    path = tmp_path / 'data.npz'
    save_dataset(path, p)
    with np.load(path, allow_pickle=False) as z:
        fields = {name: z[name] for name in z.files}
    if mutation == 'content':
        fields['train_labels'][0, 0] ^= 1
    elif mutation == 'metadata':
        m = json.loads(str(fields['metadata']))
        m['extra'] = 'not-public'
        fields['metadata'] = np.asarray(json.dumps(m))
    elif mutation == 'dtype':
        fields['train'] = fields['train'].astype(np.int64)
    else:
        fields['secret'] = np.zeros(1)
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError):
        load_saved_dataset(path, experiment_id='exp-a')
