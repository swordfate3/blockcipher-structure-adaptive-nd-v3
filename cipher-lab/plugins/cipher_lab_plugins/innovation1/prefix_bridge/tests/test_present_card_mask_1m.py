from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_frozen_recipe_and_caps() -> None:
    from cipher_lab_plugins.innovation1.present_card_mask import config

    with Path('configs/innovation1-present-card-mask-1m.json').open() as stream:
        assert json.load(stream) == config.recipe()
    assert config.counts(30) == {
        'real_blocks': 1_327_684, 'null_words': 0,
        'forward_rows': 31_114_112, 'optimizer_updates': 3_000,
    }
    assert config.counts(18, test=False)['forward_rows'] == 18 * 1_032_768
    assert config.recipe()['auxiliary'] is True


def test_mask_view_never_carries_hidden_bits_or_xor() -> None:
    from cipher_lab_plugins.innovation1.present_card_mask.masking import prepare

    inputs = np.zeros((8, 1, 2, 64), dtype=np.float32)
    inputs[::2, :, 0] = 1
    view, indices, targets = prepare(inputs, 1)
    assert view.shape == (8, 1, 6, 64)
    flags = view[:, :, :2].astype(bool)
    assert np.all(flags.sum(axis=(1, 2, 3)) == 16)
    assert np.all(flags.sum(axis=2) <= 1)
    outer = flags.any(axis=2)
    assert np.array_equal(view[:, :, 2], outer)
    assert not view[:, :, 5][outer].any()
    assert np.array_equal(targets, inputs.reshape(-1)[indices])
    altered = inputs.copy()
    altered[flags] = 1 - altered[flags]
    replay, replay_indices, replay_targets = prepare(altered, 1)
    assert np.array_equal(view, replay)
    assert np.array_equal(indices, replay_indices)
    assert np.array_equal(targets, 1 - replay_targets)
    assert not np.array_equal(indices, prepare(inputs, 2)[1])
    with pytest.raises(ValueError):
        prepare(inputs[:7], 1)


def test_exact_full_classifier_initialization_and_domain() -> None:
    import torch
    from cipher_lab_plugins.innovation1.present_card_fresh import model as pure
    from cipher_lab_plugins.innovation1.present_card_fresh import weights as pure_weights
    from cipher_lab_plugins.innovation1.present_card_mask import config, model, weights

    net = model.build()
    base = pure.build()
    assert all(torch.equal(actual, base.state_dict()[k])
               for k, actual in net.classifier.state_dict().items())
    assert sum(p.numel() for p in net.parameters()) == config.PARAMETERS
    assert all(not bool(p.count_nonzero()) for p in net.flags.parameters())
    source = {'test/fixture.py': 'a' * 64}
    metadata = weights.identity(config.SLOT, source, 'fixture', 12, .6, 'b' * 64,
                                pure.geometry(base))
    weights.require_identity(config.SLOT, metadata)
    with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
        weights.require_identity(config.SLOT, {**metadata, 'step': 1201})
    with pytest.raises(ValueError, match='CHECKPOINT_IDENTITY'):
        pure_weights.require_identity(config.SLOT, metadata)
