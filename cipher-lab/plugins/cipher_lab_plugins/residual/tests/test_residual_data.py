"""Artificial oracle/key bytes, never production cipher or private randomness."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_training_data import Capture, FiniteRandom, meter

from cipher_lab_plugins.prediction_common.bundles import decode_bundle, encode_bundle
from cipher_lab_plugins.prediction_common.data import PredictionData
from cipher_lab_plugins.residual.tests.test_residual_config import poison, residual_recipe

__all__ = ['poison']


@pytest.mark.parametrize('mode,real,null', [('cipher', 12, 0), ('null', 6, 6)])
def test_reference_once_zero_master_key_and_deferred_test(mode: str, real: int, null: int) -> None:
    recipe = replace(residual_recipe(), label_mode=mode)
    oracle, public, rng, ledger = Capture(), Capture(), FiniteRandom(), meter()
    data = PredictionData(recipe, oracle, rng, ledger, reference_oracle=public)
    for split in ('train', 'dev'):
        h = data.reference(split)
        assert h.shape == (2, 1)
        assert torch.equal(h, data.inputs(split)[:, :1])
        h.fill_(2)
        assert bool((data.reference(split) <= 1).all())
    with pytest.raises(ValueError, match='TEST'):
        data.reference('test')
    assert len(public.calls) == 2
    assert all(key == (0, 0, 0, 0) and rounds == 6 for _, key, rounds in public.calls)
    data.release_test()
    bundle = data.test_bundle()
    assert bundle.panels['SIMON'][0].shape == (2, 64)
    assert torch.equal(bundle.references['SIMON'], data.reference('test'))
    restored = decode_bundle(encode_bundle(bundle))
    assert restored.task_declarations == bundle.task_declarations
    assert torch.equal(restored.references['SIMON'], bundle.references['SIMON'])
    assert len(public.calls) == 3
    if mode == 'cipher':
        assert len({key for _, key, _ in oracle.calls}) == 1
        assert oracle.calls[0][1] != (0, 0, 0, 0)
        assert rng.requests == [16]
        assert [points for points, _, _ in oracle.calls] == [p for p, _, _ in public.calls]
    else:
        assert not oracle.calls and rng.requests == [1] * 6
    assert ledger.snapshot()['completed_real_blocks'] == real
    assert ledger.snapshot()['completed_null_words'] == null


@pytest.mark.parametrize('rounds', [1, 44])
def test_reference_chunks_at_most_8192_and_msb0_target(rounds: int) -> None:
    recipe = residual_recipe()
    recipe = replace(recipe, rounds=rounds, target_bit=63,
                     training=replace(recipe.training, train=8194, dev=1, test=1))
    oracle, public, ledger = Capture(), Capture(), meter(real=20000)
    data = PredictionData(recipe, oracle, FiniteRandom(), ledger, reference_oracle=public)
    h = data.reference('train')
    assert torch.equal(h, data.inputs('train')[:, 63:64])
    assert [len(points) for points, _, _ in public.calls] == [8192, 2]
    assert all(key == (0, 0, 0, 0) and r == rounds for _, key, r in public.calls)
    data.reference('train')
    assert len(public.calls) == 2


def test_missing_reference_and_failed_reference_are_not_retried() -> None:
    recipe = residual_recipe()
    with pytest.raises(ValueError):
        PredictionData(recipe, Capture(), FiniteRandom(), meter())
    ledger = meter()
    data = PredictionData(recipe, Capture(), FiniteRandom(), ledger,
                          reference_oracle=Capture(fail=True))
    from cipher_lab.execution import Stopped
    with pytest.raises(Stopped, match='ORACLE_FAILED'):
        data.reference('train')
    with pytest.raises(Stopped, match='DOMAIN_REUSED'):
        data.reference('train')
    assert ledger.snapshot()['attempted_real_blocks'] == 6
    assert ledger.snapshot()['completed_real_blocks'] == 4
