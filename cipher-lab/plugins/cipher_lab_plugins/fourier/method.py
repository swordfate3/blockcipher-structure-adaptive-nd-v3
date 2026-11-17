"""Single Fourier stage machine, with durable JSON selection before label gates.

This is assembly, not a training runner: search owns math, data owns all labels,
Meter owns query/resource reservations, RunStore owns all writes. No retry, DEV
adaptation, model fallback, extra keys, public references or evaluation entry.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import cast

import numpy as np

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter, Stopped
from cipher_lab_plugins.fourier.codec import (
    FourierPredictor,
    decode_predictor,
    encode_predictor,
    validate_identity,
)
from cipher_lab_plugins.fourier.config import FourierRecipe
from cipher_lab_plugins.fourier.data import Points, Signs, SpectralData
from cipher_lab_plugins.fourier.search import advance_beam, fit_coefficients, predict


@dataclass(frozen=True)
class _Selected:
    ref: str
    sha256: str
    identity: dict[str, object]


def _source(store: RunStore) -> str:
    current = store.current_source_identity()
    if current != store.source_snapshot():
        raise ArtifactError('SOURCE_MISMATCH')
    raw = json.dumps(current, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return sha256(raw).hexdigest()


def _identity(recipe: FourierRecipe, backend: str, source_sha: str,
              *, null: bool = False) -> dict[str, object]:
    return validate_identity({
        'source_sha256': source_sha, 'backend': backend, 'synthetic': recipe.synthetic,
        'task': {'cipher': 'synthetic' if recipe.synthetic else 'SIMON',
                 'rounds': recipe.rounds, 'target_bit': recipe.target_bit,
                 'input_bits': recipe.input_bits},
        'label_mode': 'pseudorandom_function' if null else 'target_bit',
    }, recipe)


def _persist(store: RunStore, name: str, model: FourierPredictor,
             identity: dict[str, object]) -> _Selected:
    content = encode_predictor(model)
    ref = store.write_model(name, content)
    # RunStore currently has no public metadata accessor. Read its actual committed
    # registration, not a second digest of proposed bytes or model self-identity.
    artifacts = cast(dict[str, dict[str, object]], store._record['artifacts'])
    registered = artifacts[ref]['sha256']
    if type(registered) is not str:
        raise ArtifactError('INVALID_SHA256')
    return _Selected(ref, registered, identity)


def _load(store: RunStore, selected: _Selected, recipe: FourierRecipe) -> FourierPredictor:
    raw = store.read_model(selected.ref, selected.sha256, 65536)
    try:
        model = decode_predictor(raw, expected_identity=selected.identity)
        if model.recipe != recipe:
            raise ValueError
        return model
    except ValueError:
        raise ArtifactError('INVALID_SPECTRAL_MODEL') from None


def _score(model: FourierPredictor, points: Points, signs: Signs, meter: Meter,
           counts: dict[str, int], side: str, *, test: bool) -> dict[str, object]:
    prefix = f'{side}_scoring_rows'
    meter.check()
    counts[prefix + '_attempted'] += len(points)
    predictions = predict(model, points, check=meter.check)
    # All metrics use the whole panel and coef-frozen baseline, never a subset.
    correct = predictions == (signs < 0)
    baseline_correct = np.equal(signs < 0, model.baseline)
    delta_rows = correct.astype(np.int8) - baseline_correct.astype(np.int8)
    n = len(points)
    result: dict[str, object] = {
        'count': n, 'correct': int(np.count_nonzero(correct)),
        'baseline': model.baseline,
        'baseline_correct': int(np.count_nonzero(baseline_correct)),
        'accuracy': float(np.mean(correct)),
        'baseline_accuracy': float(np.mean(baseline_correct)),
        'delta': int(np.sum(delta_rows, dtype=np.int64)) / n,
    }
    if test:
        delta = cast(float, result['delta'])
        lower = delta - math.sqrt(2 * math.log(100) / n)
        result.update({
            'd_sum': int(np.sum(delta_rows, dtype=np.int64)),
            'd_histogram': {str(v): int(np.count_nonzero(delta_rows == v)) for v in (-1, 0, 1)},
            'lower': lower, 'alpha': 0.01,
            'positive_exploratory': cast(float, result['accuracy']) >= 0.505 and lower > 0,
        })
    counts[prefix + '_completed'] += n
    meter.check()
    return result


def run_fourier(recipe: FourierRecipe, data: SpectralData, meter: Meter, store: RunStore,
                *, backend_identity: str) -> None:
    """Run once, stopping on the first controlled failure. Storage errors propagate.

    Arithmetic attempts reserve the full bounded operation before it starts;
    completions credit only fully returned operations (None coef is completed).
    Interrupted operations retain attempts and explicit partial-unknown status,
    never claim unobserved chunks complete. No NN/optimizer budget is used.
    """
    if backend_identity != ('demo' if recipe.synthetic else 'simon'):
        raise ValueError('Fourier backend does not match recipe')
    counts = {f'{side}_{unit}_{state}': 0
              for side in ('real', 'null')
              for unit in ('sample_bucket_products', 'coefficient_rows', 'scoring_rows')
              for state in ('attempted', 'completed')}
    real: dict[str, object] = {'layers': []}
    null: dict[str, object] = {'layers': [], 'label_mode': 'pseudorandom_function'}
    summary: dict[str, object] = {
        'method': recipe.method, 'backend_identity': backend_identity,
        'synthetic': recipe.synthetic, 'scientific_validation': False,
        'outcome': 'IN_PROGRESS', 'search_status': 'NOT_STARTED',
        'coef_status': 'NOT_STARTED_GATE', 'dev_status': 'NOT_STARTED_GATE',
        'test_status': 'NOT_STARTED_GATE', 'null_status': 'IN_PROGRESS',
        'real': real, 'null': null, 'tau': recipe.tau, 'beam_width': recipe.beam_width,
        'sampling': 'with_replacement', 'split_exact_intersection': 'UNKNOWN',
        'limitations': ['bounded_beam_not_complete_GL', 'single_key_only',
                        'null_descriptive_not_p_value', 'no_signal_is_not_impossibility'],
        'arithmetic_completion': 'full_operations_only_interrupted_partial_UNKNOWN',
    }

    def snapshot() -> None:
        store.write_snapshot({'recipe': recipe.to_dict(), 'summary': summary,
                              'counts': {**meter.snapshot(), **counts}})

    def finish() -> None:
        meter.check()
        store.finalize({'status': 'SUCCEEDED', 'summary': summary,
                        'counts': {**meter.snapshot(), **counts}})

    def conditional_null() -> None:
        if summary['null_status'] == 'IN_PROGRESS':
            summary['null_status'] = 'NOT_COMPLETED_REAL_GATE'

    phase = 'initialization'
    try:
        meter.check()
        source_sha = _source(store)
        identities = {side: _identity(recipe, backend_identity, source_sha, null=side == 'null')
                      for side in ('real', 'null')}
        summary['source_sha256'] = source_sha
        snapshot()
        beams: dict[str, tuple[int, ...]] = {'real': (0,), 'null': (0,)}
        sides = {'real': real, 'null': null}
        phase = 'search'
        summary['search_status'] = 'IN_PROGRESS'
        for depth in range(1, recipe.input_bits + 1):
            p, q, g, h, gg, hh, meta = data.paired(depth)
            for side, signs, paired_signs in (('real', g, gg), ('null', h, hh)):
                if not beams[side]:
                    continue
                assert signs is not None and paired_signs is not None
                prefix = f'{side}_sample_bucket_products'
                amount = 2 * len(beams[side]) * len(p)
                meter.check()
                counts[prefix + '_attempted'] += amount
                layer = advance_beam(beams[side], depth, p, q, signs, paired_signs,
                                     tau=recipe.tau, width=recipe.beam_width, check=meter.check)
                counts[prefix + '_completed'] += layer.sample_bucket_products
                beams[side] = layer.masks
                cast(list[object], sides[side]['layers']).append({
                    **asdict(layer), **meta,
                    'score_min': min(c.score for c in layer.candidates),
                    'score_max': max(c.score for c in layer.candidates),
                    'counts': {**meter.snapshot(), **counts},
                })
                if not beams[side] and side == 'null':
                    data.disable_null()
                    summary['null_status'] = 'NO_CANDIDATE'
            snapshot()
            if not beams['real']:
                summary.update(outcome='NO_CANDIDATE', search_status='NO_CANDIDATE')
                conditional_null()
                finish()
                return
        summary['search_status'] = 'COMPLETE'
        phase = 'coef'
        summary['coef_status'] = 'IN_PROGRESS'
        p, g, h = data.panel('coef')
        selected: dict[str, _Selected] = {}
        for side, signs in (('real', g), ('null', h)):
            if not beams[side]:
                continue
            assert signs is not None
            prefix = f'{side}_coefficient_rows'
            amount = len(p) * (1 + sum(mask != 0 for mask in beams[side]))
            meter.check()
            counts[prefix + '_attempted'] += amount
            model = fit_coefficients(p, signs, beams[side], recipe=recipe,
                                     identity=identities[side], check=meter.check)
            counts[prefix + '_completed'] += amount
            if model is None:
                sides[side]['coef_status'] = 'NO_CANDIDATE'
                if side == 'real':
                    # No real model: do not finish a conditional null experiment.
                    break
                data.disable_null()
                summary['null_status'] = 'NO_CANDIDATE'
                continue
            slot = 'spectral' if side == 'real' else 'spectral-null'
            selected[side] = _persist(store, slot, model, identities[side])
            sides[side].update({
                'coef_status': 'COMPLETE', 'model_ref': selected[side].ref,
                'model_sha256': selected[side].sha256, 'baseline': model.baseline,
                'intercept': model.intercept, 'terms': [asdict(t) for t in model.terms],
                'count': model.count,
            })
        if 'real' not in selected:
            summary.update(outcome='NO_CANDIDATE', coef_status='NO_CANDIDATE')
            conditional_null()
            finish()
            return
        summary['coef_status'] = 'COMPLETE'
        snapshot()
        phase = 'durable_dev'
        if _source(store) != source_sha:
            raise ArtifactError('SOURCE_MISMATCH')
        models = {side: _load(store, item, recipe) for side, item in selected.items()}
        # Release only after EVERY selected file has passed SHA/strict decode.
        data.release_dev(selected['real'].sha256)
        phase = 'dev'
        summary['dev_status'] = 'IN_PROGRESS'
        p, g, h = data.panel('dev')
        for side, signs in (('real', g), ('null', h)):
            if side in models:
                assert signs is not None
                sides[side]['dev'] = _score(models[side], p, signs, meter, counts, side, test=False)
        dev = cast(dict[str, object], real['dev'])
        passed = cast(float, dev['accuracy']) >= 0.505 and cast(float, dev['delta']) > 0
        summary['dev_status'] = 'COMPLETE'
        summary['dev_gate_passed'] = passed
        snapshot()
        if not passed:
            summary['outcome'] = 'NO_DEV_GATE'
            conditional_null()
            finish()
            return
        phase = 'durable_test'
        if _source(store) != source_sha:
            raise ArtifactError('SOURCE_MISMATCH')
        models = {side: _load(store, item, recipe) for side, item in selected.items()}
        data.release_test(selected['real'].sha256)
        phase = 'test'
        summary['test_status'] = 'IN_PROGRESS'
        p, g, h = data.panel('test')
        for side, signs in (('real', g), ('null', h)):
            if side in models:
                assert signs is not None
                sides[side]['test'] = _score(models[side], p, signs, meter, counts, side, test=True)
        summary['test_status'] = 'COMPLETE'
        positive = cast(dict[str, object], real['test'])['positive_exploratory']
        summary['outcome'] = 'POSITIVE_EXPLORATORY' if positive else 'NO_TEST_GAIN'
        if 'null' in selected:
            summary['null_status'] = 'COMPLETE'
        # Synthetic success is explicitly not a scientific validation.
        summary['scientific_validation'] = bool(not recipe.synthetic and positive)
        meter.check()
        finish()
    except Stopped as error:
        summary.update(outcome='STOPPED', reason=error.reason, interrupted_phase=phase,
                       scientific_validation=False)
        for stage in ('search', 'coef', 'dev', 'test'):
            if summary[f'{stage}_status'] == 'IN_PROGRESS':
                summary[f'{stage}_status'] = 'PARTIAL'
        conditional_null()
        store.finalize({'status': 'STOPPED', 'summary': summary,
                        'counts': {**meter.snapshot(), **counts}})
    # ArtifactError is deliberately NOT caught: read/write may seal the owner.
