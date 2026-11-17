"""Tiny artificial state-machine traces with the real Meter and RunStore."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from cipher_lab_plugins.fourier import method
from cipher_lab_plugins.fourier.codec import decode_predictor
from cipher_lab_plugins.fourier.config import FourierRecipe
from cipher_lab_plugins.fourier.data import Panel, SpectralData
from cipher_lab_plugins.fourier.search import advance_beam
from cipher_lab_plugins.fourier.source import source_identity
from cipher_lab_plugins.fourier.tests.fourier_fakes import (
    PrivateBytes,
    ToyOracle,
    meter,
    recipe,
    science_guard,
)

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.execution import Stopped

__all__ = ['science_guard']


def run(tmp_path: Path, *, data: SpectralData | None = None,
        r: FourierRecipe | None = None) -> dict[str, object]:
    r = r or recipe()
    m = meter()
    data = data or SpectralData(r, ToyOracle(), PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    method.run_fourier(r, data, m, store, backend_identity='demo')
    return read_saved(tmp_path, 'toy')


def test_positive_single_frozen_model_whole_panel_d_and_baseline(tmp_path: Path) -> None:
    saved = run(tmp_path)
    s = cast(dict[str, object], saved['summary'])
    assert s['outcome'] == 'POSITIVE_EXPLORATORY'
    assert s['test_status'] == 'COMPLETE'
    real = cast(dict[str, object], s['real'])
    dev = cast(dict[str, object], real['dev'])
    test = cast(dict[str, object], real['test'])
    assert dev['accuracy'] == test['accuracy'] == 1.
    assert test['count'] == sum(cast(dict[str, int], test['d_histogram']).values()) == 128
    assert test['delta'] == 1. - cast(float, test['baseline_accuracy'])
    counts = cast(dict[str, object], saved['counts'])
    assert counts['attempted_real_blocks'] == counts['completed_real_blocks'] == 896
    assert counts['attempted_forward_rows'] == counts['completed_forward_rows'] == 0
    assert counts['attempted_optimizer_updates'] == 0
    assert counts['real_scoring_rows_attempted'] == counts['real_scoring_rows_completed'] == 256
    model_path = tmp_path / 'models/toy/spectral.pt'
    encoded = json.loads(model_path.read_bytes())
    model = decode_predictor(model_path.read_bytes(), expected_identity=encoded['identity'])
    assert model.baseline == int(model.intercept_sum < 0)
    assert real['baseline'] == model.baseline
    assert s['scientific_validation'] is False


def test_real_empty_stops_queries_null_conditional_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = advance_beam
    def beam(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        from dataclasses import replace
        # First call is real. Null remains live, but cannot cause further query.
        if not hasattr(beam, 'seen'):
            setattr(beam, 'seen', True)
            return replace(result, masks=(), kept=0)
        return result
    monkeypatch.setattr(method, 'advance_beam', beam)
    saved = run(tmp_path)
    s = cast(dict[str, object], saved['summary'])
    assert s['outcome'] == 'NO_CANDIDATE'
    assert s['null_status'] == 'NOT_COMPLETED_REAL_GATE'
    assert cast(dict[str, object], saved['counts'])['completed_real_blocks'] == 256
    assert s['dev_status'] == s['test_status'] == 'NOT_STARTED_GATE'


def test_null_empty_real_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    original, calls = advance_beam, 0
    def beam(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        return replace(result, masks=(), kept=0) if calls == 2 else result
    monkeypatch.setattr(method, 'advance_beam', beam)
    saved = run(tmp_path)
    s = cast(dict[str, object], saved['summary'])
    c = cast(dict[str, object], saved['counts'])
    assert s['outcome'] == 'POSITIVE_EXPLORATORY' and s['null_status'] == 'NO_CANDIDATE'
    assert c['completed_null_words'] == 256 and c['completed_real_blocks'] == 896


def test_coef_none_retains_work_and_no_dev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(method, 'fit_coefficients', lambda *a, **k: None)
    saved = run(tmp_path)
    s, c = cast(dict[str, object], saved['summary']), cast(dict[str, object], saved['counts'])
    assert s['outcome'] == 'NO_CANDIDATE' and s['dev_status'] == 'NOT_STARTED_GATE'
    assert cast(int, c['real_coefficient_rows_completed']) > 0
    assert c['real_coefficient_rows_attempted'] == c['real_coefficient_rows_completed']
    assert c['completed_real_blocks'] == 640
    assert not (tmp_path / 'models/toy/spectral.pt').exists()


def test_dev_fail_does_not_generate_test(tmp_path: Path) -> None:
    class ChangedDev(SpectralData):
        def panel(self, split: str) -> Panel:
            p, g, h = super().panel(split)
            return p, -g if split == 'dev' else g, h
    m, r = meter(), recipe()
    data = ChangedDev(r, ToyOracle(), PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    method.run_fourier(r, data, m, store, backend_identity='demo')
    saved = read_saved(tmp_path, 'toy')
    s = cast(dict[str, object], saved['summary'])
    assert s['outcome'] == 'NO_DEV_GATE' and s['test_status'] == 'NOT_STARTED_GATE'
    assert 'test' not in cast(dict[str, object], s['real'])
    assert m.snapshot()['completed_real_blocks'] == 768


@pytest.mark.parametrize('slot', ['spectral', 'spectral-null'])
def test_corrupt_selected_disk_before_test_no_fallback(
    tmp_path: Path, slot: str,
) -> None:
    class Corrupt(SpectralData):
        def panel(self, split: str) -> Panel:
            result = super().panel(split)
            if split == 'dev':
                path = tmp_path / f'models/toy/{slot}.pt'
                assert path.exists()
                path.write_bytes(b'corrupt')
            assert split != 'test'
            return result
    m, r = meter(), recipe(tau=0.1)
    data = Corrupt(r, ToyOracle(), PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    with pytest.raises(ArtifactError):
        method.run_fourier(r, data, m, store, backend_identity='demo')
    assert json.loads((tmp_path / 'data/toy/run.json').read_bytes())['status'] == 'FAILED'
    assert m.snapshot()['completed_real_blocks'] == 768


def test_stopped_fit_attempts_not_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stop(*args: object, **kwargs: object) -> None:
        raise Stopped('CPU_LIMIT')
    monkeypatch.setattr(method, 'fit_coefficients', stop)
    saved = run(tmp_path)
    c = cast(dict[str, object], saved['counts'])
    assert saved['status'] == 'STOPPED'
    assert cast(int, c['real_coefficient_rows_attempted']) > 0
    assert c['real_coefficient_rows_completed'] == 0
    assert cast(dict[str, object], saved['summary'])['test_status'] == 'NOT_STARTED_GATE'


def test_wrong_source_fails_before_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    monkeypatch.setattr(store, 'current_source_identity',
                        lambda: {'cipher_lab/fake.py': 'a' * 64})
    m, r, private = meter(), recipe(), PrivateBytes()
    with pytest.raises(ArtifactError, match='SOURCE_MISMATCH'):
        method.run_fourier(r, SpectralData(r, ToyOracle(), private, m), m, store,
                           backend_identity='demo')
    assert private.calls == 0


def test_real_coef_none_does_not_start_null_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    def no_signal(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        return None
    monkeypatch.setattr(method, 'fit_coefficients', no_signal)
    saved = run(tmp_path, r=recipe(tau=0.1))
    assert calls == 1
    s = cast(dict[str, object], saved['summary'])
    assert s['null_status'] == 'NOT_COMPLETED_REAL_GATE'
    assert cast(dict[str, object], saved['counts'])['null_coefficient_rows_attempted'] == 0


@pytest.mark.parametrize('change', ['source', 'recipe'])
def test_persisted_model_must_match_independent_identity_and_complete_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    # Tamper BEFORE registration so owner SHA succeeds: strict identity/recipe
    # validation, not merely hashing, must still refuse DEV.
    original = RunStore.write_model
    def write(store: RunStore, name: str, content: bytes, *, replace: bool = False) -> str:
        obj = json.loads(content)
        if change == 'source':
            obj['identity']['source_sha256'] = 'b' * 64
        else:
            obj['recipe']['dev_seed'] += 1
        return original(store, name, json.dumps(obj).encode(), replace=replace)
    monkeypatch.setattr(RunStore, 'write_model', write)
    m, r = meter(), recipe()
    d = SpectralData(r, ToyOracle(), PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    with pytest.raises(ArtifactError, match='INVALID_SPECTRAL_MODEL'):
        method.run_fourier(r, d, m, store, backend_identity='demo')
    assert m.snapshot()['completed_real_blocks'] == 640
    assert 'dev' not in d._claimed


def test_stopped_scoring_attempts_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stop(*args: object, **kwargs: object) -> None:
        raise Stopped('CPU_LIMIT')
    monkeypatch.setattr(method, 'predict', stop)
    saved = run(tmp_path)
    c = cast(dict[str, object], saved['counts'])
    s = cast(dict[str, object], saved['summary'])
    assert saved['status'] == 'STOPPED' and s['dev_status'] == 'PARTIAL'
    assert c['real_scoring_rows_attempted'] == 128
    assert c['real_scoring_rows_completed'] == 0
    assert s['test_status'] == 'NOT_STARTED_GATE'


def test_stopped_bucket_attempts_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stop(*args: object, **kwargs: object) -> None:
        raise Stopped('WALL_LIMIT')
    monkeypatch.setattr(method, 'advance_beam', stop)
    saved = run(tmp_path)
    c = cast(dict[str, object], saved['counts'])
    assert saved['status'] == 'STOPPED'
    assert c['real_sample_bucket_products_attempted'] == 256
    assert c['real_sample_bucket_products_completed'] == 0
    assert c['completed_real_blocks'] == 256


def test_dev_and_test_both_use_owner_reloaded_models_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.fourier.search import predict
    original_read = RunStore.read_model
    loaded: list[object] = []
    reads: list[str] = []
    original_decode = decode_predictor
    def read(store: RunStore, ref: str, expected_sha256: str, max_bytes: int) -> bytes:
        reads.append(ref)
        return original_read(store, ref, expected_sha256, max_bytes)
    def decode(content: bytes, *, expected_identity: dict[str, object]) -> object:
        obj = original_decode(content, expected_identity=expected_identity)
        loaded.append(obj)
        return obj
    def score(model: object, *args: object, **kwargs: object) -> object:
        assert any(model is item for item in loaded)
        return predict(model, *args, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(RunStore, 'read_model', read)
    monkeypatch.setattr(method, 'decode_predictor', decode)
    monkeypatch.setattr(method, 'predict', score)
    run(tmp_path, r=recipe(tau=0.1))
    assert reads == ['model/spectral.pt', 'model/spectral-null.pt'] * 2


def test_final_write_failure_propagates_no_second_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    def fail(store: RunStore, value: object) -> None:
        nonlocal calls
        calls += 1
        raise ArtifactError('WRITE_FAILED')
    monkeypatch.setattr(RunStore, 'finalize', fail)
    with pytest.raises(ArtifactError, match='WRITE_FAILED'):
        run(tmp_path)
    assert calls == 1


def test_resource_oracle_stop_keeps_attempts_and_no_test(tmp_path: Path) -> None:
    m, r = meter(real=128), recipe()
    private, oracle = PrivateBytes(), ToyOracle()
    d = SpectralData(r, oracle, private, m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    method.run_fourier(r, d, m, store, backend_identity='demo')
    saved = read_saved(tmp_path, 'toy')
    s = cast(dict[str, object], saved['summary'])
    assert saved['status'] == 'STOPPED' and s['reason'] == 'REAL_BLOCKS_LIMIT'
    assert oracle.batches == [128] and private.calls == 1
    assert s['test_status'] == 'NOT_STARTED_GATE'
    assert m.snapshot()['completed_real_blocks'] == 128


def test_demo_cannot_be_identified_as_simon(tmp_path: Path) -> None:
    m, r = meter(), recipe()
    private = PrivateBytes()
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    with pytest.raises(ValueError):
        method.run_fourier(r, SpectralData(r, ToyOracle(), private, m), m, store,
                           backend_identity='simon')
    assert private.calls == 0


def test_constant_toy_is_no_candidate_not_failure(tmp_path: Path) -> None:
    import numpy as np
    from numpy.typing import NDArray
    def constant(points: NDArray[np.uint64], key: tuple[int, ...],
                 rounds: int) -> NDArray[np.uint64]:
        return np.zeros(len(points), dtype=np.uint64)
    m, r = meter(), recipe(tau=1.0)
    d = SpectralData(r, constant, PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    method.run_fourier(r, d, m, store, backend_identity='demo')
    saved = read_saved(tmp_path, 'toy')
    s = cast(dict[str, object], saved['summary'])
    assert saved['status'] == 'SUCCEEDED' and s['outcome'] == 'NO_CANDIDATE'
    assert s['test_status'] == s['dev_status'] == 'NOT_STARTED_GATE'
    assert 'test' not in cast(dict[str, object], s['real'])


def test_no_test_gain_is_not_positive(tmp_path: Path) -> None:
    import numpy as np
    class ChangedTest(SpectralData):
        def panel(self, split: str) -> Panel:
            p, g, h = super().panel(split)
            return (p, np.ones(len(g), dtype=np.int8), h) if split == 'test' else (p, g, h)
    m, r = meter(), recipe()
    d = ChangedTest(r, ToyOracle(), PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    method.run_fourier(r, d, m, store, backend_identity='demo')
    s = cast(dict[str, object], read_saved(tmp_path, 'toy')['summary'])
    assert s['outcome'] == 'NO_TEST_GAIN' and s['test_status'] == 'COMPLETE'


@pytest.mark.parametrize('bits', range(1, 9))
def test_small_synthetic_input_bits_high_encoding(tmp_path: Path, bits: int) -> None:
    r = recipe(input_bits=bits, pairs_per_layer=16, coef_size=16, dev_size=16, test_size=16)
    saved = run(tmp_path, r=r)
    s = cast(dict[str, object], saved['summary'])
    assert s['backend_identity'] == 'demo' and s['synthetic'] is True
    assert s['scientific_validation'] is False


def test_layer_saves_score_range_and_cumulative_ledger(tmp_path: Path) -> None:
    saved = run(tmp_path)
    s = cast(dict[str, object], saved['summary'])
    real = cast(dict[str, object], s['real'])
    layers = cast(list[dict[str, object]], real['layers'])
    assert len(layers) == 2
    for layer in layers:
        assert -1 <= cast(float, layer['score_min']) <= cast(float, layer['score_max']) <= 1
        ledger = cast(dict[str, object], layer['counts'])
        assert cast(int, ledger['completed_real_blocks']) > 0
        assert ledger['real_sample_bucket_products_attempted'] == (
            ledger['real_sample_bucket_products_completed'])


def test_fake_rss_failure_stops_before_key_and_oracle(tmp_path: Path) -> None:
    m, r, private, oracle = meter(), recipe(), PrivateBytes(), ToyOracle()
    m._rss_bytes = lambda: 1001
    d = SpectralData(r, oracle, private, m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    method.run_fourier(r, d, m, store, backend_identity='demo')
    saved = read_saved(tmp_path, 'toy')
    assert saved['status'] == 'STOPPED'
    assert cast(dict[str, object], saved['summary'])['reason'] == 'RSS_LIMIT'
    assert oracle.batches == [] and private.calls == 0


@pytest.mark.parametrize('check_site', ['run_fourier', 'finish'])
def test_positive_terminal_stop_revokes_validation_preserves_test_and_finalizes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_site: str,
) -> None:
    import inspect

    m, r = meter(), recipe()
    original_check, original_finalize = m.check, RunStore.finalize
    injected = False
    before: dict[str, object] = {}
    finalizations: list[dict[str, object]] = []

    def stop_after_positive() -> None:
        nonlocal injected
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        caller = frame.f_back
        summary = caller.f_locals.get('summary')
        if (caller.f_code.co_name == check_site and isinstance(summary, dict)
                and summary.get('outcome') == 'POSITIVE_EXPLORATORY' and not injected):
            # Synthetic normally cannot set this flag. Simulate only the stale
            # flag at this controlled terminal boundary, never a formal run.
            summary['scientific_validation'] = True
            before['real_test'] = dict(summary['real']['test'])
            before['null_test'] = dict(summary['null']['test'])
            before['counts'] = {**m.snapshot(), **caller.f_locals['counts']}
            injected = True
            raise Stopped('CPU_LIMIT')
        original_check()

    def finalize_once(store: RunStore, value: dict[str, object]) -> None:
        finalizations.append(value)
        original_finalize(store, value)

    monkeypatch.setattr(m, 'check', stop_after_positive)
    monkeypatch.setattr(RunStore, 'finalize', finalize_once)
    d = SpectralData(r, ToyOracle(), PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=source_identity)
    method.run_fourier(r, d, m, store, backend_identity='demo')
    saved = read_saved(tmp_path, 'toy')
    summary = cast(dict[str, object], saved['summary'])
    assert injected and len(finalizations) == 1
    assert saved['status'] == finalizations[0]['status'] == 'STOPPED'
    assert summary['outcome'] == 'STOPPED' and summary['reason'] == 'CPU_LIMIT'
    assert summary['scientific_validation'] is False
    assert summary['test_status'] == 'COMPLETE'
    assert cast(dict[str, object], summary['real'])['test'] == before['real_test']
    assert cast(dict[str, object], summary['null'])['test'] == before['null_test']
    assert saved['counts'] == before['counts']
    assert cast(dict[str, object], saved['counts'])['completed_real_blocks'] == 896


@pytest.mark.parametrize('changed_after,blocked_split', [('coef', 'dev'), ('dev', 'test')])
def test_claim_supplier_rechecked_before_each_label_gate(
    tmp_path: Path, changed_after: str, blocked_split: str,
) -> None:
    identity = {'cipher_lab/fake.py': 'a' * 64,
                'cipher_lab_plugins/fourier/fake.py': 'b' * 64}
    calls = 0

    def supplier() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return dict(identity)

    class ChangedSource(SpectralData):
        def panel(self, split: str) -> Panel:
            result = super().panel(split)
            if split == changed_after:
                identity['cipher_lab_plugins/fourier/fake.py'] = 'c' * 64
            return result

    m, r = meter(), recipe()
    data = ChangedSource(r, ToyOracle(), PrivateBytes(), m)
    store = RunStore.claim(tmp_path, 'toy', output_bytes=4_000_000, needs_models=True,
                           source_supplier=supplier)
    with pytest.raises(ArtifactError, match='SOURCE_MISMATCH'):
        method.run_fourier(r, data, m, store, backend_identity='demo')
    assert blocked_split not in data._claimed
    assert calls == (3 if blocked_split == 'dev' else 4)
