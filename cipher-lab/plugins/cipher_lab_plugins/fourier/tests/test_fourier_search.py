"""Tiny public truth tables, using a literal per-bit reference, not production parity."""
from __future__ import annotations

import numpy as np
import pytest
from cipher_lab_plugins.fourier.config import FourierRecipe
from cipher_lab_plugins.fourier.search import advance_beam, fit_coefficients, parity_sign, predict
from numpy.typing import NDArray


def literal(point: int, mask: int) -> int:
    bits = [((point >> bit) & 1) for bit in range(64) if (mask >> bit) & 1]
    return -1 if sum(bits) % 2 else 1


def table(bits: int) -> NDArray[np.uint64]:
    return np.array([v << (64 - bits) for v in range(2**bits)], dtype=np.uint64)


def identity(recipe: FourierRecipe) -> dict[str, object]:
    return {"source_sha256": "a" * 64, "backend": "demo", "synthetic": True,
            "task": {"cipher": "synthetic", "rounds": 0, "target_bit": 0,
                     "input_bits": recipe.input_bits}, "label_mode": "target_bit"}


def noop() -> None:
    pass


@pytest.mark.parametrize("mask", [0, 1, 1 << 63, (1 << 64) - 1, 0x8010020000000001])
def test_parity_literal_zero_extremes_and_high_order(mask: int) -> None:
    p = np.array([0, 1, 1 << 63, (1 << 64) - 1, 0x8010010000000001], dtype=np.uint64)
    actual = parity_sign(p, mask)
    assert actual.dtype == np.int8
    assert actual.tolist() == [literal(int(v), mask) for v in p]


def test_exact_high_order_spectrum_and_prediction() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, pairs_per_layer=16, dev_size=16, test_size=16,
                      input_bits=4, coef_size=16)
    p = table(4)
    mask = 0xD000000000000000
    g = np.array([literal(int(v), mask) for v in p], dtype=np.int8)
    model = fit_coefficients(p, g, tuple(int(v) for v in p), recipe=r,
                             identity=identity(r), check=noop)
    assert model is not None
    assert [(t.mask, t.signed_sum, t.coefficient) for t in model.terms] == [(mask, 16, 1.0)]
    assert model.intercept == 0 and model.baseline == 0
    assert model.coefficient_rows == 16 * 16  # intercept + 15 nonzero masks
    assert predict(model, p, check=noop).tolist() == [int(v < 0) for v in g]


def test_and_two_independent_parities_has_half_not_quarter_coefficients() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, pairs_per_layer=16, dev_size=16, test_size=16,
                      input_bits=4, coef_size=16)
    p = table(4)
    a, b = 0xC000000000000000, 0x3000000000000000
    g = np.array([-1 if literal(int(v), a) < 0 and literal(int(v), b) < 0 else 1
                  for v in p], dtype=np.int8)
    model = fit_coefficients(p, g, (0, a, b, a ^ b), recipe=r,
                             identity=identity(r), check=noop)
    assert model is not None and model.intercept == .5
    assert {t.mask: t.coefficient for t in model.terms} == {a: .5, b: .5, a ^ b: -.5}
    assert predict(model, p, check=noop).tolist() == [int(v < 0) for v in g]


def test_bucket_xor_reference_threshold_negative_ties_and_counts() -> None:
    p = table(1)
    # all pairs share the zero 62-bit suffix
    left, right = np.repeat(p, 8), np.tile(p, 8)
    g = np.array([literal(int(v), 1 << 63) for v in left], dtype=np.int8)
    h = np.array([literal(int(v), 1 << 63) for v in right], dtype=np.int8)
    layer = advance_beam((0,), 1, left, right, g, h, tau=.1, width=32, check=noop)
    for candidate in layer.candidates:
        expected = sum(int(x) * int(y) * literal(int(a), candidate.mask)
                       * literal(int(b), candidate.mask)
                       for a, b, x, y in zip(left, right, g, h, strict=True))
        assert candidate.signed_sum == expected and candidate.count == 16
        assert candidate.score == expected / 16
    assert layer.masks == (1 << 63,)
    assert (layer.tested, layer.survivors, layer.kept, layer.truncated) == (2, 1, 1, 0)
    assert layer.sample_bucket_products == 32
    zeros = np.zeros(8, dtype=np.uint64)
    ones = np.ones(8, dtype=np.int8)
    tied = advance_beam((0,), 1, zeros, zeros, ones, ones,
                        tau=.5, width=1, check=noop)
    assert tied.masks == (0,)
    assert (tied.tested, tied.survivors, tied.kept, tied.truncated) == (2, 2, 1, 1)
    negative = advance_beam((0,), 1, zeros, zeros, ones, -ones,
                            tau=.5, width=1, check=noop)
    assert negative.masks == () and all(t.score == -1 for t in negative.candidates)
    # tau=.5 => exact threshold .125, equality must fail (not >= or abs).
    signs = np.array([1] * 9 + [-1] * 7, dtype=np.int8)
    equality = advance_beam((0,), 1, np.zeros(16, dtype=np.uint64),
                           np.zeros(16, dtype=np.uint64), signs,
                           np.ones(16, dtype=np.int8), tau=.5, width=2, check=noop)
    assert equality.survivors == 0


def test_depth64_and_empty_beam() -> None:
    p = np.array([0, 1], dtype=np.uint64)
    g = np.array([1, -1], dtype=np.int8)
    layer = advance_beam((0,), 64, p, p, g, g, tau=.1, width=2, check=noop)
    assert layer.masks == (0, 1)
    assert advance_beam((), 1, p, p, g, g, tau=.1, width=2, check=noop).tested == 0


def test_constant_zero_mask_is_not_discovery_and_coef_inclusive_threshold() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, pairs_per_layer=16, dev_size=16, test_size=16,
                      input_bits=1, coef_size=8, tau=.5)
    p = np.array([0] * 4 + [1 << 63] * 4, dtype=np.uint64)
    assert fit_coefficients(p, np.ones(8, dtype=np.int8), (0, 1 << 63), recipe=r,
                            identity=identity(r), check=noop) is None
    # chi * g has sum=2 => coef=.25 exactly tau/2, retained.
    g = np.array([1, 1, 1, 1, -1, 1, 1, 1], dtype=np.int8)
    model = fit_coefficients(p, g, (1 << 63,), recipe=r, identity=identity(r), check=noop)
    assert model is not None and model.terms[0].coefficient == .25
    assert predict(model, np.array([0], dtype=np.uint64), check=noop).tolist() == [0]


@pytest.mark.parametrize("kind", ["float", "shape", "empty", "signzero", "signnan",
                                  "length", "depth", "mask", "suffix", "duplicate"])
def test_bucket_input_validation(kind: str) -> None:
    p = np.zeros(2, dtype=np.uint64)
    q = p.copy()
    g = np.ones(2, dtype=np.int8)
    h = g.copy()
    depth = 1
    prefixes: tuple[int, ...] = (0,)
    if kind == "float":
        p = p.astype(float)
    if kind == "shape":
        p = p.reshape(1, 2)  # type: ignore[assignment]
    if kind == "empty":
        p, q, g, h = p[:0], q[:0], g[:0], h[:0]
    if kind == "signzero":
        g[0] = 0
    if kind == "signnan":
        g = np.array([np.nan, 1])
    if kind == "length":
        h = h[:1]
    if kind == "depth":
        depth = 0
    if kind == "mask":
        prefixes = (1 << 63,)
    if kind == "suffix":
        q[0] = 1
    if kind == "duplicate":
        prefixes = (0, 0)
    with pytest.raises(ValueError):
        advance_beam(prefixes, depth, p, q, g, h, tau=.1, width=2, check=noop)


def test_chunk_checks_bound_every_operation_and_propagate_stop() -> None:
    p = np.zeros(4097, dtype=np.uint64)
    g = np.ones(4097, dtype=np.int8)
    calls: list[int] = []
    advance_beam((0,), 1, p, p, g, g, tau=.1, width=2,
                 check=lambda: calls.append(1))
    assert len(calls) >= 4  # 2 candidates * 2 chunks (plus validation checks)
    def stop() -> None:
        raise RuntimeError("public stop")
    with pytest.raises(RuntimeError, match="public stop"):
        advance_beam((0,), 1, p, p, g, g, tau=.1, width=2, check=stop)


@pytest.mark.parametrize("mask", [-1, 2**64, True, 1.0])
def test_invalid_mask(mask: object) -> None:
    with pytest.raises(ValueError):
        parity_sign(table(1), mask)  # type: ignore[arg-type]


def test_fitting_requires_panel_size_and_domain_and_identity() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, pairs_per_layer=16, dev_size=16, test_size=16,
                      input_bits=1, coef_size=2)
    p = table(1)
    g = np.array([1, -1], dtype=np.int8)
    for points, signs, masks, ident in [
        (p[:1], g[:1], (1 << 63,), identity(r)),
        (np.array([0, 1], dtype=np.uint64), g, (1 << 63,), identity(r)),
        (p, g, (1,), identity(r)), (p, g, (1 << 63, 1 << 63), identity(r)),
        (p, g, (1 << 63,), identity(r) | {"backend": "simon"}),
    ]:
        with pytest.raises(ValueError):
            fit_coefficients(points, signs, masks, recipe=r, identity=ident, check=noop)


def test_prediction_tie_zero_empty_and_checks() -> None:
    from cipher_lab_plugins.fourier.codec import FourierPredictor, FourierTerm
    r = FourierRecipe(synthetic=True, rounds=0, pairs_per_layer=16, dev_size=16, test_size=16,
                      input_bits=1, coef_size=2)
    m = FourierPredictor(recipe=r, identity=identity(r), count=2, intercept_sum=0,
                         intercept=0.0, baseline=0, terms=(FourierTerm(1 << 63, 2, 1.0),),
                         coefficient_rows=4)
    calls: list[int] = []
    out = predict(m, np.zeros(4097, dtype=np.uint64), check=lambda: calls.append(1))
    assert out.shape == (4097,) and len(calls) >= 2
    assert predict(m, np.array([], dtype=np.uint64), check=noop).size == 0
    def stop() -> None:
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        predict(m, table(1), check=stop)


def test_prediction_checks_each_candidate_chunk_and_exact_zero_tie() -> None:
    from cipher_lab_plugins.fourier.codec import FourierPredictor, FourierTerm
    r = FourierRecipe(synthetic=True, rounds=0, input_bits=2, pairs_per_layer=4,
                      coef_size=4, dev_size=4, test_size=4)
    m = FourierPredictor(r, identity(r), 4, 0, 0.0, 0,
                         (FourierTerm(1 << 62, 2, .5), FourierTerm(1 << 63, -2, -.5)), 12)
    calls: list[int] = []
    assert predict(m, table(2), check=lambda: calls.append(1)).tolist() == [0, 1, 0, 0]
    # Input validation plus each term before <=4096 row operations.
    assert len(calls) >= 3


def test_chunk_primitive_never_receives_more_than_4096_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.fourier import search
    original = search._parity
    sizes: list[int] = []
    def observed(points: NDArray[np.uint64], mask: int) -> NDArray[np.int8]:
        sizes.append(len(points))
        return original(points, mask)
    monkeypatch.setattr(search, "_parity", observed)
    p = np.zeros(4097, dtype=np.uint64)
    g = np.ones(4097, dtype=np.int8)
    advance_beam((0,), 1, p, p, g, g, tau=.1, width=2, check=noop)
    assert sizes == [4096, 1, 4096, 1]


def test_fit_coefficient_checks_and_no_mutation() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, input_bits=2, pairs_per_layer=4,
                      coef_size=4, dev_size=4, test_size=4)
    p = table(2)
    g = np.array([1, -1, 1, -1], dtype=np.int8)
    before = p.copy(), g.copy()
    calls: list[int] = []
    m = fit_coefficients(p, g, (1 << 62, 1 << 63), recipe=r,
                         identity=identity(r), check=lambda: calls.append(1))
    assert m is not None and m.coefficient_rows == 12 and len(calls) == 3
    assert np.array_equal(p, before[0]) and np.array_equal(g, before[1])
    def stop() -> None:
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        fit_coefficients(p, g, (1 << 62,), recipe=r, identity=identity(r), check=stop)


def test_exact_bucket_mass_across_layers_with_literal_high_order_oracle() -> None:
    # Exhaustive tiny full-domain pairs: each layer shares every possible suffix.
    bits = 3
    target = 0xA000000000000000
    beam = (0,)
    points = table(bits)
    for depth in range(1, bits + 1):
        pairs = [(int(x), int(y)) for x in points for y in points
                 if (int(x) & ((1 << (64 - depth)) - 1)) ==
                 (int(y) & ((1 << (64 - depth)) - 1))]
        left = np.array([x for x, _ in pairs], dtype=np.uint64)
        right = np.array([y for _, y in pairs], dtype=np.uint64)
        g = np.array([literal(x, target) for x, _ in pairs], dtype=np.int8)
        h = np.array([literal(y, target) for _, y in pairs], dtype=np.int8)
        layer = advance_beam(beam, depth, left, right, g, h,
                             tau=.1, width=32, check=noop)
        prefix = target & (((1 << depth) - 1) << (64 - depth))
        assert layer.masks == (prefix,)
        assert [c.score for c in layer.candidates if c.mask == prefix] == [1.0]
        beam = layer.masks


@pytest.mark.parametrize("tau,width", [(float("nan"), 1), (float("inf"), 1),
                                       (.1, True), (.1, 33), (True, 1), (1, 1)])
def test_bad_threshold_and_width(tau: float, width: int) -> None:
    p = np.zeros(2, dtype=np.uint64)
    g = np.ones(2, dtype=np.int8)
    with pytest.raises(ValueError):
        advance_beam((0,), 1, p, p, g, g, tau=tau, width=width, check=noop)


def test_rank_is_sum_descending_mask_ascending_not_absolute() -> None:
    # Four depth2 prefixes have masses 1, .5, .5, 0 with deterministic ties.
    p = np.zeros(8, dtype=np.uint64)
    q = np.array([0] * 4 + [1 << 63] * 2 + [1 << 62] * 2, dtype=np.uint64)
    g = np.ones(8, dtype=np.int8)
    layer = advance_beam((1 << 63, 0), 2, p, q, g, g, tau=.1, width=2, check=noop)
    assert layer.masks == (0, 1 << 62)
    assert (layer.tested, layer.survivors, layer.kept, layer.truncated) == (4, 3, 2, 1)
    assert tuple(c.mask for c in layer.candidates) == (0, 1 << 62, 1 << 63, 3 << 62)


def test_all_public_input_boundaries() -> None:
    for p in (np.zeros((1, 1), dtype=np.uint64), np.zeros(1, dtype=np.int64),
              np.array([float("nan")]), np.zeros(65537, dtype=np.uint64)):
        with pytest.raises(ValueError):
            parity_sign(p, 0)  # type: ignore[arg-type]
    assert parity_sign(np.array([], dtype=np.uint64), 0).size == 0
    r = FourierRecipe(synthetic=True, rounds=0, input_bits=1, pairs_per_layer=2,
                      coef_size=2, dev_size=2, test_size=2)
    m = fit_coefficients(table(1), np.array([1, -1], dtype=np.int8), (1 << 63,),
                         recipe=r, identity=identity(r), check=noop)
    assert m is not None
    with pytest.raises(ValueError):
        predict(m, np.array([1], dtype=np.uint64), check=noop)
    with pytest.raises(ValueError):
        fit_coefficients(table(1), np.array([1, 0], dtype=np.int8), (1 << 63,),
                         recipe=r, identity=identity(r), check=noop)


def test_signed_negative_parity_and_constant_negative_or_empty_candidates() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, input_bits=3, pairs_per_layer=8,
                      coef_size=8, dev_size=8, test_size=8)
    p = table(3)
    mask = 0xE000000000000000
    g = np.array([-literal(int(v), mask) for v in p], dtype=np.int8)
    m = fit_coefficients(p, g, (mask,), recipe=r, identity=identity(r), check=noop)
    assert m is not None and m.terms[0].coefficient == -1
    assert predict(m, p, check=noop).tolist() == [int(v < 0) for v in g]
    for masks in ((), (0,), (0, mask)):
        assert fit_coefficients(p, -np.ones(8, dtype=np.int8), masks, recipe=r,
                                identity=identity(r), check=noop) is None


def test_masked_signs_cannot_hide_invalid_labels_in_coefficient_panel() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, input_bits=1, pairs_per_layer=4,
                      coef_size=4, dev_size=4, test_size=4)
    p = np.array([0, 0, 1 << 63, 1 << 63], dtype=np.uint64)
    signs = np.ma.MaskedArray(np.array([1, 127, -1, 127], dtype=np.int8),
                              mask=[False, True, False, True])
    with pytest.raises(ValueError):
        fit_coefficients(p, signs, (1 << 63,), recipe=r, identity=identity(r), check=noop)


@pytest.mark.parametrize("entry", ["parity", "fit_points", "predict", "beam_points",
                                   "beam_paired_points", "beam_signs", "beam_paired_signs"])
def test_masked_arrays_rejected_at_each_public_array_boundary(entry: str) -> None:
    r = FourierRecipe(synthetic=True, rounds=0, input_bits=1, pairs_per_layer=4,
                      coef_size=4, dev_size=4, test_size=4)
    p = np.array([0, 0, 1 << 63, 1 << 63], dtype=np.uint64)
    g = np.array([1, 1, -1, -1], dtype=np.int8)
    masked_points = np.ma.MaskedArray(p, mask=[False, True, False, True])
    masked_signs = np.ma.MaskedArray(g, mask=[False, True, False, True])
    m = fit_coefficients(p, g, (1 << 63,), recipe=r, identity=identity(r), check=noop)
    assert m is not None
    with pytest.raises(ValueError):
        if entry == "parity":
            parity_sign(masked_points, 1 << 63)
        elif entry == "fit_points":
            fit_coefficients(masked_points, g, (1 << 63,), recipe=r,
                             identity=identity(r), check=noop)
        elif entry == "predict":
            predict(m, masked_points, check=noop)
        else:
            advance_beam((0,), 1,
                         masked_points if entry == "beam_points" else p,
                         masked_points if entry == "beam_paired_points" else p,
                         masked_signs if entry == "beam_signs" else g,
                         masked_signs if entry == "beam_paired_signs" else g,
                         tau=.1, width=2, check=noop)


def test_plain_readonly_arrays_remain_supported_without_mutation() -> None:
    r = FourierRecipe(synthetic=True, rounds=0, input_bits=1, pairs_per_layer=4,
                      coef_size=4, dev_size=4, test_size=4)
    p = np.array([0, 0, 1 << 63, 1 << 63], dtype=np.uint64)
    g = np.array([1, 1, -1, -1], dtype=np.int8)
    p.setflags(write=False)
    g.setflags(write=False)
    assert type(p) is np.ndarray and type(g) is np.ndarray
    assert parity_sign(p, 1 << 63).tolist() == g.tolist()
    layer = advance_beam((0,), 1, p, p, g, g, tau=.1, width=2, check=noop)
    assert layer.masks == (0, 1 << 63)
    m = fit_coefficients(p, g, (1 << 63,), recipe=r, identity=identity(r), check=noop)
    assert m is not None and m.terms[0].coefficient == 1.0
    assert predict(m, p, check=noop).tolist() == [0, 0, 1, 1]
    assert not p.flags.writeable and not g.flags.writeable
    assert p.tolist() == [0, 0, 1 << 63, 1 << 63] and g.tolist() == [1, 1, -1, -1]
