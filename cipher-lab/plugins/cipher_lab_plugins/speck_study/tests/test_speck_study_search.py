from __future__ import annotations

import pytest

from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe
from cipher_lab_plugins.speck_study.parity import parity_sign, parity_sign_tensor
from cipher_lab_plugins.speck_study.search import (
    BucketAccumulator,
    CoefficientAccumulator,
    initial_prefixes,
    select_terms,
)
from cipher_lab_plugins.speck_study.tests.speck_study_fakes import (  # noqa: F401
    np,
    science_guard,
    torch,
)


def check() -> None:
    pass


def packed(values: list[int], bits: int) -> np.ndarray:
    return np.array([v << (64-bits) for v in values], dtype=np.uint64)


def test_literal_parity_numpy_and_cpu_torch_msb() -> None:
    values = [0, 1, 2**63, 2**64-1, 0x0123456789ABCDEF]
    points = np.array(values, dtype=np.uint64)
    tensor = torch.tensor([v if v < 2**63 else v-2**64 for v in values], dtype=torch.int64)
    for mask in (0, 1, 2**63, 2**64-1, 0xF00F0123456789AB):
        expected = [(-1)**sum(((v & mask) >> i) & 1 for i in range(64)) for v in values]
        assert parity_sign(points, mask).tolist() == expected
        assert parity_sign_tensor(tensor, mask).tolist() == expected


def test_two_parity_and_spectrum_and_bucket_chunk_invariance() -> None:
    # AND on first two public bits: g=(+,+,+,-), coefficients (+.5,+.5,+.5,-.5).
    p = packed([0, 1, 2, 3], 2)
    s = np.array([1, 1, 1, -1], dtype=np.int8)
    masks = (0, 1 << 62, 1 << 63, 3 << 62)
    fit = CoefficientAccumulator(masks, 4)
    fit.add(p, s, check=check)
    assert [c.signed_sum for c in fit.finish()] == [2, 2, 2, -2]
    points = np.repeat(p, 4)
    paired = np.tile(p, 4)
    signs = np.repeat(s, 4)
    paired_signs = np.tile(s, 4)
    layers = []
    for chunks in ((16,), (3, 5, 8)):
        acc = BucketAccumulator(2, 2, initial_prefixes(2, 1), 16)
        offset = 0
        for n in chunks:
            part = slice(offset, offset+n)
            acc.add(points[part], paired[part], signs[part], paired_signs[part], check=check)
            offset += n
        layers.append(acc.finish())
    assert layers[0] == layers[1]
    assert layers[0].sums == (4, 4, 4, 4)
    assert layers[0].kept == masks
    assert layers[0].sample_bucket_products == 64


@pytest.mark.parametrize('bits,depth', [(32, 8), (32, 32), (64, 64)])
def test_complete_prefix_mask_encoding_depths(bits: int, depth: int) -> None:
    prefixes = initial_prefixes(bits)
    assert len(prefixes) == 128 and prefixes[1] == 1 << 57
    prefixes = (0, 1 << (65-depth))
    acc = BucketAccumulator(bits, depth, prefixes, 1)
    p = np.zeros(1, dtype=np.uint64)
    s = np.ones(1, dtype=np.int8)
    acc.add(p, p, s, s, check=check)
    out = acc.finish()
    assert out.masks == tuple(sorted((0, 1 << (64-depth), 1 << (65-depth), 3 << (64-depth))))
    if bits == 32:
        assert all(mask & (2**32-1) == 0 for mask in out.masks)


def test_400_to_401_limit_never_beam_truncates() -> None:
    # 201 parents -> 402 actual survivors, all retained as evidence, none runnable.
    prefixes = tuple(i << 55 for i in range(201))
    acc = BucketAccumulator(64, 10, prefixes, 1)
    p, s = np.zeros(1, dtype=np.uint64), np.ones(1, dtype=np.int8)
    acc.add(p, p, s, s, check=check)
    out = acc.finish()
    assert out.engineering_limit and out.status == 'ENGINEERING_LIMIT'
    assert len(out.kept) == 402 and out.sums == (1,)*402
    assert len(out.masks) == 402
    with pytest.raises(ValueError):
        BucketAccumulator(64, 11, out.kept, 1)


def test_strict_bucket_threshold_and_full_400_boundary() -> None:
    p = np.zeros(200, dtype=np.uint64)
    s = np.ones(200, dtype=np.int8)
    ps = np.array([1]*100+[-1]*100, dtype=np.int8)
    acc = BucketAccumulator(64, 10, (0,), 200)
    acc.add(p, p, s, ps, check=check)
    assert acc.finish().kept == ()
    # Exactly 400 is allowed; never top32.
    acc2 = BucketAccumulator(64, 10, tuple(i << 55 for i in range(200)), 1)
    acc2.add(p[:1], p[:1], s[:1], s[:1], check=check)
    assert len(acc2.finish().kept) == 400 and not acc2.finish().engineering_limit


def test_validation_native_arrays_pair_suffix_counts_and_interrupt() -> None:
    p = np.zeros(2, dtype=np.uint64)
    s = np.ones(2, dtype=np.int8)
    for bad in (np.ma.array(p), p.view(type('Sub', (np.ndarray,), {})), p.astype('>u8')):
        with pytest.raises(ValueError):
            parity_sign(bad, 1)
    for points, paired, signs in ((p+1, p, s), (p, p+1, s), (p, p, s*0)):
        acc = BucketAccumulator(32, 8, (0,), 2)
        with pytest.raises(ValueError):
            acc.add(points, paired, signs, s, check=check)
        with pytest.raises(ValueError):
            acc.finish()
    coef_acc = CoefficientAccumulator((1 << 63,), 3)
    coef_acc.add(p, s, check=check)
    with pytest.raises(ValueError):
        coef_acc.finish()
    with pytest.raises(ValueError):
        coef_acc.add(p, s, check=check)
    acc2 = BucketAccumulator(64, 8, (0,), 2)
    with pytest.raises(RuntimeError):
        acc2.add(p, p, s, s, check=lambda: (_ for _ in ()).throw(RuntimeError('stop')))
    with pytest.raises(ValueError):
        acc2.finish()


def test_check_tiles_and_batch_limit_without_production_size() -> None:
    n = 4097
    points, signs = np.zeros(n, dtype=np.uint64), np.ones(n, dtype=np.int8)
    calls: list[int] = []
    acc = BucketAccumulator(64, 8, (0,), n)
    acc.add(points, points, signs, signs, check=lambda: calls.append(1))
    assert len(calls) >= 6  # validation + two masks each at 4096 and remainder
    assert acc.finish().count == n
    with pytest.raises(ValueError):
        parity_sign(np.zeros(50_001, dtype=np.uint64), 1)


def test_r_all_and_b_top12_strict_threshold_and_dc_zero() -> None:
    r = SpeckStudyRecipe.synthetic_recipe(r_coef_size=40, b_coef_size=40)
    masks = tuple(i << 32 for i in range(16))
    acc = CoefficientAccumulator(masks, 40)
    acc.add(np.zeros(40, dtype=np.uint64), np.ones(40, dtype=np.int8), check=check)
    scores = acc.finish()
    assert len(select_terms('R', scores, r)) == 15
    assert [t.mask for t in select_terms('B', scores, r)] == list(masks[1:13])
    # Exactly .05 is excluded, not >=.
    acc2 = CoefficientAccumulator((1 << 63,), 40)
    acc2.add(np.zeros(40, dtype=np.uint64), np.array([1]*21+[-1]*19, dtype=np.int8),
             check=check)
    assert select_terms('R', acc2.finish(), r) == ()


def test_exactly401_survivors_are_evidence_not_truncated() -> None:
    # Nonrandom public algebra panel: bulk zero-XOR + a negative Walsh character.
    # All tested coefficients=.5 except target mask=1<<54 whose sum is zero.
    xor_values = [0]*512 + [i << 54 for i in range(512)]
    xor = np.array(xor_values, dtype=np.uint64)
    points = np.zeros(1024, dtype=np.uint64)
    signs = np.ones(1024, dtype=np.int8)
    products = np.array([1]*512 + [-(-1)**(i & 1) for i in range(512)], dtype=np.int8)
    acc = BucketAccumulator(64, 10, tuple(i << 55 for i in range(201)), 1024)
    acc.add(points, xor, signs, products, check=check)
    layer = acc.finish()
    assert len(layer.kept) == 401 and layer.engineering_limit
    assert len(layer.masks) == 402 and (1 << 54) not in layer.kept
    assert set(layer.sums) == {0, 512}


def test_threshold_exactly005_excluded_and_above_included() -> None:
    points = np.zeros(400, dtype=np.uint64)
    signs = np.ones(400, dtype=np.int8)
    for positives, expected in ((201, ()), (202, (0, 1 << 56))):
        product = np.array([1]*positives + [-1]*(400-positives), dtype=np.int8)
        acc = BucketAccumulator(64, 8, (0,), 400)
        acc.add(points, points, signs, product, check=check)
        assert acc.finish().kept == expected


def test_coefficient_chunk_partition_and_interrupt_poison() -> None:
    points = packed([0, 1, 2, 3], 2)
    signs = np.array([1, -1, -1, 1], dtype=np.int8)
    masks = (0, 1 << 62, 1 << 63, 3 << 62)
    whole = CoefficientAccumulator(masks, 4)
    whole.add(points, signs, check=check)
    split = CoefficientAccumulator(masks, 4)
    split.add(points[:1], signs[:1], check=check)
    split.add(points[1:], signs[1:], check=check)
    assert whole.finish() == split.finish()
    stopped = CoefficientAccumulator(masks, 4)
    with pytest.raises(RuntimeError):
        stopped.add(points, signs, check=lambda: (_ for _ in ()).throw(RuntimeError('stop')))
    with pytest.raises(ValueError):
        stopped.finish()
    with pytest.raises(ValueError):
        stopped.add(points, signs, check=check)


@pytest.mark.parametrize('factory', [
    lambda: initial_prefixes(True),
    lambda: initial_prefixes(32, 8),
    lambda: BucketAccumulator(32, 33, (0,), 1),
    lambda: BucketAccumulator(64, 8, (1,), 1),
    lambda: BucketAccumulator(64, 8, (0, 0), 1),
    lambda: CoefficientAccumulator((True,), 1),
])
def test_invalid_prefix_and_mask_constructor_contract(factory: object) -> None:
    from collections.abc import Callable
    from typing import cast

    with pytest.raises(ValueError):
        cast(Callable[[], object], factory)()


def test_literal_bucket_sums_and_shared_nonzero_suffix_at_k8() -> None:
    # Constructed public points, not cipher inputs/outputs or random samples.
    vals = [0x123456789ABCDEF0, 0xFFEEDDCCBBAA9988, 0x0102030405060708]
    pairs = [v ^ (q << 56) for v, q in zip(vals, (0x85, 0x43, 0xFE))]
    points = np.array(vals, dtype=np.uint64)
    paired = np.array(pairs, dtype=np.uint64)
    signs = np.array([1, -1, 1], dtype=np.int8)
    psigns = np.array([-1, -1, 1], dtype=np.int8)
    acc = BucketAccumulator(64, 8, (0, 1 << 63, 3 << 62), 3)
    acc.add(points, paired, signs, psigns, check=check)
    layer = acc.finish()
    for mask, total in zip(layer.masks, layer.sums):
        literal = sum(int(s)*int(t)*(-1)**sum(((p ^ q) >> bit) & 1
                          for bit in range(64) if (mask >> bit) & 1)
                      for p, q, s, t in zip(vals, pairs, signs, psigns))
        assert total == literal
    broken = paired.copy()
    broken[0] ^= np.uint64(1)
    rejected = BucketAccumulator(64, 8, (0,), 3)
    with pytest.raises(ValueError):
        rejected.add(points, broken, signs, psigns, check=check)
