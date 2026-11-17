"""Private owner tests use only tiny artificial labels and deterministic bytes."""
from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import pytest
from cipher_lab_plugins.fourier.config import FourierRecipe
from cipher_lab_plugins.fourier.data import SpectralData
from cipher_lab_plugins.fourier.tests.fourier_fakes import (
    PrivateBytes,
    ToyOracle,
    meter,
    recipe,
    science_guard,
)

from cipher_lab.execution import Stopped

__all__ = ['science_guard']


def test_once_key_shared_samples_readonly_and_repeat_domain() -> None:
    r, m, private, oracle = recipe(), meter(), PrivateBytes(), ToyOracle()
    d = SpectralData(r, oracle, private, m)
    p, q, g, h, gg, hh, meta = d.paired(1)
    assert np.all(((p ^ q) & np.uint64((1 << 63) - 1)) == 0)
    assert np.all((p & np.uint64((1 << 62) - 1)) == 0)
    assert all(type(a) is np.ndarray and not a.flags.writeable for a in (p, q, g, h, gg, hh))
    assert g.tolist() == [1 - 2 * ((int(x) >> 63) & 1) for x in p]
    assert meta['pair_collisions'] == int(np.count_nonzero(p == q))
    assert private.calls == 1 and oracle.batches == [128, 128]
    with pytest.raises(Stopped, match='DOMAIN_REUSED'):
        d.paired(1)
    cp, cg, ch = d.panel('coef')
    assert private.calls == 1
    assert h is not None and ch is not None
    answers = {int(x): int(y) for x, y in zip(p, h, strict=True)}
    assert all(answers[int(x)] == int(y) for x, y in zip(cp, ch, strict=True)
               if int(x) in answers)
    assert m.snapshot()['completed_real_blocks'] == 384
    assert m.snapshot()['completed_null_words'] == 384
    assert cg.dtype == np.int8


def test_null_exact_fixed_hash_not_split_or_layer() -> None:
    d = SpectralData(recipe(), ToyOracle(), PrivateBytes(), meter())
    for panel in (d.paired(1), d.paired(2)):
        p, _, _, signs, _, _, _ = panel
        expected = [1 - 2 * (hashlib.sha256(
            b'cipher-lab/fourier-null-v1/20261111' + int(x).to_bytes(8, 'big')
        ).digest()[0] & 1) for x in p]
        assert signs is not None
        assert signs.tolist() == expected


def test_k64_pair_boundary_without_production_size() -> None:
    # Use formal recipe only for pure sampler, never labels/production-sized work.
    d = SpectralData(FourierRecipe(), ToyOracle(), PrivateBytes(), meter())
    p, q = d._sample_pair(64, 4)
    assert p.shape == q.shape == (4,)
    assert np.any(p != q)
    with pytest.raises(ValueError):
        d.paired(True)


def test_gates_and_repeat_labels_fail_closed() -> None:
    d = SpectralData(recipe(), ToyOracle(), PrivateBytes(), meter())
    with pytest.raises(ValueError):
        d.panel('dev')
    with pytest.raises(ValueError):
        d.release_test('a' * 64)
    d.panel('coef')
    d.release_dev('a' * 64)
    with pytest.raises(ValueError):
        d.release_dev('b' * 64)
    d.panel('dev')
    with pytest.raises(ValueError):
        d.release_test('b' * 64)
    d.release_test('a' * 64)
    d.panel('test')
    with pytest.raises(Stopped, match='DOMAIN_REUSED'):
        d.panel('test')


@pytest.mark.parametrize('bad', ['raise', 'dtype', 'length', 'overflow', 'subclass'])
def test_bad_oracle_nonrefundable_private_error_sanitized(bad: str) -> None:
    def broken(*args: object) -> object:
        if bad == 'raise':
            raise RuntimeError('PRIVATE_MARKER')
        if bad == 'dtype':
            return np.zeros(128, dtype=np.int64)
        if bad == 'length':
            return np.zeros(127, dtype=np.uint64)
        if bad == 'overflow':
            return [1 << 64] * 128
        return np.ma.array(np.zeros(128, dtype=np.uint64))
    m = meter()
    d = SpectralData(recipe(), broken, PrivateBytes(), m)
    with pytest.raises(Stopped) as caught:
        d.panel('coef')
    assert caught.value.reason == ('ORACLE_FAILED' if bad == 'raise' else 'ORACLE_RESPONSE')
    assert 'PRIVATE' not in str(caught.value)
    assert m.snapshot()['attempted_real_blocks'] == 128
    assert m.snapshot()['completed_real_blocks'] == 0
    with pytest.raises(Stopped, match='DOMAIN_REUSED'):
        d.panel('coef')


def test_budget_failure_before_private_and_disabled_null() -> None:
    private, m = PrivateBytes(), meter(real=0)
    d = SpectralData(recipe(), ToyOracle(), private, m)
    with pytest.raises(Stopped, match='REAL_BLOCKS_LIMIT'):
        d.panel('coef')
    assert private.calls == 0
    m = meter(null=0)
    d = SpectralData(recipe(), ToyOracle(), PrivateBytes(), m)
    d.disable_null()
    _, _, signs = d.panel('coef')
    assert signs is None
    assert m.snapshot()['completed_real_blocks'] == 128
    assert m.snapshot()['attempted_null_words'] == 0


def test_private_exception_no_repr_and_key_only_once() -> None:
    class PrivateError(Exception):
        def __str__(self) -> str:
            raise AssertionError('private str used')
        def __repr__(self) -> str:
            raise AssertionError('private repr used')
    def bad(size: int) -> bytes:
        raise PrivateError()
    m = meter()
    d = SpectralData(recipe(), ToyOracle(), bad, m)
    with pytest.raises(Stopped, match='RANDOM_UNAVAILABLE'):
        d.panel('coef')
    assert m.snapshot()['attempted_real_blocks'] == 128


def test_child_installs_own_guards() -> None:
    code = '''
import pytest
from cipher_lab_plugins.fourier.tests.fourier_fakes import (
    install_guards, meter, recipe, PrivateBytes, ToyOracle,
)
install_guards(pytest.MonkeyPatch())
from cipher_lab_plugins.fourier.data import SpectralData
d = SpectralData(recipe(), ToyOracle(), PrivateBytes(), meter())
d.paired(1)
d.panel('coef')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={**__import__('os').environ, 'PYTHONPATH': 'src:plugins'})
    assert result.returncode == 0, result.stderr


def test_arrays_cannot_be_made_writeable_or_influence_later_labels() -> None:
    d = SpectralData(recipe(), ToyOracle(), PrivateBytes(), meter())
    p, q, g, h, gg, hh, _ = d.paired(1)
    for value in (p, q, g, h, gg, hh):
        assert value is not None
        with pytest.raises(ValueError):
            value.setflags(write=True)
    cp, cg, ch = d.panel('coef')
    assert ch is not None
    assert cg.tolist() == [1 - 2 * ((int(x) >> 63) & 1) for x in cp]


@pytest.mark.parametrize('bad', [b'few', 0, None])
def test_bad_private_bytes_rejected_without_oracle(bad: object) -> None:
    oracle, m = ToyOracle(), meter()
    d = SpectralData(recipe(), oracle, lambda n: bad, m)  # type: ignore[arg-type,return-value]
    with pytest.raises(Stopped, match='RANDOM_UNAVAILABLE'):
        d.panel('coef')
    assert oracle.batches == []
    assert m.snapshot()['attempted_real_blocks'] == 128


def test_same_domain_same_public_sample_without_shared_private_state() -> None:
    a = SpectralData(recipe(), ToyOracle(), PrivateBytes(), meter())
    b = SpectralData(recipe(), ToyOracle(), PrivateBytes(), meter())
    pa, qa, *_ = a.paired(2)
    pb, qb, *_ = b.paired(2)
    assert np.array_equal(pa, pb) and np.array_equal(qa, qb)


def test_guard_blocks_every_named_scientific_entry() -> None:
    import importlib
    import random
    import secrets

    from cipher_lab import execution
    for invoke in (lambda: secrets.token_bytes(16), lambda: secrets.randbits(128),
                   lambda: random.SystemRandom(), lambda: execution.production_rss_bytes(),
                   lambda: importlib.import_module('cipher_lab.ciphers.simon'),
                   lambda: importlib.import_module('torch')):
        with pytest.raises(AssertionError):
            invoke()


def test_child_method_complete_with_own_guards() -> None:
    code = '''
import pytest
from cipher_lab_plugins.fourier.tests.fourier_fakes import (
    install_guards, meter, recipe, PrivateBytes, ToyOracle,
)
install_guards(pytest.MonkeyPatch())
from tempfile import TemporaryDirectory
from pathlib import Path
from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab_plugins.fourier.data import SpectralData
from cipher_lab_plugins.fourier.method import run_fourier
from cipher_lab_plugins.fourier.source import source_identity
with TemporaryDirectory(prefix='task2-child-') as tmp:
    m, r = meter(), recipe()
    store = RunStore.claim(Path(tmp), 'toy', output_bytes=4000000, needs_models=True,
                           source_supplier=source_identity)
    run_fourier(r, SpectralData(r, ToyOracle(), PrivateBytes(), m), m, store,
                backend_identity='demo')
    saved = read_saved(Path(tmp), 'toy')
    assert saved['summary']['outcome'] == 'POSITIVE_EXPLORATORY'
    assert saved['counts']['completed_real_blocks'] == 896
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={**__import__('os').environ, 'PYTHONPATH': 'src:plugins'})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', ['torch', 'cipher_lab.ciphers.simon'])
def test_guard_rejects_cached_import_without_changing_module(name: str) -> None:
    import importlib

    # Default T5 bootstrap/fixture has already cached and poisoned these modules.
    cached = sys.modules[name]
    with pytest.raises(AssertionError, match='scientific import forbidden'):
        importlib.import_module(name)
    assert sys.modules[name] is cached


@pytest.mark.parametrize('name', [
    'torch.__fourier_guard_probe__', 'psutil.__fourier_guard_probe__',
    'cipher_lab.ciphers.__fourier_guard_probe__',
    'cipher_lab.models.__fourier_guard_probe__',
    'cipher_lab.training.__fourier_guard_probe__',
])
def test_guard_rejects_unloaded_import_before_loading(name: str) -> None:
    import importlib

    assert name not in sys.modules
    with pytest.raises(AssertionError, match='scientific import forbidden'):
        importlib.import_module(name)
    assert name not in sys.modules


def test_guard_keeps_allowed_absolute_and_relative_imports() -> None:
    import importlib

    assert importlib.import_module('json') is sys.modules['json']
    assert importlib.import_module('.config', 'cipher_lab_plugins.fourier') is sys.modules[
        'cipher_lab_plugins.fourier.config']
    with pytest.raises(AssertionError, match='scientific import forbidden'):
        importlib.import_module('.simon', 'cipher_lab.ciphers')


def test_guard_keeps_cached_scientific_entrypoints_poisoned() -> None:
    from collections.abc import Callable

    cipher = sys.modules['cipher_lab.ciphers.simon']
    torch = sys.modules['torch']
    invocations: tuple[Callable[[], object], ...] = (
        lambda: cipher.encrypt_batch(None, (), 0),
        lambda: torch.nn.Module.__call__(None),
        lambda: torch.nn.Linear.__init__(None, 1, 1),
        lambda: torch.Tensor.backward(None),
        lambda: torch.optim.Adam.__init__(None, []),
        lambda: torch.optim.Adam.step(None),
    )
    for invoke in invocations:
        with pytest.raises(AssertionError):
            invoke()
