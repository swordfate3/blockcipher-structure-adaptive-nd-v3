from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare import config
from cipher_lab_plugins.direct_block.cube_compare.tests.test_cube_reuse import (
    cube_reuse50_arrays,
    cube_reuse50_npz,
)

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter


@pytest.fixture
def cube_direct_source(tmp_path: Path) -> Callable[..., tuple[Path, str, dict[str, bytes]]]:
    def create(fault: str = '') -> tuple[Path, str, dict[str, bytes]]:
        root = tmp_path / 'source'
        run_id = config.DIRECT_SOURCE_RUN_ID
        store = RunStore.claim(root, run_id, needs_models=True, output_bytes=2**24,
                               source_supplier=lambda: {'artificial/source.py': '0' * 64})
        datasets: dict[str, bytes] = {}
        for split, base in zip(config.SPLITS, (2, 4, 8), strict=True):
            for r in config.LOW_ROUNDS:
                arrays = cube_reuse50_arrays(2 if split == 'dev' and fault == 'overlap' else base)
                if fault == 'plain_order' and (split, r) == ('train', 2):
                    arrays = {name: val[::-1].copy() for name, val in arrays.items()}
                arrays['ciphertext'] = np.broadcast_to(np.uint64(r), (8, 2)).copy()
                arrays['ciphertext'][:, 1] |= np.uint64(1 << 63)
                out = BytesIO()
                np.savez(out, **arrays)  # type: ignore[arg-type]
                raw = (cube_reuse50_npz(base, 'fortran')
                       if fault == 'npz' and (split, r) == ('train', 3) else out.getvalue())
                datasets[f'dataset/r{r}-{split}.npz'] = raw
                store.write_dataset(f'r{r}-{split}', raw)
        store.write_model('best', b'old fake model; never read')
        store.write_payload('old-result', {'artificial': 'never read'})
        store.finalize({'status': 'STOPPED' if fault == 'status' else 'SUCCEEDED'})
        record = root / 'data' / run_id / 'run.json'
        raw = record.read_bytes()
        if fault in ('metadata', 'run_id'):
            obj = json.loads(raw)
            if fault == 'metadata':
                obj['artifacts']['dataset/r3-test.npz']['bytes'] += 1
            else:
                obj['run_id'] = 'wrong'
            raw = json.dumps(obj).encode()
            record.write_bytes(raw)
        return root, sha256(raw).hexdigest(), datasets

    return create


def cube_direct_pins(datasets: dict[str, bytes]) -> dict[str, tuple[int, str]]:
    return {ref: (len(raw), sha256(raw).hexdigest()) for ref, raw in datasets.items()}


def test_direct64_source_only_requested_splits_flatten_and_exact_archive(
    cube_direct_source: Callable[..., tuple[Path, str, dict[str, bytes]]],
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.direct64 import DirectPanels

    root, digest, datasets = cube_direct_source()
    opened = Path.open
    refs: list[str] = []

    def observe(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.is_relative_to(root):
            assert path.name == 'run.json' or f'dataset/{path.name}' in datasets
            if path.suffix == '.npz':
                refs.append(path.name)
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', observe)
    views = DirectPanels(root, cube_meter, run_id=config.DIRECT_SOURCE_RUN_ID, run_sha256=digest,
                        artifacts=cube_direct_pins(datasets), families=(1, 1, 1))
    assert not refs
    with pytest.raises(ValueError, match='split order'):
        views.label_test()
    for split, base in (('train', 2), ('dev', 4), ('test', 8)):
        panels = views.label_test() if split == 'test' else views.build(split)  # type: ignore[arg-type]
        assert refs[-3:] == [f'r{r}-{split}.npz' for r in config.LOW_ROUNDS]
        indices = np.array([15, 0, 1, 14], dtype=np.intp)
        expected = cube_reuse50_arrays(base)['plaintext'].reshape(-1)[indices]
        for r, panel in enumerate(panels, 1):
            assert len(panel) == 16 and not panel.family_ids.flags.writeable
            assert panel.family_ids.tolist() == [base] * 16
            x, y = panel.batch(indices)
            assert x.shape == y.shape == (4, 64) and x.dtype == np.float32 and y.dtype == np.uint8
            packed_input = np.packbits(x.astype(np.uint8), axis=1)
            assert [int.from_bytes(word, 'big') for word in packed_input] == list(expected)
            high = r + (1 << 63)
            assert [int.from_bytes(word, 'big') for word in np.packbits(y, axis=1)] == [high, r] * 2
            assert panel.archive() == datasets[f'dataset/r{r}-{split}.npz']
    assert len(refs) == 9
    with pytest.raises(ValueError, match='split order'):
        views.label_test()
    assert cube_meter.snapshot()['attempted_real_blocks'] == 0


@pytest.mark.parametrize('fault', ['run_sha', 'run_id', 'status', 'metadata', 'pins', 'sha', 'size',
                                  'npz', 'plain_order', 'overlap', 'parent', 'symlink', 'junction',
                                  'changed_after_read'])
def test_direct64_source_fail_closed(
    fault: str, cube_direct_source: Callable[..., tuple[Path, str, dict[str, bytes]]],
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.direct64 import DirectPanels

    root, digest, datasets = cube_direct_source(fault)
    record = root / 'data' / config.DIRECT_SOURCE_RUN_ID / 'run.json'
    ref = 'dataset/r2-train.npz'
    target = record.parent / ref
    pins = cube_direct_pins(datasets)
    match = None
    if fault == 'run_sha':
        digest = '0' * 64
    if fault == 'pins':
        del pins['dataset/r2-test.npz']
    if fault == 'sha':
        target.write_bytes(b'X' + datasets[ref][1:])
    if fault == 'size':
        target.write_bytes(b'X')
    if fault == 'parent':
        root = root / '..' / root.name
    if fault == 'symlink':
        other = target.with_name('other.npz')
        target.rename(other)
        target.symlink_to(other)
    if fault == 'junction':
        monkeypatch.setattr(Path, 'is_junction', lambda p: p == target.parent)
    if fault == 'changed_after_read':
        opened = Path.open

        def mutate(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path == target:
                with opened(record, 'ab') as stream:
                    stream.write(b' ')
            return opened(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'open', mutate)
        match = 'HASH_MISMATCH'
    if fault in ('npz', 'plain_order'):
        match = {'npz': 'INVALID_SOURCE_PANEL', 'plain_order': 'SOURCE_ROUND_ALIGNMENT'}[fault]
    error = ValueError if fault in ('npz', 'plain_order') else ArtifactError
    with pytest.raises(error, match=match):
        provider = DirectPanels(root, cube_meter, run_id=config.DIRECT_SOURCE_RUN_ID,
                                run_sha256=digest, artifacts=pins, families=(1, 1, 1))
        provider.build('train')
        provider.build('dev')
