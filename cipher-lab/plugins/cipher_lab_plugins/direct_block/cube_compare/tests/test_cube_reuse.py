from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from cipher_lab_plugins.direct_block.cube_compare import config, data

from cipher_lab.artifacts import ArtifactError, RunStore
from cipher_lab.execution import Meter


def cube_reuse50_arrays(base: int) -> dict[str, data.Words]:
    left = np.array([base | sum(((tag >> j) & 1) << b
                                for j, b in enumerate(config.ACTIVE_BITS))
                     for tag in range(8)], dtype=np.uint64)
    pts = np.column_stack((left, left ^ np.uint64(config.MASK)))[::-1].copy()
    return {'plaintext': pts, 'ciphertext': ~pts, 'family_ids': np.full(8, base, np.uint64)}


def cube_reuse50_npz(base: int, fault: str = '') -> bytes:
    arrays = cube_reuse50_arrays(base)
    if fault == 'duplicate_pair':
        arrays['plaintext'][0] = arrays['plaintext'][1]
    if fault == 'family_id':
        arrays['family_ids'][0] ^= np.uint64(2)
    if fault == 'pair':
        arrays['plaintext'][0, 1] ^= np.uint64(2)
    values: dict[str, object] = dict(arrays)
    if fault == 'dtype':
        values['ciphertext'] = arrays['ciphertext'].astype(np.int64)
    if fault == 'object':
        values['ciphertext'] = arrays['ciphertext'].astype(object)
    if fault == 'endian':
        values['ciphertext'] = arrays['ciphertext'].astype('>u8')
    if fault == 'shape':
        values['ciphertext'] = arrays['ciphertext'].ravel()
    if fault == 'fortran':
        values['plaintext'] = np.asfortranarray(arrays['plaintext'])
    if fault == 'missing':
        del values['ciphertext']
    if fault == 'extra':
        values['unknown'] = arrays['ciphertext']
    out = BytesIO()
    np.savez(out, **values)  # type: ignore[arg-type]
    if fault in ('header', 'overflow', 'duplicate', 'compressed', 'trailing', 'directory'):
        patched = BytesIO()
        with ZipFile(out) as src, ZipFile(patched, 'w') as dst:
            for name in src.namelist():
                entry = src.read(name)
                if name == 'plaintext.npy':
                    if fault in ('header', 'overflow'):
                        hdr = (b' ' * 600 if fault == 'header' else
                               b"{'descr': '<u8', 'fortran_order': False, "
                               b"'shape': (18446744073709551616, 2), }\n")
                        entry = b'\x93NUMPY\x01\x00' + len(hdr).to_bytes(2, 'little') + hdr
                        entry = entry.ljust(len(src.read(name)), b'\x00')
                    elif fault == 'trailing':
                        entry += b'X'
                    elif fault == 'directory':
                        name = '../plaintext.npy'
                compression = ZIP_DEFLATED if fault == 'compressed' else 0
                dst.writestr(name, entry, compress_type=compression)
                if fault == 'duplicate' and name == 'plaintext.npy':
                    with pytest.warns(UserWarning, match='Duplicate name'):
                        dst.writestr(name, entry)
        return patched.getvalue()
    return out.getvalue()


@pytest.fixture
def cube_reuse50_source(tmp_path: Path) -> Callable[..., tuple[Path, str, dict[str, bytes]]]:
    def create(*, fault: str = '', overlap: bool = False,
               status: str = 'SUCCEEDED',
               two_test_families: bool = False) -> tuple[Path, str, dict[str, bytes]]:
        root = tmp_path / 'source'
        store = RunStore.claim(root, config.SOURCE_RUN_ID, needs_models=True, output_bytes=2**24,
                               source_supplier=lambda: {'artificial/source.py': '0' * 64})
        datasets = {'train': cube_reuse50_npz(2, fault),
                    'dev': cube_reuse50_npz(2 if overlap else 4), 'test': cube_reuse50_npz(8)}
        if two_test_families:
            left, right = cube_reuse50_arrays(8), cube_reuse50_arrays(16)
            buffer = BytesIO()
            arrays = {name: np.concatenate((left[name], right[name])) for name in left}
            np.savez(buffer, **arrays)  # type: ignore[arg-type]
            datasets['test'] = buffer.getvalue()
        for split, raw in datasets.items():
            store.write_dataset(split, raw)
        store.write_model('best', b'not loadable; must never read')
        store.write_payload('old-result', {'score': 'do not read'})
        store.finalize({'status': status, 'recipe': config.recipe(),
                        'summary': {'test': {'ignore': 'old-score-artificial'}}})
        raw = (root / 'data' / config.SOURCE_RUN_ID / 'run.json').read_bytes()
        return root, sha256(raw).hexdigest(), datasets

    return create


def cube_reuse50_bind(monkeypatch: pytest.MonkeyPatch, digest: str,
                     datasets: dict[str, bytes]) -> None:
    monkeypatch.setattr(config, 'FAMILIES', (1, 1, 1))
    monkeypatch.setattr(config, 'BATCH', 8)
    monkeypatch.setattr(config, 'SOURCE_RUN_SHA256', digest)
    monkeypatch.setattr(config, 'SOURCE_SPLITS', tuple((len(datasets[s]),
                                                     sha256(datasets[s]).hexdigest())
                                                     for s in config.SPLITS))


def test_cube_reuse50_reads_only_requested_npz_and_returns_exact_bytes(
    cube_reuse50_source: Callable[..., tuple[Path, str, dict[str, bytes]]],
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.reuse import SavedPanels

    root, digest, datasets = cube_reuse50_source()
    cube_reuse50_bind(monkeypatch, digest, datasets)
    opened = Path.open
    files: list[str] = []

    def observe(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.is_relative_to(root):
            files.append(path.name)
            assert path.name in ('run.json', 'train.npz', 'dev.npz', 'test.npz')
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', observe)
    panels = SavedPanels(root, cube_meter)
    assert set(files) == {'run.json'}
    with pytest.raises(ValueError, match='split order'):
        panels.label_test()
    for split, base in (('train', 2), ('dev', 4)):
        panel = panels.build(split)  # type: ignore[arg-type]
        assert panel.archive() == datasets[split]
        np.testing.assert_array_equal(panel.plaintext, cube_reuse50_arrays(base)['plaintext'])
        assert not panel.plaintext.flags.writeable
    assert set(files) == {'run.json', 'train.npz', 'dev.npz'}
    assert panels.label_test().archive() == datasets['test']
    assert files.count('test.npz') == 1
    assert panels.provenance == config.source_recipe()
    assert cube_meter.snapshot()['attempted_real_blocks'] == 0


@pytest.mark.parametrize('fault', ['duplicate_pair', 'family_id', 'pair', 'dtype', 'object',
                                  'endian', 'shape', 'fortran', 'missing', 'extra', 'header',
                                  'overflow', 'duplicate', 'compressed', 'trailing', 'directory'])
def test_cube_reuse50_rejects_invalid_npz_before_any_fit(
    fault: str, cube_reuse50_source: Callable[..., tuple[Path, str, dict[str, bytes]]],
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.reuse import SavedPanels
    from cipher_lab_plugins.direct_block.cube_compare.training import Plan, run

    root, digest, datasets = cube_reuse50_source(fault=fault)
    cube_reuse50_bind(monkeypatch, digest, datasets)
    store = RunStore.claim(tmp_path / 'new', 'fit', needs_models=True, output_bytes=2**24)
    with pytest.raises(ValueError, match='INVALID_SOURCE_PANEL'):
        run(cube_meter, store, {}, make_panels=lambda meter: SavedPanels(root, meter),
            make_engine=lambda arm: pytest.fail('invalid source entered fit'),
            plan=Plan((1, 1, 1), 1, 8))
    assert cube_meter.snapshot()['attempted_optimizer_updates'] == 0


@pytest.mark.parametrize('fault', ['run_sha', 'run_id', 'status', 'metadata', 'symlink',
                                  'parent', 'size', 'sha', 'missing', 'overlap', 'source_changed',
                                  'junction'])
def test_cube_reuse50_rejects_bad_source_identity_or_io(
    fault: str, cube_reuse50_source: Callable[..., tuple[Path, str, dict[str, bytes]]],
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.reuse import SavedPanels

    root, digest, datasets = cube_reuse50_source(overlap=fault == 'overlap',
                                                 status='STOPPED' if fault == 'status'
                                                 else 'SUCCEEDED')
    record = root / 'data' / config.SOURCE_RUN_ID / 'run.json'
    path = record.parent / 'dataset/train.npz'
    if fault == 'run_sha':
        digest = '0' * 64
    elif fault in ('metadata', 'run_id'):
        actual = json.loads(record.read_bytes())
        if fault == 'metadata':
            actual['artifacts']['dataset/test.npz']['bytes'] += 1
        else:
            actual['run_id'] = 'other'
        raw = json.dumps(actual).encode()
        record.write_bytes(raw)
        digest = sha256(raw).hexdigest()
    elif fault == 'symlink':
        target = path.with_name('other.npz')
        path.rename(target)
        path.symlink_to(target)
    elif fault == 'parent':
        root = root / '..' / root.name
    elif fault == 'junction':
        # Simulate a Windows junction in the dataset parent, not a real FFI call.
        monkeypatch.setattr(Path, 'is_junction', lambda p: p == path.parent)
    elif fault == 'size':
        path.write_bytes(b'too short')
    elif fault == 'sha':
        path.write_bytes(b'X' + datasets['train'][1:])
    elif fault == 'missing':
        path.unlink()
    cube_reuse50_bind(monkeypatch, digest, datasets)
    with pytest.raises(ArtifactError):
        panels = SavedPanels(root, cube_meter)
        if fault == 'source_changed':
            with record.open('ab') as stream:
                stream.write(b' ')
        panels.build('train')
        panels.build('dev')
