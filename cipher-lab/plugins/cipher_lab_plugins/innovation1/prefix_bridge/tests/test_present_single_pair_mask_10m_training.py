from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cipher_lab.artifacts import ArtifactError, RunStore, read_saved
from cipher_lab.execution import Meter, Stopped


@pytest.mark.parametrize('fault', ['', 'capacity', 'gate', 'tail', 'second-sha', 'second-restore',
                                  'late-data', 'second-score', 'late-source', 'late-stop',
                                  'terminal-write'])
def test_shared_fit_and_delayed_test(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    from cipher_lab_plugins.innovation1.mask_aux_single import config, data, training

    for key, val in [('TRAIN_ROWS', 12), ('DEV_ROWS', 4), ('TEST_ROWS', 8), ('EPOCHS', 2),
                     ('BATCH_SIZE', 8)]:
        monkeypatch.setattr(config, key, val)
    monkeypatch.setattr(data, '_CHUNK_PAIRS', 4)
    meter = bridge_meter
    monkeypatch.setattr(meter, '_policy', replace(meter._policy, real_blocks=(51 if fault ==
                                                                             'capacity' else 52)))
    source = {'artificial/owner.py': 'a'*64}
    store = RunStore.claim(tmp_path, 'pair', output_bytes=2**24, needs_models=True,
                           model_slots=config.SLOTS, source_supplier=lambda: dict(source))
    store.write_snapshot({'recipe': config.recipe()})
    summary: dict[str, Any] = {}
    calls: list[str] = []
    train: dict[str, list[bytes]] = {slot: [] for slot in config.SLOTS}
    dev: dict[str, list[bytes]] = {slot: [] for slot in config.SLOTS}
    test: dict[str, bytes] = {}
    samples = 0
    scores_started = False

    def gate(m: Meter) -> None:
        calls.append('gate')
        m.reserve('real_blocks', 4)
        if fault == 'gate':
            raise ValueError('gate fixture')
        m.complete('real_blocks', 4)

    def draw(bits: int) -> int:
        assert calls == ['gate'] and bits == 80
        calls.append('draw')
        return 17

    def sample(rng: Any, n: int) -> Any:
        nonlocal samples
        if samples >= 4:
            assert [s for s in calls if s.startswith('restored:')] == [
                f'restored:{slot}' for slot in config.SLOTS]
            assert all(arm['best_reloaded'] for arm in summary['fits'].values())
            calls.append('test-data')
        samples += 1
        return (np.arange(n*2, dtype=np.uint64) + np.uint64(samples << 32)).reshape(n, 2)

    def oracle(points: Any, key: Any, rounds: Any) -> Any:
        assert key == (17, 0, 0) and rounds == (7,)
        if samples == 6 and fault == 'late-data':
            return points.reshape(-1, 1)  # alias response after the first TEST chunk
        return (points ^ np.uint64(3)).reshape(-1, 1)

    class Runtime:
        initial_sha = 'b'*64

        def __init__(self, slot: str, arm: dict[str, object], fitting: bool) -> None:
            self.slot, self.arm, self.fitting = slot, arm, fitting
            self.b = slot == config.AUXILIARY
            self.parameters = 2000 + (config.AUX_PARAMETERS if self.b else 0)
            self.aux_rows = self.aux_bce_sum = 0

        def batch_update(self, x: Any, y: Any, m: Meter) -> float:
            assert x.shape == (len(y), 1, 6, 64) and not x[:, :, :3].any()
            assert len(x) == (4 if len(train[self.slot]) % 2 else 8)
            assert self.fitting and 'test-data' not in calls
            train[self.slot].append(x.tobytes() + y.tobytes())
            for _ in range(2 if self.b else 1):
                m.reserve('forward_rows', len(x))
                m.complete('forward_rows', len(x))
            if fault == 'tail' and self.b and len(x) == 4:
                raise ValueError('tail fixture')
            self.aux_rows += len(x)
            self.aux_bce_sum += len(x) * (2 if len(x) == 4 else 5)
            return 1. if len(x) == 8 else 4.

        def take_aux_epoch(self) -> tuple[float, int]:
            assert self.fitting and self.b
            rows, mean = self.aux_rows, self.aux_bce_sum / self.aux_rows
            self.aux_rows = self.aux_bce_sum = 0
            return mean, rows

        def predict(self, x: Any) -> Any:
            nonlocal scores_started
            assert x.shape[1:] == (1, 6, 64) and not x[:, :, :3].any()
            if self.fitting:
                assert len(train[self.slot]) == 2 * (len(dev[self.slot])+1)
                assert 'test-data' not in calls and len(x) == 4
                dev[self.slot].append(x.tobytes())
            else:
                scores_started = True
                assert 'test-data' in calls and len(x) == 8
                test[self.slot] = x.tobytes()
                if self.b and fault == 'second-score':
                    return np.full((len(x), 1), np.nan, np.float32)
                return np.full((len(x), 1), -1. if self.b else 1., np.float32)
            return np.zeros((len(x), 1), np.float32)

        def dump(self) -> bytes:
            assert self.fitting and len(dev[self.slot]) == 1  # strict earliest tie
            return self.slot.encode()

        def restore(self, raw: bytes) -> None:
            assert not self.fitting and 'test-data' not in calls and raw == self.slot.encode()
            if self.b and fault == 'second-restore':
                raise ValueError('restore fixture')
            calls.append(f'restored:{self.slot}')

        def activate(self) -> None:
            assert not self.fitting and 'test-data' in calls

        def release(self) -> None:
            calls.append(f'release:{self.slot}:{self.fitting}')

    def make(slot: str, arm: dict[str, object], fitting: bool) -> Any:
        calls.append(f'make:{slot}:{fitting}')
        return Runtime(slot, arm, fitting)

    path = tmp_path / 'data/pair/run.json'
    atomic = store._atomic

    def write(target: Path, raw: bytes) -> None:
        if target == path:
            record = json.loads(raw)
            if record['status'] == 'SUCCEEDED' and fault == 'terminal-write':
                raise OSError('terminal fixture')
            if record['status'] == 'RUNNING' and 'summary' in record:
                assert 'test' not in record['summary'].get('panel_hashes', {})
                assert 'test' not in record['summary']
        atomic(target, raw)
        if fault == 'second-sha' and target.name == f'{config.AUXILIARY}.pt':
            target.write_bytes(b'artificial corruption')

    monkeypatch.setattr(store, '_atomic', write)
    monkeypatch.setattr(store, 'current_source_identity', lambda: (
        {'artificial/owner.py': 'c'*64} if fault == 'late-source' and
        summary.get('phase') == 'FINALIZE' else dict(source)))
    check = meter.check

    def check_time() -> None:
        if fault == 'late-stop' and summary.get('phase') == 'FINALIZE':
            raise Stopped('DEADLINE')
        check()

    monkeypatch.setattr(meter, 'check', check_time)

    def execute() -> None:
        training.run(meter, store, summary, gate=gate, make=make, peaks=lambda: {'stub': 1},
                     make_panels=lambda m: data.make_panels(m, draw, oracle, sample=sample))

    if fault:
        reason = {'capacity': 'REAL_BLOCKS_LIMIT', 'gate': 'gate fixture', 'tail': 'tail fixture',
                  'second-sha': 'HASH_MISMATCH', 'second-restore': 'restore fixture',
                  'late-data': 'PREFIX_ORACLE_RESPONSE', 'second-score': 'invalid logits',
                  'late-source': 'SOURCE_CHANGED', 'late-stop': 'DEADLINE',
                  'terminal-write': 'WRITE_FAILED'}[fault]
        with pytest.raises((ValueError, ArtifactError, Stopped), match=reason):
            execute()
        record = json.loads(path.read_text())
        if record['status'] == 'RUNNING':
            store.finalize({'status': 'FAILED', 'counts': meter.snapshot(), 'summary': summary})
            record = json.loads(path.read_text())
        assert record['status'] == 'FAILED' and record['counts'] == meter.snapshot()
        assert 'test' not in summary and 'test' not in summary.get('panel_hashes', {})
        saved_summary = record['summary']
        assert 'test' not in saved_summary and 'test' not in saved_summary.get('panel_hashes', {})
        late = {'late-data', 'second-score', 'late-source', 'late-stop', 'terminal-write'}
        assert ('test-data' in calls) == (fault in late)
        assert scores_started == (fault in late - {'late-data'})
        counts = record['counts']
        assert (counts['attempted_real_blocks'], counts['completed_real_blocks']) == {
            'capacity': (0, 0), 'gate': (4, 0), 'tail': (36, 36), 'second-sha': (36, 36),
            'second-restore': (36, 36), 'late-data': (52, 44),
        }.get(fault, (52, 52))
        if fault == 'tail':
            assert (counts['attempted_optimizer_updates'], counts['completed_optimizer_updates']) == (6, 5)  # noqa: E501
            assert counts['completed_forward_rows'] == 56
        if fault == 'second-score':
            assert counts['attempted_forward_rows'] - counts['completed_forward_rows'] == 8
    else:
        execute()
        record = read_saved(tmp_path, 'pair')
        saved: Any = record['summary']
        assert record['status'] == 'SUCCEEDED' and samples == 6
        assert train[config.BASELINE] == train[config.AUXILIARY]
        assert dev[config.BASELINE] == dev[config.AUXILIARY]
        assert len(set(test.values())) == 1 and len(test) == 2
        assert set(saved['panel_hashes']) == {'train', 'dev', 'test'}
        assert saved['test']['unit'] == 'pair'
        assert saved['test']['forward_rows_by_slot'] == dict.fromkeys(config.SLOTS, 8)
        for slot, arm in saved['fits'].items():
            assert arm['epochs_completed'] == 2 and 'stop_reason' not in arm
            assert arm['selected']['epoch'] == 1 and arm['selected']['step'] == 2
            assert arm['selected']['train_bce'] == 2. and arm['work']['optimizer_updates'] == 4
            score = saved['test']['by_slot'][slot]
            assert score['pairs'] == 8 and score['correct_pairs'] == 4 and 'groups' not in score
            assert score['by_label']['0']['pairs'] == score['by_label']['1']['pairs'] == 4
        b = saved['fits'][config.AUXILIARY]
        assert b['aux_train'][-1] == dict(epoch=2, train_bce=4., masked_forward_rows=12,
                                        target_bits=192)
        assert b['work']['aux_target_bits'] == 384
        assert config.counts(2, 2) == dict(real_blocks=52, null_words=0, optimizer_updates=8,
                                          forward_rows=104)
        assert set(saved['timing']) == {
            'train_dev_data', 'fits', 'best_reload', 'test_data', 'test_score', 'gate'}
        from cipher_lab_plugins.innovation1.mask_aux_single.report import report_files

        files = report_files(record)
        assert '计算量不匹配' in files['report.md'].decode()
        assert json.loads(files['source.json'])['run_sha256'] == record['run_sha256']
    assert not list(tmp_path.rglob('*.npz'))


@pytest.mark.parametrize('fault', [
    'tail-second-view', 'partial-dev', 'b-snapshot', 'b-event', 'terminal-write',
])
def test_pair_fit_terminal_records_partial_work_and_last_sample(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import UTC, datetime

    from cipher_lab_plugins.direct_block.cube_compare import resources
    from cipher_lab_plugins.innovation1.mask_aux_single import config, data, training
    from cipher_lab_plugins.innovation1.prefix_bridge.cli import _run_owned

    from cipher_lab.config import Policy
    from cipher_lab.execution import Clocks

    for key, val in [('TRAIN_ROWS', 12), ('DEV_ROWS', 12), ('TEST_ROWS', 8), ('EPOCHS', 2),
                     ('BATCH_SIZE', 8)]:
        monkeypatch.setattr(config, key, val)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    clock = [0., 0.]
    failure = False
    checks_after_failure: list[bool] = []
    samples = 0
    transcript: list[tuple[str, int]] = []
    captured: list[Meter] = []
    failed_writes: list[dict[str, Any]] = []
    tail = fault == 'tail-second-view'
    terminal = fault == 'terminal-write'
    io_failure = fault in ('b-snapshot', 'b-event', 'terminal-write')

    def wall() -> float:
        if failure:
            checks_after_failure.append(True)
            raise Stopped('WALL_LIMIT')  # a fresh check must not replace the first error
        return clock[0]

    monkeypatch.setattr(resources, 'working_set_bytes', lambda: 1)

    def gate(meter: Meter) -> None:
        meter.reserve('real_blocks', 4)
        meter.complete('real_blocks', 4)

    def sample(rng: Any, n: int) -> Any:
        nonlocal samples
        samples += 1
        assert samples <= (3 if terminal else 2)
        return np.arange(n*2, dtype=np.uint64).reshape(n, 2) + np.uint64(samples << 32)

    class Runtime:
        initial_sha = 'b' * 64

        def __init__(self, b: bool, fitting: bool) -> None:
            self.b, self.fitting = b, fitting
            self.parameters = 2000 + (config.AUX_PARAMETERS if b else 0)
            self.steps = self.predicts = self.masked_rows = 0

        def batch_update(self, x: Any, y: Any, m: Meter) -> float:
            nonlocal failure
            self.steps += 1
            n = len(x)
            assert n == (8 if self.steps % 2 else 4)
            m.reserve('forward_rows', n)
            m.complete('forward_rows', n)
            if self.b:
                transcript.append(('class', n))
                crash = fault == 'tail-second-view' and self.steps == 4
                if crash:
                    clock[:] = [27., 11.]
                m.reserve('forward_rows', n)
                if crash:
                    failure = True
                    raise ValueError('second view did not complete')
                m.complete('forward_rows', n)
                self.masked_rows += n
                transcript.append(('masked', n))
            return .25

        def take_aux_epoch(self) -> tuple[float, int]:
            n, self.masked_rows = self.masked_rows, 0
            assert self.b and n == 12
            return .5, n

        def predict(self, x: Any) -> Any:
            nonlocal failure
            if not self.fitting:
                assert terminal and samples == 3
                clock[:] = [43., 20.]
                return np.zeros((len(x), 1), np.float32)
            self.predicts += 1
            n = len(x)
            assert n == (8 if self.predicts % 2 else 4)
            if self.b:
                transcript.append(('dev', n))
                if not tail and self.predicts == 3:
                    clock[:] = [31., 13.]
                if fault == 'partial-dev' and self.predicts == 4:
                    failure = True
                    raise Stopped('DEADLINE')
            return np.zeros((n, 1), np.float32)

        def dump(self) -> bytes:
            assert self.predicts == 2
            return b'artificial-best'

        def restore(self, raw: bytes) -> None:
            assert terminal and not self.fitting and raw == b'artificial-best'

        def activate(self) -> None:
            assert terminal and not self.fitting

        def release(self) -> None:
            if not self.b and self.fitting:
                clock[:] = [20., 7.]
            elif terminal and self.fitting:
                clock[:] = [34., 15.]

    def make(slot: str, arm: dict[str, object], fitting: bool) -> Any:
        assert fitting or terminal
        return Runtime(slot == config.AUXILIARY, fitting)

    def execute(meter: Meter, store: RunStore, summary: dict[str, object]) -> None:
        nonlocal failure
        captured.append(meter)
        atomic, write_all = store._atomic, store._write_all

        def fail_write() -> None:
            nonlocal failure
            assert not failure
            failure = True
            raise OSError('artificial one-shot write failure')

        def atomic_fault(path: Path, raw: bytes) -> None:
            if path.name == 'run.json':
                record = json.loads(raw)
                state, proposed = record['status'], record['summary']
                if state == 'FAILED':
                    failed_writes.append(record)
                if terminal and state == 'SUCCEEDED':
                    assert 'test' in proposed
                    fail_write()
                if fault == 'b-snapshot' and state == 'RUNNING':
                    arm = proposed.get('fits', {}).get(config.AUXILIARY, {})
                    if arm.get('epochs_completed') == 2:
                        assert 'work' not in arm  # sealed before the fit's finally
                        fail_write()
            atomic(path, raw)

        def event_fault(fd: int, raw: bytes) -> None:
            event = json.loads(raw) if raw.startswith(b'{') else {}
            point = event.get('summary', {})
            if (event.get('kind') == 'dev' and point.get('slot') == config.AUXILIARY
                    and point.get('epoch') == 2):
                fail_write()
            write_all(fd, raw)

        if io_failure:
            monkeypatch.setattr(store, '_atomic', atomic_fault)
        if fault == 'b-event':
            monkeypatch.setattr(store, '_write_all', event_fault)
        training.run(meter, store, summary, gate=gate, make=make,
                     make_panels=lambda m: data.make_panels(
                         m, lambda bits: 17, lambda x, k, r: (x ^ np.uint64(3)).reshape(-1, 1),
                         sample=sample),
                     peaks=lambda: {'artificial': 1} if terminal else pytest.fail('No TEST peaks'))

    policy = Policy(2, 120, 120, '2026-01-01T01:00:00Z', rss_stop_bytes=2**20,
                    output_bytes=2**24, **config.counts(2, 2))
    code = _run_owned(tmp_path, 'pair-partial', config.recipe(), policy,
                      Clocks(wall, lambda: clock[1], lambda: now), slots=config.SLOTS,
                      source_identity=lambda: {'artificial/a.py': 'a' * 64}, device=dict,
                      execute=execute, report=lambda saved: pytest.fail('No failure report write'))
    assert code == (3 if fault == 'partial-dev' else 4)
    assert not checks_after_failure and failure and samples == (3 if terminal else 2)
    saved = read_saved(tmp_path, 'pair-partial')
    reason, status = (('WRITE_FAILED', 'FAILED') if io_failure else
                      ('TRAIN_FAILED', 'FAILED') if tail else ('DEADLINE', 'STOPPED'))
    assert saved['status'] == status
    summary: Any = saved['summary']
    assert summary['reason'] == reason
    counts: Any = saved['counts']
    assert counts == captured[0].snapshot() == json.loads(capsys.readouterr().err)['counts']
    assert counts['attempted_real_blocks'] == counts['completed_real_blocks'] == (68 if terminal
                                                                                  else 52)
    assert counts['attempted_optimizer_updates'] == 8
    assert counts['completed_optimizer_updates'] == (7 if tail else 8)
    assert counts['attempted_forward_rows'] == (136 if terminal else 108 if tail else 120)
    completed_rows = {'tail-second-view': 104, 'partial-dev': 116, 'terminal-write': 136}
    assert counts['completed_forward_rows'] == completed_rows.get(fault, 120)
    assert counts['wall_elapsed_seconds'] == (43. if terminal else 27. if tail else 31.)
    assert counts['cpu_elapsed_seconds'] == (20. if terminal else 11. if tail else 13.)
    assert 'test' not in summary and 'test' not in summary.get('panel_hashes', {})
    artifacts = saved['artifacts']
    assert isinstance(artifacts, dict) and set(artifacts) == {f'model/{s}.pt' for s in config.SLOTS}
    assert not list(tmp_path.rglob('*.npz'))
    epoch = [('class', 8), ('masked', 8), ('class', 4), ('masked', 4), ('dev', 8), ('dev', 4)]
    assert transcript == epoch + (epoch[:3] if tail else epoch)
    if io_failure:
        assert len(failed_writes) == 1 and failed_writes[0]['counts'] == counts
        expected_a = dict(attempted_optimizer_updates=4, optimizer_updates=4,
                          attempted_train_class_rows=24, train_class_rows=24,
                          attempted_masked_forward_rows=0, masked_forward_rows=0,
                          attempted_dev_class_rows=24, dev_class_rows=24,
                          epochs_completed=2, wall_seconds=20., cpu_seconds=7.)
        expected_b = dict(expected_a, attempted_masked_forward_rows=24, masked_forward_rows=24,
                          wall_seconds=14. if terminal else 11.,
                          cpu_seconds=8. if terminal else 6.)
        assert summary == {'reason': 'WRITE_FAILED', 'pair_fit_work': {
            config.BASELINE: expected_a, config.AUXILIARY: expected_b}}
        assert failed_writes[0]['summary'] == summary
        return
    assert not failed_writes and summary['phase'] == ('TRAIN' if tail else 'DEV')
    assert set(summary['panel_hashes']) == {'train', 'dev'}
    a, b = (summary['fits'][s] for s in config.SLOTS)
    assert a['epochs_completed'] == 2 and b['epochs_completed'] == 1
    assert len(b['dev']) == 1 and b['selected']['step'] == 2 and 'stop_reason' not in b
    for key, expected in {'attempted_optimizer_updates': 4, 'optimizer_updates': 3 if tail else 4,
                          'attempted_train_class_rows': 24, 'train_class_rows': 24,
                          'attempted_masked_forward_rows': 24,
                          'masked_forward_rows': 20 if tail else 24,
                          'attempted_dev_class_rows': 12 if tail else 24,
                          'dev_class_rows': 12 if tail else 20,
                          'aux_target_bits': (20 if tail else 24) * config.TARGETS_PER_ROW}.items():
        assert b['work'][key] == expected
    assert a['work']['attempted_optimizer_updates'] == a['work']['optimizer_updates'] == 4
    assert summary['timing']['fits'][config.AUXILIARY] == {
        'wall_seconds': 7. if tail else 11., 'cpu_seconds': 4. if tail else 6.}


@pytest.mark.parametrize('provider', [
    'default', 'valid', 'extra-slot', 'extra-field', 'missing', 'boolean', 'float-count',
    'negative', 'too-many', 'nonfinite', 'large-int', 'byte-limit', 'output-limit',
    'exception', 'existing-key', 'persistent',
])
def test_pair_fit_fallback_evidence_is_optional_and_bounded(
    provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_meter: Meter,
) -> None:
    from cipher_lab_plugins.innovation1.mask_aux_single import config

    from cipher_lab import artifacts

    store = RunStore.claim(tmp_path, 'pair-evidence', output_bytes=2**20, needs_models=True,
                           model_slots=config.SLOTS,
                           source_supplier=lambda: {'artificial/a.py': 'a' * 64})
    path = tmp_path / 'data/pair-evidence/run.json'
    evidence = dict(attempted_train_class_rows=8, train_class_rows=8,
                    attempted_masked_forward_rows=0, masked_forward_rows=0,
                    attempted_dev_class_rows=4, dev_class_rows=0,
                    attempted_optimizer_updates=1, optimizer_updates=1,
                    epochs_completed=0, wall_seconds=3., cpu_seconds=2.)
    candidate: dict[str, Any] = {config.BASELINE: dict(evidence)}
    arm = candidate[config.BASELINE]
    committed = {'backend_identity': 'ciphers', 'scientific_validation': False,
                 'pending': {'safe': 1}, 'test': 'not failure evidence'}
    store._failure_evidence_key = 'pending'
    if provider == 'extra-slot':
        candidate['test'] = dict(evidence)
    elif provider == 'extra-field':
        arm['reason'] = 'untrusted'
    elif provider == 'missing':
        del arm['epochs_completed']
    elif provider == 'boolean':
        arm['epochs_completed'] = True
    elif provider == 'float-count':
        arm['train_class_rows'] = 8.
    elif provider == 'negative':
        arm['cpu_seconds'] = -1
    elif provider == 'too-many':
        arm['dev_class_rows'] = 5
    elif provider == 'nonfinite':
        arm['wall_seconds'] = float('nan')
    elif provider == 'large-int':
        arm['attempted_optimizer_updates'] = 2**64
    elif provider == 'byte-limit':
        monkeypatch.setattr(artifacts, '_MAX_PAIR_FIT_FAILURE_BYTES', 1, raising=False)
    elif provider == 'existing-key':
        committed['pair_fit_work'] = {'safe': 2}
        store._failure_evidence_key = 'pair_fit_work'

    previous = bridge_meter.snapshot()
    store.write_snapshot({'counts': previous, 'summary': committed})
    bridge_meter.reserve('forward_rows', 12)
    bridge_meter.complete('forward_rows', 8)
    bridge_meter.reserve('optimizer_updates', 1)
    bridge_meter.complete('optimizer_updates', 1)
    fresh = bridge_meter.snapshot()
    store._failure_counts_supplier = bridge_meter.snapshot
    reads: list[bool] = []

    def supply() -> object:
        reads.append(True)
        if provider == 'exception':
            raise OSError('artificial failed supplier')
        return candidate

    if provider != 'default':
        monkeypatch.setattr(store, '_failure_pair_fit_supplier', supply, raising=False)
    assert hasattr(store, '_failure_pair_fit_supplier')
    writes: list[str] = []
    atomic = store._atomic
    base = {store._failure_evidence_key: committed[store._failure_evidence_key],
            'reason': 'WRITE_FAILED', 'backend_identity': 'ciphers',
            'scientific_validation': False}

    def write(target: Path, raw: bytes) -> None:
        assert target == path
        record = json.loads(raw)
        writes.append(record['status'])
        if len(writes) == 1 or provider == 'persistent':
            raise OSError('one-shot or persistent write fault')
        atomic(target, raw)

    if provider == 'output-limit':
        failed = {**store._record, 'status': 'FAILED', 'counts': fresh, 'summary': base}
        # The original FAILED record fits exactly; the optional field would not.
        size = len(artifacts._encode(failed))
        monkeypatch.setattr(store, '_limit', store._total() + size)
    monkeypatch.setattr(store, '_atomic', write)
    with pytest.raises(ArtifactError, match='WRITE_FAILED'):
        store.write_snapshot({'summary': {'test': 'uncommitted, must not leak'}})
    with pytest.raises(ArtifactError, match='FINALIZED'):
        store.finalize({'status': 'FAILED'})
    assert writes == ['RUNNING', 'FAILED']
    assert reads == ([] if provider in ('default', 'existing-key') else [True])
    saved = json.loads(path.read_text())
    if provider == 'persistent':
        assert saved['status'] == 'RUNNING' and saved['counts'] == previous
        assert saved['summary'] == committed
    else:
        assert saved['status'] == 'FAILED' and saved['counts'] == fresh
        assert saved['summary'] == ({**base, 'pair_fit_work': candidate}
                                    if provider == 'valid' else base)
