from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from cipher_lab.artifacts import RunStore, read_saved
from cipher_lab.config import Policy
from cipher_lab.execution import Clocks, Kind, Meter


@pytest.mark.parametrize('fault', ['', 'restore', 'readback'])
def test_cycle_reads_saved_results_before_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  fault: str) -> None:
    from cipher_lab_plugins.direct_block.bit_frontier import config, data, training
    from cipher_lab_plugins.direct_block.bit_structure import config as bit
    from cipher_lab_plugins.direct_block.bit_structure.data import Bytes, Panels, Words
    from cipher_lab_plugins.direct_block.fit import Engine

    for name in ('TRAIN', 'DEV', 'TEST', 'BATCH'):
        monkeypatch.setattr(bit, name, 2)
    policy = Policy(2, 60, 60, (datetime.now(UTC) + timedelta(seconds=60)).isoformat(),
                    rss_stop_bytes=bit.RSS_BYTES, output_bytes=bit.OUTPUT_BYTES, **config.CAPS)
    meter = Meter(policy, Clocks(), lambda: 1)
    store = RunStore.claim(tmp_path, 'fixture', output_bytes=bit.OUTPUT_BYTES,
                           needs_models=True, model_slots=config.SLOTS,
                           source_supplier=lambda: {'fixture/engine.py': 'a' * 64})
    calls: list[int] = []

    def charge(kind: str, amount: int) -> None:
        meter.reserve(cast(Kind, kind), amount)
        meter.complete(cast(Kind, kind), amount)

    def fake_round(*_: object, prefix: str, **kwargs: object) -> dict[str, object]:
        depth = int(prefix[1:-1])
        calls.append(depth)
        if fault == 'restore' and depth == 5:
            raise ValueError('fake restore failure')
        expected = config.counts(1, 108)
        expected['real_blocks'] -= bit.KAT_BLOCKS
        for key, value in expected.items():
            assert type(value) is int
            charge(key, value)
        accuracy = .55 if depth == 4 else .5
        return {'epochs_completed': 108, 'test': {
            f'r{repeat}': {'arms': {'correct': {'accuracy': accuracy, 'bce': .693}},
                           'constant': {'bce': .693}} for repeat in (1, 2)}}

    def snapshot() -> dict[str, object]:
        record = read_saved(tmp_path, 'fixture')
        summary = record['summary']
        assert isinstance(summary, dict)
        saved_round = summary['rounds'][str(calls[-1])]
        assert 'near_chance' not in saved_round
        if fault == 'readback':
            saved_round['epochs_completed'] = 109
        return record

    def never_make(*_: object) -> Engine:
        raise AssertionError('fake run must not build engines')

    def unused_points(*_: object) -> Words:
        raise AssertionError('fake round must not request data')

    def unused_label(*_: object) -> Bytes:
        raise AssertionError('fake round must not request labels')

    def execute() -> None:
        panels = data.CyclePanels({depth: Panels(unused_points, unused_label)
                                   for depth in config.ROUNDS})
        training.run(meter, store, summary, gate=lambda _: charge('real_blocks', 2),
                     make_panels=lambda _: panels, make=never_make, read_snapshot=snapshot)

    monkeypatch.setattr(training, 'run_round', fake_round)
    summary: dict[str, object] = {}
    if fault:
        error = 'fake restore failure' if fault == 'restore' else 'invalid saved round'
        with pytest.raises(ValueError, match=error):
            execute()
        assert read_saved(tmp_path, 'fixture')['status'] == 'RUNNING'
        assert summary['last_completed_round'] == 4
    else:
        execute()
        assert read_saved(tmp_path, 'fixture')['status'] == 'SUCCEEDED'
        assert summary['last_completed_round'] == 5 and summary['stop_reason'] == 'NEAR_CHANCE'
    assert calls == ([4] if fault == 'readback' else [4, 5])
