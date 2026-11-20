"""Review fixes: hand-built state and saved-only resource lifecycle, no science."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from cipher_lab_plugins.innovation1.checkpoint import decode, encode, identity
from cipher_lab_plugins.innovation1.tests.test_task2 import setup, small

FIXED = ('stems.local.indices', 'stems.word.indices', 'stems.nibble.indices')


def fixture_state() -> dict[str, torch.Tensor]:
    return {**{name: torch.tensor([[0, 1], [2, 3]], dtype=torch.int64) for name in FIXED},
            'weight': torch.tensor([0.5]), 'bn.running_mean': torch.tensor([0.]),
            'bn.running_var': torch.tensor([1.]),
            'bn.num_batches_tracked': torch.tensor(0, dtype=torch.int64)}


def test_fixed_buffers_exact_but_parameters_and_bn_are_dynamic() -> None:
    meta = identity(small(), {'manual/state.py': 'a' * 64}, 'ciphers', 'M-best', 1, 0.5)
    template = fixture_state()
    state = {k: v.clone() for k, v in template.items()}
    state['weight'] += 1
    state['bn.running_mean'] += 2
    state['bn.running_var'] += 3
    state['bn.num_batches_tracked'] += 9
    fixed = {k: template[k] for k in FIXED}
    restored = decode(encode(meta, state), meta, template, fixed_buffers=fixed)
    assert torch.equal(restored['weight'], state['weight'])
    assert int(restored['bn.num_batches_tracked']) == 9
    # In-range permutation: shape, dtype and finiteness all still valid.
    state[FIXED[0]][0, 0] = 1
    with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
        decode(encode(meta, state), meta, template, fixed_buffers=fixed)


@pytest.mark.parametrize('names', [(), FIXED[:2], (*FIXED, 'weight')])
def test_missing_extra_fixed_buffer_names_fail_closed(names: tuple[str, ...]) -> None:
    meta = identity(small(), {'manual/state.py': 'a' * 64}, 'ciphers', 'M-best', 1, 0.5)
    template = fixture_state()
    with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
        decode(encode(meta, template), meta, template,
               fixed_buffers={k: template[k] for k in names})


def test_reconstructed_model_passes_fixed_buffers_before_state_load(monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1 import models
    canonical = fixture_state()
    loaded: list[Any] = []
    class ArtificialModel:
        def state_dict(self) -> Any:
            return canonical
        def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
            loaded.append(args)
        def to(self, *args: Any, **kwargs: Any) -> Any:
            return self
        def eval(self) -> Any:
            return self
    monkeypatch.setattr(models, 'construct', lambda *a: ArtificialModel())
    runtime = models.TorchRuntime(small(), 'cuda:0', inference_only=True)
    meta = identity(small(), {'manual/state.py': 'a' * 64}, 'ciphers', 'M-best', 1, 0.5)
    changed = {k: v.clone() for k, v in canonical.items()}
    changed[FIXED[1]][0, 0] = 1
    with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
        runtime.restore('M-best', encode(meta, changed), meta)
    assert not loaded and not runtime.models


def test_mutated_fixed_buffer_reload_blocks_test(tmp_path: Path, monkeypatch: Any) -> None:
    from cipher_lab_plugins.innovation1.training import run
    r, provider, runtime, store, resources = setup(tmp_path)
    canonical = fixture_state()
    observed: list[str] = []
    def restore(slot: str, raw: bytes, meta: Any) -> None:
        # Artificial same-shape mutation at selected-model reload boundary, not NN.
        changed = {k: v.clone() for k, v in canonical.items()}
        changed[FIXED[2]][0, 0] = 1
        strict_meta = identity(r, store.source_snapshot(), 'ciphers', 'M-best', 1, 0.5)
        try:
            decode(encode(strict_meta, changed), strict_meta, canonical,
                   fixed_buffers={k: canonical[k] for k in FIXED})
        except ValueError as error:
            observed.append(str(error))
            raise
    monkeypatch.setattr(runtime, 'restore', restore)
    result = run(r, provider, runtime, store, resources)
    assert result['status'] == 'FAILED' and not provider.test_calls
    assert observed == ['CHECKPOINT_INVALID']


@pytest.mark.parametrize('phase', ['after-finalize', 'read', 'render', 'write'])
@pytest.mark.parametrize('limit,reason', [('wall', 'WALL_LIMIT'), ('cpu', 'CPU_LIMIT'),
                                         ('rss', 'RSS_LIMIT')])
def test_report_budget_keeps_scientific_terminal_and_stops_command(
        tmp_path: Path, monkeypatch: Any, capsys: Any, phase: str, limit: str, reason: str) -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    from cipher_lab_plugins.innovation1 import cli, report
    from cipher_lab_plugins.innovation1.resources import Resources, policy
    from cipher_lab_plugins.innovation1.training import run

    from cipher_lab.execution import Clocks, Meter
    r, provider, runtime, store, _ = setup(tmp_path)
    now = dict(wall=0., cpu=0., rss=0)
    meter = Meter(replace(policy(r), wall_seconds=10., cpu_seconds=10., rss_stop_bytes=100),
                  Clocks(lambda: float(now['wall']), lambda: float(now['cpu']),
                         lambda: datetime(2020, 1, 1, tzinfo=UTC)), lambda: int(now['rss']))
    resources = Resources(meter)
    finalizes: list[Any] = []
    original_finalize = store.finalize
    def finalize(value: Any) -> None:
        finalizes.append(value)
        original_finalize(value)
        if phase == 'after-finalize':
            now[limit] = 1000
    monkeypatch.setattr(store, 'finalize', finalize)
    result = run(r, provider, runtime, store, resources)
    assert result['status'] == 'SUCCEEDED'
    sealed = (tmp_path / 'data/manual/run.json').read_bytes()
    if phase == 'read':
        original_read = cli._owner_record
        def read(*args: Any) -> Any:
            value = original_read(*args)
            now[limit] = 1000
            return value
        monkeypatch.setattr(cli, '_owner_record', read)
    elif phase == 'render':
        original_render = report.render
        def render(saved: Any) -> Any:
            value = original_render(saved)
            now[limit] = 1000
            return value
        monkeypatch.setattr(report, 'render', render)
    elif phase == 'write':
        original_write = store.write_report_files
        def write(files: Any) -> None:
            original_write(files)
            now[limit] = 1000
        monkeypatch.setattr(store, 'write_report_files', write)
    code = cli._finish(store, tmp_path, 'manual', resources=resources)
    import json
    output = json.loads(capsys.readouterr().out)
    assert code != 0 and output['command_status'] == 'STOPPED'
    assert output['scientific_status'] == 'SUCCEEDED'
    assert output['report_status'] != 'SAVED' and output['reason'] == reason
    assert output['report_files_written'] == (phase == 'write')
    assert output['command_resources']['wall_elapsed_seconds'] == float(now['wall'])
    assert (tmp_path / 'data/manual/run.json').read_bytes() == sealed
    assert len(finalizes) == 1 and len(provider.test_calls) == 4




@pytest.mark.parametrize('failure', ['read', 'write', 'running'])
def test_finish_failure_never_refinalizes(
        tmp_path: Path, monkeypatch: Any, capsys: Any, failure: str) -> None:
    import json

    from cipher_lab_plugins.innovation1 import cli
    from cipher_lab_plugins.innovation1.training import run

    from cipher_lab import artifacts
    r, provider, runtime, store, resources = setup(tmp_path)
    if failure != 'running':
        run(r, provider, runtime, store, resources)
    def forbidden_finalize(*args: Any) -> None:
        raise AssertionError('no second finalize')
    monkeypatch.setattr(store, 'finalize', forbidden_finalize)
    def fail(*args: Any) -> Any:
        raise artifacts.ArtifactError('READ_FAILED' if failure == 'read' else 'WRITE_FAILED')
    if failure == 'read':
        monkeypatch.setattr(cli, '_owner_record', fail)
    elif failure == 'write':
        monkeypatch.setattr(store, 'write_report_files', fail)
    assert cli._finish(store, tmp_path, 'manual', resources=resources) != 0
    output = json.loads(capsys.readouterr().out)
    assert output['command_status'] == ('FAILED' if failure == 'write' else 'UNKNOWN')
    assert output['report_status'] != 'SAVED'
    assert output['scientific_status'] == ('SUCCEEDED' if failure == 'write' else 'UNKNOWN')




def test_dedicated_dynamic_buffers_have_no_fixed_geometry() -> None:
    meta = identity(small(), {'manual/state.py': 'a' * 64}, 'ciphers',
                    'D-SIMECK-best', 1, 0.5)
    template = {'bn.running_mean': torch.tensor([0.]),
                'bn.num_batches_tracked': torch.tensor(0, dtype=torch.int64)}
    trained = {'bn.running_mean': torch.tensor([2.]),
               'bn.num_batches_tracked': torch.tensor(7, dtype=torch.int64)}
    assert int(decode(encode(meta, trained), meta, template,
                      fixed_buffers={})['bn.num_batches_tracked']) == 7
    with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
        decode(encode(meta, trained), meta, template, fixed_buffers=template)


@pytest.mark.parametrize('failure', ['read', 'render', 'write'])
def test_report_failing_io_still_samples_final_command_budget(
        tmp_path: Path, monkeypatch: Any, capsys: Any, failure: str) -> None:
    import json
    from dataclasses import replace
    from datetime import UTC, datetime

    from cipher_lab_plugins.innovation1 import cli, report
    from cipher_lab_plugins.innovation1.resources import Resources, policy
    from cipher_lab_plugins.innovation1.training import run

    from cipher_lab import artifacts
    from cipher_lab.execution import Clocks, Meter
    r, provider, runtime, store, _ = setup(tmp_path)
    now = [0.]
    resources = Resources(Meter(replace(policy(r), wall_seconds=10.),
        Clocks(lambda: now[0], lambda: 0., lambda: datetime(2020, 1, 1, tzinfo=UTC)),
        lambda: 0))
    run(r, provider, runtime, store, resources)
    sealed = (tmp_path / 'data/manual/run.json').read_bytes()
    def fail(*args: Any) -> Any:
        now[0] = 20.
        raise artifacts.ArtifactError('READ_FAILED')
    if failure == 'read':
        monkeypatch.setattr(cli, '_owner_record', fail)
    elif failure == 'render':
        monkeypatch.setattr(report, 'render', fail)
    else:
        monkeypatch.setattr(store, 'write_report_files', fail)
    assert cli._finish(store, tmp_path, 'manual', resources=resources) != 0
    output = json.loads(capsys.readouterr().out)
    assert output['command_resources']['wall_elapsed_seconds'] == 20.
    assert output['command_status'] == 'STOPPED' and output['reason'] == 'WALL_LIMIT'
    assert output['report_status'] == 'STOPPED'
    assert (tmp_path / 'data/manual/run.json').read_bytes() == sealed
