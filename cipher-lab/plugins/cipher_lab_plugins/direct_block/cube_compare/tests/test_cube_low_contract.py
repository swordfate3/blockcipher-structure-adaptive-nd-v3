from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch
from torch import nn

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Clocks

CFG = Path('configs/direct-block-cube-lstm-simon-r1r3.json')


def test_cube_low_fixed_budget_topology_and_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import cli, config

    actual = cli._fixed(CFG)
    assert actual == config.low_recipe()
    assert actual['rounds'] == [1, 2, 3] and actual['fits'] == 3 and actual['retries'] == 0
    assert actual['model_slots'] == ['r1-best', 'r2-best', 'r3-best']
    assert actual['key_protocol'] == 'same-new-key' and actual['mode'] == 'plaintext-only-source'
    assert actual['source'] == config.source_recipe()
    assert actual['total_caps'] == config.low_counts((50, 50, 50)) == dict(
        real_blocks=7077888, null_words=0, optimizer_updates=614400, forward_rows=167313408)
    assert actual['test_denominators'] == dict(active8=524288, full128=8388608, x64=4194304,
                                               y64=4194304, block0=4194304, block1=4194304,
                                               wholepair=65536)
    e = 18 + 19 + 50
    assert config.low_counts((18, 19, 50)) == dict(
        real_blocks=7077888, null_words=0, optimizer_updates=4096 * e,
        forward_rows=1114112 * e + 196608)
    for bad in ((18, 18), (18, 18, 51), (True, 18, 18), (18, 18, 18.0)):
        with pytest.raises(ValueError):
            config.low_counts(bad)  # type: ignore[arg-type]
    now = datetime(2026, 11, 24, tzinfo=UTC)
    clocks = Clocks(lambda: 0., lambda: 0., lambda: now)
    monkeypatch.setattr(cli, 'Clocks', lambda: clocks)
    end = now + timedelta(seconds=10800)
    p = cli._policy(end.isoformat(), clocks, method=config.LOW_METHOD)
    assert (p.wall_seconds, p.cpu_seconds, p.rss_stop_bytes, p.output_bytes) == (
        10800, 43200, 8 * 2**30, 512 * 2**20)
    assert actual['checkpoint_bytes'] == 16 * 2**20
    monkeypatch.setattr(RunStore, 'claim', lambda *a, **k: pytest.fail('claim attempted'))
    protected = tmp_path / 'models' / config.SOURCE_RUN_ID
    argv = ['run', '--config', str(CFG), '--root', str(protected), '--run-id', 'new']
    assert cli.main([*argv, '--deadline-utc', end.isoformat()]) == 2
    argv += ['--source-root', str(tmp_path)]
    assert cli.main([*argv, '--deadline-utc', (end + timedelta(seconds=1)).isoformat()]) == 2
    assert cli.main([*argv, '--deadline-utc', end.isoformat()]) == 2
    drift = tmp_path / 'bad.json'
    for override in ({'rounds': [1, 2, 4]}, {'retries': False}, {'fits': 1}):
        drift.write_text(json.dumps({**actual, **override}), encoding='utf-8')
        assert cli.main(['check', '--config', str(drift)]) == 2
    assert not protected.exists()
    script = Path('scripts/start-cube-sharedm.ps1').read_text(encoding='ascii')
    assert script.count(f"'{CFG.name}'") == 2
    assert '$wallSeconds = if ($reuse) { 10800 } else { 7200 }' in script
    assert '$code = $global:LASTEXITCODE' in script and '& $python @cliArgs' in script


def test_cube_low_factory_has_independent_seeded_parameters_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, engine

    instances: list[tuple[nn.Module, str]] = []

    class ShapeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            assert torch.default_generator.initial_seed() == config.INIT_SEED
            self.placeholder = nn.Parameter(torch.empty((2, 128)))

    def fake_engine(model: nn.Module, device: torch.device, *, identity: str) -> object:
        assert str(device) == 'cuda:0' and next(model.parameters()).device.type == 'cpu'
        instances.append((model, identity))
        return object()  # independent optimizer/runtime stand-in

    monkeypatch.setattr(engine, 'CubeLSTM', ShapeModel)
    monkeypatch.setattr(engine, 'TorchEngine', fake_engine)
    a, b, c = [engine.make_low_engine(r) for r in config.LOW_ROUNDS]
    assert len({id(a), id(b), id(c)}) == 3
    assert [identity for _, identity in instances] == [
        f'{config.LOW_METHOD}/r{n}' for n in (1, 2, 3)]
    assert len({next(model.parameters()).data_ptr() for model, _ in instances}) == 3


def test_cube_low_strict_checkpoint_rejects_other_round_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import checkpoint, config

    class ShapeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))

    model = ShapeModel()
    original = model.load_state_dict
    calls: list[bool] = []

    def load(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get('strict') is True)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model, 'load_state_dict', load)
    identities = [f'{config.LOW_METHOD}/r{r}' for r in config.LOW_ROUNDS]
    for identity in identities:
        raw = checkpoint.dump(model, identity=identity)
        for other in identities:
            if other != identity:
                with pytest.raises(ValueError, match='invalid checkpoint identity'):
                    checkpoint.restore(model, raw, identity=other)
    assert calls == []
    checkpoint.restore(model, raw, identity=identities[-1])
    assert calls == [True]
