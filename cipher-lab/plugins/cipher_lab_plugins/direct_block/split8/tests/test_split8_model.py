from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
import torch
from torch import Tensor, nn


def test_split8_model_shape_independence_seed_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import engine
    from cipher_lab_plugins.direct_block.split8 import config, model

    parameters: list[list[Tensor]] = []

    class ManualAdam:
        def __init__(self, params: Iterable[Tensor], **kwargs: object) -> None:
            parameters.append(list(params))
            assert kwargs == dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                                  foreach=False, fused=False)

    factory = engine.TorchEngine
    models: list[model.SplitMLP] = []

    def manual_engine(net: model.SplitMLP, device: torch.device, *, identity: str,
                      width: int) -> engine.TorchEngine:
        assert str(device) == 'cuda:0' and width == 32
        assert identity == f'{config.METHOD}/r{len(models) // 4 + 1}/b{len(models) % 4}'
        models.append(net)
        return factory(net, torch.device('cpu'), identity=identity, width=width)

    monkeypatch.setattr(torch.optim, 'Adam', ManualAdam)
    monkeypatch.setattr(engine, 'TorchEngine', manual_engine)
    engines = [model.make_engine(r, j) for r in (1, 2, 3) for j in range(4)]
    assert len({p.data_ptr() for group in parameters for p in group}) == 72
    for m in models:
        assert [(layer.in_features, layer.out_features) for layer in m.layers
                if isinstance(layer, nn.Linear)] == [(32, 128), (128, 128), (128, 2)]
        assert [type(layer) for layer in m.layers] == [nn.Linear, nn.ReLU, nn.Linear,
                                                       nn.ReLU, nn.Linear]
        assert sum(p.numel() for p in m.parameters()) == 20994 and not list(m.buffers())
    for j in range(4):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(31 + j)
            ref = model.SplitMLP()
        for r in range(3):
            assert all(torch.equal(a, b) for a, b in
                       zip(ref.parameters(), models[4 * r + j].parameters(), strict=True))
    assert not torch.equal(parameters[0][0], parameters[1][0])
    raw = engines[0].dump()
    engines[0].restore(raw)
    for other in (engines[1], engines[4]):
        with pytest.raises(ValueError, match='identity'):
            other.restore(raw)
    for n in (2, 64, 128):
        with pytest.raises(ValueError):
            engines[0]._input(np.zeros((2, n), np.float32))


def test_split8_model_manual_shape_only() -> None:
    from cipher_lab_plugins.direct_block.split8 import model

    class ManualHead(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            assert x.shape == (3, 32)
            return x[:, [0, 31]]

    net = model.SplitMLP()
    net.layers = ManualHead()  # type: ignore[assignment]
    x = torch.zeros((3, 32), dtype=torch.float32)
    x[:, 31] = 1.
    assert net(x).tolist() == [[0., 1.]] * 3
    for invalid in (torch.zeros(3, 64), torch.zeros(3, 2), torch.zeros(0, 32),
                    torch.full((3, 32), .5), torch.zeros(3, 32, dtype=torch.int32)):
        with pytest.raises(ValueError):
            net(invalid)
