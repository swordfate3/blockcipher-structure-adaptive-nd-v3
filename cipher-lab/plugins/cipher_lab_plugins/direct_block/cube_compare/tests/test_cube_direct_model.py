from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def test_direct64_model_shape_order_and_fresh_states() -> None:
    from cipher_lab_plugins.direct_block.cube_compare.lstm64 import DirectLSTM64

    model = DirectLSTM64()
    r = model.lstm
    assert type(r) is nn.LSTM and type(model.head) is nn.Linear
    assert (r.input_size, r.hidden_size, r.num_layers) == (1, 128, 2)
    assert r.batch_first and r.bias and not r.bidirectional and r.dropout == r.proj_size == 0
    assert (model.head.in_features, model.head.out_features) == (128, 64)
    assert sum(p.numel() for p in model.parameters()) == 207424
    assert not list(model.buffers())
    raw = ((np.arange(64)[None, :] >> np.arange(6)[:, None]) & 1).astype(np.float32)
    states: list[tuple[Tensor, Tensor]] = []

    class ManualLSTM(nn.Module):
        def forward(self, seq: Tensor, state: tuple[Tensor, Tensor]) -> object:
            expected = raw if len(states) != 1 else 1 - raw[:1]
            assert seq.shape == (*expected.shape, 1)
            np.testing.assert_array_equal(seq.numpy()[:, :, 0], expected)
            h, c = state
            for t in state:
                assert t.shape == (2, len(seq), 128) and t.dtype == seq.dtype
                assert t.device == seq.device and not t.numpy().any()
            states.append(state)
            assert len({t.data_ptr() for pair in states for t in pair}) == 2 * len(states)
            h.fill_(4.)
            c.fill_(5.)
            hn = seq.new_zeros((2, len(seq), 128))
            hn[-1].fill_(7.)
            return None, (hn, None)

    class ManualHead(nn.Module):
        def forward(self, hn: Tensor) -> Tensor:
            assert hn.shape == (len(hn), 128) and (hn.numpy() == 7).all()
            return hn[:, :64]

    model.lstm = ManualLSTM()  # type: ignore[assignment]
    model.head = ManualHead()  # type: ignore[assignment]
    for source in (raw, 1 - raw[:1], raw):
        assert model(torch.from_numpy(source)).shape == (len(source), 64)


@pytest.mark.parametrize('shape', [(), (64,), (2, 64, 1), (2, 63), (2, 65), (2, 128)])
def test_direct64_rejects_invalid_input_before_allocating_states_or_recurrent_call(
    shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.lstm64 import DirectLSTM64

    model = DirectLSTM64()
    inputs = torch.zeros(shape)
    placeholder = torch.zeros((2, 1, 128))
    allocations = recurrent_calls = 0

    def fake_zeros(inputs: Tensor, shape: tuple[int, ...]) -> Tensor:
        nonlocal allocations
        allocations += 1
        return placeholder

    class FakeLSTM(nn.Module):
        def forward(self, seq: Tensor, state: tuple[Tensor, Tensor]) -> object:
            nonlocal recurrent_calls
            recurrent_calls += 1
            return None, (placeholder, None)

    monkeypatch.setattr(Tensor, 'new_zeros', fake_zeros)
    model.lstm = FakeLSTM()  # type: ignore[assignment]
    model.head = nn.Identity()  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r'\[N, 64\]'):
        model(inputs)
    assert allocations == recurrent_calls == 0


def test_direct64_factory_fresh_cpu_seed_and_distinct_checkpoint_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, engine
    from cipher_lab_plugins.direct_block.cube_compare.lstm64 import DirectLSTM64

    models: list[nn.Module] = []
    identities: list[str] = []

    def manual_engine(model: nn.Module, device: torch.device, *,
                      identity: str, width: int) -> nn.Module:
        assert type(model) is DirectLSTM64 and str(device) == 'cuda:0' and width == 64
        assert next(model.parameters()).device.type == 'cpu'
        models.append(model)
        identities.append(identity)
        return model

    monkeypatch.setattr(engine, 'TorchEngine', manual_engine)
    fitted: list[object] = [engine.make_direct_engine(r) for r in (1, 2, 3)]
    assert fitted == models
    assert len({id(model) for model in models}) == 3
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(31)
        reference = DirectLSTM64()
    assert all(torch.equal(next(model.parameters()), next(reference.parameters()))
               for model in models)
    assert identities == [f'{config.DIRECT_METHOD}/r{r}' for r in (1, 2, 3)]
    with pytest.raises(ValueError):
        engine.make_direct_engine(True)


def test_direct64_runtime_width_full_bce_and_strict_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, engine
    from cipher_lab_plugins.direct_block.cube_compare.lstm import CubeLSTM
    from cipher_lab_plugins.direct_block.cube_compare.lstm64 import DirectLSTM64

    class ManualAdam:
        def __init__(self, params: Iterable[Tensor], **kwargs: object) -> None:
            assert list(params) and kwargs['lr'] == .001

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none

    monkeypatch.setattr(torch.optim, 'Adam', ManualAdam)
    cpu = torch.device('cpu')
    identity = f'{config.DIRECT_METHOD}/r1'
    first = engine.TorchEngine(DirectLSTM64(), cpu, identity=identity, width=64)
    other = engine.TorchEngine(DirectLSTM64(), cpu, identity=f'{config.DIRECT_METHOD}/r2', width=64)
    legacy = engine.TorchEngine(CubeLSTM(), cpu, identity=identity)
    inputs = np.zeros((2, 64), np.float32)
    assert first._input(inputs).shape == (2, 64)
    assert legacy._input(np.zeros((2, 128), np.float32)).shape == (2, 128)
    with pytest.raises(ValueError):
        legacy._input(inputs)
    with pytest.raises(ValueError):
        first._input(np.zeros((2, 128), np.float32))
    raw = first.dump()
    first.restore(raw)
    with pytest.raises(ValueError, match='identity'):
        other.restore(raw)
    with pytest.raises(ValueError, match='tensor'):
        legacy.restore(raw)

    class ManualModel(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x

        def __init__(self) -> None:
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(1))

    class ManualLoss:
        def detach(self) -> Tensor:
            return torch.tensor(.125)

        def isfinite(self) -> bool:
            return True

        def backward(self) -> None:
            events.append('backward')

    def fake_bce(z: Tensor, y: Tensor) -> ManualLoss:
        assert z.shape == y.shape == (2, 64) and y.dtype == torch.float32
        np.testing.assert_array_equal(y.numpy(), labels)
        events.append('bce')
        return ManualLoss()

    events: list[str] = []
    labels = (np.arange(128).reshape(2, 64) % 3 == 0).astype(np.uint8)
    monkeypatch.setattr(F, 'binary_cross_entropy_with_logits', fake_bce)
    monkeypatch.setattr(nn.utils, 'clip_grad_norm_', lambda *a, **k: events.append('clip'))
    manual = engine.TorchEngine(ManualModel(), cpu, identity=identity, width=64)
    assert manual.train_forward(inputs)(labels) == .125
    assert events == ['bce', 'backward', 'clip']
    np.testing.assert_array_equal(manual.predict(inputs), inputs)
