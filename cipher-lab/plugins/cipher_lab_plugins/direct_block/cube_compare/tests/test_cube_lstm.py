from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import Tensor, nn


@pytest.mark.parametrize('entry', ['LSTM', 'RNN', 'GRU', 'lstm', 'gru', 'lstm_cell', 'rnn_tanh',
                                   'rnn_relu', 'gru_cell', 'rnn_tanh_cell', 'rnn_relu_cell',
                                   'dispatch', 'cached', 'torch-lstm', 'variable-functions',
                                   'packet', 'input', 'data', 'bound-call'])
def test_lstm_guard_blocks_before_harmless_backend(
    cube_rnn_probe: tuple[list[str], dict[str, Callable[..., object]]], entry: str,
) -> None:
    calls, handles = cube_rnn_probe
    # Even with the guard missing, every route ends at a harmless stub.
    with pytest.raises(AssertionError, match='scientific or external operation forbidden'):
        layers = {'LSTM': nn.LSTM, 'RNN': nn.RNN, 'GRU': nn.GRU}
        if entry in layers:
            # Tiny CPU placeholder, never passed to a real recurrent kernel:
            # the guard or, in RED, the harmless fixture's forward intercepts it.
            layers[entry](2, 3).forward(torch.empty((1, 1, 2)))
        elif entry == 'dispatch':
            torch.ops.aten.lstm(None)
        elif entry == 'cached':
            getattr(nn.modules.rnn, '_rnn_impls')['RNN_TANH'](None)
        elif entry == 'torch-lstm':
            getattr(torch, 'lstm')(None)
        elif entry == 'variable-functions':
            getattr(torch._C, '_VariableFunctions').lstm(None)
        elif entry in handles:
            handles[entry](None)
        else:
            getattr(torch._VF, entry)(None)
    assert not calls


def test_cube_lstm_structure_and_manual_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.lstm import CubeLSTM

    model = CubeLSTM()
    recurrent = model.lstm
    assert type(recurrent) is nn.LSTM and type(model.head) is nn.Linear
    assert (recurrent.input_size, recurrent.hidden_size, recurrent.num_layers) == (2, 128, 2)
    assert recurrent.batch_first and recurrent.bias and not recurrent.bidirectional
    assert recurrent.dropout == recurrent.proj_size == 0
    assert (model.head.in_features, model.head.out_features) == (128, 128)
    assert sum(p.numel() for p in model.parameters()) == 216192
    assert not list(model.buffers())
    assert {name: tuple(p.shape) for name, p in recurrent.named_parameters()} == {
        f'{key}_l{layer}': shape for layer, size in enumerate((2, 128))
        for key, shape in (('weight_ih', (512, size)), ('weight_hh', (512, 128)),
                           ('bias_ih', (512,)), ('bias_hh', (512,)))
    }
    states: list[tuple[Tensor, Tensor]] = []
    heads: list[Tensor] = []
    # Bit-position signatures across samples distinguish all 64 time indices.
    x = ((np.arange(64)[None, :] >> np.arange(6)[:, None]) & 1).astype(np.float32)
    raw = np.concatenate((x, 1 - x), axis=1)

    def fake_recurrent(
        seq: Tensor, state: tuple[Tensor, Tensor],
    ) -> tuple[None, tuple[Tensor, None]]:
        assert seq.shape == (len(seq), 64, 2) and seq.dtype == torch.float32
        pairs = seq.numpy()
        expected = raw if len(seq) == 6 else 1 - raw[:1]
        for i in range(64):
            np.testing.assert_array_equal(pairs[:, i, :], expected[:, [i, 64 + i]])
        h, c = state
        assert h.shape == c.shape == (2, len(seq), 128)
        assert h.dtype == c.dtype == seq.dtype and h.device == c.device == seq.device
        assert not h.numpy().any() and not c.numpy().any()
        pointers = {t.data_ptr() for pair in [*states, state] for t in pair}
        assert len(pointers) == 2 * (len(states) + 1)
        states.append(state)
        h.fill_(3.)  # deliberately dirty the inputs and return different states
        c.fill_(4.)
        hn = seq.new_zeros((2, len(seq), 128))
        hn[0].fill_(10.)
        hn[1].fill_(20. if len(seq) == 6 else 30.)
        return None, (hn, None)

    def fake_head(last: Tensor) -> Tensor:
        assert last.shape == (len(last), 128)
        assert (last.numpy() == (20. if len(last) == 6 else 30.)).all()
        heads.append(last)
        return last

    # Replace the modules themselves, never undo the nn.LSTM or Linear guards.
    class ManualRecurrent(nn.Module):
        forward = staticmethod(fake_recurrent)

    class ManualHead(nn.Module):
        forward = staticmethod(fake_head)

    monkeypatch.setattr(model, 'lstm', ManualRecurrent())
    monkeypatch.setattr(model, 'head', ManualHead())
    first = model.forward(torch.from_numpy(raw))
    middle = model.forward(torch.from_numpy(1 - raw[:1]))
    repeated = model.forward(torch.from_numpy(raw))
    assert [len(out) for out in (first, middle, repeated)] == [6, 1, 6]
    assert first is heads[0] and middle is heads[1] and repeated is heads[2]
    np.testing.assert_array_equal(first.numpy(), repeated.numpy())


def test_lstm_factory_reuses_torch_engine_without_legacy_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import config, engine
    from cipher_lab_plugins.direct_block.cube_compare.lstm import CubeLSTM

    def fake_engine(model: nn.Module, device: torch.device, *, identity: str) -> object:
        assert type(model) is CubeLSTM and next(model.parameters()).device.type == 'cpu'
        assert str(device) == 'cuda:0' and identity == f'{config.LSTM_METHOD}/lstm'
        return sentinel

    sentinel = object()
    monkeypatch.setattr(engine, 'TorchEngine', fake_engine)
    for name in ('DirectBlockMLP', 'CubeSharedM', 'make_engine'):
        monkeypatch.setattr(engine, name, lambda *a, **k: pytest.fail('legacy model constructed'))
    assert engine.make_lstm_engine() is sentinel
