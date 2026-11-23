from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cipher_lab.execution import Meter
from cipher_lab.models.shared_m.components import GeometryStem, Residual
from cipher_lab.models.shared_m.router import Route
from cipher_lab.models.shared_m.spec import STEM_NAMES, SharedMSpec
from cipher_lab.models.shared_m.system import SharedM
from cipher_lab.representations.tasks import public_features


def _total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_cube_structure_and_parent_device() -> None:
    from cipher_lab_plugins.direct_block.cube_compare.model import CubeSharedM
    from cipher_lab_plugins.direct_block.model import DirectBlockMLP

    model = CubeSharedM()
    assert _total(SharedM(SharedMSpec(96, 96, 8, (3, 5, 7, 9, 11), 64, 16))) == 1087509
    assert _total(model) == 1876179
    assert list(model.stems) == list(STEM_NAMES)
    local = cast(GeometryStem, model.stems['local']).network.state_dict()
    for name, stem in model.stems.items():
        layer = cast(GeometryStem, stem)
        if name != 'local':
            assert all(torch.equal(value, layer.network.state_dict()[key])
                       for key, value in local.items())
        assert not bool(layer.relation_projection.weight.any())
    for stage, kernel in zip(model.suffix, (3, 5, 7, 9, 11), strict=True):
        network = cast(Residual, stage).network
        assert cast(nn.Conv1d, network[0]).kernel_size[0] == kernel
        assert cast(nn.GroupNorm, network[1]).num_groups == 8
    assert list(model.base.weight.shape) == [128, 6144]
    assert list(cast(nn.Linear, model.residual[0]).weight.shape) == [64, 6144]
    tail = cast(nn.Linear, model.residual[2])
    assert list(tail.weight.shape) == [128, 64]
    assert not bool(tail.weight.any() or tail.bias.any())
    mlp = DirectBlockMLP(input_width=128, output_width=128, device=torch.device('cpu'))
    assert _total(mlp) == 49536
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in mlp.parameters())
    with pytest.raises(ValueError, match='CPU FP32'):
        mlp(torch.zeros(4, 128, dtype=torch.float64))


def test_cube_canonical_router_suffix_and_four_slots_have_no_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.model import CubeSharedM

    model = CubeSharedM()
    rows = torch.zeros(8, 128)
    rows[torch.arange(8), torch.arange(8) % 3] = 1
    rows[torch.arange(8), 64 + torch.arange(8)] = 1
    events: list[str] = []

    class Router(nn.Module):
        def forward(self, public: Tensor) -> Route:
            events.append('router')
            torch.testing.assert_close(public, torch.tensor(public_features('SIMON')))
            return Route(torch.tensor([2, 0]), torch.tensor([.75, .25]))

    class Stem(nn.Module):
        def __init__(self, name: str, scale: int) -> None:
            super().__init__()
            self.name, self.scale = name, scale

        def forward(self, canonical: Tensor, channel: Tensor) -> Tensor:
            events.append(self.name)
            left, right = rows[:, :64], rows[:, 64:]
            want = torch.stack((left, right, (left != right).float()), 1).reshape(2, 4, 3, 64)
            torch.testing.assert_close(canonical, want)
            assert channel.shape == (2, 4, 1, 64) and not bool(channel.any())
            return (self.scale * canonical[:, :, 1:2]).expand(2, 4, 96, 64)

    class Suffix(nn.Module):
        def forward(self, hidden: Tensor) -> Tensor:
            events.append('suffix')
            assert hidden.shape == (8, 96, 64)
            torch.testing.assert_close(hidden[:, 0, :], 5 * rows[:, 64:])
            return hidden + 1

    class Head(nn.Module):
        def __init__(self, name: str, offset: int) -> None:
            super().__init__()
            self.name, self.offset = name, offset

        def forward(self, hidden: Tensor) -> Tensor:
            events.append(self.name)
            assert hidden.shape == (8, 6144)
            return torch.cat((hidden[:, :64], -hidden[:, :64]), 1) + self.offset

    monkeypatch.setattr(model, 'router', Router())
    monkeypatch.setattr(model, 'stems', nn.ModuleDict({key: Stem(key, scale) for key, scale in zip(
        STEM_NAMES, (2, 999, 6), strict=True)}))
    monkeypatch.setattr(model, 'suffix', Suffix())
    monkeypatch.setattr(model, 'base', Head('base', 0))
    monkeypatch.setattr(model, 'residual', Head('residual', 1))
    logits = cast(Tensor, model(rows))
    assert events == ['router', 'nibble', 'local', 'suffix', 'base', 'residual']
    want = 1 + 2 * torch.cat((5 * rows[:, 64:] + 1, -(5 * rows[:, 64:] + 1)), 1)
    torch.testing.assert_close(logits, want)
    assert logits.shape == (8, 128)


@pytest.mark.parametrize('bad', [torch.zeros(0, 128), torch.zeros(5, 128), torch.zeros(4, 64),
                                 torch.full((4, 128), .5), torch.full((4, 128), float('nan')),
                                 torch.zeros(4, 128, dtype=torch.float64)])
def test_cube_rejects_invalid_plaintext_before_layers(bad: Tensor) -> None:
    from cipher_lab_plugins.direct_block.cube_compare.model import CubeSharedM

    with pytest.raises(ValueError, match='plaintext'):
        CubeSharedM()(bad)


def test_cube_engine_factory_seed_and_fp32_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import engine

    seen: list[tuple[str, str, int]] = []

    initialized: dict[str, Tensor] = {}

    class FakeEngine:
        def __init__(self, model: nn.Module, device: torch.device, *, identity: str) -> None:
            assert set(p.device.type for p in model.parameters()) == {'cpu'}
            seen.append((identity, str(device), _total(model)))
            first = next(model.parameters()).detach()
            if identity in initialized:
                assert torch.equal(first, initialized[identity])
            initialized[identity] = first

    monkeypatch.setattr(engine, 'TorchEngine', FakeEngine)
    factory = engine.make_engine
    saved = torch.get_rng_state()
    a, b, again = factory('mlp'), factory('sharedm'), factory('mlp')
    assert a is not b and a is not again
    assert seen == [('direct-block-cube-sharedm-v1/mlp', 'cuda:0', 49536),
                    ('direct-block-cube-sharedm-v1/sharedm', 'cuda:0', 1876179),
                    ('direct-block-cube-sharedm-v1/mlp', 'cuda:0', 49536)]
    from cipher_lab_plugins.direct_block.model import DirectBlockMLP

    assert torch.equal(saved, torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(31)
        reference = DirectBlockMLP(input_width=128, output_width=128)
        assert torch.equal(next(reference.parameters()),
                           initialized['direct-block-cube-sharedm-v1/mlp'])
    with pytest.raises(ValueError, match='arm'):
        factory('r8')  # type: ignore[arg-type]


def test_cube_engine_precision_bootstrap_and_accounted_prediction(
    cube_meter: Meter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import engine
    from cipher_lab_plugins.direct_block.fit import predict_batch

    threads: list[tuple[str, int]] = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: threads.append(('cpu', n)))
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: threads.append(('interop', n)))
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', True)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', True)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', True)
    monkeypatch.setattr(torch, 'set_default_dtype', lambda dtype: None)
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', lambda mode: None)
    recorded = engine.configure_runtime()
    assert threads == [('cpu', 4), ('interop', 1)]
    assert recorded['matmul_tf32'] is recorded['cudnn_tf32'] is recorded['cudnn_benchmark'] is False

    class WrongPrediction:
        def predict(self, x: object) -> object:
            return np.zeros((8, 8), np.float32)

    with pytest.raises(ValueError, match='logits'):
        predict_batch(WrongPrediction(), np.zeros((8, 128), np.float32),  # type: ignore[arg-type]
                      np.zeros((8, 128), np.uint8), cube_meter)
    counts = cube_meter.snapshot()
    assert counts['attempted_forward_rows'] == 8 and counts['completed_forward_rows'] == 0


def test_cube_engine_train_forward_wires_full128_bce(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.direct_block.cube_compare import engine
    from cipher_lab_plugins.direct_block.cube_compare.config import ACTIVE_COLUMNS

    inputs = np.zeros((4, 128), np.float32)
    labels = np.zeros((4, 128), np.uint8)
    # Non-active sentinel in each half; also exercise an active site.
    assert 0 not in ACTIVE_COLUMNS and 64 not in ACTIVE_COLUMNS
    labels[0, 0], labels[1, 64], labels[2, 127] = 1, 1, 1
    logits = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128)
    loss = torch.tensor(.375, dtype=torch.float32)
    events: list[str] = []

    class ManualModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def forward(self, value: Tensor) -> Tensor:
            assert self.training
            torch.testing.assert_close(value, torch.from_numpy(inputs))
            events.append('forward')
            return logits

    model = ManualModel().eval()
    marker = next(model.parameters())

    def check_parameter(parameters: Iterable[nn.Parameter]) -> None:
        found = list(parameters)
        assert len(found) == 1 and found[0] is marker

    class ManualAdam:
        def __init__(self, parameters: Iterable[nn.Parameter], **options: object) -> None:
            check_parameter(parameters)
            assert options == {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8,
                               'weight_decay': 0., 'foreach': False, 'fused': False}
            events.append('adam-bound')

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            events.append('zero-grad')

    def bce(actual_logits: Tensor, actual_labels: Tensor) -> Tensor:
        assert actual_logits.shape == actual_labels.shape == (4, 128)
        assert actual_logits is logits
        assert actual_labels.dtype == torch.float32 and actual_labels.device.type == 'cpu'
        assert actual_labels[0, 0].item() == actual_labels[1, 64].item() == 1
        torch.testing.assert_close(actual_labels, torch.from_numpy(labels).float())
        events.append('bce')
        return loss

    def backward(value: Tensor) -> None:
        assert value is loss
        events.append('backward')

    def clip(parameters: Iterable[nn.Parameter], maximum: float,
             *, error_if_nonfinite: bool) -> None:
        check_parameter(parameters)
        assert maximum == 1. and error_if_nonfinite
        events.append('clip')

    monkeypatch.setattr(torch.optim, 'Adam', ManualAdam)
    monkeypatch.setattr(F, 'binary_cross_entropy_with_logits', bce)
    monkeypatch.setattr(torch.Tensor, 'backward', backward)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    runtime = engine.TorchEngine(model, torch.device('cpu'), identity='manual')
    complete = runtime.train_forward(inputs)
    assert events == ['adam-bound', 'zero-grad', 'forward']
    assert complete(labels) == .375
    assert events == ['adam-bound', 'zero-grad', 'forward', 'bce', 'backward', 'clip']
