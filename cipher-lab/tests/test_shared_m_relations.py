"""Static and fake-component checks only; real neural forward is not authorized."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import Tensor, nn

from cipher_lab.models.shared_m.components import Residual
from cipher_lab.models.shared_m.router import PublicRouter, Route
from cipher_lab.representations.neighborhoods import gather_neighborhood, neighborhood_indices
from cipher_lab.representations.tasks import public_features

if TYPE_CHECKING:
    from cipher_lab.models.shared_m_relations import SharedMRelations


@pytest.fixture(autouse=True)
def forbid_real_neural_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> Tensor:
        raise AssertionError("real neural forward is forbidden in Task2 tests")

    for cls in (nn.Conv1d, nn.Linear, nn.GroupNorm):
        monkeypatch.setattr(cls, "forward", forbidden)


def test_model_module_exists() -> None:
    path = Path(__file__).resolve().parents[1] / "src/cipher_lab/models/shared_m_relations.py"
    assert path.is_file(), "independent SharedMRelations model module is not implemented"


@pytest.fixture
def model() -> SharedMRelations:
    from cipher_lab.models.shared_m_relations import SharedMRelations

    return SharedMRelations()


def card() -> Tensor:
    return torch.tensor(public_features("SIMON"))


def test_fixed_constructor_and_exact_parameter_partition(model: SharedMRelations) -> None:
    assert str(inspect.signature(type(model))) == "() -> 'None'"
    assert sum(p.numel() for p in model.parameters()) == 1103547
    assert {name: sum(p.numel() for p in component.parameters())
            for name, component in model.named_children()} == {
                "stems": 37152, "router": 275, "suffix": 648000, "readout": 418120,
            }
    assert isinstance(model.router, PublicRouter)
    assert cast(nn.Linear, model.router.gate[0]).in_features == 13
    assert cast(nn.Linear, model.router.gate[0]).out_features == 16
    assert [cast(nn.Conv1d, cast(Residual, block).network[0]).kernel_size
            for block in model.suffix] == [(3,), (5,), (7,), (9,), (11,)]
    assert all(isinstance(block, Residual) for block in model.suffix)


def test_stem_geometry_networks_and_copied_but_unshared_initial_weights(
    model: SharedMRelations,
) -> None:
    assert list(model.stems) == ["local", "word", "nibble"]
    local = model.stems["local"].get_submodule("network").state_dict()
    for name, stem in model.stems.items():
        assert torch.equal(stem.get_buffer("indices"), neighborhood_indices(name))
        network = cast(nn.Sequential, stem.get_submodule("network"))
        assert [type(layer) for layer in network] == [
            nn.Conv1d, nn.GroupNorm, nn.ReLU, nn.Conv1d, nn.GroupNorm, nn.ReLU,
        ]
        first, second = cast(nn.Conv1d, network[0]), cast(nn.Conv1d, network[3])
        assert (first.in_channels, first.out_channels, first.kernel_size) == (27, 96, (1,))
        assert (second.in_channels, second.out_channels, second.kernel_size) == (96, 96, (1,))
        assert cast(nn.GroupNorm, network[1]).num_groups == 8
        assert cast(nn.GroupNorm, network[4]).num_groups == 8
        for key, value in network.state_dict().items():
            assert torch.equal(value, local[key])
        if name != "local":
            assert first.weight.data_ptr() != cast(
                nn.Conv1d, model.stems["local"].get_submodule("network.0")
            ).weight.data_ptr()
    assert not any("relation_projection" in name for name, _ in model.named_modules())


def test_readout_shapes_and_zero_residual_last_layer(model: SharedMRelations) -> None:
    head = model.readout
    assert (head.base.in_features, head.base.out_features) == (6144, 4)
    first, last = cast(nn.Linear, head.residual[0]), cast(nn.Linear, head.residual[2])
    assert (first.in_features, first.out_features) == (6144, 64)
    assert isinstance(head.residual[1], nn.ReLU)
    assert (last.in_features, last.out_features) == (64, 4)
    assert not bool(last.weight.any())
    assert last.bias is not None and not bool(last.bias.any())


class FakeProjection(nn.Module):
    """Keep a dtype/device anchor, record gathers; never run a real convolution."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.offset = offset
        self.seen: list[Tensor] = []

    def forward(self, gathered: Tensor) -> Tensor:
        self.seen.append(gathered.clone())
        encoded = gathered[:, 4:5] + 2 * gathered[:, 13:14] + 4 * gathered[:, 22:23]
        return encoded.expand(-1, 96, -1) + self.offset


class FakeRouter(PublicRouter):
    def __init__(self) -> None:
        super().__init__(16)
        self.seen: list[Tensor] = []

    def forward(self, public: Tensor) -> Route:
        self.seen.append(public.clone())
        # Deliberately reverse geometry order and use unequal weights.
        return Route(torch.tensor([2, 0]), public.new_tensor([0.25, 0.75]))


class FakeHeadBranch(nn.Module):
    def __init__(self, multiplier: float) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.seen: list[Tensor] = []

    def forward(self, flat: Tensor) -> Tensor:
        assert flat.shape[-1] == 6144
        self.seen.append(flat.clone())
        # Four distinct outputs, with sensitivity to both MSB0 endpoints.
        value = flat[..., 0] + 10 * flat[..., 63]
        return self.multiplier * torch.stack((value, value + 1, value + 2, value + 3), -1)


def fake_components(model: SharedMRelations) -> tuple[
    dict[str, FakeProjection], FakeRouter, FakeHeadBranch, FakeHeadBranch,
]:
    projections: dict[str, FakeProjection] = {}
    for index, (name, stem) in enumerate(model.stems.items()):
        projection = FakeProjection(float(index * 10))
        stem.set_submodule("network", nn.Sequential(projection))
        projections[name] = projection
    router = FakeRouter()
    model.router = router
    model.suffix = nn.Sequential(nn.Identity())
    base, residual = FakeHeadBranch(1), FakeHeadBranch(2)
    model.readout.base = base  # type: ignore[assignment]
    model.readout.residual = residual  # type: ignore[assignment]
    return projections, router, base, residual


@pytest.mark.parametrize("dtype", [
    torch.float32, torch.float64, torch.int64, torch.uint8, torch.bool,
])
def test_fake_forward_raw_xor_msb0_top2_and_independent_pairs(
    model: SharedMRelations, dtype: torch.dtype,
) -> None:
    projections, router, base, residual = fake_components(model)
    pairs = torch.zeros(2, 4, 2, 64, dtype=dtype)
    pairs[0, 1, 0, 0] = 1
    pairs[0, 2, 1, 0] = 1
    pairs[0, 3, :, 0] = 1
    pairs[1, :, 0, 63] = 1
    public = card()
    output = model(pairs, public)
    assert output.shape == (2, 4, 4)
    assert len(router.seen) == 1 and torch.equal(router.seen[0], public)
    raw = torch.cat((pairs.float(), (pairs[:, :, :1] != pairs[:, :, 1:]).float()), dim=2)
    for name in ("local", "nibble"):
        assert torch.equal(projections[name].seen[0],
                           gather_neighborhood(raw, neighborhood_indices(name)))
    assert not projections["word"].seen
    # Fake mixed feature = endpoint0 + 2*endpoint1 + 4*XOR + 5.
    encoded = raw[:, :, 0] + 2 * raw[:, :, 1] + 4 * raw[:, :, 2] + 5
    value = encoded[:, :, 0] + 10 * encoded[:, :, 63]
    expected = 3 * torch.stack((value, value + 1, value + 2, value + 3), -1)
    assert torch.equal(output, expected)
    assert base.seen[0].shape[-1] == residual.seen[0].shape[-1] == 6144
    assert torch.equal(model(pairs[:, [3, 1, 0, 2]], public), output[:, [3, 1, 0, 2]])
    assert torch.equal(torch.cat([model(pairs[i:i + 1], public) for i in range(2)]), output)
    changed = pairs.clone()
    changed[0, 0, 1, 63] = 1
    altered = model(changed, public)
    assert not torch.equal(altered[0, 0], output[0, 0])
    assert torch.equal(altered[0, 1:], output[0, 1:])
    assert torch.equal(altered[1], output[1])
    repeated = pairs[:1, :1].expand(1, 4, 2, 64)
    equal = model(repeated, public)
    assert torch.equal(equal, equal[:, :1].expand_as(equal))


@pytest.mark.parametrize("bad", [
    torch.zeros(1, 512), torch.zeros(0, 4, 2, 64), torch.zeros(1, 3, 2, 64),
    torch.full((1, 4, 2, 64), 0.5), torch.full((1, 4, 2, 64), float("nan")),
    torch.full((1, 4, 2, 64), float("inf")), torch.zeros(1, 4, 2, 64).to_sparse(),
    torch.zeros(1, 4, 2, 64, dtype=torch.complex64),
])
def test_invalid_pairs_fail_before_neural_forward(model: SharedMRelations, bad: Tensor) -> None:
    with pytest.raises(ValueError):
        model(bad, card())


@pytest.mark.parametrize("bad", [
    torch.zeros(4, 13), torch.zeros(14), torch.zeros(13),
    torch.zeros(13, dtype=torch.int64), torch.full((13,), float("nan")),
    torch.full((13,), float("inf")), torch.zeros(13).to_sparse(),
    torch.zeros(13, dtype=torch.complex64),
])
def test_invalid_public_fails_even_with_fake_router(model: SharedMRelations, bad: Tensor) -> None:
    fake_components(model)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, 2, 64), bad)


def test_public_dtype_and_device_mismatch_fail_before_neural_forward(
    model: SharedMRelations,
) -> None:
    with pytest.raises(ValueError, match="dtype"):
        model(torch.zeros(1, 4, 2, 64), card().double())
    # Meta is a synthetic device mismatch, not a GPU allocation or numerical forward.
    model.to("meta")
    with pytest.raises(ValueError, match="device"):
        model(torch.zeros(1, 4, 2, 64), card())


@pytest.mark.parametrize("bad", [torch.zeros(2, 4), torch.full((8, 4), float("nan"))])
def test_reject_invalid_fake_output(model: SharedMRelations, bad: Tensor) -> None:
    fake_components(model)

    class BadReadout(nn.Module):
        def forward(self, hidden: Tensor) -> Tensor:
            return bad

    model.readout = BadReadout()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="logits"):
        model(torch.zeros(2, 4, 2, 64), card())


def test_pure_model_boundaries_and_no_pair_pooling(model: SharedMRelations) -> None:
    source = Path(inspect.getfile(type(model))).read_text()
    tree = ast.parse(source)
    allowed = {
        "__future__", "copy", "typing", "torch",
        "cipher_lab.models.shared_m.components", "cipher_lab.models.shared_m.router",
        "cipher_lab.models.shared_m.spec", "cipher_lab.representations.neighborhoods",
        "cipher_lab.representations.tasks",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not node.level and node.module in allowed
        elif isinstance(node, ast.Import):
            assert all(alias.name in allowed for alias in node.names)
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"mean", "relation_projection", "input_view", "backstep"}
    for forbidden in ("InputRepresentation", "public_backstep", "relation_projection",
                      "input_view", "cipher_lab.ciphers", "cipher_lab.training"):
        assert forbidden not in source
