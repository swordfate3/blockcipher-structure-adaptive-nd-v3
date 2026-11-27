from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from cipher_lab_plugins.innovation1.checkpoint import encode
from cipher_lab_plugins.innovation1.prefix_bridge import runtime as bridge_runtime
from cipher_lab_plugins.innovation1.present_card_fresh import config as pure_config
from cipher_lab_plugins.innovation1.present_card_fresh import model as pure
from cipher_lab_plugins.innovation1.present_card_fresh import runtime as pure_runtime
from cipher_lab_plugins.innovation1.present_card_fresh.layout import Q
from cipher_lab_plugins.innovation1.present_card_p_layer_fusion import model, runtime, wiring
from cipher_lab_plugins.innovation1.present_card_swap import runtime as swap_runtime
from cipher_lab_plugins.innovation1.present_card_swap.views import swap_half
from torch import nn

from cipher_lab.execution import Meter


def test_wiring() -> None:
    base, shuffled, gather = map(wiring.gather_for, (wiring.BASE, wiring.SHUFFLED, wiring.P_LAYER))
    assert base == ()
    assert tuple(gather[q] for q in Q[:64]) == tuple(range(64))
    renaming = sorted(range(64), key=lambda i: sha256(
        b'present-p-layer-fusion-shuffle-v1/' + bytes([i])).digest())
    assert all(shuffled[renaming[i]] == renaming[gather[i]] for i in range(64))
    assert shuffled != gather
    for slot, indices in zip((wiring.BASE, wiring.SHUFFLED, wiring.P_LAYER),
                             (base, shuffled, gather), strict=True):
        assert wiring.wiring_sha256(slot) == sha256(bytes(indices)).hexdigest()
    with pytest.raises(ValueError, match='INVALID_SLOT'):
        wiring.gather_for('invalid')


@torch.inference_mode()
def test_fused_graph_and_zero_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the numerical guard in force with a CPU stub for the diagonal fixture.
    def diagonal_projection(layer: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return x * layer.weight.squeeze(-1).diagonal().view(1, 96, 1)

    monkeypatch.setattr(nn.Conv1d, 'forward', diagonal_projection)
    reference = pure.build().eval()
    base = model.build(wiring.BASE).eval()
    h = torch.arange(2 * 96 * 64, dtype=torch.float32).reshape(2, 96, 64)
    assert type(base) is type(reference)
    assert type(base.pair.shared.suffix) is type(reference.pair.shared.suffix)
    for slot in (wiring.BASE, wiring.SHUFFLED, wiring.P_LAYER):
        rng = torch.random.get_rng_state()
        net = model.build(slot).eval()
        assert torch.equal(torch.random.get_rng_state(), rng)
        old, new = reference.state_dict(), net.state_dict()
        assert all(torch.equal(tensor, new[name]) for name, tensor in old.items())
        if slot == wiring.BASE:
            assert new.keys() == old.keys()
            continue
        suffix = net.pair.shared.suffix
        assert isinstance(suffix, model.FusedSuffix)
        assert tuple(suffix.children())[0] is suffix.fusion
        fusion = suffix.fusion
        assert isinstance(fusion.projection, nn.Conv1d)
        assert fusion.projection.bias is None
        assert fusion.projection.weight.shape == (96, 96, 1)
        assert set(new) - set(old) == {
            'pair.shared.suffix.fusion.projection.weight', 'pair.shared.suffix.fusion.gather'}
        assert not torch.count_nonzero(fusion.projection.weight)
        assert torch.equal(fusion(h), h)
        fusion.projection.weight.copy_(torch.eye(96).unsqueeze(-1))
        mixed = h + h.index_select(-1, torch.tensor(wiring.gather_for(slot)))
        assert torch.equal(fusion(h), mixed)
        assert tuple(suffix.children())[1:] == tuple(suffix._modules[str(i)] for i in range(
            len(reference.pair.shared.suffix)))


def test_one_arm_local_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = np.zeros((pure_config.BATCH_SIZE, 1, 2, 64), dtype=np.float32)
    inputs[:, :, 1] = 1
    labels = np.zeros((len(inputs), 1), dtype=np.uint8)
    expected = swap_half(inputs, seed=swap_runtime._SWAP_SEED, step=2)
    assert 0 < int(expected[:, 0, 0, 0].sum()) < len(inputs)
    ledger = SimpleNamespace(snapshot=lambda: dict(completed_optimizer_updates=102,
                                                   attempted_optimizer_updates=103))
    seen = []

    def fake_update(self: runtime.Runtime, x: np.ndarray, y: np.ndarray,
                    meter: object) -> float:
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_array_equal(y, labels)
        assert self._learning_rate(cast(Meter, meter)) == pure_config.learning_rate(2)
        seen.append(self.slot)
        return 0.25

    monkeypatch.setattr(pure_runtime.Runtime, 'batch_update', fake_update)
    for slot in wiring.SLOTS:
        arm = runtime.Runtime(slot, origin=100)
        assert arm.batch_update(inputs, labels, cast(Meter, ledger)) == 0.25
    assert tuple(seen) == wiring.SLOTS
    assert not inputs[:, :, 0].any()


def test_checkpoint_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep bounded archive preflight and validation, with a guarded in-memory loader.
    archives: dict[bytes, dict[str, object]] = {}

    def fixture_encode(metadata: dict[str, object], state: dict[str, torch.Tensor]) -> bytes:
        blob = encode(metadata, state)
        archives[blob] = {'metadata': metadata, 'state': dict(state)}
        return blob

    def fixture_load(stream: BytesIO, *, map_location: str,
                     weights_only: bool) -> dict[str, object]:
        assert map_location == 'cpu' and weights_only
        return archives[stream.getvalue()]

    monkeypatch.setattr(bridge_runtime, 'encode', fixture_encode)
    monkeypatch.setattr(torch, 'load', fixture_load)
    # The assembly owner supplies the real config. Exercise its contract with a stub.
    monkeypatch.setattr(runtime, '_recipe_identity', lambda: (
        'innovation1-present-card-p-layer-fusion-v1', {'fixture': True}))
    source = {'public/model.py': 'a' * 64}
    point = dict(epoch=1, step=pure_config._steps(1), bce=0.69)
    records = {}
    for slot in wiring.SLOTS:
        arm: dict[str, object] = {'dev': [point], 'selected': point}
        writer = runtime.FitRuntime(slot, arm, False, source=source, run_id='fixture', origin=0)
        net = writer._runtime._fresh_cpu()
        arm.update(initial_sha256=writer.initial_sha, parameters=writer.parameters,
                   geometry=writer.geometry)
        writer._runtime.models[slot] = net
        writer._runtime._state = 'TRAINING'
        raw = writer.dump()
        reader = runtime.FitRuntime(slot, arm, False, source=source, run_id='fixture', origin=19)
        reader.restore(raw)
        assert reader.geometry == writer.geometry == writer.backbone_geometry
        assert reader.initial_sha == writer.initial_sha
        assert reader.fusion_initial_sha == writer.fusion_initial_sha
        assert reader.backbone_initial_sha == writer.backbone_initial_sha
        records[slot] = writer, arm, raw

    base, shuffled, public = (records[slot][0] for slot in wiring.SLOTS)
    assert base.backbone_initial_sha == shuffled.backbone_initial_sha == public.backbone_initial_sha
    assert base.initial_sha == base.backbone_initial_sha
    assert base.fusion_initial_sha == ''
    assert shuffled.fusion_initial_sha == public.fusion_initial_sha != ''
    assert len({base.initial_sha, shuffled.initial_sha, public.initial_sha}) == 3
    assert shuffled.parameters == public.parameters == base.parameters + 9216

    writer, arm, raw = records[wiring.P_LAYER]

    def restore(blob: bytes, *, slot: str = wiring.P_LAYER, run_id: str = 'fixture',
                bound_source: dict[str, str] | None = None,
                bound_arm: dict[str, object] | None = None) -> None:
        reader = runtime.FitRuntime(slot, arm if bound_arm is None else bound_arm, False,
                                    source=source if bound_source is None else bound_source,
                                    run_id=run_id, origin=0)
        with pytest.raises(ValueError):
            reader.restore(blob)
        assert not reader._runtime.models

    restore(raw, slot=wiring.SHUFFLED)
    restore(raw, run_id='another-run')
    restore(raw, bound_source={'public/model.py': 'b' * 64})
    restore(raw, bound_arm={**arm, 'selected': {**point, 'step': 1}})
    metadata = writer._metadata(point)
    state = writer._runtime.models[wiring.P_LAYER].state_dict()
    restore(fixture_encode({**metadata, 'recipe': {}}, state))
    restore(fixture_encode({**metadata, 'wiring_sha256': wiring.wiring_sha256(wiring.SHUFFLED)},
                           state))
    geometry = {**writer.geometry, 'public': 'a' * 64}
    restore(fixture_encode({**metadata, 'geometry': geometry}, state),
            bound_arm={**arm, 'geometry': geometry})
    wrong_initial = 'b' * 64
    restore(fixture_encode({**metadata, 'initial_sha256': wrong_initial}, state),
            bound_arm={**arm, 'initial_sha256': wrong_initial})
    for key in ('pair.shared.suffix.fusion.gather', 'pair.shared.stems.nibble.indices'):
        restore(fixture_encode(metadata, {**state, key: state[key].roll(1, 0)}))
