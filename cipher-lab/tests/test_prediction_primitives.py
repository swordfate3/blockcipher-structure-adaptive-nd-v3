"""Caller-bound primitives, using artificial outputs only."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cipher_lab_plugins.prediction_common.tests.guards import science_guard

__all__ = ['science_guard']
pytestmark = pytest.mark.usefixtures('science_guard')


def test_initializer_accepts_caller_factory_without_changing_cpu_rng() -> None:
    from cipher_lab.training.steps import initialized_network
    rng = torch.get_rng_state().clone()
    calls = []
    def factory() -> torch.nn.Module:
        calls.append('factory')
        return torch.nn.Identity()
    assert type(initialized_network(factory, 7)) is torch.nn.Identity
    assert calls == ['factory'] and torch.equal(rng, torch.get_rng_state())


def test_metered_evaluation_accepts_bound_forward_and_absolute_conversion() -> None:
    from test_training_metrics import meter

    from cipher_lab.evaluation.metrics import TrainBaseline, evaluate_bound_panels
    model = torch.nn.Identity()
    model.train()
    seen = []
    def forward(name: str, batch: torch.Tensor) -> torch.Tensor:
        seen.append((name, len(batch), torch.is_grad_enabled()))
        return torch.zeros(len(batch), 1)
    output: dict[str, torch.Tensor] = {}
    ledger = meter()
    result = evaluate_bound_panels(model, ('toy',),
        {'toy': (torch.zeros(3, 64), torch.ones(3, 1))}, {'toy': TrainBaseline(2, 1)},
        ledger, 2, forward=forward, absolute=lambda name, logits: logits + 1, outputs=output)
    assert seen == [('toy', 2, False), ('toy', 1, False)]
    assert model.training and result['rows'] == 3
    assert torch.equal(output['toy'], torch.ones(3, 1))
    assert ledger.snapshot()['completed_forward_rows'] == 3


def test_guard_uses_claim_supplier_after_plugin_source_change(tmp_path: Path) -> None:
    from cipher_lab_plugins.prediction_common.metrics import PublicModelSpec
    from cipher_lab_plugins.prediction_common.steps import IdentityChanged, TrainingRecord
    from cipher_lab_plugins.residual.tasks import ResidualPredictionTask
    from cipher_lab_plugins.residual.tests.test_residual_config import residual_recipe
    from test_training_steps import make_meter

    from cipher_lab.artifacts import RunStore
    recipe = residual_recipe()
    source = {'cipher_lab_plugins/residual/cli.py': 'a'*64}
    store = RunStore.claim(tmp_path, 'guard', output_bytes=100000, needs_models=True,
                           source_supplier=lambda: dict(source))
    task = ResidualPredictionTask('SIMON', recipe.rounds, recipe.target_bit,
                                  recipe.label_mode, recipe.reference)
    record = TrainingRecord(recipe, PublicModelSpec(recipe.model, (task,)),
                            make_meter(), store, 'demo')
    record.begin()
    source['cipher_lab_plugins/residual/cli.py'] = 'b'*64
    with pytest.raises(IdentityChanged):
        record.guard()


@pytest.mark.parametrize('owner,filename', [('residual', 'cli.py'),
    ('feature_probe', 'model.py'), ('frontier', 'simon_frontier.py'),
    ('prediction_common', 'checkpoints.py')])
def test_source_closure_covers_necessary_plugin_and_shared_modules(owner: str, filename: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.prediction_common.source import source_identity

    import cipher_lab.artifacts as artifacts
    monkeypatch.setattr(artifacts, 'source_identity', lambda: {'cipher_lab/fake.py': 'a'*64})
    before = source_identity()
    key = f'cipher_lab_plugins/{owner}/{filename}'
    assert key in before
    original = Path.read_bytes
    def changed(path: Path) -> bytes:
        raw = original(path)
        return raw + b'\n# artificial source mutation' if path.as_posix().endswith(
            f'/{owner}/{filename}') else raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    assert source_identity()[key] != before[key]
    assert not any('/tests/' in name for name in before)


@pytest.mark.parametrize('owner', ['feature_probe', 'frontier'])
def test_wrapper_initialization_preserves_layers_state_keys_and_parameter_order(owner: str) -> None:
    from cipher_lab_plugins.feature_probe.config import FeatureMLPSpec
    from cipher_lab_plugins.feature_probe.model import FeatureBitMLP
    from cipher_lab_plugins.frontier.config import FrontierMLPSpec
    from cipher_lab_plugins.frontier.model import FrontierBitMLP
    from cipher_lab_plugins.frontier.tests.frontier_fakes import plan

    from cipher_lab.training.steps import initialized_network
    if owner == 'feature_probe':
        dimension = 13
        model = initialized_network(lambda: FeatureBitMLP(
            FeatureMLPSpec(2, 'simon-r3-residual13-v1')), 31)
    else:
        public_plan = plan()
        dimension = public_plan.dimension
        model = initialized_network(lambda: FrontierBitMLP(
            FrontierMLPSpec(2, public_plan.identity, public_plan)), 31)
    # This is the original ordered layer constructor, not a forward or optimizer.
    expected = initialized_network(lambda: torch.nn.Sequential(
        torch.nn.Linear(dimension, 2), torch.nn.ReLU(), torch.nn.Linear(2, 2),
        torch.nn.ReLU(), torch.nn.Linear(2, 1)), 31)
    state = expected.state_dict()
    assert list(model.state_dict()) == ['layers.'+key for key in state]
    assert all(torch.equal(value, state[key.removeprefix('layers.')])
               for key, value in model.state_dict().items())
    assert list(dict(model.named_modules())) == ['', 'layers', 'layers.0', 'layers.1',
                                                 'layers.2', 'layers.3', 'layers.4']


@pytest.mark.parametrize('owner', ['residual', 'feature_probe', 'frontier'])
def test_base_strict_loaders_reject_plugin_formats_before_torch_load(owner: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import importlib

    from cipher_lab_plugins.prediction_common.bundles import encode_bundle
    from cipher_lab_plugins.prediction_common.checkpoints import Selection, encode_checkpoint
    from cipher_lab_plugins.prediction_common.metrics import PublicModelSpec, TrainBaseline
    from cipher_lab_plugins.prediction_common.steps import initialized_model

    from cipher_lab.data.bundles import decode_bundle
    from cipher_lab.evaluation.checkpoints import load_checkpoint
    from cipher_lab.training_config import MLPSpec
    module = importlib.import_module(
        f'cipher_lab_plugins.{owner}.tests.test_{owner}_formats')
    panel = module.bundle()
    if owner == 'frontier':
        checkpoint = module.content('C')
    else:
        spec = (module.spec('C') if owner == 'feature_probe' else
                PublicModelSpec(MLPSpec(2), (module.task(),), 'residual'))
        checkpoint = encode_checkpoint(initialized_model(spec, 31), spec,
            spec.task_declarations, Selection(1, 1, .7), {'SIMON': TrainBaseline(2, 1)}, 'a'*64)
    with pytest.raises(ValueError):
        decode_bundle(encode_bundle(panel))
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError('base invoked torch.load on external format')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, hashlib.sha256(checkpoint).hexdigest())


def test_source_identity_fails_closed_on_source_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.prediction_common.source import source_identity

    import cipher_lab.artifacts as artifacts
    from cipher_lab.artifacts import ArtifactError
    monkeypatch.setattr(artifacts, 'source_identity', lambda: {'cipher_lab/fake.py': 'a'*64})
    def unreadable(path: Path) -> bytes:
        raise OSError('synthetic source read failure')
    monkeypatch.setattr(Path, 'read_bytes', unreadable)
    with pytest.raises(ArtifactError):
        source_identity()
