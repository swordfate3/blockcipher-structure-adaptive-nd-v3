"""Explicit provider/backend assembly only; methods retain scientific scheduling.

Called inside CLI's post-claim setup boundary. No production imports on demo.
"""
from __future__ import annotations

from collections.abc import Callable

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab_plugins.feature_probe.config import FeatureProbeRecipe
from cipher_lab_plugins.frontier.config import FrontierRecipe
from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan
from cipher_lab_plugins.residual.config import ResidualPredictionRecipe


def prepare_training(recipe: ResidualPredictionRecipe | FeatureProbeRecipe | FrontierRecipe,
                     policy_meter: Meter, store: RunStore,
                     backend: str, *, plan: PublicFeaturePlan | None = None
                     ) -> Callable[[], object]:
    from cipher_lab_plugins.prediction_common.data import PredictionData

    if backend == 'demo':
        from cipher_lab.demo import DemoBytes, fake_encrypt_many

        private_random: Callable[[int], bytes] = DemoBytes()
        oracles = {name: fake_encrypt_many for name in ('SIMECK', 'PRESENT', 'SIMON', 'SKINNY')}
        # Public zero-master-key reference: the demo fake never becomes a cipher.
        reference_oracle = fake_encrypt_many
    else:
        from secrets import token_bytes

        from cipher_lab.ciphers.simon import encrypt_many

        private_random = token_bytes
        oracles = {'SIMON': encrypt_many}
        # Data always supplies key=(0,0,0,0) for the reference, i.e. a zero
        # MASTER key with a real public-key schedule, never zero round keys.
        reference_oracle = encrypt_many
    if isinstance(recipe, FeatureProbeRecipe):
        from cipher_lab_plugins.feature_probe.training import run_feature_probe

        # One data owner: one private key, one panel and one cached public h.
        probe_data = PredictionData(recipe, oracles['SIMON'], private_random, policy_meter,
                                    reference_oracle=reference_oracle)
        return lambda: run_feature_probe(recipe, probe_data, policy_meter, store,
                                         backend_identity=backend)
    if isinstance(recipe, FrontierRecipe):
        from cipher_lab_plugins.frontier.training import run_frontier

        if plan is None:
            raise ValueError('MISSING_PUBLIC_PLAN')
        # One data owner: one private key, one panel and one cached public h.
        # The CLI already enforced output_bound before this construction.
        frontier_data = PredictionData(recipe, oracles['SIMON'], private_random,
                                       policy_meter, reference_oracle=reference_oracle)
        return lambda: run_frontier(recipe, frontier_data, policy_meter, store,
                                    backend_identity=backend, plan=plan)
    if isinstance(recipe, ResidualPredictionRecipe):
        from cipher_lab_plugins.residual.training import run_paired_prediction

        # One data owner provides the private key, one panel and the cached h.
        data = PredictionData(recipe, oracles['SIMON'], private_random, policy_meter,
                              reference_oracle=reference_oracle)
        return lambda: run_paired_prediction(recipe, data, policy_meter, store,
                                             backend_identity=backend)
    raise ValueError('unsupported experiment')
