"""Explicit provider/backend assembly only; methods retain scientific scheduling.

Called inside CLI's post-claim setup boundary. No production imports on demo.
"""
from __future__ import annotations

from collections.abc import Callable

from cipher_lab.artifacts import RunStore
from cipher_lab.execution import Meter
from cipher_lab.training_config import (
    PredictionRecipe,
    SharedMRecipe,
)


def prepare_training(recipe: PredictionRecipe | SharedMRecipe, policy_meter: Meter,
                     store: RunStore, backend: str) -> Callable[[], object]:
    from cipher_lab.data.differential import DifferentialData
    from cipher_lab.data.prediction import PredictionData

    if backend == 'demo':
        from cipher_lab.demo import DemoBytes, fake_encrypt_many

        private_random: Callable[[int], bytes] = DemoBytes()
        oracles = {name: fake_encrypt_many for name in ('SIMECK', 'PRESENT', 'SIMON', 'SKINNY')}
        # Public zero-master-key reference: the demo fake never becomes a cipher.
    else:
        from secrets import token_bytes

        from cipher_lab.ciphers.simon import encrypt_many

        private_random = token_bytes
        oracles = {'SIMON': encrypt_many}
        # Data always supplies key=(0,0,0,0) for the reference, i.e. a zero
        # MASTER key with a real public-key schedule, never zero round keys.
        if isinstance(recipe, SharedMRecipe):
            from cipher_lab.ciphers.present import encrypt_many as present
            from cipher_lab.ciphers.simeck import encrypt_many as simeck
            from cipher_lab.ciphers.skinny import encrypt_many as skinny

            oracles.update(SIMECK=simeck, PRESENT=present, SKINNY=skinny)
    if isinstance(recipe, PredictionRecipe):
        from cipher_lab.training.prediction import run_prediction

        data = PredictionData(recipe, oracles['SIMON'], private_random, policy_meter)
        return lambda: run_prediction(recipe, data, policy_meter, store, backend_identity=backend)
    from cipher_lab.training.shared_m import run_shared_m

    shared_data = DifferentialData(recipe, oracles, private_random, policy_meter)
    return lambda: run_shared_m(recipe, shared_data, policy_meter, store, backend_identity=backend)
