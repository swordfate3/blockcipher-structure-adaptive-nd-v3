"""Test-only pytest hooks: retain science cases, default deselect, poison safety net.

This is not a product execution plugin or an authorization mechanism. Subprocess
fixtures must install their own guards: monkeypatch does not cross processes.
"""
from __future__ import annotations

import importlib
import itertools
import os
import random
import secrets
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from checkpoint_child_guard import TeacherImportBlocker

# Exact original functions, including pre-forward validation in mixed model files.
# Do not fake these cases into passing: their original numerical assertions remain.
SCIENCE_CASES: dict[str, frozenset[str]] = {
    'test_mlp.py': frozenset({
        'test_two_equal_relu_layers_and_backward_without_update',
        'test_reject_invalid_input', 'test_nonfinite_model_output_is_not_success',
    }),
    'test_shared_m.py': frozenset({
        'test_public_router_stable_top2_and_gate_gradient',
        'test_shared_forward_backward_no_update',
        'test_rounds_affect_representation_not_route_or_parameters',
        'test_shared_invalid_bits', 'test_spec_identity_and_family_round_validation',
        'test_pair_readout_nonlinearity_precedes_pair_average',
        'test_task_route_sample_independence_batch_alignment_pair_symmetry',
        'test_default_forward_and_public_dtype_rejection',
        'test_reject_corrupt_router_or_output_without_updates',
    }),
    'test_execution.py': frozenset({'test_unsupported_production_rss_is_not_zero'}),
    'test_checkpoints.py': frozenset({
        'test_prediction_roundtrip_exact_interface_cpu_rng_and_logits',
        'test_shared_m_roundtrip_and_wiring_identity',
    }),
    'test_training_steps.py': frozenset({
        'test_step_attempt_completion_clip_and_fixed_errors',
        'test_step_partial_forward_failure_not_refunded',
        'test_forward_limit_keeps_reserved_update_without_optimizer_step',
    }),
    'test_training_metrics.py': frozenset({
        'test_batched_evaluation_charges_rows_preserves_modes_and_no_grad',
        'test_failed_forward_not_refunded_or_completed_and_restores_mode',
        'test_shared_m_public13_rounds_and_row_unit',
        'test_partial_failure_keeps_completed_previous_batch',
    }),
    'test_prediction_training.py': frozenset({'test_prediction_real_two_updates_closed_loop'}),
    'test_shared_m_training.py': frozenset({'test_shared_real_four_updates_closed_loop'}),
    'test_training_cli.py': frozenset({
        'test_demo_run_then_explicit_evaluate_no_new_training_or_binary_copy',
        'test_explicit_other_panel_allowed_but_task_mismatch_and_sha_rejected',
    }),
}
# Pure cipher/KAT suites belong to core but require separate permission. The
# pure-cipher SPECK suite remains excluded too; public representation tests
# such as test_simon_batch_structure are independently guarded and stay safe.
SCIENCE_FILES = frozenset({
    'test_cipher.py', 'test_ciphers.py', 'test_simon.py', 'test_simeck.py',
    'test_present.py', 'test_skinny.py', 'test_speck.py', 'test_speck_study_cipher.py',
})


def is_science(path: Path, name: str) -> bool:
    return path.name in SCIENCE_FILES or name in SCIENCE_CASES.get(path.name, ())


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        name = getattr(item, 'originalname', None) or item.name.split('[', 1)[0]
        if is_science(item.path, name):
            item.add_marker(pytest.mark.science)


def forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError('UNAUTHORIZED_SCIENCE: bind explicit artificial test inputs/outputs')


def pytest_sessionstart() -> None:
    # Collection imports Torch/manual tensor parameters. Initialize its lazy NumPy
    # dependency with fake bytes, never OS/private entropy (no network execution).
    with pytest.MonkeyPatch.context() as bootstrap:
        bootstrap.setattr(os, 'urandom', lambda size: bytes(size))
        bootstrap.setattr(secrets, 'randbits', lambda bits: 0)
        importlib.import_module('torch')
        importlib.import_module('numpy.random')


@pytest.fixture(autouse=True)
def default_science_poison(monkeypatch: pytest.MonkeyPatch,
                           tmp_path_factory: pytest.TempPathFactory) -> None:
    # CPU format/parameter initialization is allowed, never numerical forward.
    # Complements the existing T1 owner guard; no product runtime changes.
    import torch

    # Technical temp-file UUIDs must not request OS entropy after the poison.
    counter = itertools.count(1)
    monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(counter)))
    import cipher_lab.artifacts as artifacts
    monkeypatch.setattr(artifacts, 'uuid4', lambda: uuid.UUID(int=next(counter)))
    # NumPy's unused legacy RNG can initialize lazily; use explicit fake bytes.
    monkeypatch.setattr(os, 'urandom', lambda size: bytes(size))
    monkeypatch.setattr(secrets, 'randbits', lambda bits: 0)
    import numpy.random  # noqa: F401
    monkeypatch.setattr(secrets, 'randbits', forbidden)
    for owner, names in (
        (secrets, ('token_bytes', 'token_hex', 'token_urlsafe', 'randbelow', 'choice')),
        (os, ('urandom',)), (random, ('SystemRandom', '_urandom')),
        (torch.Tensor, ('backward',)), (torch.autograd, ('backward', 'grad')),
        (torch.cuda, ('is_available', 'init', '_lazy_init')),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        for name in ('__init__', 'step', 'zero_grad'):
            monkeypatch.setattr(cls, name, forbidden)
    for layer_cls in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d,
                torch.nn.Conv3d, torch.nn.GroupNorm):
        monkeypatch.setattr(layer_cls, 'forward', forbidden)
    for name in ('linear', 'conv1d', 'conv2d', 'conv3d', 'group_norm'):
        monkeypatch.setattr(torch.nn.functional, name, forbidden)
    # Catch other real Torch layers (including toy networks), but allow custom
    # explicit artificial test components and nonnumerical container/identity.
    original_call = cast(Callable[..., Any], torch.nn.Module._call_impl)

    def guarded_call(model: Any, *args: object, **kwargs: object) -> Any:
        if type(model).__module__.startswith('torch.nn.') and type(model) not in (
                torch.nn.Sequential, torch.nn.Identity):
            forbidden()
        return original_call(model, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, '_call_impl', guarded_call)
    from cipher_lab.models import BitMLP, FeatureBitMLP, FrontierBitMLP, SharedM
    for model_cls in (BitMLP, FeatureBitMLP, FrontierBitMLP, SharedM):
        monkeypatch.setattr(model_cls, 'forward', forbidden)
    for family in ('simon', 'simeck', 'present', 'skinny', 'speck', 'speck_tensor'):
        module = importlib.import_module('cipher_lab.ciphers.' + family)
        for name in ('encrypt_many', 'encrypt_batch', 'encrypt',
                     'encrypt_scalar', 'encrypt_tensor'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    import cipher_lab.execution as execution
    monkeypatch.setattr(execution, 'production_rss_bytes', forbidden)
    # Catch already imported aliases as well. Later imports see poisoned entries.
    for name, module in tuple(sys.modules.items()):
        if name.startswith(('cipher_lab.', 'cipher_lab_plugins.')):
            for attr in ('production_rss_bytes', '_nvidia_smi_runner', '_windows_api'):
                if hasattr(module, attr):
                    monkeypatch.setattr(module, attr, forbidden)
    # Decode only in-memory artificial archives or test-owned temporary files;
    # never read actual saved weights/NPZ from project data/models/results.
    import numpy as np

    base = tmp_path_factory.getbasetemp().resolve()
    for loader_owner in (np, torch):
        original_load = loader_owner.load

        def guarded_load(source: object, *args: object,
                         _load: Any = original_load, **kwargs: object) -> Any:
            candidate = source if isinstance(source, (str, bytes, os.PathLike)) else getattr(
                source, 'name', None)
            if isinstance(candidate, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(candidate)).resolve()
                if not path.is_relative_to(base):
                    forbidden()
            return _load(source, *args, **kwargs)

        monkeypatch.setattr(loader_owner, 'load', guarded_load)
    # Do not fabricate preimported identities: production checks membership.
    # Actual imports are rejected; explicit fake sys.modules injection still works.
    monkeypatch.setattr(sys, 'meta_path', [TeacherImportBlocker(), *sys.meta_path])


def pytest_report_header() -> str:
    return ('Science cases retained; default -m "not science"; '
            'marker selection is NOT authorization.')
