"""Artificial checks of the test-only safety policy, never real science."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize(('filename', 'name'), [
    ('test_mlp.py', 'test_two_equal_relu_layers_and_backward_without_update'),
    ('test_shared_m.py', 'test_public_router_stable_top2_and_gate_gradient'),
    ('test_execution.py', 'test_unsupported_production_rss_is_not_zero'),
    ('test_checkpoints.py', 'test_prediction_roundtrip_exact_interface_cpu_rng_and_logits'),
    ('test_checkpoints.py', 'test_shared_m_roundtrip_and_wiring_identity'),
    ('test_training_steps.py', 'test_step_attempt_completion_clip_and_fixed_errors'),
    ('test_training_steps.py', 'test_step_partial_forward_failure_not_refunded'),
    ('test_training_metrics.py', 'test_partial_failure_keeps_completed_previous_batch'),
    ('test_training_metrics.py',
     'test_batched_evaluation_charges_rows_preserves_modes_and_no_grad'),
    ('test_training_cli.py',
     'test_explicit_other_panel_allowed_but_task_mismatch_and_sha_rejected'),
    ('test_prediction_training.py', 'test_prediction_real_two_updates_closed_loop'),
    ('test_shared_m_training.py', 'test_shared_real_four_updates_closed_loop'),
    ('test_cipher.py', 'test_known_answer'),
    ('test_speck_study_cipher.py', 'test_public_single_round_literal_arithmetic'),
])
def test_real_cases_are_classified_without_executing_them(filename: str, name: str) -> None:
    policy = importlib.import_module('plugin_isolation')
    assert policy.is_science(Path('tests') / filename, name)


@pytest.mark.parametrize(('filename', 'name'), [
    ('test_training_steps.py', 'test_strict_selection_earliest_tie'),
    ('test_training_metrics.py', 'test_threshold_stable_bce_and_train_baseline'),
    ('test_checkpoints.py', 'test_metadata_whitelist'),
    ('test_simon_batch_structure.py',
     'test_artificial_single_round_rotations_and_xor_do_not_mutate'),
    ('test_shared_m_relations.py', 'test_fake_forward_raw_xor_msb0_top2_and_independent_pairs'),
    ('test_prediction_training.py', 'test_tie_keeps_first_saved_and_load_precedes_release'),
])
def test_artificial_cases_are_not_hidden_by_file_wide_exclusion(filename: str, name: str) -> None:
    policy = importlib.import_module('plugin_isolation')
    assert not policy.is_science(Path('tests') / filename, name)


def test_classification_names_still_match_retained_original_functions() -> None:
    import ast

    policy = importlib.import_module('plugin_isolation')
    root = Path(__file__).resolve().parent
    for filename, names in policy.SCIENCE_CASES.items():
        declared = {node.name for node in ast.parse((root / filename).read_text()).body
                    if isinstance(node, ast.FunctionDef)}
        assert names <= declared, f'stale science classification in {filename}'


def test_guard_traps_real_entries_without_calling_originals() -> None:
    import secrets

    import numpy as np
    import torch

    from cipher_lab import execution
    from cipher_lab.ciphers import simon
    from cipher_lab.models import BitMLP
    from cipher_lab.training_config import MLPSpec

    calls: list[Any] = [
        lambda: secrets.token_bytes(1), lambda: execution.production_rss_bytes(),
        lambda: simon.encrypt_many((0,), (0, 0, 0, 0), 1),
        lambda: BitMLP(MLPSpec(2))(torch.zeros(1, 64)),
        lambda: torch.nn.Linear(1, 1)(torch.zeros(1, 1)),
        lambda: torch.nn.ReLU()(torch.zeros(1)),
        lambda: BitMLP(MLPSpec(2))(torch.zeros(1, 63)),
        lambda: torch.autograd.backward(torch.zeros(1, requires_grad=True)),
        lambda: torch.optim.Adam([]), lambda: torch.cuda.is_available(),
        lambda: np.load('/not-a-test-owned-artificial-panel.npz'),
        lambda: torch.load('/not-a-test-owned-artificial-checkpoint.pt'),
    ]
    for call in calls:
        with pytest.raises(AssertionError, match='UNAUTHORIZED_SCIENCE'):
            call()


@pytest.mark.parametrize('name', ['tensorflow', 'keras', 'h5py'])
def test_import_poison_does_not_forge_preimported_identity(name: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    assert name not in sys.modules
    with pytest.raises(AssertionError, match='UNAUTHORIZED_SCIENCE'):
        importlib.import_module(name)
    assert name not in sys.modules
    fake = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, fake)
    assert importlib.import_module(name) is fake


def test_checkpoint_child_installs_guard_before_product_imports() -> None:
    import ast

    tree = ast.parse(Path(__file__).with_name('test_checkpoints.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'test_import_and_load_do_not_import_collection_modules')
    script = next(node.value.value for node in ast.walk(function)
                  if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str))
    assert 'install_checkpoint_guard()' in script
    assert script.index('install_checkpoint_guard()') < script.index('from cipher_lab.evaluation')
    assert 'collection import forbidden' in script


def test_checkpoint_child_guard_traps_without_importing_collection() -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    script = '''
from checkpoint_child_guard import install_checkpoint_guard
install_checkpoint_guard()
import importlib, os, secrets, sys, torch, numpy as np
from cipher_lab import execution
assert not any(n.startswith(('cipher_lab.ciphers', 'cipher_lab.training.')) for n in sys.modules)
for name in ('tensorflow', 'keras', 'h5py'):
    assert name not in sys.modules
calls = [lambda: os.urandom(1), lambda: secrets.token_bytes(1),
         lambda: torch.nn.Linear(1, 1)(torch.zeros(1, 1)),
         lambda: torch.nn.ReLU()(torch.zeros(1)),
         lambda: torch.zeros(1, requires_grad=True).backward(),
         lambda: torch.optim.Adam([]), lambda: execution.production_rss_bytes(),
         lambda: torch.cuda.is_available(),
         lambda: np.load('/not-artificial.npz'), lambda: torch.load('/not-artificial.pt')]
for name in ('tensorflow', 'keras', 'h5py', 'cipher_lab.ciphers.simon',
             'cipher_lab.training.steps', 'cipher_lab_plugins.speck_study.teacher'):
    calls.append(lambda name=name: importlib.import_module(name))
for call in calls:
    try:
        call()
    except AssertionError as error:
        assert 'UNAUTHORIZED_SCIENCE' in str(error)
    else:
        raise AssertionError('child guard missed forbidden entry')
assert not any(n.startswith(('cipher_lab.ciphers', 'cipher_lab.training.')) for n in sys.modules)
'''
    paths = os.pathsep.join(str(root / name) for name in ('src', 'plugins', 'tests'))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            timeout=30, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                             'PYTHONPATH': paths})
    assert result.returncode == 0, result.stdout + result.stderr
