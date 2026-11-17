"""Schema4 compact metadata + independent canonical public-plan ZIP member.

Not a nested ZIP reader: the outer archive is the restricted Torch member
structure. Physical ZIP and pickle guards remain in checkpoints; this module
owns only frontier metadata, plan binding, and single-layer canonical encoding.
"""
from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from typing import cast

import torch

from cipher_lab.training_config import MLPSpec
from cipher_lab_plugins.frontier.config import FrontierMLPSpec, parse_frontier_identity
from cipher_lab_plugins.frontier.simon_frontier import PublicFeaturePlan, encode_plan
from cipher_lab_plugins.frontier.tasks import FrontierTask
from cipher_lab_plugins.prediction_common.bundles import TaskDeclaration
from cipher_lab_plugins.prediction_common.metrics import (
    Model,
    PublicModelSpec,
    TrainBaseline,
    validate_baseline,
)

PLAN_MEMBER = 'archive/public-plan.json'


def frontier_metadata(node: object, plan: PublicFeaturePlan
                      ) -> tuple[PublicModelSpec, object, dict[str, TrainBaseline], str]:
    from cipher_lab_plugins.prediction_common.checkpoints import (
        Selection,
        _object,
        _same_public,
        _sha,
        _wiring,
    )
    obj = _object(node, {'schema', 'method', 'model_family', 'model_spec', 'wiring',
                         'task_declarations', 'selection', 'baseline', 'recipe_sha', 'state',
                         'plan_identity'})
    if (type(obj['schema']) is not int or obj['schema'] != 4
            or obj['method'] != 'bit-public-feature-frontier-v1'
            or parse_frontier_identity(obj['plan_identity']) != plan.identity):
        raise ValueError
    raw_tasks = obj['task_declarations']
    if type(raw_tasks) is not list or len(raw_tasks) != 1:
        raise ValueError
    t = _object(raw_tasks[0], {'cipher', 'rounds', 'target_bit', 'label_mode',
                               'reference', 'identity'})
    task = FrontierTask(cast(str, t['cipher']), cast(int, t['rounds']),
        cast(int, t['target_bit']), cast(str, t['label_mode']), cast(str, t['reference']),
        parse_frontier_identity(t['identity']))
    if task.identity != plan.identity:
        raise ValueError
    model: MLPSpec | FrontierMLPSpec
    if obj['model_family'] == 'bit-mlp-v1':
        raw = _object(obj['model_spec'], {'hidden_width'})
        model = MLPSpec(cast(int, raw['hidden_width']))
    elif obj['model_family'] == 'frontier-bit-mlp-v1':
        raw = _object(obj['model_spec'], {'hidden_width', 'identity'})
        model = FrontierMLPSpec(cast(int, raw['hidden_width']),
                                parse_frontier_identity(raw['identity']), plan)
    else:
        raise ValueError
    wiring = _object(obj['wiring'], {'input_bits', 'output_logits', 'hidden_layers',
        'activation', 'bit_order', 'output_mode', 'arm_id', 'representation',
        'network_input_width', 'plan_identity'})
    spec = PublicModelSpec(model, (task,), cast(str, wiring['output_mode']),
        cast(str, wiring['arm_id']), cast(str, wiring['representation']))
    if not _same_public(wiring, _wiring(spec)):
        raise ValueError
    selected = _object(obj['selection'], {'epoch', 'updates', 'selected_dev_bce'})
    selection = Selection(cast(int, selected['epoch']), cast(int, selected['updates']),
                          cast(float, selected['selected_dev_bce']))
    raw_baseline = _object(obj['baseline'], {'SIMON'})
    count = _object(raw_baseline['SIMON'], {'rows', 'ones'})
    baseline = {'SIMON': TrainBaseline(cast(int, count['rows']), cast(int, count['ones']))}
    validate_baseline(spec, baseline)
    return spec, selection, baseline, _sha(obj['recipe_sha'])


def encode_frontier_checkpoint(model: Model, public_model_spec: PublicModelSpec,
        task_declarations: tuple[TaskDeclaration, ...], selection: object,
        baseline: Mapping[str, TrainBaseline], recipe_sha: str, *,
        plan: PublicFeaturePlan) -> bytes:
    """No filesystem and no forward; canonical flat ZIP with CRC on every member."""
    from cipher_lab_plugins.frontier.simon_frontier import decode_plan
    from cipher_lab_plugins.prediction_common.checkpoints import (
        MAX_BYTES,
        Selection,
        _family,
        _new,
        _preflight,
        _sha,
        _validate_state,
        _wiring,
    )
    from cipher_lab_plugins.prediction_common.metrics import validate_model
    try:
        public_model_spec.__post_init__()
        canonical = encode_plan(plan)
        validated = decode_plan(canonical)
        if (public_model_spec.method != 'bit-public-feature-frontier-v1'
                or task_declarations != public_model_spec.task_declarations
                or type(selection) is not Selection
                or cast(FrontierTask, task_declarations[0]).identity != validated.identity):
            raise ValueError
        selection.__post_init__()
        _sha(recipe_sha)
        validate_baseline(public_model_spec, baseline)
        validate_model(model, public_model_spec)
        expected = _new(public_model_spec, meta=True).state_dict()
        if sum(t.numel()*t.element_size() for t in expected.values()) > MAX_BYTES:
            raise ValueError
        reference = _new(public_model_spec)
        _validate_state(dict(model.state_dict()), reference.state_dict(), reference)
        node: dict[str, object] = {
            'schema': 4, 'method': public_model_spec.method,
            'model_family': _family(public_model_spec),
            'model_spec': public_model_spec.model.to_dict(), 'wiring': _wiring(public_model_spec),
            'task_declarations': [t.to_dict() for t in task_declarations],
            'selection': {'epoch': selection.epoch, 'updates': selection.updates,
                          'selected_dev_bce': selection.selected_dev_bce},
            'baseline': {n: {'rows': v.rows, 'ones': v.ones} for n, v in baseline.items()},
            'recipe_sha': recipe_sha, 'plan_identity': validated.identity.to_dict(),
            'state': {k: v.detach().to(device='cpu', copy=True).contiguous()
                      for k, v in model.state_dict().items()}}
        payload = io.BytesIO()
        torch.save(node, payload, pickle_protocol=2)
        output = io.BytesIO()
        # Flatten restricted Torch members. No arbitrary nested archive survives.
        with zipfile.ZipFile(payload) as source, zipfile.ZipFile(output, 'w',
                compression=zipfile.ZIP_STORED) as target:
            for name in source.namelist():
                target.writestr(name, source.read(name))
            target.writestr(PLAN_MEMBER, canonical)
        result = output.getvalue()
        if len(result) > MAX_BYTES:
            raise ValueError
        _preflight(result)  # Apply exactly the reader's guard before owner persistence.
        return result
    except Exception:
        raise ValueError('invalid checkpoint') from None
