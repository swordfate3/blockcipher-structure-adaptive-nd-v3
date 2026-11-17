"""Explicit one-shot engineering preflight, never a training launcher.

8 public cipher blocks / 48 model rows / 8 updates; warmup=retry=0.
Run only after separate machine authorization. Help/import is stdlib-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

GPU_UUID = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
GPU_PCI = '0000:3b:00.0'
LIMITS = dict(wall_seconds=600, cpu_seconds=2400, rss_bytes=16 * 2**30,
              gpu_bytes=20 * 2**30, artifact_bytes=16 * 2**20,
              real_blocks=8, forward_rows=48, optimizer_updates=8)
VECTOR_SOURCE = '/workspace/backups/v3-before-rebuild-20261019/tests/test_ciphers.py'
VECTOR_SOURCE_SHA256 = 'db93bd1626bfd1b7d3bbc45659f43c1e39c6d71937f2c8780b74b2181a600955'
# These exact UTF-8 fixture bytes are hashed as evidence, not read from an archive at runtime.
FIXTURE_BYTES = (
    b'[\n{"cipher":"SIMON","rounds":44,"key_words":[50462976,185207048,319951120,4'
    b'54695192],"plaintext":"656b696c20646e75","expected":"44c8fc20b9dfa07a","lev'
    b'el":"full-public-kat"},\n{"cipher":"SIMON","rounds":12,"key_words":[50462976'
    b',185207048,319951120,454695192],"plaintext":"656b696c20646e75","expected":"'
    b'3b2bc82c320b0062","level":"reduced-replay-not-independent-official"},\n{"cip'
    b'her":"PRESENT","rounds":31,"key_words":[0,0,0],"plaintext":"000000000000000'
    b'0","expected":"5579c1387b228445","level":"full-public-kat"},\n{"cipher":"PRE'
    b'SENT","rounds":8,"key_words":[0,0,0],"plaintext":"0000000000000000","expect'
    b'ed":"6ba06c48b513e6cc","level":"reduced-replay-not-independent-official"},\n'
    b'{"cipher":"SKINNY","rounds":32,"key_words":[4234678840,4112947238],"plainte'
    b'xt":"06034f957724d19d","expected":"bb39dfb2429b8ac7","level":"full-public-k'
    b'at"},\n{"cipher":"SKINNY","rounds":7,"key_words":[4234678840,4112947238],"pl'
    b'aintext":"06034f957724d19d","expected":"276128939906f992","level":"reduced-'
    b'replay-not-independent-official"},\n{"cipher":"SIMECK","rounds":44,"key_word'
    b's":[50462976,185207048,319951120,454695192],"plaintext":"656b696c20646e75",'
    b'"expected":"45ce69025f7ab7ed","level":"full-public-kat"},\n{"cipher":"SIMECK'
    b'","rounds":8,"key_words":[50462976,185207048,319951120,454695192],"plaintex'
    b't":"656b696c20646e75","expected":"a14867236133b40e","level":"reduced-replay'
    b'-not-independent-official"}\n]'
)
# Public synthetic inputs, unrelated to cipher outputs. No scientific label generator.
INPUT_FIXTURE_BYTES = bytes((b + p + s + k) % 2 for b in range(2) for p in range(4)
                            for s in range(2) for k in range(64)) + bytes([0, 1])


class Failure(RuntimeError):
    def __init__(self, reason: str) -> None:
        if reason not in ('KAT_MISMATCH', 'OUTPUT_INVALID', 'NUMERICAL_MISMATCH',
                          'UPDATE_INVALID', 'SOURCE_CHANGED', 'DEVICE_IDENTITY'):
            raise ValueError('invalid public code')
        self.reason = reason
        super().__init__(reason)


class Production:
    """Tool-local adapter; does not broaden TorchRuntime's production CPU API."""
    kind = 'production-public-fixture'

    def __init__(self, recipe: Any, resources: Any, device: str, uuid: str, pci: str) -> None:
        from cipher_lab_plugins.innovation1.application import production_resources
        from cipher_lab_plugins.innovation1.device import actual_attestor
        device_resources = production_resources(recipe, device, uuid, pci)
        self.gpu = device_resources.gpu
        resources.gpu = self.gpu
        resources.check()
        from cipher_lab_plugins.innovation1.resources import ResourceStop
        try:
            self.evidence = actual_attestor()
        except Exception:
            raise ResourceStop() from None
        from cipher_lab_plugins.innovation1.resources import pci as normalize_pci
        if (self.evidence.get('gpu_uuid') != uuid
                or normalize_pci(self.evidence.get('pci_bus_id', '')) != normalize_pci(pci)
                or type(self.evidence.get('logical_device')) is not int
                or self.evidence['logical_device'] != 0
                or self.evidence.get('pid') != os.getpid()
                or self.evidence.get('evidence') != 'cuda-current-context'):
            raise Failure('DEVICE_IDENTITY')
        self.runtime: Any = None
        self.x: Any = None
        self.y: Any = None

    def encrypt(self, vector: dict[str, Any]) -> Any:
        from cipher_lab.ciphers import present, simeck, simon, skinny
        oracle = {'SIMON': simon.encrypt_many, 'SIMECK': simeck.encrypt_many,
                  'PRESENT': present.encrypt_many, 'SKINNY': skinny.encrypt_many}[vector['cipher']]
        return oracle((int(vector['plaintext'], 16),), tuple(vector['key_words']), vector['rounds'])

    def build(self, recipe: Any) -> None:
        import torch
        from cipher_lab_plugins.innovation1.models import TorchRuntime
        self.runtime = TorchRuntime(recipe, 'cuda:0')  # five real production Adam instances
        # Explicit public synthetic tensor; no Batch/Provider/DEV/TEST or RNG calls.
        self.x_cpu = torch.tensor([[[list(INPUT_FIXTURE_BYTES[
            ((b * 4 + p) * 2 + side) * 64:((b * 4 + p) * 2 + side + 1) * 64])
            for side in range(2)] for p in range(4)] for b in range(2)],
                                 dtype=torch.float32)
        self.x = self.x_cpu.to('cuda:0')
        self.y = torch.tensor([[float(v)] for v in INPUT_FIXTURE_BYTES[-2:]],
                              dtype=torch.float32, device='cuda:0')

    def evaluate(self, device: str, slot: str, task: Any) -> Any:
        if device == 'cuda:0':
            return self.runtime.predict(slot, task, self.x)
        from cipher_lab_plugins.innovation1.models import TorchRuntime, construct
        # CPU-only view is deliberately local, not a new public runtime constructor.
        view = object.__new__(TorchRuntime)
        view.recipe, view.device = self.runtime.recipe, self.x_cpu.device
        model = construct(view.recipe, slot)
        state = {k: v.detach().cpu().clone()
                 for k, v in self.runtime.models[slot].state_dict().items()}
        model.load_state_dict(state, strict=True)
        view.models, view.optimizers = {slot: model.eval()}, {}
        return view.predict(slot, task, self.x_cpu)

    def update(self, slot: str, task: Any, lr: float, resources: Any) -> float:
        return float(self.runtime.update(slot, task, self.x, self.y, lr, resources))


def _source() -> dict[str, str]:
    from cipher_lab_plugins.innovation1.source import source_identity
    source = source_identity()
    path = Path(__file__).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise Failure('SOURCE_CHANGED')
    source['tools/innovation1/preflight.py'] = sha256(path.read_bytes()).hexdigest()
    return source


def _logits(value: Any) -> list[float]:
    # Enforce exact [2,1], finite public scores without another forward/metric call.
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if (type(value) is not list or len(value) != 2 or any(
            type(row) is not list or len(row) != 1 or type(row[0]) not in (int, float)
            or not math.isfinite(row[0]) for row in value)):
        raise Failure('OUTPUT_INVALID')
    return [float(row[0]) for row in value]


def _write_evidence(path: Path, content: bytes) -> None:
    if len(content) > LIMITS['artifact_bytes']:
        raise ValueError('evidence too large')
    with path.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def run_preflight(config: Path, output: Path, *, device: str, gpu_uuid: str, gpu_pci: str,
                  setup: Any = Production, clocks: Any = None, rss: Any = None) -> dict[str, Any]:
    """A single explicit invocation; fakes can only enter through Python test injection.

    Saved status is pre-write evidence, command status includes post-write checks.
    No fallback, repeat, warmup, scientific labels, saved weights or training launch.
    """
    from cipher_lab_plugins.innovation1.config import Recipe, learning_rate, load_config
    from cipher_lab_plugins.innovation1.resources import (
        Resources,
        ResourceStop,
        pci,
        production_rss_bytes,
    )

    from cipher_lab.config import Policy
    from cipher_lab.execution import Clocks, Meter, Stopped
    # Clock origin precedes source/config reads, output claim and all numerical setup.
    policy = Policy(2, 600., 2400., (datetime.now(UTC) + timedelta(seconds=600)).isoformat(),
                    8, 0, LIMITS['rss_bytes'], LIMITS['artifact_bytes'], 8, 48)
    resources: Any = None
    initialization_error: Stopped | None = None
    try:
        resources = Resources(Meter(policy, clocks or Clocks(), rss or production_rss_bytes))
    except Stopped as error:
        initialization_error = error
    def snapshot() -> Any:
        return None if resources is None else resources.snapshot()
    stage, status, reason = 'output-claim', 'FAILED', 'EXECUTION_FAILED'
    evidence: dict[str, Any] = dict(schema=1, method='innovation1-engineering-preflight-v1',
        scientific_validation=False, limits=dict(LIMITS), tolerance=dict(atol=1e-5, rtol=1e-4),
        warmup=0, retries=0, operations=[], source={}, config_sha256=None,
        fixture_sha256=sha256(FIXTURE_BYTES).hexdigest(),
        input_fixture_sha256=sha256(INPUT_FIXTURE_BYTES).hexdigest(),
        fixture_source=dict(path=VECTOR_SOURCE, sha256=VECTOR_SOURCE_SHA256),
        device={}, runtime_kind='UNVERIFIED',
        counts_scope='saved snapshot is pre-write; stdout command counts include post-write budget')
    try:
        if any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError('unsafe path')
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return dict(command_status='FAILED', evidence_status='NOT_SAVED',
                    reason='OUTPUT_OCCUPIED', stage=stage, exit_code=2)
    except Exception:
        return dict(command_status='FAILED', evidence_status='NOT_SAVED',
                    reason='OUTPUT_CLAIM_FAILED', stage=stage, exit_code=2)
    try:
        stage = 'clock-setup'
        if initialization_error is not None:
            raise initialization_error
        stage = 'settings'
        resources.check()
        raw = config.read_bytes()
        evidence['config_sha256'] = sha256(raw).hexdigest()
        recipe = load_config(config)
        if raw != config.read_bytes() or recipe.to_dict() != Recipe().to_dict():
            raise Failure('SOURCE_CHANGED')
        evidence['source'] = _source()
        if device != 'cuda:0' or gpu_uuid != GPU_UUID or pci(gpu_pci) != GPU_PCI:
            raise Failure('DEVICE_IDENTITY')
        stage = 'device-setup'
        runtime = setup(recipe, resources, device, gpu_uuid, gpu_pci)
        resources.gpu = runtime.gpu
        evidence['runtime_kind'], evidence['device'] = runtime.kind, runtime.evidence
        resources.check()
        stage = 'kat'
        for vector in json.loads(FIXTURE_BYTES):
            operation = dict(stage=stage, cipher=vector['cipher'], rounds=vector['rounds'],
                             level=vector['level'], attempted=True, completed=False)
            resources.reserve('real_blocks', 1)
            evidence['operations'].append(operation)
            actual = runtime.encrypt(vector)
            if (type(actual) not in (tuple, list) or len(actual) != 1
                    or type(actual[0]) is not int or actual[0] != int(vector['expected'], 16)):
                raise Failure('KAT_MISMATCH')
            resources.complete('real_blocks', 1)
            operation['completed'] = True
            resources.check()
        stage = 'model-setup'
        runtime.build(recipe)
        resources.check()
        for task in recipe.tasks:
            for slot in ('M-best', f'D-{task.cipher}-best'):
                stage = 'eval'
                scores = []
                for evaluation_device in ('cpu', 'cuda:0'):
                    operation = dict(stage=stage, device=evaluation_device, slot=slot,
                        task=task.cipher, groups=2, attempted=True, completed=False)
                    resources.reserve('forward_rows', 2)
                    evidence['operations'].append(operation)
                    scores.append(_logits(runtime.evaluate(evaluation_device, slot, task)))
                    resources.complete('forward_rows', 2)
                    operation['completed'] = True
                    resources.check()
                stage = 'compare'
                if any(abs(cpu - gpu) > 1e-5 + 1e-4 * abs(cpu)
                       for cpu, gpu in zip(scores[0], scores[1], strict=True)):
                    raise Failure('NUMERICAL_MISMATCH')
        stage = 'update'
        for task in recipe.tasks:
            for slot in ('M-best', f'D-{task.cipher}-best'):
                operation = dict(stage=stage, slot=slot, task=task.cipher, groups=2,
                                 attempted=True, completed=False)
                resources.reserve('optimizer_updates', 1)
                evidence['operations'].append(operation)
                loss = runtime.update(slot, task, learning_rate(recipe, 0), resources)
                resources.complete('optimizer_updates', 1)
                operation['completed'] = True
                if type(loss) not in (int, float) or not math.isfinite(loss):
                    raise Failure('UPDATE_INVALID')
                resources.check()
        stage = 'identity-final'
        if _source() != evidence['source'] or config.read_bytes() != raw:
            raise Failure('SOURCE_CHANGED')
        resources.check()
        status, reason, stage = 'PASS', 'COMPLETE', 'complete'
    except (Stopped, ResourceStop) as error:
        status, reason = 'STOPPED', error.reason
    except Failure as error:
        status, reason = 'FAILED', error.reason
    except Exception:
        status, reason = 'FAILED', 'EXECUTION_FAILED'
    evidence.update(status=status, reason=reason, stage=stage, counts=snapshot())
    content = (json.dumps(evidence, sort_keys=True, allow_nan=False) + '\n').encode()
    # Serialization is part of this same budget, not an unmetered tail. Even
    # on STOP, persist the bounded public failure evidence, never do more science.
    try:
        if resources is not None:
            resources.check()
    except (Stopped, ResourceStop) as error:
        status, reason = 'STOPPED', error.reason
        evidence.update(status=status, reason=reason, counts=snapshot())
        content = (json.dumps(evidence, sort_keys=True, allow_nan=False) + '\n').encode()
    write_reason = None
    try:
        _write_evidence(output / 'preflight.json', content)
    except Exception:
        write_reason = 'EVIDENCE_WRITE_FAILED'
    command_status = 'FAILED' if write_reason else status
    command_reason = write_reason or reason
    try:
        if resources is not None:
            resources.check()
    except (Stopped, ResourceStop) as error:
        command_status, command_reason = 'STOPPED', error.reason
    return dict(command_status=command_status,
                evidence_status='UNKNOWN' if write_reason else status, reason=command_reason,
                write_reason=write_reason,
                stage=stage, counts=snapshot(),
                evidence_sha256=sha256(content).hexdigest(),
                evidence_bytes=len(content), counts_scope=evidence['counts_scope'],
                exit_code=(6 if write_reason and command_status == 'FAILED' else
                           {'PASS': 0, 'STOPPED': 3, 'FAILED': 4}[command_status]))


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', required=True, choices=('cuda:0',))
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--gpu-pci', required=True)
    args = parser.parse_args(argv)
    result = run_preflight(args.config, args.output_dir, device=args.device,
                           gpu_uuid=args.gpu_uuid, gpu_pci=args.gpu_pci)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return int(result['exit_code'])


if __name__ == '__main__':
    raise SystemExit(main())
