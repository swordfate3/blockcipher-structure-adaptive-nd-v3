"""One device-setup diagnosis, no model/cipher/optimizer work or training launch.

Saved evidence is pre-write; stdout command status includes post-write checks.
Stdout diagnostic_status/reason retain the setup outcome before write/closeout;
command_status/reason describe command closeout, with write_reason separate.
OS exit must be captured natively by the caller, not inferred from JSON stdout.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

GPU_UUID = 'GPU-171dfbbe-c1a7-d93b-549d-8824cb8a18c7'
GPU_PCI = '0000:3b:00.0'
LIMITS = dict(wall_seconds=120, cpu_seconds=480, rss_bytes=16 * 2**30,
              gpu_bytes=20 * 2**30, artifact_bytes=16 * 2**20,
              real_blocks=0, null_words=0, forward_rows=0, optimizer_updates=0)
STAGES = frozenset(('entry-check', 'gate-setup', 'resource-setup',
                   'initial-resource-check', 'torch-import', 'torch-configure',
                   'cuda-select', 'cuda-init', 'context-attest', 'identity-check',
                   'device-properties', 'memory-fraction', 'final-resource-check', 'returned'))


def _source() -> dict[str, str]:
    from cipher_lab_plugins.innovation1.source import source_identity
    value = source_identity()
    path = Path(__file__).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('source unavailable')
    value['tools/innovation1/device_check.py'] = sha256(path.read_bytes()).hexdigest()
    return value


def _write(path: Path, raw: bytes) -> None:
    if len(raw) > LIMITS['artifact_bytes']:
        raise ValueError('oversize evidence')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run_check(config: Path, output: Path, *, device: str, gpu_uuid: str, gpu_pci: str,
              setup: Any = None, clocks: Any = None, rss: Any = None) -> dict[str, Any]:
    from cipher_lab_plugins.innovation1.application import production_resources
    from cipher_lab_plugins.innovation1.config import Recipe, load_config
    from cipher_lab_plugins.innovation1.resources import (
        DETAIL_CODES,
        ResourceStop,
        pci,
        production_rss_bytes,
    )

    from cipher_lab.config import Policy
    from cipher_lab.execution import Clocks, Meter, Stopped
    meter: Any = None
    initialization_error: Exception | None = None
    try:
        policy = Policy(2, 120., 480., (datetime.now(UTC) + timedelta(seconds=120)).isoformat(),
                        0, 0, LIMITS['rss_bytes'], LIMITS['artifact_bytes'], 0, 0)
        meter = Meter(policy, clocks or Clocks(), rss or production_rss_bytes)
    except Exception as error:
        initialization_error = error
    def snapshot() -> Any:
        return None if meter is None else meter.snapshot()
    def check() -> None:
        if meter is not None:
            meter.check()  # never use production Resources.check: no extra GPU sampling
    evidence: dict[str, Any] = dict(schema=1, method='innovation1-device-check-v1',
        scientific_validation=False, limits=dict(LIMITS), source={}, config_sha256=None,
        events=[], device={}, counts=None,
        count_scope=('outer zero-budget ledger covers diagnosis boundaries; production setup '
                     'uses its own admission meter, not evidence of zero GPU/context access'),
        status_scope=('saved status/counts are pre-write; '
                      'stdout command status/counts are post-write'))
    stage, status, reason, detail = 'output-claim', 'FAILED', 'EXECUTION_FAILED', None
    try:
        if any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return dict(command_status='FAILED', evidence_status='NOT_SAVED',
                    reason='OUTPUT_OCCUPIED', stage=stage, exit_code=2)
    except Exception:
        return dict(command_status='FAILED', evidence_status='NOT_SAVED',
                    reason='OUTPUT_CLAIM_FAILED', stage=stage, exit_code=2)

    def observe(event: dict[str, Any]) -> None:
        nonlocal stage
        if (event.get('stage') not in STAGES
                or event.get('event') not in ('begin', 'success', 'stop')):
            raise ValueError('invalid diagnostic event')
        stage = event['stage']
        # Only fixed schema keys; values come from the trusted production copy.
        allowed = {'stage', 'event', 'elapsed_seconds', 'started_seconds',
                   'duration_seconds', 'reason',
                   'detail_code', 'gate_last', 'gate_latest', 'attestation'}
        value = {k: v for k, v in event.items() if k in allowed}
        if value.get('detail_code') is not None and value['detail_code'] not in DETAIL_CODES:
            raise ValueError('invalid detail')
        evidence['events'].append(value)
        if value.get('attestation') is not None:
            evidence['device'] = value['attestation']
        check()  # same outer Meter only; never resample the GPU gate

    try:
        stage = 'clock-setup'
        if initialization_error is not None:
            raise initialization_error
        stage = 'source-identity'
        evidence['source'] = _source()
        raw = config.read_bytes()
        evidence['config_sha256'] = sha256(raw).hexdigest()
        check()
        stage = 'settings'
        recipe = load_config(config)
        if recipe.to_dict() != Recipe().to_dict() or device != 'cuda:0' or gpu_uuid != GPU_UUID:
            raise ValueError
        if pci(gpu_pci) != GPU_PCI or config.read_bytes() != raw:
            raise ValueError
        stage = 'device-setup'
        check()
        (production_resources if setup is None else setup)(
            recipe, device, gpu_uuid, gpu_pci, observer=observe)
        check()
        stage = 'identity-final'
        if _source() != evidence['source'] or config.read_bytes() != raw:
            raise ValueError
        check()
        stage, status, reason = 'complete', 'PASS', 'COMPLETE'
    except (ResourceStop, Stopped) as error:
        status, reason = 'STOPPED', error.reason
        detail = error.detail_code if isinstance(error, ResourceStop) else None
    except Exception:
        status, reason = 'FAILED', 'EXECUTION_FAILED'
    diagnostic_status, diagnostic_reason = status, reason
    evidence.update(status=status, reason=reason, detail_code=detail,
                    stage=stage, counts=snapshot())
    prewrite_reason = None
    try:
        check()
    except (ResourceStop, Stopped) as error:
        prewrite_reason = error.reason
        if status == 'PASS':
            status, reason = 'STOPPED', error.reason
    evidence.update(status=status, reason=reason, prewrite_reason=prewrite_reason,
                    counts=snapshot())
    content = (json.dumps(evidence, sort_keys=True, allow_nan=False) + '\n').encode()
    write_reason = None
    try:
        _write(output/'device-check.json', content)
    except Exception:
        write_reason = 'EVIDENCE_WRITE_FAILED'
    command_status = 'FAILED' if write_reason else status
    command_reason = write_reason or reason
    postwrite_reason = None
    try:
        check()
    except (ResourceStop, Stopped) as error:
        postwrite_reason = error.reason
        if command_status == 'PASS':
            command_status, command_reason = 'STOPPED', error.reason
    return dict(command_status=command_status, reason=command_reason,
        diagnostic_status=diagnostic_status, diagnostic_reason=diagnostic_reason,
        prewrite_reason=prewrite_reason, postwrite_reason=postwrite_reason,
        evidence_status='UNKNOWN' if write_reason else status, detail_code=detail,
        write_reason=write_reason, counts=snapshot(), evidence_sha256=sha256(content).hexdigest(),
        stage=stage, status_scope=evidence['status_scope'],
        exit_code=(6 if write_reason and command_status == 'FAILED' else
                   {'PASS': 0, 'STOPPED': 3, 'FAILED': 4}[command_status]))


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda:0',), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--gpu-pci', required=True)
    args = parser.parse_args(argv)
    result = run_check(args.config, args.output_dir, device=args.device,
                       gpu_uuid=args.gpu_uuid, gpu_pci=args.gpu_pci)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return int(result['exit_code'])


if __name__ == '__main__':
    raise SystemExit(main())
