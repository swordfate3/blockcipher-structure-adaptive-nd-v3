"""Exact TF admission through guarded, frozen REAL native application assembly.

Fake READY / fake formal_r0 only; no real TF, cipher, KAT, DLL or GPU calls.
The child source-only closure and its existing science traps remain in force.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_speck_cli import test_clean_source_fake_cli_check_run_read_saved_strict_codec as run_chain


@pytest.mark.parametrize(('version', 'formal_result', 'expected', 'calls'), [
    ('2.10.0', True, True, 1),
    ('2.10.0', False, False, 1),
    ('2.5.0', True, False, 0),
    ('2.5.9', True, False, 0),
    ('2.10.1', True, False, 0),
    ('2.11.0', True, False, 0),
    ('2.10.0.synthetic', True, False, 0),
    ('2.10.00', True, False, 0),
    (' 2.10.0', True, False, 0),
])
def test_exact_tf_admission_calls_only_fake_formal_r0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    version: str, formal_result: bool, expected: bool, calls: int,
) -> None:
    monkeypatch.setenv('SPECK_TEST_VERSION_GATE', json.dumps(dict(
        tensorflow=version, formal_result=formal_result,
        expected_result=expected, expected_calls=calls)))
    run_chain(tmp_path)


def test_empty_ready_version_rejected_before_gpu_verification() -> None:
    script = r"""
import sys, importlib.abc, json
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','numpy','tensorflow','h5py') or (
            fullname.startswith('cipher_lab.ciphers')):
            raise AssertionError('real science import forbidden '+fullname)
sys.meta_path.insert(0,Guard())
from cipher_lab.speck_study.teacher import TeacherClient
from cipher_lab.speck_study.teacher_network import MODEL_SPEC
from cipher_lab.speck_study.teacher_protocol import ProtocolError
client=object.__new__(TeacherClient)
def forbidden(*args):
    raise AssertionError('invalid READY must fail before GPU verification')
payload=json.dumps(dict(python='3.9.13',tensorflow='',model=MODEL_SPEC,
    weights_sha256='a'*64,gpu={},allocator_mib=8192)).encode()
try:
    client._validate_ready(payload,'a'*64,'GPU-test','pci',forbidden)
except ProtocolError:
    pass
else:
    raise AssertionError('empty TF version admitted')
assert 'tensorflow' not in sys.modules
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=root,
        env={**os.environ, 'PYTHONPATH':str(root/'src'), 'PYTHONDONTWRITEBYTECODE':'1'},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout+result.stderr


def test_shipped_policy_is_exact_existing_schema_and_static_cli_only() -> None:
    from test_speck_cli import ROOT, child

    from cipher_lab.config import load_policy
    p = ROOT/'configs/speck-study-policy.json'
    policy = load_policy(p)
    assert policy.to_dict() == dict(schema=2, wall_seconds=28800, cpu_seconds=28800,
        deadline_utc='2099-01-01T00:00:00Z', real_blocks=542000000, null_words=0,
        rss_stop_bytes=32*2**30, output_bytes=4*2**30, optimizer_updates=0,
        forward_rows=1161000000)
    result = child(['check','--config','configs/speck-study-paper.json','--policy',str(p)])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['estimate']['planned_real_blocks'] == 541000512
    assert report['estimate']['teacher_rows'] == 1161000000
    assert set(report['planned_calls_fit_policy'].values()) == {True}


def test_shipped_policy_forward_capacity_and_study_cpu_record_only() -> None:
    from datetime import datetime, timezone

    from cipher_lab.config import load_policy
    from cipher_lab.execution import Clocks, Meter, Stopped

    root = Path(__file__).resolve().parents[1]
    policy = load_policy(root/'configs/speck-study-policy.json')
    cpu = [0.0]
    clocks = Clocks(lambda: 100.0, lambda: cpu[0],
                    lambda: datetime(2030, 1, 1, tzinfo=timezone.utc))
    study = Meter(policy, clocks, lambda: 0, enforce_cpu=False)
    default = Meter(policy, clocks, lambda: 0)
    cpu[0] = float(policy.cpu_seconds+1)
    study.reserve('forward_rows', 1161000000)  # arithmetic only, no inference
    study.complete('forward_rows', 1161000000)
    assert study.snapshot()['completed_forward_rows'] == 1161000000
    assert study.snapshot()['cpu_elapsed_seconds'] == cpu[0]
    with pytest.raises(Stopped, match='CPU_LIMIT'):
        default.check()
    with pytest.raises(Stopped, match='FORWARD_ROWS_LIMIT'):
        study.reserve('forward_rows', 1)
