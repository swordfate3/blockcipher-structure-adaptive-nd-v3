"""Finite injected randomness/fake oracles; never scientific encryption or fitting."""
from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from cipher_lab.config import Policy
from cipher_lab.data.bundles import PredictionTask, decode_bundle, encode_bundle
from cipher_lab.data.differential import DifferentialData
from cipher_lab.data.prediction import PredictionData
from cipher_lab.execution import Clocks, Meter, Stopped
from cipher_lab.training_config import (
    MLPSpec,
    PredictionRecipe,
    SharedMRecipe,
    SharedMSpec,
    TaskSpec,
    TrainSpec,
)

torch.set_num_threads(1)


def meter(real: int = 10000, null: int = 10000) -> Meter:
    return Meter(Policy(2, 10, 10, "2031-01-01T00:00:00Z", real, null, 100, 100, 0, 0),
                 Clocks(lambda: 0., lambda: 0.,
                        lambda: datetime(2030, 1, 1, tzinfo=UTC)), lambda: 0)


def recipe(mode: str = "cipher", bit: int = 0) -> PredictionRecipe:
    return PredictionRecipe(1, "bit-prediction-v1", MLPSpec(4),
                            TrainSpec(2, 2, 2, 2, 1, .01, 5, 7, 11), 13, bit, mode)


def shared_recipe() -> SharedMRecipe:
    return SharedMRecipe(1, "shared-m-four-task-v1", SharedMSpec(8, 8, 2, (3,), 4, 4),
                         recipe().training,
                         tuple(TaskSpec(c, r, d) for c, r, d in (
                             ("SIMECK", 16, 64), ("PRESENT", 7, 0xD00000),
                             ("SIMON", 13, 64), ("SKINNY", 8, 0x2000))))


class FiniteRandom:
    def __init__(self, count: int = 24) -> None:
        self.remaining = count
        self.requests: list[int] = []

    def __call__(self, size: int) -> bytes:
        assert self.remaining > 0, "finite fixture exhausted"
        self.remaining -= 1
        self.requests.append(size)
        return self.remaining.to_bytes(size, "big")


class Capture:
    def __init__(self, *, response: Sequence[int] | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        self.response = response
        self.fail = fail

    def __call__(self, points: tuple[int, ...], key: tuple[int, ...],
                 rounds: int) -> Sequence[int]:
        self.calls.append((points, key, rounds))
        if self.fail:
            raise RuntimeError("SECRET-CANARY")
        return self.response if self.response is not None else points


@pytest.fixture(autouse=True)
def poison_production(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> tuple[int, ...]:
        raise AssertionError("production cipher must not run")
    for name in ("simon", "simeck", "present", "skinny"):
        monkeypatch.setattr(f"cipher_lab.ciphers.{name}.encrypt_many", forbidden)


@pytest.mark.parametrize("bit", [0, 31, 32, 63])
def test_prediction_global_inputs_fixed_private_key_msb0_and_test_gate(bit: int) -> None:
    oracle, random, ledger = Capture(), FiniteRandom(1), meter()
    data = PredictionData(recipe(bit=bit), oracle, random, ledger)
    assert len(oracle.calls) == 2  # train/dev ONLY, even though all inputs exist
    all_inputs = [tuple(row) for split in ("train", "dev", "test")
                  for row in data.inputs(split).tolist()]
    assert len(all_inputs) == len(set(all_inputs)) == 6
    assert ledger.snapshot()["attempted_real_blocks"] == 4
    for split in ("train", "dev"):
        assert torch.equal(data.labels(split), data.inputs(split)[:, bit:bit+1])
    for action in (lambda: data.labels("test"), data.test_bundle):
        with pytest.raises(ValueError, match="TEST"):
            action()
    assert len(oracle.calls) == 2
    data.release_test()
    assert len(oracle.calls) == 2  # release changes permission, does not generate
    bundle = data.test_bundle()
    assert torch.equal(bundle.panels["SIMON"][1], data.inputs("test")[:, bit:bit+1])
    assert bundle.task_declarations == (PredictionTask("SIMON", 13, bit, "cipher"),)
    assert len({call[1] for call in oracle.calls}) == 1
    assert oracle.calls[0][1] == (0, 0, 0, 0)
    assert random.requests == [16]
    assert ledger.snapshot()["completed_real_blocks"] == 6
    assert ledger.snapshot()["attempted_null_words"] == 0
    data.labels("test")
    data.release_test()
    assert len(oracle.calls) == 3  # cache does not collect twice
    altered = data.inputs("train")
    altered.fill_(2)
    assert bool((data.inputs("train") <= 1).all())
    bundle.panels["SIMON"][1].fill_(2)
    assert bool((data.labels("test") <= 1).all())
    restored = decode_bundle(encode_bundle(data.test_bundle()))
    assert restored.task_declarations == bundle.task_declarations


def test_null_one_word_per_bit_no_key_no_oracle() -> None:
    random, oracle, ledger = FiniteRandom(6), Capture(fail=True), meter(real=0)
    data = PredictionData(recipe("null"), oracle, random, ledger)
    assert random.requests == [1] * 4
    data.release_test()
    assert data.labels("test").tolist() == [[1.], [0.]]
    assert random.requests == [1] * 6 and not oracle.calls
    assert ledger.snapshot()["completed_null_words"] == 6
    assert ledger.snapshot()["attempted_real_blocks"] == 0


def test_public_seeds_do_not_choose_key() -> None:
    calls = []
    inputs = []
    for offset in (0, 1):
        oracle = Capture()
        training = replace(recipe().training, init_seed=offset, sample_seed=offset,
                           shuffle_seed=offset)
        data = PredictionData(replace(recipe(), training=training), oracle,
                              FiniteRandom(1), meter())
        calls.append(oracle.calls[0][1])
        inputs.append(data.inputs("train"))
    assert calls[0] == calls[1]
    assert not torch.equal(inputs[0], inputs[1])
    other = Capture()
    PredictionData(recipe(), other, FiniteRandom(2), meter())
    assert other.calls[0][1] != calls[0]


@pytest.mark.parametrize("response", [(), (True, 0), (-1, 0), (1 << 64, 0), (0,)])
def test_prediction_invalid_oracle_response_charged(response: tuple[int, ...]) -> None:
    ledger = meter()
    with pytest.raises(Stopped, match="ORACLE_RESPONSE"):
        PredictionData(recipe(), Capture(response=response), FiniteRandom(1), ledger)
    assert ledger.snapshot()["attempted_real_blocks"] == 2
    assert ledger.snapshot()["completed_real_blocks"] == 0


def test_secret_errors_and_budget_precedes_private_random() -> None:
    for mode in ("cipher", "null"):
        random, oracle = FiniteRandom(0), Capture()
        ledger = meter(real=0, null=0)
        with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT|NULL_WORDS_LIMIT"):
            PredictionData(recipe(mode), oracle, random, ledger)
        assert random.requests == [] and not oracle.calls
    ledger = meter()
    with pytest.raises(Stopped) as error:
        PredictionData(recipe(), Capture(fail=True), FiniteRandom(1), ledger)
    assert str(error.value) == "ORACLE_FAILED" and error.value.__cause__ is None
    assert ledger.snapshot()["attempted_real_blocks"] == 2
    assert ledger.snapshot()["completed_real_blocks"] == 0


@pytest.mark.parametrize("raw", [b"", b"short", bytearray(16)])
def test_private_random_response_sanitized(raw: bytes) -> None:
    ledger = meter()
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE"):
        PredictionData(recipe(), Capture(), lambda _: raw, ledger)
    assert ledger.snapshot()["attempted_real_blocks"] == 2


def test_failed_test_attempt_cannot_be_retried() -> None:
    oracle, ledger = Capture(), meter()
    data = PredictionData(recipe(), oracle, FiniteRandom(1), ledger)
    data.release_test()
    oracle.fail = True
    with pytest.raises(Stopped, match="ORACLE_FAILED"):
        data.labels("test")
    oracle.fail = False
    with pytest.raises(Stopped, match="DOMAIN_REUSED"):
        data.labels("test")
    assert ledger.snapshot()["attempted_real_blocks"] == 6


def test_public_input_collision_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("random.Random.getrandbits", lambda self, bits: 0)
    oracle, random = Capture(), FiniteRandom(0)
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE"):
        PredictionData(recipe(), oracle, random, meter())
    assert not oracle.calls and not random.requests


@pytest.mark.parametrize("bad", ["TEST", "", "confirm"])
def test_invalid_split_is_not_a_domain(bad: str) -> None:
    data = PredictionData(recipe(), Capture(), FiniteRandom(1), meter())
    with pytest.raises(ValueError):
        data.inputs(bad)
    with pytest.raises(ValueError):
        data.labels(bad)


def test_shared_prebuilt_balance_pair_order_key_domains_and_meter() -> None:
    oracles = {t.cipher: Capture() for t in shared_recipe().tasks}
    private, ledger = FiniteRandom(), meter()
    data = DifferentialData(shared_recipe(), oracles, private, ledger)
    assert ledger.snapshot()["completed_real_blocks"] == 4*3*2*8
    assert ledger.snapshot()["attempted_null_words"] == 0
    assert private.requests == [16]*6 + [10]*6 + [16]*6 + [8]*6
    for task in shared_recipe().tasks:
        capture = oracles[task.cipher]
        assert len(capture.calls) == 6  # includes prebuilt TEST, not prediction timing
        assert len({call[1] for call in capture.calls}) == 6
        for split_index, split in enumerate(("train", "dev", "test")):
            x, y = data.panel(split, task)
            assert x.shape == (2, 4, 2, 64) and y.shape == (2, 1)
            assert sorted(y[:, 0].tolist()) == [0., 1.]
            calls = capture.calls[split_index*2:split_index*2+2]
            for row, (points, key, rounds) in enumerate(calls):
                assert rounds == task.rounds and len(points) == 8
                assert len(key) == {"PRESENT": 3, "SKINNY": 2}.get(task.cipher, 4)
                pairs = list(zip(points[::2], points[1::2], strict=True))
                if y[row, 0] == 1:
                    assert all(a ^ b == task.difference for a, b in pairs)
                else:
                    assert any(a ^ b != task.difference for a, b in pairs)
                expected = torch.tensor([[(p >> shift) & 1 for shift in range(63, -1, -1)]
                                         for p in points], dtype=torch.float32).reshape(4, 2, 64)
                assert torch.equal(x[row], expected)
            x.fill_(2)
            assert bool((data.panel(split, task)[0] <= 1).all())
    bundle = data.test_bundle()
    assert bundle.task_declarations == shared_recipe().tasks
    assert set(bundle.panels) == set(oracles)
    assert decode_bundle(encode_bundle(bundle)).task_declarations == shared_recipe().tasks
    assert ledger.snapshot()["attempted_real_blocks"] == 192
    with pytest.raises(ValueError):
        data.panel("dev", replace(shared_recipe().tasks[0], rounds=15))
    with pytest.raises(ValueError):
        data.panel("DEV", shared_recipe().tasks[0])


def test_shared_public_material_same_private_keys_different() -> None:
    a = {t.cipher: Capture() for t in shared_recipe().tasks}
    b = {t.cipher: Capture() for t in shared_recipe().tasks}
    DifferentialData(shared_recipe(), a, FiniteRandom(24), meter())
    DifferentialData(shared_recipe(), b, FiniteRandom(25), meter())
    for cipher in a:
        assert [c[0] for c in a[cipher].calls] == [c[0] for c in b[cipher].calls]
        assert [c[1] for c in a[cipher].calls] != [c[1] for c in b[cipher].calls]


def test_shared_duplicate_key_retry_limit_and_failed_attempts() -> None:
    oracle, ledger = Capture(), meter()
    requests = 0
    def duplicate(size: int) -> bytes:
        nonlocal requests
        requests += 1
        assert requests <= 129
        return bytes(size)
    with pytest.raises(Stopped, match="KEY_REUSE_LIMIT"):
        DifferentialData(shared_recipe(), {t.cipher: oracle for t in shared_recipe().tasks},
                         duplicate, ledger)
    assert len(oracle.calls) == 1
    assert ledger.snapshot()["attempted_real_blocks"] == 16
    assert ledger.snapshot()["completed_real_blocks"] == 8


@pytest.mark.parametrize("fail,response,reason", [(True, None, "ORACLE_FAILED"),
    (False, (0,) * 7, "ORACLE_RESPONSE"), (False, (0,) * 7 + (True,), "ORACLE_RESPONSE")])
def test_shared_failure_is_fixed_and_nonrefundable(fail: bool, response: tuple[int, ...] | None,
                                                   reason: str) -> None:
    ledger = meter()
    oracle = Capture(fail=fail, response=response)
    with pytest.raises(Stopped, match=reason) as error:
        DifferentialData(shared_recipe(), {t.cipher: oracle for t in shared_recipe().tasks},
                         FiniteRandom(), ledger)
    assert error.value.__cause__ is None
    assert ledger.snapshot()["attempted_real_blocks"] == 8
    assert ledger.snapshot()["completed_real_blocks"] == 0


def test_shared_budget_before_random_and_bad_oracle_map() -> None:
    oracle, private = Capture(), FiniteRandom(0)
    with pytest.raises(ValueError):
        DifferentialData(shared_recipe(), {}, private, meter())
    with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT"):
        DifferentialData(shared_recipe(), {t.cipher: oracle for t in shared_recipe().tasks},
                         private, meter(real=0))
    assert not private.requests and not oracle.calls


def test_new_ciphers_static_only_no_old_imports() -> None:
    base = Path(__file__).resolve().parents[1] / "src/cipher_lab/ciphers"
    for name in ("simeck", "present", "skinny"):
        source = (base / f"{name}.py").read_text()
        tree = ast.parse(source)
        assert any(isinstance(node, ast.FunctionDef) and node.name == "encrypt_many"
                   for node in tree.body)
        imports = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert all("cipher_experiments" not in ast.unparse(node) for node in imports)
        assert all("torch" not in ast.unparse(node) for node in imports)


def test_null_partial_success_is_credited_before_next_private_failure() -> None:
    private = FiniteRandom(1)
    ledger = meter()
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE"):
        PredictionData(recipe("null"), Capture(), private, ledger)
    assert ledger.snapshot()["attempted_null_words"] == 2
    assert ledger.snapshot()["completed_null_words"] == 1


def test_shared_duplicate_key_then_fresh_retry() -> None:
    fixtures = iter([1, 1, 2] + list(range(3, 25)))
    def private(size: int) -> bytes:
        return next(fixtures).to_bytes(size, "big")
    oracles = {t.cipher: Capture() for t in shared_recipe().tasks}
    DifferentialData(shared_recipe(), oracles, private, meter())
    assert oracles["SIMECK"].calls[0][1] == (1, 0, 0, 0)
    assert oracles["SIMECK"].calls[1][1] == (2, 0, 0, 0)


@pytest.mark.parametrize("phase", ["random", "oracle"])
def test_cooperative_expiry_never_calls_oracle_after_slow_key_and_credits_valid_response(
    phase: str,
) -> None:
    wall = 0.
    ledger = Meter(Policy(2, 1, 10, "2031-01-01T00:00:00Z", 100, 100, 100, 100, 0, 0),
                   Clocks(lambda: wall, lambda: 0.,
                          lambda: datetime(2030, 1, 1, tzinfo=UTC)), lambda: 0)
    capture = Capture()
    def private(size: int) -> bytes:
        nonlocal wall
        if phase == "random":
            wall = 2.
        return bytes(size)
    def oracle(points: tuple[int, ...], key: tuple[int, ...], rounds: int) -> Sequence[int]:
        nonlocal wall
        result = capture(points, key, rounds)
        wall = 2.
        return result
    with pytest.raises(Stopped, match="WALL_LIMIT"):
        PredictionData(recipe(), oracle, private, ledger)
    assert len(capture.calls) == (0 if phase == "random" else 1)
    assert ledger.snapshot()["attempted_real_blocks"] == 2
    assert ledger.snapshot()["completed_real_blocks"] == (0 if phase == "random" else 2)


def test_prediction_chunk_budget_and_partial_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cipher_lab.data.prediction._CHUNK", 1)
    capture, private, ledger = Capture(), FiniteRandom(1), meter(real=1)
    with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT"):
        PredictionData(recipe(), capture, private, ledger)
    assert len(capture.calls) == 1 and len(capture.calls[0][0]) == 1
    assert private.requests == [16]
    assert ledger.snapshot()["attempted_real_blocks"] == 1
    assert ledger.snapshot()["completed_real_blocks"] == 1


def test_prediction_oracle_chunks_never_repeat_public_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cipher_lab.data.prediction._CHUNK", 1)
    capture = Capture()
    data = PredictionData(recipe(), capture, FiniteRandom(1), meter())
    data.release_test()
    data.labels("test")
    assert len(capture.calls) == 6
    assert len({point for call in capture.calls for point in call[0]}) == 6
    assert len({call[1] for call in capture.calls}) == 1


def test_shared_meter_counts_first_group_when_next_group_refused() -> None:
    capture, private, ledger = Capture(), FiniteRandom(1), meter(real=8)
    with pytest.raises(Stopped, match="REAL_BLOCKS_LIMIT"):
        DifferentialData(shared_recipe(), {t.cipher: capture for t in shared_recipe().tasks},
                         private, ledger)
    assert private.requests == [16]
    assert len(capture.calls) == 1
    assert ledger.snapshot()["attempted_real_blocks"] == 8
    assert ledger.snapshot()["completed_real_blocks"] == 8
