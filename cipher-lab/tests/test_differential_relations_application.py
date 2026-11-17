from datetime import UTC
from typing import Any

import pytest
from cipher_lab_plugins.differential_relations import application
from test_differential_relations_training import Ledger


def test_gate_uses_both_production_batch_and_scalar_and_charges() -> None:
    calls: list[Any] = []
    meter = Ledger()

    def scalar(points: Any, key: Any, rounds: Any) -> Any:
        calls.append(("scalar", points, rounds))
        return tuple(0x44C8FC20B9DFA07A if rounds == 44 else p ^ 9 for p in points)

    def batch(points: Any, key: Any, rounds: Any) -> Any:
        calls.append(("batch", points, rounds))
        return scalar(points, key, rounds)

    application.correctness_gate(meter, scalar, batch)
    assert any(c[0] == "batch" and c[2] == 44 for c in calls)
    assert meter.attempted == meter.completed
    assert 1 < meter.completed["real_blocks"] <= 256


def test_gate_failure_does_not_complete_failed_call() -> None:
    meter = Ledger()
    with pytest.raises(ValueError, match="KAT_FAILED"):
        application.correctness_gate(meter, lambda *a: (0,), lambda *a: (0,))
    assert meter.attempted["real_blocks"] > meter.completed.get("real_blocks", 0)


@pytest.fixture
def fullfake(tmp_path: Any, monkeypatch: Any) -> Any:
    """Explicit tiny artificial matrix, not a modified production config file."""
    import itertools
    import json
    import os
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import torch
    from cipher_lab_plugins.differential_relations.config import Recipe
    from cipher_lab_plugins.differential_relations.data import OrbitSplits

    from cipher_lab import artifacts
    from cipher_lab.execution import Clocks

    calls: list[Any] = []
    tick = [0.0]
    clock = Clocks(lambda: tick[0], lambda: tick[0], lambda: datetime(2026, 11, 16, tzinfo=UTC))

    # Fail hard if any un-injected scientific/external default is touched.
    def forbidden(*a: Any, **k: Any) -> Any:
        pytest.fail("forbidden real scientific/resource/entropy call")

    import secrets

    from cipher_lab_plugins.differential_relations import resources

    from cipher_lab.ciphers import simon

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Linear, "forward", forbidden)
    monkeypatch.setattr(simon, "encrypt_many", forbidden)
    monkeypatch.setattr(simon, "encrypt_batch", forbidden)
    monkeypatch.setattr(secrets, "token_bytes", forbidden)
    monkeypatch.setattr(resources, "rss_bytes", forbidden)
    monkeypatch.setattr(resources, "_nvidia_smi_runner", forbidden)
    counter = itertools.count()
    monkeypatch.setattr(artifacts, "uuid4", lambda: SimpleNamespace(hex=f"{next(counter):032x}"))
    uuid = "GPU-12345678-1234-1234-1234-123456789ABC"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    # Original frozen config is untouched. All dimensions below are fake fixture inputs.
    values = Recipe().to_dict()
    values.update(
        train_small=4, train_large=8, dev=4, test=4, updates=2, dev_every=1, batch_groups=1
    )
    options = SimpleNamespace(**values)
    options.to_dict = lambda: dict(values)
    splits = OrbitSplits(
        np.arange(8, dtype=np.uint64),
        np.arange(8, 12, dtype=np.uint64),
        np.arange(12, 16, dtype=np.uint64),
        4,
    )

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", torch.tensor([0.0]))

        def forward(self, *a: Any) -> Any:
            forbidden()

    class FakeBackend:
        def __init__(self, *a: Any) -> None:
            pass

        def initialize(self, seed: Any, lr: Any) -> Any:
            calls.append(("initialize", seed, lr))
            return FakeModel()

        def update(self, model: Any, points: Any, labels: Any) -> Any:
            calls.append(("update", tuple(map(int, points)), tuple(map(tuple, labels.tolist()))))
            tick[0] += 1

        def predict(self, model: Any, points: Any) -> Any:
            calls.append(("forward", tuple(map(int, points))))
            tick[0] += 1
            return np.zeros((len(points), 4))

        def release(self, model: Any) -> Any:
            calls.append(("release",))

    class GPU:
        def check(self, limit: Any) -> Any:
            calls.append(("gpu",))
            return {"uuid": uuid, "pci": "00000000:01:00.0", "used_bytes": 1}

    def batch(points: Any, key: Any, rounds: Any) -> Any:
        calls.append(("oracle", points, rounds))
        if rounds == 44:
            return (0x44C8FC20B9DFA07A,)
        return tuple((p ^ 64) if p & 64 else 0 for p in points)

    def key() -> Any:
        calls.append(("key",))
        return object()

    def invoke(name: Any = "fake", backend: Any = FakeBackend, **override: Any) -> Any:
        store = artifacts.RunStore.claim(
            tmp_path,
            name,
            output_bytes=2**24,
            needs_models=True,
            model_slots=tuple(f"F{i}-best" for i in range(1, 6)),
        )
        kwargs = dict(
            store=store,
            recipe=options,
            root=tmp_path,
            experiment_id=name,
            device="cuda:0",
            gpu_uuid=uuid,
            gpu_pci="0000:01:00.0",
            clocks=clock,
            rss=lambda: 1,
            gpu=GPU(),
            oracles=(batch, batch),
            key_factory=key,
            splits_factory=lambda r: splits,
            backend_factory=backend,
            device_validator=lambda d: None,
        )
        kwargs.update(override)
        rc = application.execute(**kwargs)
        record = json.loads((tmp_path / "data" / name / "run.json").read_text())
        return rc, record

    return SimpleNamespace(
        invoke=invoke,
        calls=calls,
        clock=tick,
        recipe=options,
        backend=FakeBackend,
        root=tmp_path,
        model=FakeModel,
        uuid=uuid,
        monkeypatch=monkeypatch,
        forbidden=forbidden,
        Path=Path,
        os=os,
    )


def test_fullfake_five_fits_durable_freeze_one_test(fullfake: Any) -> None:
    rc, saved = fullfake.invoke()
    assert rc == 0, saved
    summary = saved["summary"]
    assert summary["completion"] == "COMPLETE" and summary["main_fit"] == "F3"
    assert [f["fit_id"] for f in summary["fits"]] == ["F1", "F2", "F3", "F4", "F5"]
    assert summary["fits"][2]["learning_rate"] == 0.001  # exact DEV tie
    assert all(f["selected_step"] == 1 and f["updates"] == 2 for f in summary["fits"])
    assert len(summary["selected_sha256"]) == 5
    assert len([c for c in fullfake.calls if c[0] == "key"]) == 1
    test_oracles = [c for c in fullfake.calls if c[0] == "oracle" and c[2] == 6 and 12 in c[1]]
    assert len(test_oracles) == 1
    final_test = [c for c in fullfake.calls if c[0] == "forward" and c[1] == (12, 13, 14, 15)]
    assert len(final_test) == 5
    assert saved["counts"]["completed_optimizer_updates"] == 10
    assert saved["counts"]["completed_forward_rows"] == 152
    assert saved["counts"]["completed_real_blocks"] == 50
    assert (fullfake.root / "result/fake/report.html").is_file()
    assert (fullfake.root / "data/fake/events.jsonl").is_file()
    assert len(list((fullfake.root / "models/fake").glob("*.pt"))) == 5
    assert summary["fits"][0]["budget"]["wall_seconds"] > 0
    assert summary["fits"][0]["budget"]["wall_seconds"] > 6  # includes deferred TEST


@pytest.mark.parametrize(
    "failure", ["update", "readback", "resource", "gate", "wall", "fitwall", "source"]
)
def test_first_failure_stops_no_retry_no_test(
    fullfake: Any, monkeypatch: Any, failure: Any
) -> None:
    from cipher_lab_plugins.differential_relations import checkpoint

    from cipher_lab import artifacts

    backend = fullfake.backend
    overrides: dict[str, Any] = {}
    if failure == "update":

        class Broken(backend):  # type: ignore[valid-type,misc]
            def update(self, *a: Any) -> Any:
                raise RuntimeError("unlogged secret detail")

        backend = Broken
    elif failure == "readback":
        monkeypatch.setattr(
            checkpoint,
            "load_checkpoint",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad SHA")),
        )
    elif failure == "resource":
        overrides["rss"] = lambda: None
    elif failure == "gate":
        overrides["oracles"] = (lambda *a: (0,), lambda *a: (0,))
    elif failure in ("wall", "fitwall"):

        class Slow(backend):  # type: ignore[valid-type,misc]
            def update(self, *a: Any) -> Any:
                fullfake.clock[0] += 28800 if failure == "wall" else 5400

        backend = Slow
    elif failure == "source":
        monkeypatch.setattr(artifacts, "source_identity", lambda: {"cipher_lab/fake.py": "a" * 64})
        # claim captures same fake source, so change it after the initial snapshot.
        original = application.plugin_source
        n = [0]

        def changed() -> Any:
            n[0] += 1
            return original() if n[0] == 1 else {"changed": True}

        monkeypatch.setattr(application, "plugin_source", changed)
    rc, saved = fullfake.invoke(backend=backend, **overrides)
    assert rc == 4
    assert saved["summary"]["completion"] == "PARTIAL"
    assert len(saved["summary"]["fits"]) <= 1
    assert not any(c[0] == "forward" and c[1] == (12, 13, 14, 15) for c in fullfake.calls)
    assert not any(c[0] == "oracle" and c[2] == 6 and 12 in c[1] for c in fullfake.calls)
    assert "unlogged secret detail" not in str(saved)
    if failure in ("resource", "gate", "source"):
        assert not any(c[0] == "key" for c in fullfake.calls)


def test_all_train_constant_zero_fit_and_test(fullfake: Any) -> None:
    def constant(points: Any, key: Any, rounds: Any) -> Any:
        if rounds == 44:
            return (0x44C8FC20B9DFA07A,)
        return tuple(0 for p in points)

    rc, saved = fullfake.invoke(oracles=(constant, constant))
    assert rc == 0
    assert saved["summary"]["completion"] == "ALL_TRAIN_CONSTANT"
    assert saved["summary"]["fits"] == []
    assert not any(c[0] in ("initialize", "update", "forward") for c in fullfake.calls)
    assert "test_hash" not in saved["summary"]


def test_binding_failure_before_entropy(fullfake: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    rc, saved = fullfake.invoke()
    assert rc == 4 and saved["summary"]["completion"] == "PARTIAL"
    assert not any(c[0] in ("key", "oracle") for c in fullfake.calls)


def test_f3_f5_same_batches_and_f5_real_test_targets(fullfake: Any) -> None:
    rc, saved = fullfake.invoke()
    assert rc == 0
    updates = [c for c in fullfake.calls if c[0] == "update"]
    assert [c[1] for c in updates[4:6]] == [c[1] for c in updates[8:10]]
    assert saved["summary"]["fits"][4]["identity"]["target_version"] == "row-permutation-v1"
    assert len(saved["summary"]["accuracy_comparisons"]) == 8
    assert len(saved["summary"]["baselines"]["large"]["constant"]["test"]["relations"]) == 4


def test_outside_fit_delay_stops_before_total_closing_window(fullfake: Any) -> None:
    class Slow(fullfake.backend):  # type: ignore[misc]
        def release(self, model: Any) -> Any:
            super().release(model)
            fullfake.clock[0] = 28500

    rc, saved = fullfake.invoke(backend=Slow)
    assert rc == 4
    assert len(saved["summary"]["fits"]) == 1
    # release() is outside the active fit segment: shared 1800s expires first.
    assert saved["summary"]["reason"] == "SHARED_WALL_LIMIT"
    assert not any(c[0] == "forward" and c[1] == (12, 13, 14, 15) for c in fullfake.calls)


def test_final_test_uses_existing_fit_budget(fullfake: Any) -> None:
    class SlowTest(fullfake.backend):  # type: ignore[misc]
        def predict(self, model: Any, points: Any) -> Any:
            if tuple(points) == (12, 13, 14, 15):
                fullfake.clock[0] += 5395
            return super().predict(model, points)

    rc, saved = fullfake.invoke(backend=SlowTest)
    assert rc == 4
    assert saved["summary"]["reason"] == "ARM_WALL_LIMIT"
    assert saved["summary"]["completion"] == "PARTIAL"
    assert len([c for c in fullfake.calls if c[0] == "forward" and c[1] == (12, 13, 14, 15)]) == 1
    assert "main_comparisons" not in saved["summary"]


def test_five_fit_fullfake_saved_evaluate_and_cross_experiment_rejection(
    fullfake: Any, monkeypatch: Any
) -> None:
    from cipher_lab_plugins.differential_relations import evaluate

    # The saved-reader fixture accepts only this explicit artificial recipe.
    monkeypatch.setattr(evaluate, "Recipe", lambda: fullfake.recipe)
    rc, saved = fullfake.invoke()
    assert rc == 0

    class FakePredictor:
        def model(self) -> Any:
            return fullfake.model()

        def predict(self, model: Any, points: Any) -> Any:
            return fullfake.backend().predict(model, points)

    root = fullfake.root
    inputs = dict(
        run_record=root / "data/fake/run.json",
        checkpoint=root / "models/fake/F3-best.pt",
        dataset=root / "data/fake/dataset/final.npz",
        output=root / "evaluation",
        predictor_factory=FakePredictor,
    )
    original = (root / "data/fake/run.json").read_bytes()
    result = evaluate.evaluate_saved(**inputs)
    assert result["summary"]["checkpoint_sha256"] == saved["summary"]["selected_sha256"]["F3"]
    assert (root / "data/fake/run.json").read_bytes() == original
    assert (root / "evaluation/evaluation.json").is_file()
    assert (root / "evaluation/report.html").is_file()
    with pytest.raises(FileExistsError):
        evaluate.evaluate_saved(**inputs)
    rc, other = fullfake.invoke(name="other")
    assert rc == 0
    inputs.update(output=root / "bad-evaluation", checkpoint=root / "models/other/F3-best.pt")
    with pytest.raises(ValueError, match="CHECKPOINT_REFERENCE"):
        evaluate.evaluate_saved(**inputs)
    assert not (root / "bad-evaluation").exists()


def test_cli_fake_claim_to_report(fullfake: Any, monkeypatch: Any) -> None:
    from cipher_lab_plugins.differential_relations import cli

    from cipher_lab.artifacts import RunStore

    # Exercise public CLI argument forwarding and real claim; inject only execution seams.
    real = application.execute

    def execution(**kwargs: Any) -> Any:
        # Use a separately claimed fake fixture cycle to avoid production scientific defaults.
        assert kwargs["gpu_uuid"] == fullfake.uuid
        assert kwargs["device"] == "cuda:0"
        # The actual passed owner is used for the fake complete loop.
        from datetime import datetime

        import numpy as np
        from cipher_lab_plugins.differential_relations.data import OrbitSplits

        from cipher_lab.execution import Clocks

        class GPU:
            def check(self, limit: Any) -> Any:
                return {"used_bytes": 0}

        def batch(points: Any, key: Any, rounds: Any) -> Any:
            return (
                (0x44C8FC20B9DFA07A,)
                if rounds == 44
                else tuple(p ^ 64 if p & 64 else 0 for p in points)
            )

        kwargs.update(
            recipe=fullfake.recipe,
            clocks=Clocks(lambda: 0.0, lambda: 0.0, lambda: datetime(2026, 11, 16, tzinfo=UTC)),
            rss=lambda: 1,
            gpu=GPU(),
            oracles=(batch, batch),
            key_factory=lambda: object(),
            splits_factory=lambda r: OrbitSplits(
                np.arange(8, dtype=np.uint64),
                np.arange(8, 12, dtype=np.uint64),
                np.arange(12, 16, dtype=np.uint64),
                4,
            ),
            backend_factory=fullfake.backend,
            device_validator=lambda d: None,
        )
        return real(**kwargs)

    monkeypatch.setattr(application, "execute", execution)
    monkeypatch.setattr(
        cli,
        "_claim",
        lambda root, id, limit: RunStore.claim(
            root,
            id,
            output_bytes=2**24,
            needs_models=True,
            model_slots=tuple(f"F{i}-best" for i in range(1, 6)),
        ),
    )
    rc = cli.main(
        [
            "run",
            "--config",
            "configs/differential-relations-simon6.json",
            "--root",
            str(fullfake.root),
            "--id",
            "via-cli",
            "--device",
            "cuda:0",
            "--gpu-uuid",
            fullfake.uuid.lower().replace("gpu-", "GPU-"),
            "--gpu-pci",
            "0000:01:00.0",
        ]
    )
    assert rc == 0
    assert (fullfake.root / "result/via-cli/report.html").is_file()


def test_sealed_storage_failure_never_retries_terminal(fullfake: Any, monkeypatch: Any) -> None:
    from cipher_lab.artifacts import ArtifactError, RunStore

    terminal_calls = []
    original = RunStore.finalize

    def terminal(self: Any, value: Any) -> None:
        terminal_calls.append(value)
        original(self, value)

    def broken_read(self: Any, *a: Any, **k: Any) -> Any:
        self._failed("READ_FAILED")
        raise ArtifactError("READ_FAILED")

    monkeypatch.setattr(RunStore, "read_model", broken_read)
    monkeypatch.setattr(RunStore, "finalize", terminal)
    rc, saved = fullfake.invoke()
    assert rc == 4 and saved["status"] == "FAILED"
    assert terminal_calls == []


def test_saved_evaluate_rejects_all_frozen_sha_mismatch(fullfake: Any, monkeypatch: Any) -> None:
    import json

    from cipher_lab_plugins.differential_relations import evaluate

    monkeypatch.setattr(evaluate, "Recipe", lambda: fullfake.recipe)
    rc, saved = fullfake.invoke()
    assert rc == 0
    # Tamper F4 in original record, while evaluating F3: all five must bind.
    saved["summary"]["selected_sha256"]["F4"] = "a" * 64
    record = fullfake.root / "data/fake/run.json"
    record.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="FROZEN_MATRIX_IDENTITY"):
        evaluate.evaluate_saved(
            run_record=record,
            checkpoint=fullfake.root / "models/fake/F3-best.pt",
            dataset=fullfake.root / "data/fake/dataset/final.npz",
            output=fullfake.root / "tampered",
            predictor_factory=lambda: pytest.fail("forward before identity check"),
        )


def test_gpu_throttle_records_max_and_forces_closing_sample(fullfake: Any) -> None:
    reads = []

    class Probe:
        def check(self, limit: Any) -> Any:
            reads.append(fullfake.clock[0])
            return {"used_bytes": len(reads), "uuid": fullfake.uuid}

    rc, saved = fullfake.invoke(gpu=Probe())
    assert rc == 0
    summary = saved["summary"]
    assert summary["gpu_min_interval_seconds"] == 1.0
    assert summary["gpu_max_observed_bytes"] == len(reads)
    assert summary["gpu_probe_count"] == len(reads)
    assert reads[0] == 0 and reads[-1] == fullfake.clock[0]
    # Public gate and label batches at t=0 reuse first reading, not per-pair spawning.
    assert reads.count(0) == 1


@pytest.mark.parametrize("fit_id", ["F1", "F5"])
def test_saved_small_and_null_targets(fullfake: Any, monkeypatch: Any, fit_id: str) -> None:
    from cipher_lab_plugins.differential_relations import evaluate

    monkeypatch.setattr(evaluate, "Recipe", lambda: fullfake.recipe)
    rc, saved = fullfake.invoke()
    assert rc == 0

    class FakePredictor:
        def model(self) -> Any:
            return fullfake.model()

        def predict(self, model: Any, points: Any) -> Any:
            return fullfake.backend().predict(model, points)

    result = evaluate.evaluate_saved(
        run_record=fullfake.root / "data/fake/run.json",
        checkpoint=fullfake.root / f"models/fake/{fit_id}-best.pt",
        dataset=fullfake.root / "data/fake/dataset/final.npz",
        output=fullfake.root / f"eval-{fit_id}",
        predictor_factory=FakePredictor,
    )
    assert result["summary"]["fits"][0]["fit_id"] == fit_id


@pytest.mark.parametrize("stop", [False, True])
def test_one_necessary_dataset_on_completion_or_controlled_stop(fullfake: Any, stop: bool) -> None:
    from cipher_lab_plugins.differential_relations.checkpoint import load_saved_dataset

    class Broken(fullfake.backend):  # type: ignore[misc]
        def update(self, *a: Any) -> None:
            raise RuntimeError("stop")

    rc, saved = fullfake.invoke(backend=Broken if stop else fullfake.backend)
    assert rc == (4 if stop else 0)
    files = list((fullfake.root / "data/fake/dataset").glob("*.npz"))
    assert len(files) == 1
    decoded = load_saved_dataset(files[0], experiment_id="fake")
    assert (decoded.test_labels is None) == stop
    assert saved["summary"]["dataset_ref"].startswith("dataset/")


def test_application_temporaries_are_under_root_and_cleaned(
    fullfake: Any, monkeypatch: Any
) -> None:
    from tempfile import TemporaryDirectory

    created = []

    def temporary(*a: Any, **k: Any) -> Any:
        assert k.get("dir") == fullfake.root, "system temp forbidden"
        context = TemporaryDirectory(*a, **k)
        created.append(fullfake.Path(context.name))
        return context

    monkeypatch.setattr(application, "TemporaryDirectory", temporary)
    rc, saved = fullfake.invoke()
    assert rc == 0, saved
    assert created and all(not path.exists() for path in created)


def test_deadline_does_not_start_dataset_preservation(fullfake: Any) -> None:
    class Expired(fullfake.backend):  # type: ignore[misc]
        def update(self, *a: Any) -> None:
            fullfake.clock[0] = 28801
            raise RuntimeError("stop")

    rc, saved = fullfake.invoke(backend=Expired)
    assert rc == 4
    assert not list((fullfake.root / "data/fake").rglob("*.npz"))
    assert saved["summary"]["dataset_preservation"] == "NOT_SAVED_DEADLINE"
    assert saved["summary"]["timing"]["actual_elapsed_seconds"] == 28801
    assert saved["summary"]["timing"]["overrun_seconds"] == 1
    assert saved["summary"]["closing_gpu_sample"]["status"] == "NOT_ATTEMPTED"


@pytest.mark.parametrize("stage", ["serialize", "store"])
def test_dataset_late_boundary_stops_before_more_work(
    fullfake: Any, monkeypatch: Any, stage: str
) -> None:
    from cipher_lab_plugins.differential_relations import data

    from cipher_lab.artifacts import RunStore

    if stage == "serialize":
        original = data.save_dataset

        def serialize(*a: Any, **k: Any) -> Any:
            value = original(*a, **k)
            fullfake.clock[0] = 28801
            return value

        monkeypatch.setattr(data, "save_dataset", serialize)
    else:
        writer = RunStore.write_dataset

        def store(self: Any, *a: Any, **k: Any) -> Any:
            value = writer(self, *a, **k)
            fullfake.clock[0] = 28801
            return value

        monkeypatch.setattr(RunStore, "write_dataset", store)
    rc, saved = fullfake.invoke()
    assert rc == 4 and saved["status"] != "SUCCEEDED"
    assert saved["summary"]["timing"]["overrun_seconds"] == 1
    assert not any(c[0] == "forward" and c[1] == (12, 13, 14, 15) for c in fullfake.calls)
    if stage == "serialize":
        assert not list((fullfake.root / "data/fake").rglob("*.npz"))


def test_late_report_returns_failure_without_writing_html(
    fullfake: Any, monkeypatch: Any, capsys: Any
) -> None:
    from cipher_lab_plugins.differential_relations import report

    original = report.render_report

    def render(*a: Any) -> Any:
        result = original(*a)
        fullfake.clock[0] = 28801
        return result

    monkeypatch.setattr(report, "render_report", render)
    rc, saved = fullfake.invoke()
    assert rc == 4
    assert not (fullfake.root / "result/fake/report.html").exists()
    assert saved["summary"]["timing"]["whole_cycle_within_wall"] is None
    assert "CLOSING_FAILED" in capsys.readouterr().err


def test_bad_gpu_is_not_retried_at_close(fullfake: Any) -> None:
    from cipher_lab_plugins.differential_relations.resources import ResourceError

    calls = []

    class Bad:
        def check(self, limit: Any) -> Any:
            calls.append(1)
            raise ResourceError("GPU_UNAVAILABLE")

    rc, saved = fullfake.invoke(gpu=Bad())
    assert rc == 4 and len(calls) == 1
    assert saved["summary"]["closing_gpu_sample"] == {
        "status": "NOT_ATTEMPTED",
        "reason": "GPU_UNAVAILABLE",
    }


def test_every_fit_saves_test_paired_counts_and_train_constant_flags(fullfake: Any) -> None:
    rc, saved = fullfake.invoke()
    assert rc == 0
    summary = saved["summary"]
    assert summary["train_constant_relations"] == [False, True, True, True]
    for entry in summary["fits"]:
        assert len(entry["accuracy_comparisons"]) == 8
        assert entry["baseline_size"] == ("small" if entry["fit_id"] in ("F1", "F2") else "large")
        assert entry["comparison_role"] == (
            "prespecified-main-secondary" if entry["fit_id"] == "F3" else "descriptive-only"
        )
        assert all(
            "model_only_correct" in c and "baseline_only_correct" in c
            for c in entry["accuracy_comparisons"]
        )
    assert summary["accuracy_comparisons"] == summary["fits"][2]["accuracy_comparisons"]


@pytest.mark.parametrize("failed_step", [500, 1500])
def test_non_dev_failure_preserves_actual_updates_and_budget(
    fullfake: Any, failed_step: int
) -> None:
    fullfake.recipe.updates = 2048
    fullfake.recipe.dev_every = 1024

    class Interrupted(fullfake.backend):  # type: ignore[misc]
        def update(self, *a: Any) -> None:
            count = len([c for c in fullfake.calls if c[0] == "update"])
            if count == failed_step - 1:
                raise RuntimeError("interrupted")
            # Artificial microseconds, not an altered formal wall limit.
            super().update(*a)
            fullfake.clock[0] -= 0.999

    rc, saved = fullfake.invoke(backend=Interrupted)
    assert rc == 4
    entry = saved["summary"]["fits"][0]
    assert entry["updates"] == failed_step - 1
    assert saved["counts"]["completed_optimizer_updates"] == entry["updates"]
    assert entry["budget"]["wall_seconds"] > 0
    assert entry["budget"]["active_segment_included"] is False


def test_test_generation_failure_saves_train_dev_without_frozen(
    fullfake: Any, monkeypatch: Any
) -> None:
    from cipher_lab_plugins.differential_relations import data

    attempts = []

    def broken(self: Any, token: Any) -> Any:
        attempts.append(1)
        raise ValueError("ORACLE_FAILED")

    monkeypatch.setattr(data.DataSession, "test_labels", broken)
    rc, saved = fullfake.invoke()
    assert rc == 4 and attempts == [1]
    decoded = data.load_dataset(fullfake.root / "data/fake/dataset/final.npz", experiment_id="fake")
    assert decoded[2] is None
    assert saved["summary"]["dataset_preservation"] == "SAVED"


def test_auxiliary_saved_evaluation_never_claims_main_signal(
    fullfake: Any, monkeypatch: Any
) -> None:
    from cipher_lab_plugins.differential_relations import evaluate

    monkeypatch.setattr(evaluate, "Recipe", lambda: fullfake.recipe)
    rc, saved = fullfake.invoke()
    assert rc == 0

    class Predictor:
        def model(self) -> Any:
            return fullfake.model()

        def predict(self, model: Any, points: Any) -> Any:
            return fullfake.backend().predict(model, points)

    result = evaluate.evaluate_saved(
        run_record=fullfake.root / "data/fake/run.json",
        checkpoint=fullfake.root / "models/fake/F5-best.pt",
        dataset=fullfake.root / "data/fake/dataset/final.npz",
        output=fullfake.root / "auxiliary",
        predictor_factory=Predictor,
    )
    summary = result["summary"]
    assert "main_comparisons" not in summary
    assert "selected_comparisons" in summary
    assert len(summary["baselines"]["large"]["constant"]["test"]["relations"]) == 4


@pytest.mark.parametrize("stage", ["finalize", "report-write"])
def test_late_frozen_io_returns_four_and_discloses_timing(
    fullfake: Any, monkeypatch: Any, capsys: Any, stage: str
) -> None:
    from cipher_lab.artifacts import RunStore

    if stage == "finalize":
        original = RunStore.finalize

        def finish(self: Any, *a: Any, **k: Any) -> Any:
            result = original(self, *a, **k)
            fullfake.clock[0] = 28802
            return result

        monkeypatch.setattr(RunStore, "finalize", finish)
    else:
        writer = RunStore.write_html_report

        def report(self: Any, *a: Any, **k: Any) -> Any:
            result = writer(self, *a, **k)
            fullfake.clock[0] = 28802
            return result

        monkeypatch.setattr(RunStore, "write_html_report", report)
    rc, saved = fullfake.invoke()
    assert rc == 4
    assert saved["summary"]["timing"]["whole_cycle_within_wall"] is None
    assert 'overrun_seconds": 2.0' in capsys.readouterr().err


def test_test_budget_timeout_snapshot_includes_deferred_segment(fullfake: Any) -> None:
    class Slow(fullfake.backend):  # type: ignore[misc]
        def predict(self, model: Any, points: Any) -> Any:
            if tuple(points) == (12, 13, 14, 15):
                fullfake.clock[0] += 5395
            return super().predict(model, points)

    rc, saved = fullfake.invoke(backend=Slow)
    assert rc == 4
    entry = saved["summary"]["fits"][0]
    assert entry["budget"]["wall_seconds"] >= 5400
    assert entry["budget"]["active_segment_included"] is False
    assert (
        sum(f["updates"] for f in saved["summary"]["fits"])
        == saved["counts"]["completed_optimizer_updates"]
    )


def test_saved_evaluate_uses_exact_output_parent_temps_and_cleans(
    fullfake: Any, monkeypatch: Any
) -> None:
    from tempfile import TemporaryDirectory

    from cipher_lab_plugins.differential_relations import evaluate

    monkeypatch.setattr(evaluate, "Recipe", lambda: fullfake.recipe)
    rc, saved = fullfake.invoke()
    assert rc == 0
    created = []

    def temporary(*a: Any, **k: Any) -> Any:
        assert k.get("dir") == fullfake.root
        result = TemporaryDirectory(*a, **k)
        created.append(fullfake.Path(result.name))
        return result

    monkeypatch.setattr(evaluate, "TemporaryDirectory", temporary)

    class Predictor:
        def model(self) -> Any:
            return fullfake.model()

        def predict(self, model: Any, points: Any) -> Any:
            return fullfake.backend().predict(model, points)

    evaluate.evaluate_saved(
        run_record=fullfake.root / "data/fake/run.json",
        checkpoint=fullfake.root / "models/fake/F3-best.pt",
        dataset=fullfake.root / "data/fake/dataset/final.npz",
        output=fullfake.root / "evaluated",
        predictor_factory=Predictor,
    )
    assert len(created) == 2 and all(not p.exists() for p in created)


def test_failed_closing_gpu_sample_not_relabelled_as_never_attempted(fullfake: Any) -> None:
    from cipher_lab_plugins.differential_relations.resources import ResourceError

    reads = []

    class Probe:
        def check(self, limit: Any) -> Any:
            reads.append(fullfake.clock[0])
            if fullfake.clock[0] >= 35:
                raise ResourceError("GPU_UNAVAILABLE")
            return {"used_bytes": 1}

    rc, saved = fullfake.invoke(gpu=Probe())
    assert rc == 4
    assert saved["summary"]["closing_gpu_sample"] == {
        "status": "FAILED",
        "reason": "GPU_UNAVAILABLE",
    }


@pytest.mark.parametrize("stage", ["sampling", "baseline", "dataset", "report"])
def test_shared_wall_limit_independent_of_fast_fits(
    fullfake: Any, monkeypatch: Any, stage: str, capsys: Any
) -> None:
    from cipher_lab_plugins.differential_relations import data, metrics, report

    if stage == "sampling":
        original_sample = data.sample_orbits

        # Explicit injected artificial split factory; never call production sampling.
        def sample(recipe: Any) -> Any:
            import numpy as np

            fullfake.clock[0] += 1801
            return data.OrbitSplits(
                np.arange(8, dtype=np.uint64),
                np.arange(8, 12, dtype=np.uint64),
                np.arange(12, 16, dtype=np.uint64),
                4,
            )

        override = {"splits_factory": sample}
        assert callable(original_sample)
    else:
        override = {}
        module, name = (
            (metrics, "fit_baselines")
            if stage == "baseline"
            else (data, "save_dataset")
            if stage == "dataset"
            else (report, "render_report")
        )
        original = getattr(module, name)

        def slow(*a: Any, **k: Any) -> Any:
            value = original(*a, **k)
            fullfake.clock[0] += 1801
            return value

        monkeypatch.setattr(module, name, slow)
    rc, saved = fullfake.invoke(**override)
    assert rc == 4
    if stage == "report":
        assert not (fullfake.root / "result/fake/report.html").exists()
        assert "SHARED_WALL_LIMIT" in capsys.readouterr().err
    else:
        assert saved["summary"]["reason"] == "SHARED_WALL_LIMIT"
        assert saved["summary"]["shared_wall_seconds"] >= 1801
        assert saved["status"] != "SUCCEEDED"
        assert not list((fullfake.root / "data/fake").rglob("*.npz"))
    assert saved["summary"]["shared_wall_limit_seconds"] == 1800


def test_fit_active_time_is_not_charged_twice_as_shared(fullfake: Any) -> None:
    class LongFits(fullfake.backend):  # type: ignore[misc]
        def update(self, *a: Any) -> None:
            super().update(*a)
            fullfake.clock[0] += 399

    rc, saved = fullfake.invoke(backend=LongFits)
    assert rc == 0
    assert saved["counts"]["wall_elapsed_seconds"] > 1800
    assert saved["summary"]["shared_wall_seconds"] < 1
    assert all(f["budget"]["wall_seconds"] >= 800 for f in saved["summary"]["fits"])
