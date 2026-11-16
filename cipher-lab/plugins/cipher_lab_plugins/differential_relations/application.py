"""One fixed five-fit protocol: assembly, timing, identity and owner persistence."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def plugin_source() -> dict[str, object]:
    """Explicit local Python files; self-description, not independent attestation."""
    names = (
        "__init__",
        "config",
        "data",
        "metrics",
        "checkpoint",
        "training",
        "application",
        "evaluate",
        "resources",
        "report",
        "cli",
    )
    base = Path(__file__).absolute().parent
    hashes = {}
    actual = {p.name for p in base.glob("*.py")}
    if actual != {name + ".py" for name in names}:
        raise ValueError("PLUGIN_SOURCE_CHANGED")
    for name in names:
        path = base / (name + ".py")
        if path.is_symlink() or not path.is_file():
            raise ValueError("PLUGIN_SOURCE_CHANGED")
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    return {"kind": "local-source-self-description-not-git-attestation", "files": hashes}


def correctness_gate(meter: Any, scalar: Callable[..., Any], batch: Callable[..., Any]) -> None:
    """One official full-round KAT and public r6 scalar/array consistency.

    Vector: SIMON/SPECK Implementation Guide, SIMON64/128; low-first key
    words and high-x/low-y packing match docs/provenance.json SIMON entry.
    Scalar and array share a design ancestry: NOT independent-design validation.
    """
    key = (0x03020100, 0x0B0A0908, 0x13121110, 0x1B1A1918)
    point = (0x656B696C20646E75,)
    expected = (0x44C8FC20B9DFA07A,)
    for oracle in (batch, scalar):
        meter.reserve("real_blocks", 1)
        if oracle(point, key, 44) != expected:
            raise ValueError("KAT_FAILED")
        meter.complete("real_blocks", 1)
    points = (
        0,
        64,
        1,
        65,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFBF,
        0x0123456789ABCDEF,
        0x0123456789ABCDAF,
    )
    meter.reserve("real_blocks", len(points))
    reference = scalar(points, key, 6)
    if (
        type(reference) is not tuple
        or len(reference) != len(points)
        or any(type(v) is not int or not 0 <= v < 2**64 for v in reference)
    ):
        raise ValueError("ORACLE_RESPONSE")
    meter.complete("real_blocks", len(points))
    meter.reserve("real_blocks", len(points))
    if batch(points, key, 6) != reference:
        raise ValueError("ORACLE_MISMATCH")
    meter.complete("real_blocks", len(points))
    meter.check()


def production_oracles() -> tuple[Any, Any]:
    import numpy as np

    from cipher_lab.ciphers.simon import encrypt_batch, encrypt_many

    def batch(points: tuple[int, ...], key: Any, rounds: int) -> tuple[int, ...]:
        return tuple(
            int(v) for v in encrypt_batch(np.asarray(points, dtype=np.uint64), key, rounds)
        )

    return encrypt_many, batch


def private_key() -> tuple[int, ...]:
    import secrets

    raw = secrets.token_bytes(16)
    return tuple(int.from_bytes(raw[i : i + 4], "little") for i in range(0, 16, 4))


def validate_cuda(device: str) -> None:
    """Actual runtime availability before KAT/key; no implicit alternative device."""
    import torch

    if device != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("CUDA_UNAVAILABLE")
    torch.cuda.get_device_properties(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class _SharedWallLimit(Exception):
    reason = "SHARED_WALL_LIMIT"


def _budget_snapshot(budget: Any) -> dict[str, object]:
    saved: dict[str, object] = budget.snapshot()
    active = budget.start is not None and budget.latest is not None
    if active:
        saved["wall_seconds"] = budget.total[0] + budget.latest[0] - budget.start[0]
        saved["cpu_seconds"] = budget.total[1] + budget.latest[1] - budget.start[1]
    saved["active_segment_included"] = bool(active)
    return saved


def execute(
    *,
    store: Any,
    recipe: Any,
    root: Path,
    experiment_id: str,
    device: str,
    gpu_uuid: str,
    gpu_pci: str,
    clocks: Any = None,
    rss: Any = None,
    gpu: Any = None,
    oracles: Any = None,
    key_factory: Callable[[], Any] = private_key,
    splits_factory: Any = None,
    backend_factory: Any = None,
    device_validator: Callable[[str], None] = validate_cuda,
) -> int:
    """All injectable seams default to production, never a tiny scientific recipe.

    Exceptions stop the matrix with no retry. A storage failure may seal the
    owner before a terminal can be written; its existing FAILED/RUNNING record
    remains authoritative and is never described as scientific completion.
    """
    from cipher_lab.artifacts import read_saved, source_identity
    from cipher_lab.config import Policy
    from cipher_lab.execution import Clocks, Meter, Stopped

    from .resources import GPUProbe, ResourceError, rss_bytes, validate_device_binding

    summary: dict[str, Any] = {
        "method": recipe.method,
        "experiment_id": experiment_id,
        "completion": "PARTIAL",
        "reason": "STARTING",
        "main_fit": "F3",
        "fits": [],
        "retry_count": 0,
        "shared_wall_seconds": 0.0,
        "shared_wall_limit_seconds": 1800,
        "shared_wall_semantics": (
            "total wall minus completed and current active fit segments; "
            "fit IO stays in fit, all other work including closing IO is shared"
        ),
        "closing_gpu_sample": {"status": "NOT_ATTEMPTED", "reason": "NOT_REACHED"},
        "cpu_hard_limit": False,
        "limits_semantics": "cooperative wall/RSS/GPU; native calls are not hard-killed",
        "dataset_storage": (
            "one final.npz at completion or controlled stop; hard crash/OOM can lose unsaved data; "
            "no private-key resume"
        ),
        "f5_targets": {
            "train": "row-permutation-v1",
            "dev": "row-permutation-v1",
            "test": "real-v1",
        },
    }
    prepared: Any = None
    null: Any = None
    test: Any = None
    frozen: Any = None
    dataset_attempted = False
    session: Any = None
    meter: Any = None
    active_backend: Any = None
    budgets: dict[str, Any] = {}
    status = "FAILED"
    public_recipe: dict[str, Any] = recipe.to_dict()
    try:
        root = Path(os.path.abspath(root))
        for path in (*reversed(root.parents), root):
            if path.is_symlink() or (
                path.exists() and getattr(path.stat(), "st_file_attributes", 0) & 0x400
            ):
                raise ValueError("UNSAFE_ROOT")
        if not root.is_dir():
            raise ValueError("ROOT_MISSING")
        clocks = Clocks() if clocks is None else clocks
        policy = Policy(
            schema=2,
            wall_seconds=recipe.wall_seconds,
            cpu_seconds=recipe.wall_seconds * 1024,
            deadline_utc=(clocks.utc() + timedelta(seconds=recipe.wall_seconds)).isoformat(),
            real_blocks=recipe.real_blocks,
            null_words=0,
            rss_stop_bytes=recipe.rss_bytes,
            output_bytes=recipe.artifact_bytes,
            optimizer_updates=recipe.optimizer_updates,
            forward_rows=recipe.forward_pairs,
        )
        meter = Meter(policy, clocks, rss_bytes if rss is None else rss, enforce_cpu=False)
        start_wall = meter._start_wall

        def shared_wall() -> float:
            now = clocks.wall()
            fit_wall = sum(
                budget.total[0] + (now - budget.start[0] if budget.start is not None else 0.0)
                for budget in budgets.values()
            )
            value = float(max(0.0, now - start_wall - fit_wall))
            summary["shared_wall_seconds"] = value
            return value

        def boundary() -> None:
            # Total limit has priority; shared uses clock differences, never a nested arm.
            if clocks.wall() - start_wall >= recipe.wall_seconds:
                shared_wall()
                raise Stopped("WALL_LIMIT")
            meter.check()
            if shared_wall() >= 1800:
                raise _SharedWallLimit()

        initial_source = plugin_source()
        public_recipe["plugin_source"] = initial_source
        core_source = store.source_snapshot()

        def unchanged() -> None:
            boundary()
            if plugin_source() != initial_source or source_identity() != core_source:
                raise ValueError("SOURCE_CHANGED")

        store.write_snapshot(
            {
                "recipe": public_recipe,
                "policy": policy.to_dict(),
                "summary": summary,
                "counts": meter.snapshot(),
            }
        )
        boundary()
        store.append_event(
            {
                "kind": "START",
                "summary": {"device": device, "gpu_uuid": gpu_uuid, "gpu_pci": gpu_pci},
            }
        )
        boundary()
        validate_device_binding(device, gpu_uuid, gpu_pci, os.environ)
        probe = GPUProbe(gpu_uuid, gpu_pci) if gpu is None else gpu

        def closing_gpu_check() -> None:
            summary["closing_gpu_sample"] = {"status": "ATTEMPTED", "reason": "NORMAL_CLOSE"}
            try:
                gpu_check(force=True)
            except (Exception, KeyboardInterrupt) as error:
                reason = (
                    error.reason
                    if isinstance(error, (ResourceError, Stopped, _SharedWallLimit))
                    else "FAILED"
                )
                summary["closing_gpu_sample"] = {"status": "FAILED", "reason": reason}
                raise
            summary["closing_gpu_sample"] = {"status": "READ", "reason": "NORMAL_CLOSE"}

        last_gpu_read: float | None = None
        summary["gpu_probe_count"] = 0
        summary["gpu_min_interval_seconds"] = 1.0

        def gpu_check(*, force: bool = False) -> None:
            nonlocal last_gpu_read
            boundary()
            now = clocks.wall()
            if force or last_gpu_read is None or now - last_gpu_read >= 1.0:
                summary["resource"] = probe.check(recipe.gpu_bytes)
                summary["gpu_probe_count"] += 1
                summary["gpu_max_observed_bytes"] = max(
                    summary.get("gpu_max_observed_bytes", 0), summary["resource"]["used_bytes"]
                )
                last_gpu_read = clocks.wall()
                summary["gpu_last_elapsed_seconds"] = meter.snapshot()["wall_elapsed_seconds"]
                boundary()

        summary["gpu_sampling"] = (
            "oracle/neural batch boundaries, at most once per second; "
            "not continuous peak enforcement"
        )

        class CheckedMeter:
            """RSS per reservation; GPU only at bounded batch boundaries."""

            def check(self) -> None:
                boundary()

            def reserve(self, kind: str, amount: int) -> None:
                self.check()
                if kind == "forward_rows":
                    gpu_check()
                meter.reserve(kind, amount)

            def complete(self, kind: str, amount: int) -> None:
                meter.complete(kind, amount)

        checked = CheckedMeter()
        gpu_check(force=True)
        device_validator(device)
        unchanged()
        # Only now import scientific assembly and production adapters.
        from . import checkpoint, data, evaluate, training

        scalar, raw_batch = production_oracles() if oracles is None else oracles

        def batch(points: tuple[int, ...], key: Any, rounds: int) -> tuple[int, ...]:
            gpu_check()
            result: tuple[int, ...] = raw_batch(points, key, rounds)
            return result

        correctness_gate(checked, scalar, batch)
        unchanged()
        splits = (
            data.sample_orbits(data.SplitSizes.from_recipe(recipe), seed=recipe.sample_seed)
            if splits_factory is None
            else splits_factory(recipe)
        )
        checked.check()
        # A single unlogged private key; only DataSession retains it.
        session = data.DataSession(
            experiment_id,
            splits,
            key=key_factory(),
            meter=checked,
            batch_oracle=lambda points, key: batch(points, key, 6),
        )
        prepared = session.prepare()
        summary.update(
            train_hash=prepared.train_hash,
            dev_hash=prepared.dev_hash,
            rejected_duplicates=splits.rejected_duplicates,
            train_constant_relations=[
                bool((prepared.train_labels[:, i] == prepared.train_labels[0, i]).all())
                for i in range(4)
            ],
        )

        def dataset_save(
            name: str,
            null: Any = None,
            test: Any = None,
            frozen: Any = None,
        ) -> tuple[str, dict[str, object]]:
            nonlocal dataset_attempted
            checked.check()
            dataset_attempted = True
            summary["dataset_preservation"] = "SERIALIZING"
            with TemporaryDirectory(dir=root, prefix="relations-dataset-") as temporary:
                path = Path(temporary) / "data.npz"
                metadata = data.save_dataset(
                    path, prepared, null=null, test_labels=test, frozen=frozen
                )
                checked.check()
                raw = path.read_bytes()
                checked.check()
                ref = store.write_dataset(name, raw)
                summary["dataset_ref"] = ref
                summary["dataset_preservation"] = "SAVED"
                checked.check()
            checked.check()
            return ref, metadata

        if prepared.all_train_constant:
            summary["dataset_ref"], _ = dataset_save("final")
            closing_gpu_check()
            summary.update(completion="ALL_TRAIN_CONSTANT", reason="ALL_TRAIN_CONSTANT")
            status = "STOPPED"
        else:
            from .metrics import fit_baselines

            null = data.make_null_targets(
                prepared.train_labels,
                prepared.dev_labels,
                train_seed=recipe.null_train_seed,
                dev_seed=recipe.null_dev_seed,
            )
            summary["null_targets"] = {
                "train": null.train_record,
                "dev": null.dev_record,
                "train_hash": null.train_hash,
                "dev_hash": null.dev_hash,
            }
            small_n = splits.train_small
            checked.check()
            small_baseline = fit_baselines(splits.small, prepared.train_labels[:small_n])
            checked.check()
            large_baseline = fit_baselines(splits.train, prepared.train_labels)
            checked.check()
            baselines = {"small": small_baseline, "large": large_baseline}
            selected: dict[str, str] = {}
            fits: dict[str, Any] = {}
            identities: dict[str, Any] = {}
            lr = recipe.learning_rates[0]
            for index in range(1, 6):
                checked.check()
                unchanged()
                if meter.snapshot()["wall_elapsed_seconds"] >= recipe.wall_seconds - 300:
                    raise Stopped("WALL_LIMIT")
                fit_id = f"F{index}"
                small = index <= 2
                n = small_n if small else len(splits.train)
                train_orbits = splits.train[:n]
                train_labels = null.train if index == 5 else prepared.train_labels[:n]
                dev_labels = null.dev if index == 5 else prepared.dev_labels
                if index <= 2:
                    lr = recipe.learning_rates[index - 1]
                elif index == 3:
                    lr = (
                        recipe.learning_rates[0]
                        if fits["F1"].best_bce <= fits["F2"].best_bce
                        else recipe.learning_rates[1]
                    )
                seed = recipe.init_b if index == 4 else recipe.init_a
                identity = checkpoint.Identity(
                    experiment_id,
                    fit_id,
                    data.content_sha256(train_orbits, prepared.train_labels[:n]),
                    prepared.dev_hash,
                    data.array_sha256(train_labels),
                    data.array_sha256(dev_labels),
                    "row-permutation-v1" if index == 5 else "real-v1",
                )
                identities[fit_id] = identity
                budget = meter.cumulative_budget(
                    wall_seconds=recipe.fit_wall_seconds, cpu_seconds=recipe.wall_seconds * 1024
                )
                budgets[fit_id] = budget
                entry: dict[str, Any] = {
                    "fit_id": fit_id,
                    "train_rows": n,
                    "learning_rate": lr,
                    "init_seed": seed,
                    "identity": identity.to_dict(),
                    "updates": 0,
                    "selected_step": None,
                    "dev_curve": [],
                    "final_metrics": {},
                }
                summary["fits"].append(entry)

                def save(model: Any, step: int) -> tuple[str, str]:
                    checked.check()
                    unchanged()
                    with TemporaryDirectory(dir=root, prefix="relations-save-") as temporary:
                        path = Path(temporary) / "selected.pt"
                        digest = checkpoint.save_checkpoint(path, model, identity)
                        checked.check()
                        raw = path.read_bytes()
                        checked.check()
                        ref = store.write_model(f"{fit_id}-best", raw, replace=True)
                        checked.check()
                    checked.check()
                    return ref, digest

                def restore(model: Any, ref: str, digest: str) -> None:
                    checked.check()
                    raw = store.read_model(ref, digest, recipe.artifact_bytes)
                    with TemporaryDirectory(dir=root, prefix="relations-load-") as temporary:
                        path = Path(temporary) / "selected.pt"
                        path.write_bytes(raw)
                        checked.check()
                        checkpoint.load_checkpoint(path, model, identity, expected_sha256=digest)
                    checked.check()

                def progress(values: dict[str, object]) -> None:
                    checked.check()
                    entry.update(values)
                    entry["budget"] = _budget_snapshot(budget)
                    store.write_snapshot({"summary": summary, "counts": meter.snapshot()})
                    checked.check()
                    store.append_event(
                        {
                            "kind": "DEV",
                            "counts": meter.snapshot(),
                            "summary": {"fit_id": fit_id, **values},
                        }
                    )
                    checked.check()

                active_backend = (
                    training.TorchBackend(device, data.encode_pairs)
                    if backend_factory is None
                    else backend_factory(device, data.encode_pairs)
                )
                result = training.fit(
                    backend=active_backend,
                    options=recipe,
                    seed=seed,
                    lr=lr,
                    train=train_orbits,
                    train_labels=train_labels,
                    dev=splits.dev,
                    dev_labels=dev_labels,
                    meter=checked,
                    budget=budget,
                    save=save,
                    restore=restore,
                    progress=progress,
                    on_step=lambda step: entry.update(updates=step),
                )
                if result.updates != recipe.updates:
                    raise ValueError("INCOMPLETE_FIT")
                # Selected TRAIN/DEV and later TEST reuse the exact same accumulator.
                with budget:
                    for split, points, labels in (
                        ("train", train_orbits, train_labels),
                        ("dev", splits.dev, dev_labels),
                    ):
                        logits = training.forward(
                            active_backend, result.model, points, meter=checked
                        )
                        entry["final_metrics"][split] = evaluate.panel(logits, labels)
                active_backend.release(result.model)
                result.model = None
                fits[fit_id] = result
                selected[fit_id] = result.sha256
                entry["budget"] = _budget_snapshot(budget)
                store.write_snapshot({"summary": summary, "counts": meter.snapshot()})
            unchanged()
            # Re-read ALL five actual files before issuing the TEST capability.
            for fit_id, result in fits.items():
                checked.check()
                raw = store.read_model(result.ref, result.sha256, recipe.artifact_bytes)
                if sha256(raw).hexdigest() != selected[fit_id]:
                    raise ValueError("SELECTED_SHA_CHANGED")
            frozen = data.FrozenFits(experiment_id, selected)
            test = session.test_labels(frozen)
            session.release_key()
            summary["dataset_ref"], metadata = dataset_save(
                "final", null=null, test=test, frozen=frozen
            )
            summary.update(test_hash=metadata["test_hash"], selected_sha256=selected)
            summary["baselines"] = {}
            for size, baseline in baselines.items():
                checked.check()
                points = splits.small if size == "small" else splits.train
                labels = prepared.train_labels[: len(points)]
                panels = {
                    "train": evaluate.baseline_panels(baseline, points, labels),
                    "dev": evaluate.baseline_panels(baseline, splits.dev, prepared.dev_labels),
                    "test": evaluate.baseline_panels(baseline, splits.test, test),
                }
                summary["baselines"][size] = {
                    kind: {split: panels[split][kind] for split in ("train", "dev", "test")}
                    for kind in ("constant", "stump")
                }
                checked.check()
            for entry in summary["fits"]:
                checked.check()
                unchanged()
                fit_id = entry["fit_id"]
                result = fits[fit_id]
                # Inference loader never creates Adam or imports another training path.
                predictor: Any = (
                    evaluate.SavedPredictor(device) if backend_factory is None else active_backend
                )
                with budgets[fit_id]:
                    if backend_factory is None:
                        model = predictor.model()
                    else:
                        model = predictor.initialize(entry["init_seed"], entry["learning_rate"])
                    raw = store.read_model(result.ref, result.sha256, recipe.artifact_bytes)
                    with TemporaryDirectory(dir=root, prefix="relations-test-") as temporary:
                        path = Path(temporary) / "selected.pt"
                        path.write_bytes(raw)
                        checked.check()
                        checkpoint.load_checkpoint(
                            path, model, identities[fit_id], expected_sha256=result.sha256
                        )
                    logits = training.forward(predictor, model, splits.test, meter=checked)
                    entry["final_metrics"]["test"] = evaluate.panel(logits, test)
                    baseline_size = "small" if fit_id in ("F1", "F2") else "large"
                    baseline = baselines[baseline_size]
                    comparison = evaluate.comparisons(logits, test, baseline, splits.test)
                    entry["accuracy_comparisons"] = comparison["accuracy"]
                    entry["baseline_size"] = baseline_size
                    entry["comparison_role"] = (
                        "prespecified-main-secondary" if fit_id == "F3" else "descriptive-only"
                    )
                    checked.check()
                    if fit_id == "F3":
                        summary["main_comparisons"] = comparison["paired_bce"]
                        summary["accuracy_comparisons"] = entry["accuracy_comparisons"]
                    if hasattr(model, "cpu"):
                        model.cpu()
                    del model
                entry["budget"] = _budget_snapshot(budgets[fit_id])
                store.write_snapshot({"summary": summary, "counts": meter.snapshot()})
            checked.check()
            unchanged()
            closing_gpu_check()
            summary.update(completion="COMPLETE", reason="FIXED_MATRIX_COMPLETED")
            status = "SUCCEEDED"
    except (Exception, KeyboardInterrupt) as error:
        # Fixed public codes only; no message/repr/traceback from oracle/entropy.
        if isinstance(error, (Stopped, _SharedWallLimit)):
            if (
                dataset_attempted
                and "dataset_ref" not in summary
                and error.reason in ("WALL_LIMIT", "DEADLINE", "SHARED_WALL_LIMIT")
            ):
                summary["dataset_preservation"] = "NOT_SAVED_DEADLINE"
            summary["reason"] = error.reason
            status = "STOPPED"
        elif isinstance(error, ResourceError):
            summary["reason"] = error.reason
            status = "STOPPED"
        elif isinstance(error, KeyboardInterrupt):
            summary["reason"] = "INTERRUPTED"
            status = "STOPPED"
        else:
            summary["reason"] = "EXECUTION_FAILED"
            status = "FAILED"
    finally:
        if session is not None:
            session.release_key()
        for fit_entry in summary["fits"]:
            if fit_entry["fit_id"] in budgets:
                fit_entry["budget"] = _budget_snapshot(budgets[fit_entry["fit_id"]])
    if (
        status != "SUCCEEDED"
        and summary["completion"] != "ALL_TRAIN_CONSTANT"
        and summary["closing_gpu_sample"]["status"] == "NOT_ATTEMPTED"
    ):
        summary["closing_gpu_sample"] = {"status": "NOT_ATTEMPTED", "reason": summary["reason"]}
    if prepared is not None and not dataset_attempted and not getattr(store, "_sealed", False):
        try:
            checked.check()
            summary["dataset_ref"], metadata = dataset_save(
                "final", null=null, test=test, frozen=frozen if test is not None else None
            )
            if test is not None:
                summary.update(
                    test_hash=metadata["test_hash"], selected_sha256=metadata["selected_sha256"]
                )
        except (Stopped, _SharedWallLimit) as error:
            summary["dataset_preservation"] = (
                "NOT_SAVED_DEADLINE"
                if error.reason in ("WALL_LIMIT", "DEADLINE")
                else "NOT_SAVED_SHARED_LIMIT"
                if error.reason == "SHARED_WALL_LIMIT"
                else "NOT_SAVED_RESOURCE_STOP"
            )
        except Exception:
            summary["dataset_preservation"] = "FAILED_NO_RETRY"
    if getattr(store, "_sealed", False):
        return 4

    def timing() -> dict[str, Any]:
        if meter is None:
            return {
                "actual_elapsed_seconds": None,
                "overrun_seconds": None,
                "whole_cycle_within_wall": None,
            }
        elapsed = float(clocks.wall() - meter._start_wall)
        return {
            "actual_elapsed_seconds": elapsed,
            "shared_wall_seconds": shared_wall(),
            "shared_wall_limit_seconds": 1800,
            "shared_overrun_seconds": max(0.0, summary["shared_wall_seconds"] - 1800),
            "overrun_seconds": max(0.0, elapsed - recipe.wall_seconds),
            "last_scientific_check_seconds": meter.snapshot()["wall_elapsed_seconds"],
            "whole_cycle_within_wall": None,
            "scope": "pre-terminal observation; terminal/report IO not included",
            "report_status": "NOT_YET_ATTEMPTED_FROZEN_RUN",
        }

    try:
        # Terminal is the ONLY permitted IO after expiry. No large data/report work.
        # It cannot include its own final write time; stdout/stderr carries that bound.
        summary["timing"] = timing()
        if meter is not None:
            try:
                checked.check()
            except (Stopped, _SharedWallLimit) as error:
                if status == "SUCCEEDED" or summary["completion"] == "ALL_TRAIN_CONSTANT":
                    summary.update(completion="PARTIAL", reason=error.reason)
                    status = "STOPPED"
            summary["timing"] = timing()
        counts = {} if meter is None else meter.snapshot()
        if (
            meter is not None
            and summary["timing"]["actual_elapsed_seconds"] < recipe.wall_seconds
            and summary["shared_wall_seconds"] < 1800
        ):
            store.append_event(
                {
                    "kind": "EXIT",
                    "summary": {"completion": summary["completion"], "reason": summary["reason"]},
                    "counts": counts,
                }
            )
            try:
                checked.check()
            except (Stopped, _SharedWallLimit) as error:
                summary.update(completion="PARTIAL", reason=error.reason)
                status = "STOPPED"
            summary["timing"] = timing()
            counts = meter.snapshot()
        store.finalize(
            {"status": status, "recipe": public_recipe, "summary": summary, "counts": counts}
        )
        if meter is None:
            return 4
        checked.check()
        from .report import render_report

        saved = read_saved(root, experiment_id)
        checked.check()
        html = render_report(saved)
        checked.check()
        store.write_html_report(html)
        checked.check()
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "CLOSING_FAILED",
                    "reason": error.reason
                    if isinstance(error, (Stopped, _SharedWallLimit))
                    else "CLOSING_IO_FAILED",
                    "timing": timing(),
                    "original_run_frozen": bool(getattr(store, "_sealed", False)),
                }
            ),
            file=sys.stderr,
        )
        return 4
    final_timing = timing()
    final_timing["scope"] = "after report write/check; original run remains frozen"
    final_timing["whole_cycle_within_wall"] = True
    final_timing["report_status"] = "WRITTEN_AND_CHECKED"
    print(json.dumps({"status": "CLOSING_OK", "timing": final_timing}))
    return 0 if summary["completion"] in ("COMPLETE", "ALL_TRAIN_CONSTANT") else 4
