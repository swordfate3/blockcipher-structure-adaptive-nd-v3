"""Single-writer R -> conditional B stage machine. No sampling or spectral math.

Resources must expose check(science=...), begin_closing(), formal_r0(data), and
acknowledge_closing(). Task5 supplies bounded IPC ACK transport, never parent
RunStore writes. formal_r0 must attest actual env/GPU/weights/KAT; recipe is not
permission. Returning TeacherClient.close() is the Task2 confirmed-reaped contract.
"""

from __future__ import annotations

from typing import Any

from cipher_lab.artifacts import ArtifactError
from cipher_lab.execution import Stopped
from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe, estimate_study
from cipher_lab_plugins.speck_study.paper import PUBLISHED_TERMS
from cipher_lab_plugins.speck_study.resources import ResourceStop
from cipher_lab_plugins.speck_study.spectral import EngineeringLimit, SpectralMath
from cipher_lab_plugins.speck_study.storage import durable_model


class GateEnd(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


def _near(value: float, reference: float, tolerance: float) -> bool:
    return abs(value - reference) <= tolerance


def reproduction_gate(
    recipe: SpeckStudyRecipe, model: Any, agreement: Any, classification: Any
) -> dict[str, Any]:
    published = dict(PUBLISHED_TERMS)
    extracted = {t.mask: t.signed_sum / model.count for t in model.terms}
    common = sorted(published.keys() & extracted.keys())
    errors = {str(m): abs(extracted[m] - published[m] / 10000) for m in common}
    matched = (
        recipe.r_terms_min <= len(extracted) <= recipe.r_terms_max
        and len(common) >= recipe.r_mask_intersection_min
        and all(e <= recipe.r_coefficient_error_max for e in errors.values())
        and _near(agreement["accuracy"], recipe.r_agreement_reference, recipe.r_agreement_tolerance)
        and _near(
            classification["accuracy"],
            recipe.r_classification_accuracy_reference,
            recipe.r_classification_accuracy_tolerance,
        )
    )
    return dict(
        matched=matched, terms=len(extracted), intersection=len(common), coefficient_errors=errors
    )


def run_study(
    recipe: SpeckStudyRecipe, data: Any, teacher: Any, math: Any, resources: Any, store: Any
) -> dict[str, Any]:
    """Run once. All exceptions sanitized; storage failures propagate already sealed.

    Null/optimizer work never issued. A normal negative gate is not a partial;
    stop/unknown always clears validation/positive but retains aggregate artifacts.
    """
    backend = "speck-study"
    source = store.source_snapshot()
    summary: dict[str, Any] = dict(
        backend_identity=backend,
        scientific_validation=False,
        reproduction_matched=False,
        prediction_positive=False,
        reason="RUNNING",
        stage="R0",
        synthetic=recipe.synthetic,
        test=None,
        dev=None,
        choices=1,
        retry_count=0,
        sampling="with-replacement-independent-domains; overlap UNKNOWN; not unseen",
    )
    status = "SUCCEEDED"
    teacher_closed = False
    artifact_failed = False

    def snapshot() -> None:
        store.write_snapshot(
            {
                "recipe": recipe.to_dict(),
                "policy": estimate_study(recipe),
                "counts": data.counts(),
                "summary": dict(summary),
            }
        )

    def stage(name: str) -> None:
        resources.check()
        summary["stage"] = name
        snapshot()

    def close_teacher() -> None:
        nonlocal teacher_closed
        if not teacher_closed:
            teacher.close()  # failure/UNKNOWN must not release B
            teacher_closed = True

    try:
        if data.recipe != recipe or data.store is not store:
            raise ValueError("owner recipe/store mismatch")
        snapshot()
        resources.check()
        if not recipe.synthetic and resources.formal_r0(data) is not True:
            raise GateEnd("R0_UNVERIFIED")
        engine = math if math is not None else SpectralMath(data, teacher, store)
        stage("R1")
        first = engine.r1()
        summary["r1"] = first
        if not (
            recipe.r_teacher_accuracy_min
            <= first["teacher"]["accuracy"]
            <= recipe.r_teacher_accuracy_max
            and _near(
                first["eq4"]["accuracy"],
                recipe.r_eq4_accuracy_reference,
                recipe.r_eq4_accuracy_tolerance,
            )
        ):
            raise GateEnd("REPRODUCTION_MISMATCH")
        stage("R2")
        masks = engine.search("R")
        if not masks:
            raise GateEnd("R_EMPTY")
        stage("R3")
        r_model = engine.coefficients("R", masks)
        if r_model is None:
            raise GateEnd("R_NO_COEFFICIENTS")
        r_model = durable_model(
            store,
            r_model,
            expected_kind="R",
            expected_recipe=recipe,
            expected_source=source,
            expected_backend=backend,
        )
        summary["r_model"] = dict(
            count=r_model.count,
            intercept=r_model.intercept,
            terms=[dict(mask=t.mask, signed_sum=t.signed_sum) for t in r_model.terms],
        )
        agreement = engine.agreement(r_model)
        classification = engine.classification(r_model)
        summary.update(
            r_agreement=agreement,
            r_classification=classification,
            r_gate=reproduction_gate(recipe, r_model, agreement, classification),
        )
        if not summary["r_gate"]["matched"]:
            raise GateEnd("REPRODUCTION_MISMATCH")
        summary["reproduction_matched"] = True
        stage("Rgate")
        close_teacher()  # MUST close/reap TF before opening B key/cipher phase
        resources.check()
        data.release_prediction(reproduction_matched=True, teacher_reaped=teacher_closed)
        data.begin_prediction()
        stage("Bsearch")
        masks = engine.search("B")
        if not masks:
            raise GateEnd("B_EMPTY")
        stage("Bcoef")
        model = engine.coefficients("B", masks)
        if model is None:
            raise GateEnd("B_NO_COEFFICIENTS")
        stage("Bdurable")
        model = durable_model(
            store,
            model,
            expected_kind="B",
            expected_recipe=recipe,
            expected_source=source,
            expected_backend=backend,
        )
        summary["b_model"] = dict(
            count=model.count,
            intercept=model.intercept,
            terms=[dict(mask=t.mask, signed_sum=t.signed_sum) for t in model.terms],
        )
        stage("Bdev")
        dev = engine.evaluate("Bdev", model)
        summary["dev"] = dev
        if not (
            dev["accuracy"] >= recipe.b_dev_accuracy_min
            and dev["delta"] > recipe.b_dev_delta_min_exclusive
        ):
            raise GateEnd("DEV_GATE_FAILED")
        stage("Btest")
        test = engine.evaluate("Btest", model)
        summary["test"] = test
        summary["prediction_positive"] = (
            test["accuracy"] >= recipe.b_test_accuracy_min
            and test["lower"] > recipe.b_test_lower_min_exclusive
        )
        summary["reason"] = (
            "POSITIVE_EXPLORATORY" if summary["prediction_positive"] else "TEST_GATE_FAILED"
        )
        summary["scientific_validation"] = not recipe.synthetic
        resources.check()  # late stop invalidates completed-looking scientific claim
    except ArtifactError:
        artifact_failed = True
        summary.update(scientific_validation=False, prediction_positive=False)
        raise
    except GateEnd as end:
        summary["reason"] = end.reason
        if end.reason == "R0_UNVERIFIED":
            status = "STOPPED"
        # Even a negative gate must finish before the science admission cutoff.
        try:
            resources.check()
            summary["scientific_validation"] = (
                not recipe.synthetic and end.reason != "R0_UNVERIFIED"
            )
        except (Stopped, ResourceStop) as error:
            status = "STOPPED"
            summary.update(
                reason=error.reason, scientific_validation=False, prediction_positive=False
            )
    except EngineeringLimit:
        status = "STOPPED"
        summary.update(
            reason="ENGINEERING_LIMIT", scientific_validation=False, prediction_positive=False
        )
    except (Stopped, ResourceStop) as error:
        status = "STOPPED"
        summary.update(reason=error.reason, scientific_validation=False, prediction_positive=False)
    except Exception:
        status = "FAILED"
        summary.update(
            reason="STUDY_FAILED", scientific_validation=False, prediction_positive=False
        )
    except BaseException:
        # Process-level interruptions are not normal gate completion. Save a
        # fixed public non-success terminal below, then propagate the original
        # interruption; never certify a partial or completed-looking result.
        status = "STOPPED"
        summary.update(
            reason="STUDY_INTERRUPTED", scientific_validation=False, prediction_positive=False
        )
        raise
    finally:
        # Quiesce first; neither ACK nor saving can permit new science. Child is
        # sole writer. Parent request is sampled by resources.check(); Task5 wires it.
        data.close()
        resources.begin_closing()
        try:
            close_teacher()
            resources.check(science=False)
        except Exception:
            status = "STOPPED"
            summary.update(
                reason="CLEANUP_UNKNOWN", scientific_validation=False, prediction_positive=False
            )
        if summary["scientific_validation"]:
            # Save/cleanup can cross the science boundary even after the last chunk.
            # Admission is now intentionally closed; compare the supervisor's last
            # successfully sampled wall (do not attempt science in closing).
            now = getattr(resources, "last_wall", None)
            deadline = getattr(resources, "science_deadline", None)
            if now is not None and deadline is not None and now >= deadline:
                status = "STOPPED"
                summary.update(
                    reason="SCIENCE_DEADLINE",
                    scientific_validation=False,
                    prediction_positive=False,
                )
        if not artifact_failed:
            # No finalize after ArtifactError: RunStore sealed its own failure.
            store.finalize({"status": status, "counts": data.counts(), "summary": dict(summary)})
            # ACK means no more science AND sole owner has saved. Not terminal proof.
            resources.acknowledge_closing()
    return summary
