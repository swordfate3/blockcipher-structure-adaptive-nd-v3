from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pytest
from speck_method_fakes import (
    Cipher,
    Meter,
    Private,
    Resources,
    Sampler,
    Store,
    Teacher,
    np,
    science_guard,  # noqa: F401
)

from cipher_lab.artifacts import ArtifactError
from cipher_lab.execution import Stopped
from cipher_lab.speck_study.codec import StudyPredictor, StudyTerm
from cipher_lab.speck_study.data import StudyData
from cipher_lab.speck_study.method import run_study
from cipher_lab.speck_study.paper import PUBLISHED_TERMS
from cipher_lab.speck_study.report import render_report
from cipher_lab.speck_study.scoring import Evaluation, predict
from cipher_lab.speck_study.spectral import EngineeringLimit, SpectralMath
from cipher_lab.speck_study.storage import durable_model, source_digest
from cipher_lab.speck_study_config import SpeckStudyRecipe, estimate_study


def tiny() -> SpeckStudyRecipe:
    return SpeckStudyRecipe.synthetic_recipe(
        r_bits=3,
        b_bits=3,
        initial_depth=1,
        pairs_per_layer=8,
        r_classification_size=8,
        r_coef_size=8,
        r_agreement_size=8,
        b_coef_size=8,
        b_dev_size=8,
        b_test_size=8,
    )


def setup(recipe: SpeckStudyRecipe | None = None) -> tuple[Any, ...]:
    r, store, cipher, private = Resources(), Store(), Cipher(), Private()
    meter = Meter(r)
    d = StudyData(recipe or tiny(), cipher, private, Sampler(), meter, store)
    return d, Teacher(r), r, store, cipher, private, meter


def test_r1_pair_is_two_cipher_blocks_immutable_reuse_no_new_labels() -> None:
    d, t, _, s, c, p, m = setup()
    rows = list(d.make_classification(t, chunk_rows=3))
    assert sum(len(x.points) for x in rows) == 8
    assert sum(n for n, _ in c.calls) == 16
    assert p.sizes == [24, 24, 16]
    assert m.counts["real_blocks"] == [16, 16]
    assert m.counts["teacher"] == [8, 8]
    assert len(s.raw) == 3
    before = (len(c.calls), t.calls, list(p.sizes))
    assert [x.points.tolist() for x in d.classification_chunks()] == [
        x.points.tolist() for x in rows
    ]
    assert (len(c.calls), t.calls, p.sizes) == before
    assert not rows[0].points.flags.writeable
    with pytest.raises(ValueError):
        rows[0].points.setflags(write=True)
    ref = next(iter(s.raw))
    s.raw[ref] = b"tampered"
    with pytest.raises(ArtifactError):
        list(d.classification_chunks())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_teacher_attempt_not_refunded(bad: float) -> None:
    d, t, _, _, _, _, m = setup()
    t.bad = bad
    with pytest.raises(Stopped, match="ORACLE_RESPONSE"):
        list(d.make_classification(t))
    assert m.counts["teacher"] == [8, 0]


def test_b_key_requires_gate_and_teacher_reaped_once_no_teacher_for_b() -> None:
    d, t, _, _, _, p, _ = setup()
    with pytest.raises(ValueError):
        d.begin_prediction()
    d.release_prediction(reproduction_matched=True, teacher_reaped=True)
    d.begin_prediction()
    assert p.sizes == [8]
    with pytest.raises(ValueError):
        d.begin_prediction()
    list(d.paired_chunks("Bsearch", 2, chunk_rows=3))
    assert t.calls == 0
    with pytest.raises(ValueError):
        list(d.panel_chunks("Btest"))


def test_stage_domain_stream_chunk_invariant_repeated_domain_rejected() -> None:
    def stream(chunk: int) -> tuple[list[int], list[int]]:
        d, *_ = setup()
        a, b = [], []
        for x, y in d.paired_chunks("Rsearch", 2, chunk_rows=chunk):
            a.extend(x.tolist())
            b.extend(y.tolist())
        with pytest.raises(Stopped, match="DOMAIN_REUSED"):
            list(d.paired_chunks("Rsearch", 2))
        return a, b

    assert stream(3) == stream(5)
    a, b = stream(8)
    assert all((x ^ y) & ((1 << 62) - 1) == 0 for x, y in zip(a, b))


def test_private_failure_sanitized_and_domain_not_retried() -> None:
    d, *_ = setup()

    def failed(size: int) -> bytes:
        raise RuntimeError("SECRET_KEY_MATERIAL")

    d.private_random = failed
    d.release_prediction(reproduction_matched=True, teacher_reaped=True)
    with pytest.raises(Stopped, match="RANDOM_UNAVAILABLE") as info:
        d.begin_prediction()
    assert "SECRET" not in str(info.value)
    with pytest.raises(ValueError):
        d.begin_prediction()


def test_r0_max_budget_static_and_teacher_count_not_zero() -> None:
    r = SpeckStudyRecipe()
    assert estimate_study(r)["teacher_rows"] == 1_161_000_000
    assert estimate_study(r)["real_blocks"] == 541_000_512
    d, *_ = setup(r)
    with pytest.raises(ValueError):
        d.public_r0(
            np.zeros(257, dtype=np.uint32),
            np.zeros(4, dtype=np.uint16),
            22,
            implementation="reference",
        )


def test_published_eq4_literal_not_table5_and_vector_tie0() -> None:
    assert len(PUBLISHED_TERMS) == 12
    assert PUBLISHED_TERMS[0] == (0x4020502070206020, 1824)
    assert PUBLISHED_TERMS[7] == (0x7020602070206020, 3549)
    assert sum(c < 0 for _, c in PUBLISHED_TERMS) == 6
    bits, ties = predict(np.array([0, 1 << 63], dtype=np.uint64), ((1 << 63, 1),))
    assert bits.tolist() == [0, 1]
    assert ties == 0
    bits, ties = predict(np.array([0], dtype=np.uint64), ((1, 1), (2, -1)))
    assert bits.tolist() == [0] and ties == 1


def test_evaluation_full_n_difference_histogram_baseline_zero_hoeffding() -> None:
    e = Evaluation()
    e.add(np.array([0, 1, 1, 0], dtype=np.uint8), np.array([0, 1, 0, 1], dtype=np.uint8), 0)
    result = e.finish(4)
    assert result["correct"] == 2 and result["baseline_correct"] == 2
    assert result["d_histogram"] == {"-1": 1, "0": 2, "1": 1}
    assert result["delta"] == 0
    assert result["lower"] == -math.sqrt(2 * math.log(100) / 4)
    with pytest.raises(ValueError):
        e.finish(5)


def test_mismatch_no_b_key_and_teacher_closed() -> None:
    d, t, r, s, _, p, _ = setup()
    result = run_study(d.recipe, d, t, None, r, s)
    assert result["reason"] == "REPRODUCTION_MISMATCH"
    assert result["scientific_validation"] is False
    assert p.sizes == [64]
    assert t.closed and r.acked and r.closing
    assert s.finalizations == 1


def test_saved_only_report_escape_not_test_zero_and_exact_source() -> None:
    saved = {
        "run_id": "<script>",
        "run_ref": "data/x/run.json",
        "run_sha256": "a" * 64,
        "summary": {"reason": "NOT_STARTED", "scientific_validation": False},
        "counts": {},
        "status": "STOPPED",
    }
    files = render_report(saved)
    assert set(files) == {"report.md", "index.html", "source.json"}
    assert b"<script>" not in files["index.html"]
    assert "未访问" in files["report.md"].decode()
    assert "single-key" in files["report.md"].decode()
    source = json.loads(files["source.json"])
    assert source == {k: saved[k] for k in ("run_id", "run_ref", "run_sha256")} | {"schema": 1}
    assert hashlib.sha256(files["index.html"]).hexdigest()


class ScriptedMath:
    """Fixed manual gate summaries, never an empirical experiment/tuned toy."""

    def __init__(self, data: StudyData, store: Store, teacher: Teacher) -> None:
        self.data, self.store, self.teacher = data, store, teacher
        self.empty = ""
        self.no_coef = ""
        self.cap = False
        self.dev_fail = False
        self.test_fail = False
        self.stop_late = False
        self.visited: list[str] = []

    def r1(self) -> dict[str, Any]:
        return {"teacher": {"accuracy": 0.514}, "eq4": {"accuracy": 0.5087755}}

    def search(self, kind: str) -> tuple[int, ...]:
        self.visited.append(kind + "search")
        if self.cap:
            raise EngineeringLimit
        return () if self.empty == kind else (1 << 63,)

    def coefficients(self, kind: str, masks: tuple[int, ...]) -> StudyPredictor | None:
        if self.no_coef == kind:
            return None
        terms: tuple[StudyTerm, ...]
        if kind == "B":
            assert self.teacher.closed
            self.data.release_panel("Bcoef")
            terms = (StudyTerm(1 << 63, 8),)
        else:
            # count20000 => exact published signed coefficients, all12.
            terms = tuple(
                sorted((StudyTerm(m, c * 2) for m, c in PUBLISHED_TERMS), key=lambda t: t.mask)
            )
        return StudyPredictor(
            kind,
            "teacher-reproduction" if kind == "R" else "absolute-bit-prediction",
            source_digest(self.store.source_snapshot()),
            self.data.recipe,
            terms,
            20000 if kind == "R" else 8,
        )

    def agreement(self, model: StudyPredictor) -> dict[str, Any]:
        return {"accuracy": 0.8027425}

    def classification(self, model: StudyPredictor) -> dict[str, Any]:
        return {"accuracy": 0.5087755}

    def evaluate(self, stage: str, model: StudyPredictor) -> dict[str, Any]:
        self.visited.append(stage)
        assert self.teacher.closed and "model/B-best.pt" in self.store.raw
        self.data.release_panel(stage)
        if stage == "Bdev":
            return {"accuracy": 0.5 if self.dev_fail else 0.6, "delta": 0 if self.dev_fail else 0.1}
        if self.stop_late:
            self.teacher.resources.requested = True
        return {"accuracy": 0.6, "delta": 0.1, "lower": -0.01 if self.test_fail else 0.09}


def scripted() -> tuple[Any, ...]:
    recipe = SpeckStudyRecipe.synthetic_recipe(
        pairs_per_layer=8,
        r_classification_size=8,
        r_coef_size=20000,
        r_agreement_size=8,
        b_coef_size=8,
        b_dev_size=8,
        b_test_size=8,
    )
    args = setup(recipe)
    d, t, _, s, *_ = args
    return (*args, ScriptedMath(d, s, t))


@pytest.mark.parametrize(
    ("field", "value", "reason", "has_bkey", "test_visited"),
    [
        ("empty", "R", "R_EMPTY", False, False),
        ("cap", True, "ENGINEERING_LIMIT", False, False),
        ("no_coef", "R", "R_NO_COEFFICIENTS", False, False),
        ("empty", "B", "B_EMPTY", True, False),
        ("no_coef", "B", "B_NO_COEFFICIENTS", True, False),
        ("dev_fail", True, "DEV_GATE_FAILED", True, False),
        ("test_fail", True, "TEST_GATE_FAILED", True, True),
        ("stop_late", True, "WALL_LIMIT", True, True),
        ("test_fail", False, "POSITIVE_EXPLORATORY", True, True),
    ],
)
def test_complete_manual_stage_machine(
    field: str, value: object, reason: str, has_bkey: bool, test_visited: bool
) -> None:
    d, t, r, s, _, p, _, engine = scripted()
    setattr(engine, field, value)
    result = run_study(d.recipe, d, t, engine, r, s)
    assert result["reason"] == reason
    assert ("Btest" in engine.visited) == test_visited
    assert p.sizes == ([8] if has_bkey else [])
    assert t.closed and r.acked and r.closing
    assert s.finalizations == 1
    assert result["scientific_validation"] is False  # all fake, no science
    if reason == "WALL_LIMIT":
        assert not result["prediction_positive"]
    with pytest.raises(Stopped):
        d.begin_prediction()


def test_teacher_cleanup_unknown_never_releases_bkey() -> None:
    d, t, r, s, _, p, _, engine = scripted()
    t.fail_close = True
    result = run_study(d.recipe, d, t, engine, r, s)
    assert result["reason"] == "CLEANUP_UNKNOWN"
    assert p.sizes == [] and result["scientific_validation"] is False


def test_artifact_error_sealed_no_second_finalize_or_ack() -> None:
    d, t, r, s, _, _, _, engine = scripted()
    s.fail_read = True
    with pytest.raises(ArtifactError):
        run_study(d.recipe, d, t, engine, r, s)
    assert s.finalizations == 0 and not r.acked
    assert d.closed and t.closed


@pytest.mark.parametrize("identity", ["source", "backend", "recipe", "task", "sha", "bytes"])
def test_durable_model_identity_tamper_never_memory_fallback(identity: str) -> None:
    d, t, _, s, *_ = scripted()
    engine = ScriptedMath(d, s, t)
    model = engine.coefficients("R", (1 << 63,))
    assert model is not None
    s.write_snapshot({"recipe": d.recipe.to_dict(), "summary": {"backend_identity": "speck-study"}})
    source = s.source_snapshot()
    if identity == "source":
        source = {"cipher_lab/fake.py": "b" * 64}
    elif identity == "backend":
        s.saved["summary"]["backend_identity"] = "demo"
    elif identity == "recipe":
        s.saved["recipe"] = {}
    elif identity == "task":
        from dataclasses import replace

        model = replace(model, task="absolute-bit-prediction")
    elif identity in ("sha", "bytes"):
        original = s.read_model

        def tamper(ref: str, digest: str, size: int) -> bytes:
            return bytes(original(ref, digest, size)) + b" "  # valid whitespace is still wrong SHA

        s.read_model = tamper
    with pytest.raises((ArtifactError, ValueError)):
        durable_model(
            s,
            model,
            expected_kind="R",
            expected_recipe=d.recipe,
            expected_source=source,
            expected_backend="speck-study",
        )


def test_partial_oracle_budget_and_incomplete_r1_not_empty_negative() -> None:
    d, t, _, s, c, _, m = setup()

    def failed(points: object, keys: object, rounds: int) -> object:
        raise RuntimeError("SECRET")

    d.cipher = failed
    result = run_study(d.recipe, d, t, None, d.meter.resources, s)
    assert result["reason"] == "ORACLE_FAILED"
    assert m.counts["real_blocks"] == [8, 0]
    assert result["scientific_validation"] is False
    assert not c.calls and result["test"] is None


def test_negative_teacher_row_count_hardlabel_mismatch_and_tie_protocol() -> None:
    for response in (((), ()), ((0.5,) * 8, (1,) * 8), ((0.5,) * 8, (-1,) * 8)):
        d, t, _, _, _, _, m = setup()
        t.predict = lambda points: response
        with pytest.raises(Stopped, match="ORACLE_RESPONSE"):
            list(d.make_classification(t))
        assert m.counts["teacher"] == [8, 0]


def test_search_shared_points_full_layer_saved_and_partial_not_empty() -> None:
    d, t, _, s, _, _, m = setup()
    engine = SpectralMath(d, t, s)
    masks = engine.search("R")
    assert isinstance(masks, tuple)
    layers = [e for e in s.events if isinstance(e, dict) and e["kind"] == "R-layer"]
    assert layers and layers[0]["summary"]["count"] == 8
    assert len(layers[0]["summary"]["masks"]) == 4
    assert m.counts["teacher"][0] == 16 * len(layers)
    assert m.counts["teacher"][1] == m.counts["teacher"][0]


def test_formal_r0_not_replaced_by_fake_synthetic_pass() -> None:
    d, t, r, s, _, p, _ = setup(SpeckStudyRecipe())
    r.formal_r0 = lambda data: False
    result = run_study(d.recipe, d, t, None, r, s)
    assert result["reason"] == "R0_UNVERIFIED"
    assert not p.sizes and not t.calls and result["scientific_validation"] is False


def test_teacher_forbidden_after_b_start_without_charging_forward() -> None:
    d, t, _, _, _, _, m = setup()
    d.release_prediction(reproduction_matched=True, teacher_reaped=True)
    d.begin_prediction()
    with pytest.raises(ValueError):
        d.teacher_signs(t, np.zeros(1, dtype=np.uint64))
    assert not m.counts["teacher"][0] and not t.calls


def test_late_gate_stop_clears_validation_before_save_ack() -> None:
    d, t, r, s, _, _, _, engine = scripted()

    def last(model: StudyPredictor) -> dict[str, Any]:
        r.requested = True
        return {"accuracy": 0.1}  # mismatch after science cutoff, not a normal negative

    engine.classification = last
    result = run_study(d.recipe, d, t, engine, r, s)
    assert result["reason"] == "WALL_LIMIT"
    assert not result["scientific_validation"]


def test_predict_count_rejects_nonbinary_official_and_bad_lengths() -> None:
    e = Evaluation()
    with pytest.raises(ValueError):
        e.add(
            np.zeros(2, dtype=np.uint8),
            np.zeros(2, dtype=np.uint8),
            0,
            official_predictions=np.zeros(1, dtype=np.uint8),
        )


def test_durable_identity_kind_is_externally_expected_not_model_self() -> None:
    d, _, _, s, *_ = setup()
    model = StudyPredictor(
        "B",
        "absolute-bit-prediction",
        source_digest(s.source_snapshot()),
        d.recipe,
        (StudyTerm(1 << 63, 8),),
        8,
    )
    s.write_snapshot({"recipe": d.recipe.to_dict(), "summary": {"backend_identity": "speck-study"}})
    with pytest.raises(ValueError):
        durable_model(
            s,
            model,
            expected_kind="R",
            expected_recipe=d.recipe,
            expected_source=s.source_snapshot(),
            expected_backend="speck-study",
        )


def test_stop_preserves_study_parity_attempt_complete_counts() -> None:
    d, t, r, s, _, _, m, engine = scripted()
    from cipher_lab.speck_study.resources import StudyMeter

    adapter = StudyMeter(m, r)
    d.meter = adapter
    adapter.reserve("parity", 800)
    adapter.complete("parity", 400)
    engine.empty = "R"
    run_study(d.recipe, d, t, engine, r, s)
    assert s.saved["counts"]["attempted_parity"] == 800
    assert s.saved["counts"]["completed_parity"] == 400


def test_coefficient_shared_panel_and_vector_prediction_does_not_requery_teacher() -> None:
    d, t, _, s, _, _, m = setup()
    engine = SpectralMath(d, t, s)
    model = engine.coefficients("R", (1 << 63,))
    assert t.calls == 8 and m.counts["teacher"] == [8, 8]
    coef = [e for e in s.events if isinstance(e, dict) and e["kind"] == "R-coef-chunk"]
    assert coef[-1]["summary"]["count"] == 8
    assert coef[-1]["summary"]["intercept"] == 0
    assert model is None or model.kind == "R"


def test_r0_total_attempts_per_implementation_and_combined_cap_fake_only() -> None:
    d, _, _, _, c, _, m = setup(SpeckStudyRecipe())
    c.public_r0 = lambda points, keys, rounds, implementation: points.copy()
    points = np.zeros(256, dtype=np.uint32)
    keys = np.zeros(4, dtype=np.uint16)
    d.public_r0(points, keys, 22, implementation="reference")
    d.public_r0(points, keys, 22, implementation="batch")
    assert m.counts["real_blocks"] == [512, 512]
    with pytest.raises(ValueError):
        d.public_r0(points[:1], keys, 22, implementation="batch")


def test_all_twelve_published_masks_literal_msb_indices_equation4() -> None:
    indices = (
        (1, 10, 17, 19, 26, 33, 34, 35, 42, 49, 50, 58),
        (1, 3, 4, 10, 17, 26, 33, 35, 36, 42, 49, 58),
        (1, 3, 10, 11, 17, 26, 33, 35, 42, 43, 49, 58),
        (1, 2, 3, 10, 17, 18, 26, 33, 42, 49, 51, 58),
        (1, 3, 4, 10, 17, 26, 33, 34, 36, 42, 49, 50, 51, 58),
        (1, 3, 4, 10, 17, 20, 26, 33, 34, 35, 42, 49, 50, 58),
        (1, 3, 4, 10, 17, 26, 33, 34, 35, 42, 49, 50, 52, 58),
        (1, 2, 3, 10, 17, 18, 26, 33, 34, 35, 42, 49, 50, 58),
        (1, 2, 4, 10, 17, 18, 19, 26, 33, 35, 36, 42, 49, 58),
        (1, 2, 3, 10, 17, 18, 20, 26, 33, 35, 36, 42, 49, 58),
        (1, 2, 3, 10, 17, 18, 26, 33, 35, 36, 42, 49, 52, 58),
        (1, 2, 3, 4, 10, 11, 17, 18, 26, 33, 34, 35, 36, 42, 43, 49, 50, 58),
    )
    assert tuple(sum(1 << (63 - i) for i in row) for row in indices) == tuple(
        m for m, _ in PUBLISHED_TERMS
    )
    assert tuple(c for _, c in PUBLISHED_TERMS) == (
        1824,
        3454,
        -918,
        1810,
        -1782,
        900,
        -844,
        3549,
        -1753,
        -868,
        865,
        -880,
    )


def test_retained_real_runstore_registered_dataset_model_read_tamper_fake_only(
    tmp_path: Any, monkeypatch: Any
) -> None:
    import cipher_lab.artifacts as artifacts
    from cipher_lab.speck_study.storage import StudyStore

    # uuid filenames are technical identities, but even these use a fake source.
    counter = iter(range(10000))

    class FakeUuid:
        @property
        def hex(self) -> str:
            return f"{next(counter):032x}"

    monkeypatch.setattr(artifacts, "uuid4", FakeUuid)
    owner = artifacts.RunStore.claim(
        tmp_path, "fake-owner", output_bytes=4_000_000, needs_models=True
    )
    store = StudyStore(owner, tmp_path, "fake-owner")
    d, teacher, _, _, cipher, private, meter = setup()
    d.store = store
    list(d.make_classification(teacher, chunk_rows=3))
    assert sum(len(c.points) for c in d.classification_chunks()) == 8
    owner.write_snapshot(
        {
            "recipe": d.recipe.to_dict(),
            "summary": {"backend_identity": "speck-study", "scientific_validation": False},
        }
    )
    model = StudyPredictor(
        "B",
        "absolute-bit-prediction",
        source_digest(owner.source_snapshot()),
        d.recipe,
        (StudyTerm(1 << 63, 8),),
        8,
    )
    loaded = durable_model(
        store,
        model,
        expected_kind="B",
        expected_recipe=d.recipe,
        expected_source=owner.source_snapshot(),
        expected_backend="speck-study",
    )
    assert loaded == model
    path = tmp_path / "models/fake-owner/B-best.pt"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ArtifactError):
        owner.read_model(
            "model/B-best.pt", hashlib.sha256(path.read_bytes()).hexdigest(), 256 * 1024
        )
    # read failure sealed owner: no fallback/finalize allowed.
    with pytest.raises(ArtifactError, match="FINALIZED"):
        owner.finalize({"status": "FAILED"})


def test_b_test_full_n_positive_bound_and_negative_signed_delta_report() -> None:
    evaluation = Evaluation()
    for _ in range(100):
        evaluation.add(np.ones(100, dtype=np.uint8), np.ones(100, dtype=np.uint8), 0)
    result = evaluation.finish(10000)
    assert result["accuracy"] == 1 and result["delta"] == 1 and result["lower"] > 0
    assert sum(result["d_histogram"].values()) == 10000
    saved = {
        "run_id": "negative",
        "run_ref": "data/negative/run.json",
        "run_sha256": "a" * 64,
        "summary": {
            "test": {"accuracy": 0.4, "delta": -0.1, "lower": -0.11, "terms": [{"signed_sum": -8}]}
        },
        "status": "SUCCEEDED",
    }
    rendered = render_report(saved)
    assert b"-0.1" in rendered["report.md"] and b"-8" in rendered["report.md"]


def test_b_reports_model_terms_dc_and_no_neural_zero_claim() -> None:
    d, t, r, s, _, _, _, engine = scripted()
    result = run_study(d.recipe, d, t, engine, r, s)
    assert result["b_model"]["terms"][0]["signed_sum"] == 8
    assert result["b_model"]["intercept"] == 0
    assert result["r_model"]["terms"][0]["signed_sum"] != 0


def test_invalid_official_r0_does_not_claim_normal_success() -> None:
    d, t, r, s, *_ = setup(SpeckStudyRecipe())
    r.formal_r0 = lambda data: False
    run_study(d.recipe, d, t, None, r, s)
    assert s.saved["status"] == "STOPPED"


def test_full_bucket_cap_saved_all_kept_not_topk(monkeypatch: Any) -> None:
    import cipher_lab.speck_study.spectral as spectral
    from cipher_lab.speck_study.search import BucketLayer

    d, t, _, s, *_ = setup()

    class CapAccumulator:
        def __init__(self, *args: Any) -> None:
            self.masks = tuple(range(800))
            self.sums = [8] * 800
            self.count = 0

        def add(self, points: Any, *args: Any, **kwargs: Any) -> None:
            self.count += len(points)

        def finish(self, *args: Any) -> BucketLayer:
            return BucketLayer(64, 8, self.masks, tuple(self.sums), 8, tuple(range(401)), True)

    monkeypatch.setattr(spectral, "BucketAccumulator", CapAccumulator)
    with pytest.raises(EngineeringLimit):
        SpectralMath(d, t, s).search("R")
    layer = [e for e in s.events if isinstance(e, dict) and e["kind"] == "R-layer"][0]
    assert len(layer["summary"]["kept"]) == 401
    assert len(layer["summary"]["masks"]) == 800


def test_public_sampler_random_access_pcg_seam_fake_only(monkeypatch: Any) -> None:
    from cipher_lab.speck_study.data import PublicSampler

    class FakePcg:
        def __init__(self, seed: int) -> None:
            self.position = seed % 256

        def advance(self, count: int) -> None:
            self.position += count

        def random_raw(self, count: int) -> Any:
            return np.arange(self.position, self.position + count, dtype=np.uint64)

    monkeypatch.setattr(np.random, "PCG64", FakePcg)
    sampler = PublicSampler()
    full = sampler(123, "Rsearch:8", 0, 7, 2)
    split = np.concatenate([sampler(123, "Rsearch:8", 0, 3, 2), sampler(123, "Rsearch:8", 3, 4, 2)])
    assert np.array_equal(full, split)
    assert not np.array_equal(full, sampler(123, "Rsearch:9", 0, 7, 2))


def test_b_panel_release_sequence_reuse_and_key_never_in_sampler_arguments() -> None:
    d, t, _, _, _, p, m = setup()
    d.release_prediction(reproduction_matched=True, teacher_reaped=True)
    d.begin_prediction()
    assert p.sizes == [8]
    with pytest.raises(ValueError):
        d.release_panel("Btest")
    for stage in ("Bcoef", "Bdev", "Btest"):
        d.release_panel(stage)
        for points in d.panel_chunks(stage, chunk_rows=3):
            signs = d.signs(stage, points, None)
            assert set(signs.tolist()) <= {-1, 1}
        with pytest.raises(Stopped, match="DOMAIN_REUSED"):
            list(d.panel_chunks(stage))
    assert m.counts["real_blocks"] == [24, 24]
    assert t.calls == 0 and p.sizes == [8]


def test_stop_inside_teacher_reply_preserves_attempt_complete_and_public_partial() -> None:
    d, t, r, s, _, _, m = setup()
    original = t.predict

    def stop(points: list[int]) -> Any:
        result = original(points)
        r.requested = True
        return result

    t.predict = stop
    result = run_study(d.recipe, d, t, None, r, s)
    assert result["reason"] == "WALL_LIMIT"
    assert m.counts["teacher"] == [8, 0]
    assert m.counts["real_blocks"] == [16, 16]
    assert not d.refs and result["test"] is None
    assert not result["scientific_validation"]


def test_r3_term_intersection_coefficient_and_accuracy_gates() -> None:
    from dataclasses import replace

    from cipher_lab.speck_study.method import reproduction_gate

    d, t, _, s, *_ = scripted()
    model = ScriptedMath(d, s, t).coefficients("R", ())
    assert model is not None
    agreement, classification = {"accuracy": 0.8027425}, {"accuracy": 0.5087755}
    assert reproduction_gate(d.recipe, model, agreement, classification)["matched"]
    assert not reproduction_gate(
        d.recipe, replace(model, terms=model.terms[:10]), agreement, classification
    )["matched"]
    wrong = replace(model, terms=tuple(StudyTerm(x.mask, x.signed_sum + 1000) for x in model.terms))
    assert not reproduction_gate(d.recipe, wrong, agreement, classification)["matched"]
    assert not reproduction_gate(d.recipe, model, {"accuracy": 0.79}, classification)["matched"]
    assert not reproduction_gate(d.recipe, model, agreement, {"accuracy": 0.5})["matched"]


def test_recipe_mismatch_never_touches_secret_or_teacher() -> None:
    d, t, r, s, _, p, _ = setup()
    result = run_study(SpeckStudyRecipe(), d, t, None, r, s)
    assert result["reason"] == "STUDY_FAILED"
    assert not p.sizes and not t.calls


def test_data_rejects_b_label_domain_forgery_before_oracle() -> None:
    d, t, _, _, c, *_ = setup()
    d.release_prediction(reproduction_matched=True, teacher_reaped=True)
    d.begin_prediction()
    with pytest.raises(ValueError):
        d.signs("Btest", np.zeros(1, dtype=np.uint64), t)
    assert not c.calls


def test_fix1_r1_keyboard_interrupt_saves_non_success_then_propagates() -> None:
    d, t, r, s, _, _, _, engine = scripted()
    interrupted = KeyboardInterrupt("private interruption detail")

    def interrupt_r1() -> dict[str, Any]:
        raise interrupted

    engine.r1 = interrupt_r1
    with pytest.raises(KeyboardInterrupt) as caught:
        run_study(d.recipe, d, t, engine, r, s)
    assert caught.value is interrupted
    assert s.finalizations == 1
    assert s.saved["status"] == "STOPPED"
    summary = s.saved["summary"]
    assert summary["reason"] == "STUDY_INTERRUPTED"
    assert summary["scientific_validation"] is False
    assert summary["prediction_positive"] is False
    assert "private interruption detail" not in json.dumps(s.saved)
    assert d.closed and t.closed and r.closing and r.acked


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_fix1_positive_final_check_interrupt_clears_both_flags_then_propagates(
    interrupt_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    # Formal-shaped recipe, but every component including formal_r0 is an existing
    # guarded fake. No real key/cipher/teacher/scientific sampling is executed.
    d, t, r, s, _, _, _, engine = scripted()
    d.recipe = SpeckStudyRecipe()
    assert isinstance(engine, ScriptedMath)
    original_coefficients = engine.coefficients

    def formal_shaped_coefficients(kind: str, masks: tuple[int, ...]) -> StudyPredictor:
        model = original_coefficients(kind, masks)
        assert model is not None
        count = d.recipe.r_coef_size if kind == "R" else d.recipe.b_coef_size
        scale = count // model.count
        return replace(
            model,
            count=count,
            terms=tuple(StudyTerm(term.mask, term.signed_sum * scale) for term in model.terms),
        )

    monkeypatch.setattr(engine, "coefficients", formal_shaped_coefficients)
    original_evaluate = engine.evaluate
    original_check = r.check
    interrupted = interrupt_type("private interruption detail")
    armed = False

    def evaluate(stage: str, model: StudyPredictor) -> dict[str, Any]:
        nonlocal armed
        result = original_evaluate(stage, model)
        if stage == "Btest":
            armed = True
        return result

    def check(*, science: bool = True) -> None:
        if science and armed:
            raise interrupted
        original_check(science=science)

    monkeypatch.setattr(engine, "evaluate", evaluate)
    r.check = check
    with pytest.raises(interrupt_type) as caught:
        run_study(d.recipe, d, t, engine, r, s)
    assert caught.value is interrupted
    assert armed and "Btest" in engine.visited
    assert s.finalizations == 1
    assert s.saved["status"] == "STOPPED"
    summary = s.saved["summary"]
    assert summary["test"]["accuracy"] == 0.6  # preserve completed fake result evidence
    assert summary["reason"] == "STUDY_INTERRUPTED"
    assert summary["scientific_validation"] is False
    assert summary["prediction_positive"] is False
    assert "private interruption detail" not in json.dumps(s.saved)
    assert d.closed and t.closed and r.closing and r.acked
