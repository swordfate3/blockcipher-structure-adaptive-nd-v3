"""Study-specific streaming GL/coef/scoring assembly, not a training workflow.

No keys/cipher access; consumes only owner-released public points and signs.
Injected fake math may implement this same narrow interface in engineering tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cipher_lab.speck_study.codec import StudyPredictor
from cipher_lab.speck_study.paper import PUBLISHED_TERMS
from cipher_lab.speck_study.scoring import Evaluation, predict_both
from cipher_lab.speck_study.search import (
    BucketAccumulator,
    CoefficientAccumulator,
    initial_prefixes,
    select_terms,
)
from cipher_lab.speck_study.storage import source_digest


class SpectralMath:
    def __init__(self, data: Any, teacher: Any, store: Any) -> None:
        self.data, self.teacher, self.store = data, teacher, store
        self.recipe, self.meter = data.recipe, data.meter
        self.source = store.source_snapshot()

    def _save(self, kind: str, aggregate: Any) -> None:
        self.store.append_event({"kind": kind, "counts": self.data.counts(), "summary": aggregate})

    def _predict(self, points: Any, terms: Any) -> tuple[Any, int, Any]:
        n = len(points)
        self.meter.reserve("scoring", n)
        self.meter.reserve("parity", n * len(terms))
        # One integer sum, both fixed tie conventions; no doubled parity work.
        predictions, ties, official = predict_both(points, terms, check=self.meter.check)
        self.meter.complete("parity", n * len(terms))
        self.meter.complete("scoring", n)
        return predictions, ties, official

    def r1(self) -> dict[str, Any]:
        teacher_eval, eq4 = Evaluation(), Evaluation()
        for chunk in self.data.make_classification(self.teacher):
            teacher_eval.add(chunk.hardlabels, chunk.labels, 0)
            pred, ties, official = self._predict(chunk.points, PUBLISHED_TERMS)
            eq4.add(pred, chunk.labels, ties, official_predictions=official)
            self._save("R1-chunk", {"teacher": teacher_eval.snapshot(), "eq4": eq4.snapshot()})
        return {
            "teacher": teacher_eval.finish(self.recipe.r_classification_size),
            "eq4": eq4.finish(self.recipe.r_classification_size),
        }

    def search(self, kind: str) -> tuple[int, ...]:
        bits = self.recipe.r_bits if kind == "R" else self.recipe.b_bits
        prefixes = initial_prefixes(bits, self.recipe.initial_depth)
        for depth in range(self.recipe.initial_depth + 1, bits + 1):
            accumulator = BucketAccumulator(bits, depth, prefixes, self.recipe.pairs_per_layer)
            for points, paired in self.data.paired_chunks(kind + "search", depth):
                a = self.data.signs(kind + "search", points, self.teacher)
                b = self.data.signs(kind + "search", paired, self.teacher)
                products = len(points) * len(accumulator.masks)
                self.meter.reserve("parity", products)
                accumulator.add(points, paired, a, b, check=self.meter.check)
                self.meter.complete("parity", products)
                self._save(
                    kind + "-search-chunk",
                    dict(
                        depth=depth,
                        count=accumulator.count,
                        masks=list(accumulator.masks),
                        sums=accumulator.sums,
                    ),
                )
            layer = accumulator.finish(self.recipe.tau, self.recipe.max_prefixes)
            self._save(
                kind + "-layer",
                dict(
                    depth=depth,
                    count=layer.count,
                    masks=list(layer.masks),
                    sums=list(layer.sums),
                    kept=list(layer.kept),
                    engineering_limit=layer.engineering_limit,
                    minimum=min(layer.sums),
                    maximum=max(layer.sums),
                ),
            )
            if layer.engineering_limit:
                raise EngineeringLimit
            prefixes = layer.kept
            if not prefixes:
                return ()
        return prefixes

    def coefficients(self, kind: str, masks: tuple[int, ...]) -> StudyPredictor | None:
        if kind == "B":
            self.data.release_panel("Bcoef")
        count = self.recipe.r_coef_size if kind == "R" else self.recipe.b_coef_size
        # Include descriptive DC without consuming another point/oracle query.
        masks_with_dc = tuple(sorted(set(masks) | {0}))
        # Task1 cap includes DC, so accumulate DC independently when all400 candidates exist.
        dc = CoefficientAccumulator((0,), count)
        fit = CoefficientAccumulator(tuple(m for m in masks_with_dc if m), count)
        for points in self.data.panel_chunks(kind + "coef"):
            signs = self.data.signs(kind + "coef", points, self.teacher)
            products = len(points) * (len(fit.masks) + 1)
            self.meter.reserve("parity", products)
            fit.add(points, signs, check=self.meter.check)
            dc.add(points, signs, check=self.meter.check)
            self.meter.complete("parity", products)
            self._save(
                kind + "-coef-chunk",
                dict(
                    count=fit.count,
                    masks=list(fit.masks),
                    sums=fit.sums,
                    dc_sum=dc.sums[0],
                    intercept=0,
                ),
            )
        scores = fit.finish()
        dc.finish()
        terms = select_terms(kind, scores, self.recipe)
        if not terms:
            return None
        return StudyPredictor(
            kind,
            "teacher-reproduction" if kind == "R" else "absolute-bit-prediction",
            source_digest(self.source),
            self.recipe,
            terms,
            count,
        )

    def agreement(self, model: StudyPredictor) -> dict[str, Any]:
        evaluation = Evaluation()
        terms = tuple((t.mask, t.signed_sum) for t in model.terms)
        for points in self.data.panel_chunks("Ragreement"):
            signs = self.data.signs("Ragreement", points, self.teacher)
            labels = ((1 - signs) // 2).astype(np.uint8)
            pred, ties, official = self._predict(points, terms)
            evaluation.add(pred, labels, ties, official_predictions=official)
            self._save("R-agreement-chunk", evaluation.snapshot())
        return evaluation.finish(self.recipe.r_agreement_size)

    def classification(self, model: StudyPredictor) -> dict[str, Any]:
        evaluation = Evaluation()
        terms = tuple((t.mask, t.signed_sum) for t in model.terms)
        for chunk in self.data.classification_chunks():
            pred, ties, official = self._predict(chunk.points, terms)
            evaluation.add(pred, chunk.labels, ties, official_predictions=official)
            self._save("R-classification-chunk", evaluation.snapshot())
        return evaluation.finish(self.recipe.r_classification_size)

    def evaluate(self, stage: str, model: StudyPredictor) -> dict[str, Any]:
        self.data.release_panel(stage)
        evaluation = Evaluation()
        terms = tuple((t.mask, t.signed_sum) for t in model.terms)
        for points in self.data.panel_chunks(stage):
            signs = self.data.signs(stage, points, None)  # never B -> teacher
            labels = ((1 - signs) // 2).astype(np.uint8)
            pred, ties, official = self._predict(points, terms)
            evaluation.add(pred, labels, ties, official_predictions=official)
            self._save(stage + "-chunk", evaluation.snapshot())
        size = self.recipe.b_dev_size if stage == "Bdev" else self.recipe.b_test_size
        return evaluation.finish(size)


class EngineeringLimit(RuntimeError):
    """All kept evidence saved, never truncate to top400."""
