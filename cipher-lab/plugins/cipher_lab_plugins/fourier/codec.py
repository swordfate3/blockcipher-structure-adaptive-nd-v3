"""Bounded canonical JSON predictor format; exclusively standard-library dependencies.

No keys, oracle state, filesystem access, NumPy, or Torch. Identity is an exact
public contract, not a guess from filenames. An absent discovery has no model.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import cast

from cipher_lab_plugins.fourier.config import FourierRecipe, parse_fourier

MAX_MODEL_BYTES = 65536


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: expected bounded integer")
    return value


def _object(value: object, names: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != names:
        raise ValueError(f"{name}: missing or unknown fields")
    return cast("dict[str, object]", value)


def validate_identity(identity: object, recipe: FourierRecipe) -> dict[str, object]:
    """Return a detached strict public identity; neither arbitrary extras nor secrets."""
    obj = _object(identity, {"source_sha256", "backend", "synthetic", "task", "label_mode"},
                  "identity")
    source = obj["source_sha256"]
    if type(source) is not str or re.fullmatch(r"[0-9a-f]{64}", source) is None:
        raise ValueError("identity: expected lowercase source SHA256")
    if type(obj["synthetic"]) is not bool or obj["synthetic"] != recipe.synthetic:
        raise ValueError("identity: synthetic mismatch")
    backend = "demo" if recipe.synthetic else "simon"
    if type(obj["backend"]) is not str or obj["backend"] != backend:
        raise ValueError("identity: backend mismatch")
    if type(obj["label_mode"]) is not str or obj["label_mode"] not in (
        "target_bit", "pseudorandom_function"
    ):
        raise ValueError("identity: unsupported label_mode")
    task = _object(obj["task"], {"cipher", "rounds", "target_bit", "input_bits"}, "task")
    cipher = "synthetic" if recipe.synthetic else "SIMON"
    if type(task["cipher"]) is not str or task["cipher"] != cipher:
        raise ValueError("identity: task cipher mismatch")
    for name in ("rounds", "target_bit", "input_bits"):
        expected = getattr(recipe, name)
        _integer(task[name], name, expected, expected)
    return {**obj, "task": dict(task)}


@dataclass(frozen=True, slots=True)
class FourierTerm:
    mask: int
    signed_sum: int
    coefficient: float


@dataclass(frozen=True, slots=True)
class FourierPredictor:
    recipe: FourierRecipe
    identity: dict[str, object]
    count: int
    intercept_sum: int
    intercept: float
    baseline: int
    terms: tuple[FourierTerm, ...]
    coefficient_rows: int


def _coefficient(value: object, signed_sum: object, n: int, name: str) -> None:
    total = _integer(signed_sum, name + " signed_sum", -n, n)
    if (total + n) % 2:
        raise ValueError(f"{name}: impossible sign sum parity")
    if type(value) not in (int, float):
        raise ValueError(f"{name}: expected finite coefficient")
    numeric = cast("int | float", value)
    if not -1 <= numeric <= 1 or not math.isfinite(numeric) or numeric != total / n:
        raise ValueError(f"{name}: coefficient disagrees with signed_sum/count")


def validate_predictor(model: FourierPredictor) -> None:
    """Also used by public prediction; direct dataclass construction cannot bypass checks."""
    if type(model) is not FourierPredictor or type(model.recipe) is not FourierRecipe:
        raise ValueError("predictor: expected typed model/recipe")
    model.recipe.__post_init__()
    validate_identity(model.identity, model.recipe)
    n = _integer(model.count, "count", model.recipe.coef_size, model.recipe.coef_size)
    _coefficient(model.intercept, model.intercept_sum, n, "intercept")
    _integer(model.baseline, "baseline", 0, 1)
    if model.baseline != int(model.intercept_sum < 0):
        raise ValueError("baseline: must be frozen coefficient-panel majority, tie zero")
    if type(model.terms) is not tuple or not 1 <= len(model.terms) <= model.recipe.beam_width:
        raise ValueError("terms: expected 1..beam_width nonzero discoveries")
    previous = 0
    suffix = (1 << (64 - model.recipe.input_bits)) - 1
    for term in model.terms:
        if type(term) is not FourierTerm:
            raise ValueError("term: expected typed FourierTerm")
        mask = _integer(term.mask, "mask", 1, 2**64 - 1)
        if mask <= previous or mask & suffix:
            raise ValueError("mask: noncanonical order/duplicate or outside input domain")
        previous = mask
        _coefficient(term.coefficient, term.signed_sum, n, "term")
        if term.signed_sum == 0 or abs(term.coefficient) < model.recipe.tau / 2:
            raise ValueError("term: below fixed nonzero coefficient threshold")
    # Includes intercept, rejected candidates, and zero-mask deduplication.
    rows = _integer(model.coefficient_rows, "coefficient_rows", n * (len(model.terms) + 1),
                    n * (model.recipe.beam_width + 1))
    if rows % n:
        raise ValueError("coefficient_rows: expected whole candidate panels")


def encode_predictor(model: FourierPredictor) -> bytes:
    validate_predictor(model)
    obj = {"schema": "fourier-predictor-v1", "bit_order": "uint64-msb0-prefix",
           "recipe": model.recipe.to_dict(), "identity": model.identity, "count": model.count,
           "intercept_sum": model.intercept_sum, "intercept": model.intercept,
           "baseline": model.baseline, "coefficient_rows": model.coefficient_rows,
           "terms": [{"mask": term.mask, "signed_sum": term.signed_sum,
                      "coefficient": term.coefficient} for term in model.terms]}
    content = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if len(content) > MAX_MODEL_BYTES:
        raise ValueError("predictor: exceeds 64 KiB")
    return content


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("predictor: duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> object:
    raise ValueError("predictor: non-finite JSON number")


def decode_predictor(content: bytes, *, expected_identity: dict[str, object]) -> FourierPredictor:
    if type(content) is not bytes or not 0 < len(content) <= MAX_MODEL_BYTES:
        raise ValueError("predictor: expected 1..65536 bytes")
    try:
        value = json.loads(content, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("predictor: invalid JSON") from None
    obj = _object(value, {"schema", "bit_order", "recipe", "identity", "count", "intercept_sum",
                          "intercept", "baseline", "coefficient_rows", "terms"}, "predictor")
    if obj["schema"] != "fourier-predictor-v1" or obj["bit_order"] != "uint64-msb0-prefix":
        raise ValueError("predictor: unsupported schema or bit order")
    recipe = parse_fourier(obj["recipe"])
    actual = validate_identity(obj["identity"], recipe)
    expected = validate_identity(expected_identity, recipe)
    if actual != expected:
        raise ValueError("predictor: public identity mismatch")
    raw_terms = obj["terms"]
    if type(raw_terms) is not list or not 1 <= len(raw_terms) <= 32:
        raise ValueError("predictor: expected 1..32 terms")
    terms = []
    for raw in raw_terms:
        item = _object(raw, {"mask", "signed_sum", "coefficient"}, "term")
        terms.append(FourierTerm(cast("int", item["mask"]), cast("int", item["signed_sum"]),
                                 cast("float", item["coefficient"])))
    model = FourierPredictor(recipe, actual, cast("int", obj["count"]),
                             cast("int", obj["intercept_sum"]), cast("float", obj["intercept"]),
                             cast("int", obj["baseline"]), tuple(terms),
                             cast("int", obj["coefficient_rows"]))
    validate_predictor(model)
    return model
