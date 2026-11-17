import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cipher_lab_plugins.differential_relations import evaluate


def test_evaluate_import_boundary() -> None:
    source = Path(evaluate.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert not any(
                x in (node.module or "").split(".")
                for x in ("data", "training", "application", "ciphers", "secrets", "random")
            )


def test_saved_encoding_matches_public_pair_layout() -> None:
    x = np.array([0, 1, 2, 3], dtype=np.uint64)
    result = evaluate.encode_saved(x)
    assert result.shape == (1, 4, 2, 64)
    assert result.dtype == np.float32
    assert result[0, 0, 0].sum() == 0 and result[0, 0, 1, 57] == 1


def test_new_output_refuses_overwrite(tmp_path: Any) -> None:
    with pytest.raises(FileExistsError):
        evaluate.write_evaluation(tmp_path, {"summary": {}})


def test_evaluate_all_temp_calls_have_explicit_parent() -> None:
    tree = ast.parse(Path(evaluate.__file__).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "TemporaryDirectory"
    ]
    assert len(calls) == 2
    assert all(
        any(k.arg == "dir" and ast.unparse(k.value) == "output.parent" for k in n.keywords)
        for n in calls
    )


def test_evaluate_requires_existing_safe_output_parent(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="OUTPUT_PARENT"):
        evaluate.evaluate_saved(
            run_record=tmp_path / "absent.json",
            checkpoint=tmp_path / "absent.pt",
            dataset=tmp_path / "absent.npz",
            output=tmp_path / "missing" / "out",
        )
