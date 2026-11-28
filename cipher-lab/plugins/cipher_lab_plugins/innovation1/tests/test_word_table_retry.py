"""Verify the retry identity and failure diagnostics without scientific work."""
import traceback
from pathlib import Path, PureWindowsPath
from unittest.mock import Mock

import pytest
from cipher_lab_plugins.innovation1.config import load_config
from cipher_lab_plugins.innovation1.four_single import interleave_config as cfg
from cipher_lab_plugins.innovation1.four_single import interleave_runtime, training


def test_retry_is_same_recipe_in_a_new_root() -> None:
    retry = load_config(Path('configs/innovation1-feistel-word-tables-10m-retry.json'))
    assert retry is cfg.WORD_RETRY
    cfg.official(cfg.WORD_RETRY, cfg.WORD_RETRY.run_id)
    cfg.validate_launch(PureWindowsPath(cfg.WORD_RETRY.output_root), cfg.GPU_UUID,
                        recipe=cfg.WORD_RETRY)
    with pytest.raises(ValueError):
        cfg.validate_launch(PureWindowsPath(cfg.WORD_INTERLEAVE.output_root), cfg.GPU_UUID,
                            recipe=cfg.WORD_RETRY)
    old, new = cfg.WORD_INTERLEAVE.to_dict(), retry.to_dict()
    assert old.keys() == new.keys()
    assert {key for key in old if old[key] != new[key]} == {'run_id', 'output_root'}
    assert retry.streams == cfg.WORD_INTERLEAVE.streams


@pytest.mark.parametrize('stderr_fails', [False, True])
def test_retry_records_exception_and_seals_failure(monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str],
                                                  stderr_fails: bool) -> None:
    recipe = cfg.WORD_RETRY
    store = Mock(_run_id=recipe.run_id)
    resources = Mock()
    monkeypatch.setattr(interleave_runtime, 'assemble_pools', Mock())
    monkeypatch.setattr(interleave_runtime, 'InterleaveRuntime', Mock())
    failure = Mock(side_effect=RuntimeError('artificial transfer error'))
    monkeypatch.setattr(training, '_fit', failure)
    if stderr_fails:
        monkeypatch.setattr(traceback, 'print_exc', Mock(side_effect=RuntimeError))
    training.execute(recipe, store, resources, launch={})
    finalized = store.finalize.call_args.args[0]
    assert finalized['status'] == 'FAILED'
    assert finalized['summary']['reason'] == 'EXECUTION_FAILED'
    assert finalized['summary']['test'] == {}
    error = capsys.readouterr().err
    if not stderr_fails:
        assert 'Traceback (most recent call last)' in error
        assert 'RuntimeError: artificial transfer error' in error
