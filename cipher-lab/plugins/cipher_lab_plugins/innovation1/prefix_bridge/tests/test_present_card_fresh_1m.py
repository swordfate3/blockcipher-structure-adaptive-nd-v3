from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def test_locked_recipe() -> None:
    from cipher_lab_plugins.innovation1.present_card_fresh import config

    config_path = Path('configs/innovation1-present-card-fresh-1m.json')
    assert json.loads(config_path.read_text()) == config.recipe()
    assert config.counts(18, test=False) == dict(real_blocks=1065540, null_words=0,
                                               optimizer_updates=1800, forward_rows=9589824)
    assert config.counts(30) == dict(real_blocks=1327684, null_words=0,
                                   optimizer_updates=3000, forward_rows=16114112)
    assert config.learning_rate(0) == .001
    assert config.learning_rate(2999) == .0001
    for step in (1, 99, 1799, 2998):
        assert config.learning_rate(step) == .0001 + 0.5 * (.001 - .0001) * (
            1 + math.cos(math.pi * step / 2999))
    for invalid in (-1, 3000, True, 1.0):
        with pytest.raises(ValueError):
            config.learning_rate(invalid)  # type: ignore[arg-type]


def test_real_batch_boundaries_without_materializing_a_panel() -> None:
    from cipher_lab_plugins.direct_block.fit import batch_indices
    from cipher_lab_plugins.innovation1.present_card_fresh import config

    for rows, full, tail in ((config.TRAIN_ROWS, 100, 0), (config.DEV_ROWS, 6, 2768),
                             (config.TEST_ROWS, 26, 1072)):
        offset = 0
        sizes = []
        for chunk in batch_indices(rows, config.BATCH_SIZE):
            assert chunk[0] == offset and chunk[-1] == offset + len(chunk) - 1
            assert len(chunk) % 4 == 0
            sizes.append(len(chunk))
            offset += len(chunk)
        assert offset == rows
        assert sizes == [5000] * full + ([tail] if tail else [])
