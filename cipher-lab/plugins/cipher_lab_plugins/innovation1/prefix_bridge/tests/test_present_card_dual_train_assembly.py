from __future__ import annotations

import json
from pathlib import Path


def test_fixed_recipe_and_accounting() -> None:
    from cipher_lab_plugins.innovation1.present_card_dual_train import cli, config

    path = Path('configs/innovation1-present-card-dual-train.json')
    assert json.loads(path.read_text(encoding='utf-8')) == config.recipe()
    assert cli.main(['check', '--config', str(path)]) == 0
    assert config.SLOTS == ('swap-50', 'dual-train')
    assert config.CAPS == dict(real_blocks=1_327_684, null_words=0,
                               optimizer_updates=6_000, forward_rows=49_587_520)
    first = config.counts((18,), test=False)
    fit = config.counts((18, 19), test=False)
    diag = config.counts((18, 19), test=False, diagnostic=True)
    test = config.counts((18, 19))
    assert first['forward_rows'] == 18 * (500_000 + 2 * 32_768)
    assert fit['forward_rows'] - first['forward_rows'] == 19 * (1_000_000 + 2 * 32_768)
    assert diag['forward_rows'] - fit['forward_rows'] == 4 * 32_768
    assert test['forward_rows'] - diag['forward_rows'] == 4 * 131_072
