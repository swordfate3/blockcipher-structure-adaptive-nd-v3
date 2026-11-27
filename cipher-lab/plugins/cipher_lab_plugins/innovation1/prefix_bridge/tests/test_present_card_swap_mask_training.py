from __future__ import annotations

import json
from pathlib import Path


def test_static_recipe() -> None:
    from cipher_lab_plugins.innovation1.present_card_swap_mask import cli, config

    path = Path('configs/innovation1-present-card-swap-mask.json')
    assert json.loads(path.read_text(encoding='utf-8')) == config.recipe()
    assert cli.main(['check', '--config', str(path)]) == 0
    assert config.SLOTS == ('swap-50', 'swap-50-mask')
    assert config.CAPS == dict(real_blocks=1_327_684, null_words=0,
                               optimizer_updates=6_000, forward_rows=47_359_296)
    assert config.recipe()['max_auxiliary_targets'] == 240_000_000
