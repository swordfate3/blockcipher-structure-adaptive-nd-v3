from __future__ import annotations

import json
from pathlib import Path


def test_frozen_bit_structure_config() -> None:
    from cipher_lab_plugins.direct_block.bit_structure import config

    assert config.CAPS == dict(real_blocks=212994, optimizer_updates=46080,
                               forward_rows=13467648, null_words=0)
    assert len(set(config.SLOTS)) == 6
    assert config.counts((18,) * 6, test=False) == dict(
        real_blocks=147458, optimizer_updates=27648, forward_rows=7962624, null_words=0)
    path = Path(__file__).resolve().parents[5] / 'configs/direct-block-simon32-bit-structure.json'
    assert json.loads(path.read_text(encoding='utf-8')) == config.recipe()
