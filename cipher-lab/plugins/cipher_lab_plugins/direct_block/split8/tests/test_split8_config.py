from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_split8_fixed_recipe_caps_and_launcher() -> None:
    from cipher_lab_plugins.direct_block.split8 import config

    fixed = Path('configs/direct-block-simon32-split8.json')
    assert json.loads(fixed.read_text()) == config.recipe()
    assert config.COLS == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert config.CAPS == dict(real_blocks=1081346, null_words=0, optimizer_updates=614400,
                               forward_rows=177733632)
    assert (config.WALL_SECONDS, config.CPU_SECONDS, config.RSS_BYTES,
            config.OUTPUT_BYTES, config.CHECKPOINT_BYTES) == (10800, 43200, 2**33, 2**29, 2**24)
    epochs = (18, 19, 20, 50) * 3
    assert config.counts(epochs) == {**config.CAPS, 'optimizer_updates': sum(epochs) * 1024,
                                     'forward_rows': sum(epochs) * 294912 + 786432}
    for bad in ((18,) * 11, (17,) * 12, (51,) * 12, (18.,) * 12):
        with pytest.raises(ValueError):
            config.counts(bad)  # type: ignore[arg-type]
    script = Path('scripts/start-cube-sharedm.ps1').read_text(encoding='ascii')
    assert script.count(f"'{fixed.name}'") == 2
    assert 'cipher_lab_plugins.direct_block.split8.cli' in script
    assert 'if ($split8) { $wallSeconds = 10800 }' in script
    assert "if ($reuse) { $cliArgs += @('--source-root', $SourceRoot) }" in script
    assert '& $python @cliArgs 1> $stdout 2> $stderr\n$code = $global:LASTEXITCODE' in script
