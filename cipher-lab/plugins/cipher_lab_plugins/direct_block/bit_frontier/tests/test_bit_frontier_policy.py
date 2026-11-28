from __future__ import annotations

import pytest


def test_frozen_count_caps() -> None:
    from cipher_lab_plugins.direct_block.bit_frontier import config

    assert config.CAPS == {'real_blocks': 1064962, 'null_words': 0,
                           'optimizer_updates': 230400, 'forward_rows': 67338240}
    assert len(set(config.SLOTS)) == 30
    assert config.counts(1, 108) == dict(real_blocks=212994, null_words=0,
                                         optimizer_updates=27648, forward_rows=8159232)


@pytest.mark.parametrize(('accuracy', 'bce', 'expected'), [
    (0.49, 0.696, True), (0.51, 0.689, True), (0.509, 0.687, False),
    (0.5, 0.688, True), (0.5, 0.687999999999, False),
    (0.5101, 0.694, False), (0.4899, 0.694, False),
])
def test_all_panels_required(accuracy: float, bce: float, expected: bool) -> None:
    from cipher_lab_plugins.direct_block.bit_frontier.training import near_chance

    def panel(acc: float, loss: float) -> dict[str, object]:
        return {'arms': {'correct': {'accuracy': acc, 'bce': loss}}, 'constant': {'bce': 0.693}}

    assert near_chance({'r1': panel(0.5, 0.69), 'r2': panel(accuracy, bce)}) is expected
    assert not near_chance({'r1': panel(0.52, 0.69), 'r2': panel(0.5, 0.69)})
