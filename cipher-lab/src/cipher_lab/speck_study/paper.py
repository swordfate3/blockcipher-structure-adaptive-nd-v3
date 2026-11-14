"""Eq(4), p451, Yuan/Wu/Zhang/Wu, DOI 10.46586/tosc.v2026.i1.441-467.

Literal MSB0 masks and coefficients scaled by 10000, in published equation
order. NOT Table5 (a different estimate). The spec fixes Eq4/Table3 as reference.
Read-only extraction: /tmp/pi-web-pdf/fourier-analysis-of-neural-distinguishers.md
(page11). No executable notebook or saved scientific arrays are imported.
"""

from __future__ import annotations

# Concatenated [c0L,c0R,c1L,c1R], MSB0. Preserve signed four-decimal literals.
PUBLISHED_TERMS: tuple[tuple[int, int], ...] = (
    (0x4020502070206020, 1824),
    (0x5820402058204020, 3454),
    (0x5030402050304020, -918),
    (0x7020602040205020, 1810),
    (0x5820402068207020, -1782),
    (0x5820482070206020, 900),
    (0x5820402070206820, -844),
    (0x7020602070206020, 3549),
    (0x6820702058204020, -1753),
    (0x7020682058204020, -868),
    (0x7020602058204820, 865),
    (0x7830602078306020, -880),
)
