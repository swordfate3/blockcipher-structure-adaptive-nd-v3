"""Pure saved-value presentation: no Torch, oracle, training or metric imports."""
from __future__ import annotations

import json
import math
from html import escape
from typing import cast

from .config import LSTM_METHOD, REUSE_METHOD

_FLOW = '''<svg viewBox="0 0 920 96" role="img" aria-label="Experiment sequence">
<rect x="1" y="1" width="918" height="94" rx="10" fill="#edf3ff"/>
<text x="18" y="34">Public preflight → one set of cube panels → MLP fit → Shared-M fit</text>
<text x="18" y="68">Both DEV best from disk → generate TEST labels → paired scoring → publish</text>
</svg>'''
_REUSE_FLOW = _FLOW.replace(
    'Public preflight → one set of cube panels', 'Device admission → reuse fixed TRAIN/DEV panels',
).replace('generate TEST labels', 'read and decode source TEST')
_LSTM_FLOW = _REUSE_FLOW.replace('MLP fit → Shared-M fit', 'one LSTM fit').replace(
    'Both DEV best from disk', 'Restore LSTM DEV best from disk')
_LSTM_METHOD_NOTE = ('One LSTM fit using 64 ordered MSB0 bit pairs, two unidirectional layers, '
                     'and fresh zero states. TEST follows the strict reload of its only best.')
_LSTM_COMPARISON = ('LSTM vs. TRAIN majority only, using a family-cluster paired difference '
                    'with a conservative 97.5% normal interval.')
_PAIRED_COMPARISON = ('Family-cluster paired differences: 97.5% normal intervals for Shared-M '
                      'vs. majority and Shared-M vs. MLP, Bonferroni familywise 5%. '
                      'Per-family rows are in the saved record.')
_LSTM_REFERENCE = ('Historical saved results (descriptive only): MLP 49.9969%, Shared-M 49.8363%. '
                   'Source run cube-sharedm-simon7-50epoch-20261124-01, SHA256 '
                   'befe14e5fb467412965d98cc0f121a827370d82860d9399e8cf4df145ff9ef72. '
                   'No historical model was fitted or evaluated here; no paired intervals '
                   'against the historical models are reported.')
_REUSE_LIMIT = ('The configured TEST panel was previously reported. This is adaptive exploration, '
                'not independent confirmation.')
_CONFIGURED_SOURCE_NOTE = ('Expected identity from the saved recipe, not evidence that all source '
                           'panels were verified in this run.')
_LEGEND = {'train_bce': '#06755b', 'full_bce': '#324b9e', 'active_bce': '#a44100'}
_COLORS = {'majority': '#666e77', 'mlp': '#324b9e', 'sharedm': '#06755b', 'lstm': '#a44100'}


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _reuse(saved: dict[str, object]) -> bool:
    # A storage fault can seal the record and strip summary; the recipe survives.
    recipe = _mapping(saved.get('recipe'))
    return (recipe.get('mode') == 'saved-panels'
            and recipe.get('method') in (REUSE_METHOD, LSTM_METHOD))


def _lstm(saved: dict[str, object]) -> bool:
    recipe = _mapping(saved.get('recipe'))
    return recipe.get('mode') == 'saved-panels' and recipe.get('method') == LSTM_METHOD


def _configured_source(saved: dict[str, object]) -> str:
    return escape(_json(_mapping(saved.get('recipe')).get('source')))


def _list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _lstm_comparison(test: dict[str, object]) -> str:
    if not test:
        return f'Planned comparison: {_LSTM_COMPARISON}'
    rows = ' Per-family rows are in the saved record.' if _list(test.get('families')) else ''
    return f'Comparison: {_LSTM_COMPARISON}{rows}'


def _num(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _curves(arms: dict[str, object], names: tuple[str, ...]) -> str:
    figures = []
    for name in names:
        points = [_mapping(point) for point in _list(_mapping(arms.get(name)).get('dev'))]
        values = [_num(point.get(key)) for point in points for key in _LEGEND]
        if not any(value is not None for value in values):
            continue
        top = max((value for value in values if value is not None), default=1.) or 1.
        elements = ['<path d="M45 20V160H548" fill="none" stroke="#222"/>',
                    f'<text x="5" y="24">{top:.3g}</text>', '<text x="20" y="165">0</text>']
        # At most nine evenly spaced labels, including both endpoints. The curves
        # below still render every saved sample at its original position.
        last = len(points) - 1
        gaps = min(last, 8)
        for i in (j * last // max(1, gaps) for j in range(gaps + 1)):
            x = 60 + 470 * i / max(1, last)
            epoch = escape(str(points[i].get('epoch', '')))
            elements.append(f'<text x="{x:.2f}" y="184" text-anchor="middle">{epoch}</text>')
        for j, (key, color) in enumerate(_LEGEND.items()):
            coords = []
            for i, point in enumerate(points):
                value = _num(point.get(key))
                if value is None:
                    continue
                x = 60 + 470 * i / max(1, len(points) - 1)
                y = 160 - 140 * (value / top)
                coords.append(f'{x:.2f},{y:.2f}')
                elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}">'
                                f'<title>{key}: {value}</title></circle>')
            elements.append(f'<polyline points="{" ".join(coords)}" fill="none" '
                            f'stroke="{color}" stroke-width="2"/>')
            elements.append(f'<text x="{55 + 160*j}" y="211" fill="{color}">{key}</text>')
        figures.append(f'<h3>{name}: BCE by saved epoch</h3><svg viewBox="0 0 560 220" '
                       f'role="img" aria-label="Saved {name} BCE curves">'
                       + ''.join(elements) + '</svg>')
    return ''.join(figures)


def _scores(test: dict[str, object], names: tuple[str, ...]) -> str:
    accuracy = _mapping(test.get('accuracy'))
    bars = []
    for i, arm in enumerate(('majority', *names)):
        color = _COLORS[arm]
        value = _num(accuracy.get(arm))
        if value is None or not 0 <= value <= 1:
            continue
        y = 24 + 36 * i
        bars.append(f'<text x="4" y="{y+19}">{arm}</text>'
                    f'<rect x="100" y="{y}" width="380" height="24" fill="#eef1f4"/>'
                    f'<rect x="100" y="{y}" width="{380*value:.3f}" height="24" '
                    f'fill="{color}"/><text x="490" y="{y+19}">{value:.4%}</text>')
    if not bars:
        return ''
    return ('<svg viewBox="0 0 580 145" role="img" aria-label="Saved active-bit accuracy">'
            + ''.join(bars) + '</svg>')


def _json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_report(saved: dict[str, object]) -> bytes:
    """Return HTML bytes, using only committed values; no metric recomputation."""
    summary = _mapping(saved.get('summary'))
    arms = _mapping(summary.get('arms'))
    test = _mapping(summary.get('test')) if saved.get('status') == 'SUCCEEDED' else {}
    counts = escape(_json(saved.get('counts')))
    public = {key: value for key, value in summary.items() if key not in ('test', 'arms')}
    reuse = _reuse(saved)
    lstm = _lstm(saved)
    names = ('lstm',) if lstm else ('mlp', 'sharedm')
    flow = _LSTM_FLOW if lstm else _REUSE_FLOW if reuse else _FLOW
    method_note = f'<p>{_LSTM_METHOD_NOTE}</p>' if lstm else ''
    reference = f'<p>{_LSTM_REFERENCE}</p>' if lstm else ''
    comparison = _lstm_comparison(test) if lstm else _PAIRED_COMPARISON
    confirmation = ('Previously reported TEST is not a sealed confirmation.' if lstm
                    else 'TEST is saved, not a sealed confirmation.')
    contrast = ('The final hidden state compresses a fixed bit order into 128 features.' if lstm
                else 'This compares architecture and capacity, not an isolated routing effect.')
    configured = (f'<h2>Configured data source</h2><p>{_CONFIGURED_SOURCE_NOTE}</p>'
                  f'<pre>{_configured_source(saved)}</pre>') if reuse else ''
    limit = f'<p>{_REUSE_LIMIT}</p>' if reuse else ''
    detail = {key: value for key, value in test.items() if key != 'families'}
    outcome = (_scores(test, names) + '<pre>' + escape(_json(detail)) + '</pre>') if test else (
        '<p>No final TEST score was committed.</p>')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>SIMON7 cube-pair comparison</title>
<style>body{{max-width:960px;margin:2rem auto;padding:0 1rem;font:16px system-ui;color:#172635}}
svg{{width:100%;max-width:920px}}pre{{background:#f3f6fa;padding:1em;overflow:auto}}</style>
<h1>SIMON7 cube-pair comparison</h1>
<p>Run {escape(str(saved.get('run_id')))} · saved status {escape(str(saved.get('status')))}.
Run-store status is not process exit evidence; see the launcher's native exit record.</p>
<p>Source: {escape(str(saved.get('run_ref')))} · SHA256 {escape(str(saved.get('run_sha256')))}</p>
{configured}<h2>Method</h2>{flow}{method_note}
<p>Sixteen variants per family; eight complementary pairs. One synthetic private key.
Full 128-bit BCE for TRAIN; DEV active-8 BCE chooses the earliest minimum separately per arm.
Active input positions (LSB0): (0, 21, 42, 60). Scored output columns (MSB0):
(63, 42, 21, 3, 127, 106, 85, 67). TRAIN-only majority, ties go to zero.</p>
<h2>Actual accounting and gate</h2><pre>{counts}</pre><pre>{escape(_json(public))}</pre>
<p>Accounting uses the last cooperative sample; null means unknown. On storage failure,
only last committed accounting may survive. No hard RSS or wall isolation is claimed.</p>
<h2>TRAIN convergence and DEV model selection</h2>{_curves(arms, names)}
<pre>{escape(_json(arms))}</pre>
<h2>Final TEST (active-8)</h2>{outcome}{limit}
<p>{comparison} Treat gains as exploratory only; a non-gain applies to this fixed recipe only.</p>
{reference}<p>Single key and seed, reduced rounds, synthetic data. {confirmation}
{contrast} This is not a cross-key result or evidence of a practical cipher break.</p>
</html>'''.encode()


def report_files(saved: dict[str, object]) -> dict[str, bytes]:
    """The core's frozen record provenance plus compact Markdown and HTML."""
    source = {'schema': 1, **{key: saved[key] for key in ('run_id', 'run_ref', 'run_sha256')}}
    summary = _mapping(saved.get('summary'))
    test = _mapping(summary.get('test')) if saved.get('status') == 'SUCCEEDED' else {}
    outcome = escape(_json({key: test[key] for key in ('accuracy', 'comparisons') if key in test}))
    if not test:
        outcome = 'No final TEST score was committed.'
    md = (f'# SIMON7 cube-pair comparison\n\nSaved status: {escape(str(saved.get("status")))}.'
          ' See native exit separately. [Charts and method](index.html).\n\n'
          f'## Actual accounting\n\n```json\n{escape(_json(saved.get("counts")))}\n```\n\n'
          f'## Final TEST\n\n{outcome}\n\nSingle-key, single-seed, reduced-round exploration.\n')
    if _reuse(saved):
        arms = _mapping(summary.get('arms'))
        actual = {name: {**{key: arm.get(key) for key in ('epochs_completed', 'stop_reason')},
                         'best_epoch': _mapping(arm.get('selected')).get('epoch')}
                  for name, value in arms.items() if (arm := _mapping(value))}
        md += (f'\n## Configured data source\n\n{_CONFIGURED_SOURCE_NOTE}\n\n'
               f'{_configured_source(saved)}\n\n## Actual fits\n\n'
               f'{escape(_json(actual)) if actual else "No fit details survived in the record."}'
               f'\n\n{_REUSE_LIMIT}\n')
    if _lstm(saved):
        md += f'\n{_LSTM_METHOD_NOTE}\n\n{_lstm_comparison(test)}\n\n{_LSTM_REFERENCE}\n'
    return {'index.html': render_report(saved), 'report.md': md.encode(),
            'source.json': (_json(source) + '\n').encode()}
