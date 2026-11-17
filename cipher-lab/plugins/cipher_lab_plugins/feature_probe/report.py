"""Saved-only plain-language report for the fixed three-arm public-feature probe.

Consumes only summary arms/comparisons/pairing/arm_resources written by the probe
method plus recipe/counts/meter/source. No tensors, model/data access, label
generation, metric recomputation or file I/O. Legacy empty top-level
metrics/learning_curve/selected placeholders are never presented as results.

The main body is written for readers without experiment vocabulary: it explains
the question, the three methods and the guessed-share cards first. Raw field
names, four-cell tables, identity hashes and the glossary live in a folded
appendix. Every saved string is escaped and missing values are shown as unsaved,
never replaced with zero.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from cipher_lab.training.report import _json, _map, _number, _text

_SPLITS = ('train', 'dev', 'test')
_SPLIT_LABEL = {'train': '训练集', 'dev': '开发集', 'test': '测试集'}
_ARMS = ('A', 'B', 'C')
_ARM_NAME = {'A': '直接预测', 'B': '原始输入学习纠错', 'C': '13个公开组合学习纠错'}
_ARM_INPUT = {
    'A': '看64位原始明文，直接猜目标位',
    'B': '看64位原始明文，先猜“目标位与公开参考的区别”，再把公开参考合并回去',
    'C': '先把明文换成13个公开组合，再像乙一样先猜区别、再合并公开参考',
}
_COLOR = {'A': '#1f4e79', 'B': '#9c2f2f', 'C': '#2f7d32'}
_MISSING = '未保存（不补 0）'
_BASE_NAME = {'C-A': '甲', 'C-B': '乙'}


def _arm(summary: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _map(_map(summary.get('arms')).get(name))


def _task(summary: Mapping[str, object], name: str, split: str) -> Mapping[str, object]:
    metrics = _map(_arm(summary, name).get('metrics'))
    return _map(_map(_map(metrics.get(split)).get('tasks')).get('SIMON'))


def _acc(summary: Mapping[str, object], name: str, split: str) -> float | None:
    value = _number(_task(summary, name, split).get('acc'))
    return value if value is not None and 0. <= value <= 1. else None


def _percent(value: float | None) -> str:
    return _MISSING if value is None else f'{value * 100:.3f}%'


def _decimal(value: object, digits: int = 6) -> str:
    number = _number(value)
    return _MISSING if number is None else f'{number:.{digits}f}'


def _comparison(summary: Mapping[str, object], key: str, split: str) -> Mapping[str, object]:
    return _map(_map(_map(summary.get('comparisons')).get(key)).get(split))


def _cards(summary: Mapping[str, object]) -> str:
    cards: list[str] = []
    for name in _ARMS:
        test = _percent(_acc(summary, name, 'test'))
        train = _percent(_acc(summary, name, 'train'))
        dev = _percent(_acc(summary, name, 'dev'))
        cards.append(
            '<div style="flex:1;min-width:230px;border:1px solid #c9d6e2;border-radius:8px;'
            'padding:12px;margin:6px;background:#f7fafc">'
            f'<div style="font-size:15px;color:#334">{_text(_ARM_NAME[name])}</div>'
            f'<div style="font-size:34px;font-weight:bold">{_text(test)}</div>'
            '<div style="font-size:13px;color:#556">测试集猜对比例</div>'
            f'<div style="font-size:13px;color:#556">训练集 {_text(train)} · '
            f'开发集 {_text(dev)}</div></div>')
    return '<div style="display:flex;flex-wrap:wrap">' + ''.join(cards) + '</div>'


def _label_mode(saved: Mapping[str, object]) -> str:
    """Explicit tri-state: cipher, null, or not saved (never inferred as cipher)."""
    mode = _map(saved.get('recipe')).get('label_mode')
    return mode if mode in ('cipher', 'null') else 'missing'


def _answer_source(saved: Mapping[str, object],
                   summary: Mapping[str, object]) -> str:
    """Answer path: null/missing, or cipher split by saved backend identity.

    The top-level disclaimer never excuses an internally wrong method
    paragraph, so demo/adapter/unverified cipher text differs explicitly.
    """
    mode = _label_mode(saved)
    if mode != 'cipher':
        return mode
    backend = summary.get('backend_identity')
    if backend == 'demo':
        return 'cipher-demo'
    if backend in ('simon', 'ciphers'):
        return 'cipher-adapter'
    return 'cipher-unverified'


def _label_note(source: str) -> str:
    if source == 'cipher-demo':
        return ('这条保存记录里的答案是模拟的：每条明文的私有答案与公开参考各计一次'
                '“加密”，共两次加密计数；demo 只用公开假函数，不执行真实密码，'
                '也不使用真实密钥，本页不认证为实际加密结果。')
    if source == 'cipher-adapter':
        return ('这条保存记录声明的答案是生产适配器给出的：每条明文计一次私有答案加密、'
                '一次零主密钥公开参考加密，共两次加密；后端由调用方声明，'
                '本页不认证该适配器或密码实现。')
    if source == 'cipher-unverified':
        return ('保存记录没有可核验的后端身份：答案与公开参考的来源无法确认，'
                '本页不推定它们是真实密码结果，也不据此补算或认证。')
    if source == 'null':
        return ('答案不是加密结果：每条明文只做一次公开参考加密，'
                '答案另用一次独立的空标签随机抽样；它和明文、密钥都无关，'
                '用于对照三条路线会不会自己“看出”并不存在的规律。')
    return ('保存记录没有写明 label_mode（答案来源），本页不推定它是真实密文答案'
            '还是独立空标签，也不据此补算任何次数。')


def _method_svg(parameter_note: str, source: str) -> str:
    subject = ('同一批明文 + 模拟的目标与公开参考（不执行真实密码）'
               if source == 'cipher-demo' else
               '同一批明文 + 调用方声明的生产适配器目标与公开参考（本页未认证）'
               if source == 'cipher-adapter' else
               '同一批明文 + 来源未核验的目标与公开参考'
               if source == 'cipher-unverified' else
               '同一批明文 + 一组独立的随机标签' if source == 'null' else
               '同一批明文 + 保存记录未写明来源的标签')
    flow = (f'{subject}（训练/开发/测试三段互不重叠） → '
            '甲/乙/丙三条结构相同的网络各自训练，每次只看一小批')
    flow2 = ('每个轮次结束后在开发集算一次误差分数，只保留误差最低的那个模型 → '
             '三个模型都严格校验加载成功后才第一次看测试集')
    flow3 = ('同一批测试预测同时给出三条路线的猜对比例与两两逐条对照 → '
             '报告只读已保存结果，不重新训练、不重新预测、不重新加密')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 280" '
            'role="img" aria-label="三条路线方法流程图">'
            '<rect width="1200" height="280" fill="#eef3f8"/>'
            f'<text x="12" y="32" font-size="14">{_text(flow)}</text>'
            f'<text x="12" y="64" font-size="14">{_text(flow2)}</text>'
            f'<text x="12" y="96" font-size="14">{_text(flow3)}</text>'
            f'<text x="12" y="128" font-size="13">{_text(_label_note(source))}</text>'
            f'<text x="12" y="168" font-size="13">{_text(_ARM_INPUT["A"])}</text>'
            f'<text x="12" y="194" font-size="13">{_text(_ARM_INPUT["B"])}</text>'
            f'<text x="12" y="220" font-size="13">{_text(_ARM_INPUT["C"])}</text>'
            + (f'<text x="560" y="168" font-size="13">{_text(parameter_note)}</text>'
               if parameter_note else '') + '</svg>')


def _bars_svg(summary: Mapping[str, object]) -> str:
    rows: list[str] = []
    index = 0
    for split in _SPLITS:
        for name in _ARMS:
            y = 24 + index * 34
            value = _acc(summary, name, split)
            rows.append(f'<text x="8" y="{y + 12}" font-size="13">'
                        f'{_text(_SPLIT_LABEL[split])} · {_text(_ARM_NAME[name])}</text>')
            if value is None:
                rows.append(f'<text x="330" y="{y + 12}" font-size="13">{_MISSING}</text>')
            else:
                rows.append(f'<rect x="330" y="{y}" width="{value * 500:.2f}" height="16" '
                            f'fill="{_COLOR[name]}" opacity="0.85"/>'
                            f'<text x="{330 + value * 500 + 8:.2f}" y="{y + 12}" '
                            f'font-size="13">{value * 100:.2f}%</text>')
            index += 1
    height = 24 + index * 34
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 1000 {height}" role="img" aria-label="已保存的猜对比例">'
            f'<rect width="1000" height="{height}" fill="#fbfbfb"/>'
            + ''.join(rows) + '</svg>')


def _series(summary: Mapping[str, object]) -> list[tuple[str, str, list[tuple[int, float]]]]:
    series: list[tuple[str, str, list[tuple[int, float]]]] = []
    for name in _ARMS:
        raw_entries = _arm(summary, name).get('learning_curve')
        entries: list[object] = raw_entries if isinstance(raw_entries, list) else []
        for field in ('train_bce', 'dev'):
            points: list[tuple[int, float]] = []
            for entry in entries:
                node = _map(entry)
                epoch = node.get('epoch')
                raw = (node.get('train_bce') if field == 'train_bce'
                       else _map(node.get('dev')).get('mean_bce'))
                value = _number(raw)
                if type(epoch) is int and value is not None:
                    points.append((epoch, value))
            series.append((name, field, points))
    return series


def _curve_svg(summary: Mapping[str, object]) -> str:
    series = _series(summary)
    values = [value for _, _, points in series for _, value in points]
    if not values:
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 120" '
                'role="img" aria-label="学习曲线"><rect width="900" height="120" '
                'fill="#f7f7f7"/><text x="12" y="40">无已保存学习曲线'
                '（不重算、不补 0）</text></svg>')
    low, high = min(values), max(values)
    if high == low:
        high = low + 1.
    epochs = sorted({epoch for _, _, points in series for epoch, _ in points})
    left, right, top, bottom = 80., 860., 30., 260.

    def x(epoch: int) -> float:
        if len(epochs) == 1:
            return (left + right) / 2
        return left + (right - left) * (epoch - epochs[0]) / (epochs[-1] - epochs[0])

    def y(value: float) -> float:
        return bottom - (bottom - top) * (value - low) / (high - low)

    parts: list[str] = [f'<text x="8" y="{top + 4}" font-size="12">误差分数 {high:.4f}</text>',
                        f'<text x="8" y="{bottom}" font-size="12">误差分数 {low:.4f}</text>']
    for epoch in epochs:
        parts.append(f'<text x="{x(epoch):.1f}" y="{bottom + 22:.1f}" font-size="12">'
                     f'第 {epoch} 轮</text>')
    legend: list[str] = []
    for name, field, points in series:
        if not points:
            continue
        color = _COLOR[name]
        coords = ' '.join(f'{x(epoch):.1f},{y(value):.1f}' for epoch, value in points)
        dash = '' if field == 'train_bce' else ' stroke-dasharray="6 4"'
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} '
                     f'points="{coords}"/>')
        for epoch, value in points:
            parts.append(f'<circle cx="{x(epoch):.1f}" cy="{y(value):.1f}" r="3" '
                         f'fill="{color}"><title>{_text(_ARM_NAME[name])} 第 {epoch} 轮 '
                         f'{"训练集" if field == "train_bce" else "开发集"}'
                         f'={value:.6g}</title></circle>')
        part = '训练集' if field == 'train_bce' else '开发集'
        legend.append(f'<text x="{640 if len(legend) < 3 else 790}" '
                      f'y="{30 + 20 * (len(legend) % 3)}" font-size="12" fill="{color}">'
                      f'{_text(_ARM_NAME[name] + "·" + part)}</text>')
    parts.extend(legend)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300" '
            'role="img" aria-label="保存的逐轮学习曲线">'
            '<rect width="900" height="300" fill="#fbfbfb"/>'
            '<line x1="80" y1="260" x2="860" y2="260" stroke="#888"/>'
            + ''.join(parts) + '</svg>')


def _comparison_text(summary: Mapping[str, object], key: str) -> str:
    base = _BASE_NAME[key]
    stats = _comparison(summary, key, 'test')
    if not stats:
        return f'丙 与 {base} 的测试集逐条对照：{_MISSING}'
    delta = _number(stats.get('delta'))
    delta_text = _MISSING if delta is None else f'{delta * 100:.3f} 个百分点'
    b = _text(stats.get('b', _MISSING))
    c = _text(stats.get('c', _MISSING))
    return (f'丙 与 {base} 在测试集上的猜对比例差为 {delta_text}；逐条看，'
            f'丙对而{base}错的条数是 {b}，{base}对而丙错的条数是 {c}。')


def _confirmation_scope(source: str) -> str:
    if source == 'cipher-demo':
        return '跨数据或跨模拟条件的确认'
    if source == 'cipher-adapter':
        return '跨密钥确认'
    if source == 'null':
        return '跨标签来源或跨条件的确认'
    return '跨条件的确认'


def _verdict_note(summary: Mapping[str, object], source: str) -> str:
    """The fixed C-A rule must stay visible; never recompute or auto-decide."""
    rule = ('预定主比较规则：只看最终测试集上 丙 与 甲 的猜对比例差；只有差大于 0 且'
            '配对精确检验的单侧尾概率不超过 0.025 时，才记为“有正向探索证据”。'
            '这个单侧尾概率不是“方法无效”的概率，也不是猜对比例本身；它只描述'
            '在两条路线其实一样时，恰好出现这么多单向分歧有多罕见。'
            f'这是一次固定条件下的探索证据，不构成{_confirmation_scope(source)}。')
    stats = _comparison(summary, 'C-A', 'test')
    delta = _number(stats.get('delta'))
    p = _number(stats.get('p_one_sided'))
    if delta is None or p is None:
        return rule + ('本次保存结果缺少主比较所需的数值，证据不完整，不能按规则裁决，'
                       '本页也不自动补算或判定。')
    return rule + ('本次两个已保存数值都存在；本页只展示它们，不重新计算、也不自动判定，'
                   '是否满足规则交由后续独立审查。')


def _four_cell_rows(summary: Mapping[str, object]) -> list[str]:
    rows: list[str] = []
    for key in ('C-A', 'C-B'):
        base = _BASE_NAME[key]
        for split in _SPLITS:
            stats = _comparison(summary, key, split)
            if not stats:
                rows.append(f'{key} / {_SPLIT_LABEL[split]}：{_MISSING}')
                continue
            rows.append(
                f'{key} / {_SPLIT_LABEL[split]}（丙 与 {base}）：'
                f'丙对·{base}对={_text(stats.get("both_correct", _MISSING))}；'
                f'丙对·{base}错={_text(stats.get("b", _MISSING))}；'
                f'丙错·{base}对={_text(stats.get("c", _MISSING))}；'
                f'丙错·{base}错={_text(stats.get("both_wrong", _MISSING))}')
    return rows


def _comparison_rows(summary: Mapping[str, object]) -> list[str]:
    rows: list[str] = []
    for key in ('C-A', 'C-B'):
        for split in _SPLITS:
            stats = _comparison(summary, key, split)
            if not stats:
                rows.append(f'{key} / {_SPLIT_LABEL[split]}：{_MISSING}')
                continue
            loss = _map(stats.get('loss_difference'))
            rows.append(
                f'{key} / {_SPLIT_LABEL[split]}：n={_text(stats.get("n", _MISSING))}；'
                f'b={_text(stats.get("b", _MISSING))}；c={_text(stats.get("c", _MISSING))}；'
                f'sum_d={_text(stats.get("sum_d", _MISSING))}；'
                f'sum_d2={_text(stats.get("sum_d2", _MISSING))}；'
                f'delta={_decimal(stats.get("delta"))}；'
                f'p_one_sided={_decimal(stats.get("p_one_sided"), 6)}；'
                f'normal_95_interval={_text(stats.get("normal_95_interval", _MISSING))}；'
                f'loss_difference：direction={_text(loss.get("direction", _MISSING))} '
                f'sum={_decimal(loss.get("sum"))} '
                f'sum_squares={_decimal(loss.get("sum_squares"))} '
                f'mean={_decimal(loss.get("mean"))}')
    return rows


def _arm_rows(summary: Mapping[str, object]) -> list[str]:
    rows: list[str] = []
    for name in _ARMS:
        arm = _arm(summary, name)
        selected = _map(arm.get('selected'))
        model = _map(arm.get('model_spec'))
        resource = _map(_map(summary.get('arm_resources')).get(name))
        rows.append(
            f'{name}（{_ARM_NAME[name]}）：'
            f'输入={_text(arm.get("input_representation", _MISSING))}；'
            f'输出={_text(arm.get("output_mode", _MISSING))}；'
            f'参数量={_text(arm.get("parameter_count", _MISSING))}；'
            f'隐藏宽度={_text(model.get("hidden_width", _MISSING))}；'
            f'选中轮次={_text(selected.get("epoch", _MISSING))}；'
            f'选中时更新次数={_text(selected.get("updates", _MISSING))}；'
            f'选中时开发集误差分数={_decimal(selected.get("selected_dev_bce"))}；'
            f'选中模型校验值={_text(selected.get("sha256", _MISSING))}；'
            f'该臂累计墙钟={_decimal(resource.get("wall_seconds"), 3)} 秒；'
            f'累计CPU={_decimal(resource.get("cpu_seconds"), 3)} 秒；'
            f'上限={_decimal(resource.get("wall_limit_seconds"), 3)} 秒')
    return rows


def _valid_state_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and set(value) <= set('0123456789abcdef'))


def _same_identities(values: Mapping[str, object], arms: tuple[str, ...]) -> str:
    """Agreement needs every relevant arm's checksum; absence is not agreement."""
    picked = [values.get(name) for name in arms]
    if not all(_valid_state_sha(value) for value in picked):
        return '证据不完整（未保存）'
    return '相同' if len({str(value) for value in picked}) == 1 else '不同'


def _pairing_rows(summary: Mapping[str, object]) -> list[str]:
    pairing = _map(summary.get('pairing'))
    identities = _map(pairing.get('initial_state_sha256'))
    shared = _map(pairing.get('shared_layers_sha256'))
    same_ab = _same_identities(identities, ('A', 'B'))
    same_shared = _same_identities(shared, ('A', 'B', 'C'))
    rows = [f'初始化规则：{_text(pairing.get("initialization_rule", _MISSING))}',
            f'初始化种子={_text(pairing.get("init_seed", _MISSING))}；'
            f'打乱种子={_text(pairing.get("shuffle_seed", _MISSING))}',
            f'甲与乙整份初始参数校验值：{_text(same_ab)}',
            f'三条路线共有的隐藏层/输出层初值校验值：{_text(same_shared)}']
    panels = _map(pairing.get('panels'))
    for split in _SPLITS:
        identity = _map(panels.get(split))
        if not identity:
            rows.append(f'{_SPLIT_LABEL[split]}面板：{_MISSING}')
            continue
        rows.append(f'{_SPLIT_LABEL[split]}面板：行数={_text(identity.get("rows", _MISSING))}；'
                    f'明文校验值={_text(identity.get("x_sha256", _MISSING))}；'
                    f'答案校验值={_text(identity.get("y_sha256", _MISSING))}；'
                    f'公开参考校验值={_text(identity.get("h_sha256", _MISSING))}')
    return rows


def _feature_note() -> str:
    return ('13个公开组合只由明文的位和公开的异或、与运算推出，不读真实密钥、轮密钥、'
            '实际中间状态或其他密文位，属于任何人都能自己算的公开计算；'
            '公开组合个数少不等于模型与另外两条路线参数完全匹配。')


def _feature_formula() -> str:
    return ('公开组合的完整定义（X、Y是明文高、低半字，字内从最低位开始编号，'
            '加号表示异或，相邻书写表示与）：A_j = Y_j + X_{j-1}X_{j-8} + X_{j-2}；'
            'l=A29，m=A22，n=A15，p=X30+A28，q=X23+A21，s=A28，t=A21；'
            '13个输入固定顺序为 1, l, m, n, p, q, s, t, lm, mn, mq, pm, ln+lq+pn。'
            '这里的小写字母只是这13个坐标的名字，不是统计检验里的 p 值。')


def _question_parts(source: str) -> list[str]:
    if source == 'cipher-demo':
        first = ('我们要让一个网络只看明文，猜一个模拟出来的目标位。demo 用公开假函数'
                 '生成目标与公开参考，不执行真实密码，也不是真实密钥。问题是：如果先替'
                 '网络把“只用明文就能算出来的公开部分”整理好，剩下要学的部分会不会更容易学？')
        prefix = '同一份模拟数据上训练三条结构相同的网络'
    elif source == 'cipher-adapter':
        first = ('我们要让一个网络只看明文，猜三轮加密后密文某一位是0还是1。'
                 '答案由明文和一把我们不知道的密钥一起决定；后端由调用方声明，'
                 '本页不认证密码实现。问题是：如果先替网络把“只用明文就能算出来的'
                 '公开部分”整理好，剩下要学的部分会不会更容易学？')
        prefix = '同一把密钥、同一批明文上训练三条结构相同的网络'
    elif source == 'cipher-unverified':
        first = ('我们要让一个网络只看明文，猜一个标签位。保存记录没有可核验的后端身份，'
                 '本页不推定答案是真实密文、模拟值还是别的来源，只照原样展示已保存的数字。')
        prefix = '同一批明文与标签上训练三条结构相同的网络'
    elif source == 'null':
        first = ('这是一个对照实验：要让网络只看明文，猜一个与明文、密钥都无关的'
                 '独立随机标签位。正常情况下三条路线都只能接近一半；它用来检查训练和'
                 '评价流程会不会凭空制造出“看起来会预测”的优势。')
        prefix = '同一批明文与标签上训练三条结构相同的网络'
    else:
        first = ('我们要让网络只看明文，猜一个标签位。保存记录没有写明这个标签位的来源，'
                 '本页不推定它是真实密文答案还是独立随机标签，只照原样展示已保存的数字。')
        prefix = '同一批明文与标签上训练三条结构相同的网络'
    second = (f'{prefix}，只改“输入怎么给、要它学什么”：甲直接猜答案；'
              '乙仍看原始明文，但先学“答案和公开参考的区别”；'
              '丙先把明文换成13个公开组合，再学“答案和公开参考的区别”。'
              '三条路线各自保存开发集误差最低的模型，三个模型都严格校验加载成功后才'
              '第一次看测试集。')
    third = ('为什么值得这样比：甲是“什么都不预先整理”的参照；乙把“只是换一种学习目标”'
             '的影响单独拿出来；丙才加入公开组合输入。若丙比乙还好，公开组合本身可能有帮助；'
             '若丙只比甲好、不比乙好，只能说明“先学区别再合并”这套思路在这组条件下更有用，'
             '不能把功劳算在13个组合上。')
    return [first, second, third]


def _cost_note(source: str) -> str:
    if source == 'cipher-demo':
        detail = ('其中“加密明文块次数”是模拟计数：每条明文的私有答案与公开参考各计一次'
                  '（共两次），demo 不执行真实密码；测试集三个模型的最终评价也计入前向行数。')
    elif source == 'cipher-adapter':
        detail = ('加密次数包含每条明文的私有答案与公开参考两次；后端由调用方声明，'
                  '本页不认证密码实现；测试集三个模型的最终评价也计入前向行数。')
    elif source == 'cipher-unverified':
        detail = ('保存记录没有可核验的后端身份，无法确认加密与空标签次数的来源；'
                  '四单位原始计数见身份附录，测试集最终评价仍计入前向行数。')
    elif source == 'null':
        detail = ('加密次数只包含每条明文的公开参考一次，答案另计独立的空标签抽样；'
                  '测试集三个模型的最终评价也计入前向行数。')
    else:
        detail = ('保存记录缺少 label_mode，无法推定每条明文的加密与空标签次数；'
                  '四单位原始计数见身份附录，测试集最终评价仍计入前向行数。')
    return '成本四单位：加密明文块次数、独立空标签抽样、优化更新次数、模型前向行数。' + detail


def render_feature_probe_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Exactly three files for the original RunStore; never writes or reads here."""
    summary = _map(saved.get('summary'))
    title = '三轮公开组合输入对照报告'
    raw_status = saved.get('status', 'UNKNOWN')
    status = ('已完成' if raw_status == 'SUCCEEDED' else
              '已停止' if raw_status == 'STOPPED' else
              '失败' if raw_status == 'FAILED' else
              '告警' if raw_status == 'ALARM' else
              '未知（仍在运行，没有最终记录）' if raw_status == 'RUNNING' else '未知')
    raw_backend = summary.get('backend_identity')
    backend = ('合成演示数据（不是真实密码）' if raw_backend == 'demo' else
               '调用方声明的生产适配器（本页未认证）'
               if raw_backend in ('simon', 'ciphers') else '未核验')
    synthetic = summary.get('synthetic') is True or raw_backend == 'demo'
    identity = ('本页是合成工程演示，用于检查接线，不是真实密码结果。' if synthetic
                else '后端身份由调用方声明，密码实现未在本页核验；不认证科学有效性。'
                if raw_backend in ('simon', 'ciphers') else
                '保存记录没有可核验的后端身份；不认证真实密码或科学有效性。')
    answer_source = _answer_source(saved, summary)
    fixed = ('模拟的答案与公开参考（不执行真实密码）' if answer_source == 'cipher-demo' else
             '调用方声明的生产适配器与一把未公开密钥'
             if answer_source == 'cipher-adapter' else
             '来源未核验的标签与公开参考' if answer_source == 'cipher-unverified' else
             '固定一组独立随机标签' if answer_source == 'null' else '答案来源未保存')
    other = ('换模拟数据或假函数' if answer_source == 'cipher-demo' else
             '换密钥' if answer_source == 'cipher-adapter' else
             '换成别的随机标签' if answer_source == 'null' else '换成别的条件')
    warning = (f'限制：只跑了三轮、{fixed}、固定初始化、每种做法只训练一次。'
               f'因此它可以说明这次固定条件下三条路线谁猜得更准，但不能推出{other}、'
               '四轮或六轮也成立，也不构成新颖性证明；不要求超过以往任何成绩才算有价值。'
               '猜对比例接近一半时的微小差别不算有用预测。资源上限是协作式检查，'
               '不是操作系统的硬限制。本页只展示已保存结果。')
    cards = _cards(summary)
    parameter_note = ('丙的首层输入是13维、甲/乙是64维，所以三者参数总量不同，'
                      '不能说是完全公平的参数对照。')
    curve_note = ('曲线只连接已保存的每一轮训练集误差分数与开发集误差分数，'
                  '实线是训练集、虚线是开发集；不重新前向、不重算，也不补缺失点。'
                  '开发集误差最低的模型被保存，最终测试结果来自该保存模型。')
    detail = {'recipe': saved.get('recipe', {}), '实际计数': saved.get('counts', {}),
              'arms': summary.get('arms', {}),
              'comparisons': summary.get('comparisons', {}),
              'pairing': summary.get('pairing', {}),
              'arm_resources': summary.get('arm_resources', {}),
              'meter': summary.get('meter', {}), 'refs': summary.get('refs', {}),
              'source': saved.get('source', {}), 'policy': saved.get('policy', {}),
              '原因': summary.get('reason', '')}
    glossary = [
        '猜对比例：猜对的条数除以总条数；一半是瞎猜的期望水平。',
        '误差分数：模型给答案打的分数与实际答案之间的偏离程度，越小越好；'
        '它叫二元交叉熵，报告内部字段名 bce。',
        '训练集：模型学习用的明文与答案。开发集：每一轮后用来挑“这轮更好吗”的明文与答案，'
        '报告内部字段名 dev。测试集：三个模型都冻结后才第一次看的明文与答案，'
        '报告内部字段名 test；它不是训练过程能改的东西。',
        '甲（报告内部臂名 A，直接）：看原始明文直接猜答案。'
        '乙（报告内部臂名 B，原始输入纠错）：先学“答案与公开参考的区别”，再合并公开参考。'
        '丙（报告内部臂名 C，13个公开组合纠错）：先换成13个公开组合，再学区别并合并。',
        '公开参考：用零主密钥（不是零轮密钥）按同样轮数算出的公开值，'
        '每一次都照常计入加密次数。',
        '配对检验值（报告内部字段 p_one_sided）：只比较两条路线对同一条明文的判断，'
        '用精确条件分布算出的单侧尾概率；越小说明差别越难用“恰好抽到这种分歧”解释。'
        '它是单侧尾概率，不是“方法无效”的概率，也不在正文据此宣布结论。',
        _label_note(answer_source),
        '预定主比较规则：最终测试集上丙与甲的猜对比例差大于 0，且配对精确检验的'
        f'单侧尾概率不超过 0.025。这是一次固定条件下的探索规则，不是'
        f'{_confirmation_scope(answer_source)}，也不要求超过历史任何成绩。',
        'pp：百分点（percentage point），即猜对比例之差乘以100。',
    ]
    body_parts = [
        f'<h1>{_text(title)}：{_text(saved.get("run_id", "未知"))}</h1>'
        f'<p>执行状态：{_text(status)}；数据来源：{_text(backend)}</p>'
        f'<p>{_text(identity)}</p><p>{_text(warning)}</p>',
        '<h2>这个实验想回答什么</h2>'
        + ''.join(f'<p>{_text(paragraph)}</p>' for paragraph in _question_parts(answer_source))
        + f'<h2>方法图</h2>{_method_svg(parameter_note, answer_source)}',
        f'<h2>猜对比例</h2>{cards}'
        f'<p>{_text(_feature_note())}</p>',
        '<h2>主比较：丙 与 甲</h2>'
        f'<p>{_text(_comparison_text(summary, "C-A"))}</p>'
        f'<p>{_text(_verdict_note(summary, answer_source))}</p>'
        '<p>这是唯一预先固定的主比较；次要比较只用于解释，不把训练集或开发集的差异'
        '当作挑选显著结果的依据。</p>'
        '<h2>次要比较（只用于解释）：丙 与 乙</h2>'
        f'<p>{_text(_comparison_text(summary, "C-B"))}</p>'
        '<p>它帮助区分“公开组合输入”与“先学区别再合并”两种原因，'
        '不另选结论，也不进入上面的主比较规则。</p>',
        f'<h2>已保存的猜对比例图</h2>{_bars_svg(summary)}',
        f'<h2>全部学习曲线</h2><p>{_text(curve_note)}</p>{_curve_svg(summary)}',
        '<h2>结论边界</h2>'
        '<p>本页不声称跨密钥确认、不声称四轮或六轮也成立、不声称方法是新的。'
        '三条路线都有各自选模机会，测试集只在最后看一次；但一次固定条件下的'
        '留出观察仍可能被这次抽到的明文影响，所以它只是探索证据。'
        '若三条路线差不多、且都在一半附近，这次矩阵就应停下，'
        '讨论“训练集能不能拟合”与“有没有学到可泛化的公开规律”，'
        '而不是直接否定所有公开表示或转去换模型。</p>',
        '<h2>附录</h2>',
        '<details><summary>技术词典（正文避免使用的术语在这里解释）</summary>'
        + ''.join(f'<p>{_text(line)}</p>' for line in glossary) + '</details>',
        '<details><summary>13个公开组合与初始化</summary>'
        f'<p>{_text(_feature_formula())}</p>'
        '<p>初始化不是“整模型同一随机数”：每个线性层用公开种子与层名、输入维数派生'
        '独立随机流，甲/乙初始参数逐字相同，丙的共有隐藏层/输出层与甲/乙相同，'
        '但首层输入维数不同，因此丙的参数更少（首层13×宽+宽，而甲/乙是64×宽+宽）。'
        '这不是按结果挑初值，也不能声称参数完全匹配。</p>'
        + ''.join(f'<p>{row}</p>' for row in _pairing_rows(summary)) + '</details>',
        '<details><summary>每条路线的参数、选模与资源</summary>'
        + ''.join(f'<p>{row}</p>' for row in _arm_rows(summary)) + '</details>',
        '<details><summary>四格表：逐条对照（丙对/丙错 × 对照对/对照错）</summary>'
        + ''.join(f'<p>{row}</p>' for row in _four_cell_rows(summary)) + '</details>',
        '<details><summary>比较原始数值与成本</summary>'
        + ''.join(f'<p>{row}</p>' for row in _comparison_rows(summary))
        + f'<p>{_text(_cost_note(answer_source))}</p></details>',
        '<details><summary>身份、来源与原始保存值</summary>'
        '<p>source.json 绑定已冻结的 run.json 校验值；不提供模型或数据下载，'
        '本页不解码权重、不读取数据文件、不重算任何指标。</p>'
        f'<pre>{_json(detail)}</pre></details>',
    ]
    body = ''.join(body_parts)
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_text(title)}</title><style>body{{max-width:1200px;margin:auto;'
            'padding:1em;font-family:sans-serif;line-height:1.8}svg{width:100%}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}'
            'details{margin:0.8em 0}</style><body>' + body + '</body></html>')
    source = {'schema': 1, 'run_id': saved['run_id'], 'run_ref': saved['run_ref'],
              'run_sha256': saved['run_sha256']}
    return {'report.md': (f'# {title}\n\n' + body + '\n').encode(),
            'index.html': html.encode(),
            'source.json': (json.dumps(source, ensure_ascii=False, sort_keys=True,
                                       indent=2) + '\n').encode()}
