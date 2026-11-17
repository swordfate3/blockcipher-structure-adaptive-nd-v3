"""Saved-only plain-Chinese report for the bounded public-combination frontier.

Consumes only summary arms/comparisons/pairing/arm_resources/plan_identity/
output_bound written by the frontier method plus recipe/counts/meter/source.
No tensors, model/data access, label generation, metric recomputation or file
I/O. The renderer never adjudicates the experiment: it prints the fixed rules
and the saved numbers, and missing values stay unsaved rather than zero.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from cipher_lab.training.report import _json, _map, _number, _text

_SPLITS = ('train', 'dev', 'test')
_SPLIT_LABEL = {'train': '训练集', 'dev': '开发集', 'test': '测试集'}
_ARMS = ('A', 'B', 'C')
_COLOR = {'A': '#1f4e79', 'B': '#9c2f2f', 'C': '#2f7d32'}
_MISSING = '未保存（不补 0）'
_BASE_NAME = {'C-A': '甲', 'C-B': '乙'}
_ROUND_NAME = {4: '四轮', 5: '五轮', 6: '六轮'}


def _arm_name(name: str, summary: Mapping[str, object]) -> str:
    if name == 'A':
        return '直接预测'
    if name == 'B':
        return '原始输入学习纠错'
    dimension = _map(summary.get('plan_identity')).get('dimension')
    return (f'{dimension}个公开组合学习纠错' if type(dimension) is int
            else '该轮公开组合学习纠错')


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


def _p(value: object) -> str:
    """Exact saved tail probability in scientific notation; never rounded to 0.

    A nonzero saved value keeps significant digits (e.g. 1.000000e-08), so the
    page cannot print 0.000000 while silently meaning underflow. Only a saved
    value that is exactly zero carries the conditional underflow annotation;
    missing values are shown as unsaved and make no probability claim.
    """
    number = _number(value)
    if number is None:
        return _MISSING
    if number == 0.:
        return f'{number:.6e}（保存值恰为 0：这是浮点下溢，不是概率精确为 0）'
    return f'{number:.6e}'


def _comparison(summary: Mapping[str, object], key: str, split: str) -> Mapping[str, object]:
    return _map(_map(_map(summary.get('comparisons')).get(key)).get(split))


def _round_label(saved: Mapping[str, object]) -> str:
    rounds = _map(saved.get('recipe')).get('rounds')
    return _ROUND_NAME.get(rounds, '该轮') if type(rounds) is int else '该轮'


def _cards(summary: Mapping[str, object]) -> str:
    cards: list[str] = []
    for name in _ARMS:
        test = _percent(_acc(summary, name, 'test'))
        cards.append(
            '<div style="flex:1;min-width:230px;border:1px solid #c9d6e2;border-radius:8px;'
            'padding:12px;margin:6px;background:#f7fafc">'
            f'<div style="font-size:15px;color:#334">{_text(_arm_name(name, summary))}</div>'
            f'<div style="font-size:34px;font-weight:bold">{_text(test)}</div>'
            '<div style="font-size:13px;color:#556">测试集猜对比例</div>'
            f'<div style="font-size:13px;color:#556">训练集 '
            f'{_text(_percent(_acc(summary, name, "train")))} · 开发集 '
            f'{_text(_percent(_acc(summary, name, "dev")))}</div></div>')
    return '<div style="display:flex;flex-wrap:wrap">' + ''.join(cards) + '</div>'


def _label_mode(saved: Mapping[str, object]) -> str:
    mode = _map(saved.get('recipe')).get('label_mode')
    return mode if mode in ('cipher', 'null') else 'missing'


def _answer_source(saved: Mapping[str, object],
                   summary: Mapping[str, object]) -> str:
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
        return ('这条保存记录里的答案是模拟的：每条明文的私有答案与公开零主密钥参考'
                '各计一次“加密”，共两次加密计数；demo 只用公开假函数，不执行真实密码，'
                '也不使用真实密钥，本页不认证为实际加密结果。')
    if source == 'cipher-adapter':
        return ('这条保存记录声明的答案是生产适配器给出的：每条明文计一次私有答案加密、'
                '一次零主密钥公开参考加密，共两次加密；后端由调用方声明，'
                '本页不认证该适配器或密码实现。')
    if source == 'cipher-unverified':
        return ('保存记录没有可核验的后端身份：答案与公开参考的来源无法确认，'
                '本页不推定它们是真实密码结果，也不据此补算或认证。')
    if source == 'null':
        return ('答案不是加密结果：每条明文只做一次公开零主密钥参考加密，'
                '答案另用一次独立的空标签随机抽样；它和明文、密钥都无关，'
                '用于对照三条路线会不会自己“看出”并不存在的规律。')
    return ('保存记录没有写明 label_mode（答案来源），本页不推定它是真实密文答案'
            '还是独立空标签，也不据此补算任何次数。')


def _question_parts(source: str, rounds: str, dimension: object) -> list[str]:
    size = f'{dimension}个' if type(dimension) is int else '一组'
    if source == 'cipher-demo':
        first = ('我们要让一个网络只看明文，猜一个模拟出来的目标位。demo 用公开假函数'
                 '生成目标与公开参考，不执行真实密码，也不是真实密钥。'
                 f'这一轮是{rounds}，公开组合来自该轮的公开计划。')
        prefix = '同一份模拟数据上训练三条结构相同的网络'
    elif source == 'cipher-adapter':
        first = (f'我们要让一个网络只看明文，猜{rounds}加密后密文某一位是0还是1。'
                 '答案由明文和一把我们不知道的密钥一起决定；后端由调用方声明，'
                 '本页不认证密码实现。')
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
        first = ('我们要让一个网络只看明文，猜一个标签位。保存记录没有写明这个标签位的来源，'
                 '本页不推定它是真实密文答案还是独立随机标签，只照原样展示已保存的数字。')
        prefix = '同一批明文与标签上训练三条结构相同的网络'
    second = (f'{prefix}，只改“输入怎么给、要它学什么”：甲直接猜答案；'
              '乙仍看原始64位明文，但先学“答案和公开参考的区别”；'
              f'丙先把明文换成该轮的{size}公开组合，再学“答案和公开参考的区别”。'
              '公开组合只由明文位和公开的异或、与运算推出，不读真实密钥、轮密钥、'
              '实际中间状态或其他密文位，也不把公开参考 h 送进网络。'
              '三条路线各自保存开发集误差最低的模型，三个模型都严格校验加载成功后才'
              '第一次看测试集。')
    third = ('为什么值得这样比：甲是“什么都不预先整理”的参照；乙把“只是换一种学习目标”'
             '的影响单独拿出来；丙才加入公开组合输入。若丙比乙还好，公开组合本身可能有帮助；'
             '若丙只比甲好、不比乙好，只能说明“先学区别再合并”这套思路在这组条件下更有用，'
             '不能把功劳算在公开组合上。')
    return [first, second, third]


def _method_svg(source: str, rounds: str, dimension: object) -> str:
    size = f'{dimension}个' if type(dimension) is int else '一组'
    flow = (f'{rounds}公开组合计划 → 同一面板训练 甲/乙/丙 三臂（公开布尔组合只从明文推出）'
            ' → 每轮开发集误差分数选模 → 三个 best 严格加载成功后才释放测试集')
    flow2 = (f'甲：64位明文直接预测；乙：64位明文学习“答案与公开参考的区别”再合并；'
             f'丙：{size}公开组合学习区别再合并。公开参考 h 只用于翻转恢复，不进入网络。')
    flow3 = ('报告只读已保存结果，不重新训练、不重新预测、不重新加密、不重算指标。')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 220" '
            'role="img" aria-label="三臂方法流程图">'
            '<rect width="1200" height="220" fill="#eef3f8"/>'
            f'<text x="12" y="34" font-size="14">{_text(flow)}</text>'
            f'<text x="12" y="70" font-size="14">{_text(flow2)}</text>'
            f'<text x="12" y="106" font-size="14">{_text(flow3)}</text>'
            f'<text x="12" y="146" font-size="13">{_text(_label_note(source))}</text>'
            '</svg>')


def _bars_svg(summary: Mapping[str, object]) -> str:
    rows: list[str] = []
    index = 0
    for split in _SPLITS:
        for name in _ARMS:
            y = 24 + index * 34
            value = _acc(summary, name, split)
            rows.append(f'<text x="8" y="{y + 12}" font-size="13">'
                        f'{_text(_SPLIT_LABEL[split])} · '
                        f'{_text(_arm_name(name, summary))}</text>')
            if value is None:
                rows.append(f'<text x="420" y="{y + 12}" font-size="13">'
                            f'{_text(_MISSING)}</text>')
            else:
                rows.append(f'<rect x="420" y="{y}" width="{value * 400:.2f}" height="16" '
                            f'fill="{_COLOR[name]}" opacity="0.85"/>'
                            f'<text x="{420 + value * 400 + 8:.2f}" y="{y + 12}" '
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
                f'fill="#f7f7f7"/><text x="12" y="40">无已保存学习曲线'
                f'（{_text(_MISSING)}，不重算）</text></svg>')
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
                         f'fill="{color}"><title>{_text(_arm_name(name, summary))} 第 {epoch} 轮'
                         f'{"训练集" if field == "train_bce" else "开发集"}'
                         f'={value:.6g}</title></circle>')
        part = '训练集' if field == 'train_bce' else '开发集'
        legend.append(f'<text x="{640 if len(legend) < 3 else 790}" '
                      f'y="{30 + 20 * (len(legend) % 3)}" font-size="12" fill="{color}">'
                      f'{_text(_arm_name(name, summary) + "·" + part)}</text>')
    parts.extend(legend)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300" '
            'role="img" aria-label="保存的逐轮学习曲线">'
            '<rect width="900" height="300" fill="#fbfbfb"/>'
            '<line x1="80" y1="260" x2="860" y2="260" stroke="#888"/>'
            + ''.join(parts) + '</svg>')


def _comparison_text(summary: Mapping[str, object], key: str, split: str = 'test') -> str:
    base = _BASE_NAME[key]
    stats = _comparison(summary, key, split)
    if not stats:
        return f'丙 与 {base} 的{_SPLIT_LABEL[split]}逐条对照：{_MISSING}'
    delta = _number(stats.get('delta'))
    delta_text = _MISSING if delta is None else f'{delta * 100:.3f} 个百分点'
    return (f'丙 与 {base} 在{_SPLIT_LABEL[split]}上的猜对比例差为 {delta_text}；逐条看，'
            f'丙对而{base}错的条数是 {_text(stats.get("b", _MISSING))}，'
            f'{base}对而丙错的条数是 {_text(stats.get("c", _MISSING))}。')


def _test_rule_note(summary: Mapping[str, object]) -> str:
    rule = ('预定主比较（仅测试集，探索性）：只看最终测试集上 丙 与 甲 的猜对比例差；'
            '只有差大于 0 且配对精确检验的单侧尾概率不超过 0.025/3，才记为'
            '“有正向探索证据”。除以3只为覆盖本周期最多三轮主比较，'
            '不校正开发集选模，也不是对自适应周期整体的独立确认。'
            '这个单侧尾概率不是“方法无效”的概率，也不是猜对比例本身。')
    stats = _comparison(summary, 'C-A', 'test')
    delta, p = _number(stats.get('delta')), _number(stats.get('p_one_sided'))
    if delta is None or p is None:
        return rule + f'本次保存结果缺少主比较所需数值（{_MISSING}），证据不完整，' \
                      '不能按规则裁决；本页不自动补算或判定。'
    return rule + ('本次两个已保存数值都存在；本页只展示它们，不重新计算、也不自动判定，'
                   '是否满足规则交由后续独立审查。')


def _dev_gate_note(summary: Mapping[str, object]) -> str:
    stats = _comparison(summary, 'C-A', 'dev')
    saved = (f'本次开发集已保存：丙={_text(_percent(_acc(summary, "C", "dev")))}；'
             f'甲={_text(_percent(_acc(summary, "A", "dev")))}；'
             f'丙对而甲错的条数 b={_text(stats.get("b", _MISSING))}；'
             f'丙错而甲对的条数 c={_text(stats.get("c", _MISSING))}；'
             f'单侧尾概率 p={_p(stats.get("p_one_sided"))}。')
    return ('升轮只看最终选中模型的开发集保存统计：丙的猜对比例不低于55%，且与甲的差为正，'
            '且精确单侧尾概率不超过 0.025/3；不满足就记门槛未达、后续轮不启动。'
            '55% 是本次保守的“值得再升轮”资源门槛，不是密码学通用成功标准；'
            '开发集同时参与了32次选模，所以这个精确二项尾数只是预定资源门槛，'
            '不是选模后仍然校准的显著性检验，除以3也不校正开发集选模。' + saved)


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
            delta = _number(stats.get('delta'))
            delta_text = _MISSING if delta is None else f'{delta * 100:.3f}pp'
            rows.append(
                f'{key} / {_SPLIT_LABEL[split]}：n={_text(stats.get("n", _MISSING))}；'
                f'b={_text(stats.get("b", _MISSING))}；c={_text(stats.get("c", _MISSING))}；'
                f'both_correct={_text(stats.get("both_correct", _MISSING))}；'
                f'both_wrong={_text(stats.get("both_wrong", _MISSING))}；'
                f'sum_d={_text(stats.get("sum_d", _MISSING))}；'
                f'sum_d2={_text(stats.get("sum_d2", _MISSING))}；'
                f'delta={delta_text}；'
                f'p={_p(stats.get("p_one_sided"))}；'
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
            f'{name}（{_arm_name(name, summary)}）：'
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
    picked = [values.get(name) for name in arms]
    if not all(_valid_state_sha(value) for value in picked):
        return '证据不完整（未保存）'
    return '相同' if len({str(value) for value in picked}) == 1 else '不同'


def _pairing_rows(summary: Mapping[str, object]) -> list[str]:
    pairing = _map(summary.get('pairing'))
    identities = _map(pairing.get('initial_state_sha256'))
    shared = _map(pairing.get('shared_layers_sha256'))
    rows = [f'初始化规则：{_text(pairing.get("initialization_rule", _MISSING))}',
            f'初始化种子={_text(pairing.get("init_seed", _MISSING))}；'
            f'打乱种子={_text(pairing.get("shuffle_seed", _MISSING))}',
            f'甲与乙整份初始参数校验值：{_text(_same_identities(identities, ("A", "B")))}',
            f'三条路线共有的隐藏层/输出层初值校验值：'
            f'{_text(_same_identities(shared, ("A", "B", "C")))}']
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


def _plan_rows(summary: Mapping[str, object]) -> list[str]:
    identity = _map(summary.get('plan_identity'))
    bound = _map(summary.get('output_bound'))
    return [
        f'轮数={_text(identity.get("rounds", _MISSING))}；'
        f'目标位={_text(identity.get("target_bit", _MISSING))}；'
        f'表示={_text(identity.get("representation", _MISSING))}；'
        f'公开组合维数={_text(identity.get("dimension", _MISSING))}',
        f'顺序校验值={_text(identity.get("order_sha256", _MISSING))}；'
        f'公开计划校验值={_text(identity.get("plan_sha256", _MISSING))}',
        f'工程输出上界：三份 best 加一份最大临时副本与16MiB元数据预留的峰值='
        f'{_text(bound.get("peak_bytes", _MISSING))} 字节；'
        f'元数据预留={_text(bound.get("metadata_bytes", _MISSING))} 字节',
    ]


def _status_text(saved: Mapping[str, object]) -> str:
    status = saved.get('status', 'UNKNOWN')
    return ('已完成' if status == 'SUCCEEDED' else
            '已停止' if status == 'STOPPED' else
            '失败' if status == 'FAILED' else
            '告警' if status == 'ALARM' else
            '未知（仍在运行，没有最终记录）' if status == 'RUNNING' else '未知')


def _limits_note(saved: Mapping[str, object], summary: Mapping[str, object]) -> str:
    rounds = _map(saved.get('recipe')).get('rounds')
    if summary.get('reason') == 'ENGINEERING_LIMIT':
        return ('本记录是工程未启动（ENGINEERING_LIMIT）：在构造任何面板或训练前，'
                '最坏输出/元数据预分配已超出本周期工程上限，因此没有开始该轮训练。'
                '这不是训练阴性、不是预测失败、不是最低或最高理论轮数，也不补 0 当成绩。')
    if type(rounds) is int and rounds >= 6:
        return ('六轮在本周期受结构/工程上限约束；若没有可审查的精确压缩，六轮只报告'
                '工程未启动，不宣布预测阴性或理论边界。')
    return (f'本页只展示{_round_label(saved)}这一次固定条件下已保存的结果，'
            '不同轮密钥独立，不能把跨轮差异全归轮数，也不构成跨密钥确认、'
            '全部明文域结论或新颖性证明。')


def render_frontier_report(saved: Mapping[str, object]) -> dict[str, bytes]:
    """Exactly three files for the original RunStore; never writes or reads here."""
    summary = _map(saved.get('summary'))
    rounds = _round_label(saved)
    identity = _map(summary.get('plan_identity'))
    dimension = identity.get('dimension')
    title = f'{rounds}公开组合神经位预测报告'
    status = _status_text(saved)
    raw_backend = summary.get('backend_identity')
    backend = ('合成演示数据（不是真实密码）' if raw_backend == 'demo' else
               '调用方声明的生产适配器（本页未认证）'
               if raw_backend in ('simon', 'ciphers') else '未核验')
    synthetic = summary.get('synthetic') is True or raw_backend == 'demo'
    identity_note = ('本页是合成工程演示，用于检查接线，不是真实密码结果。' if synthetic
                     else '后端身份由调用方声明，密码实现未在本页核验；不认证科学有效性。'
                     if raw_backend in ('simon', 'ciphers') else
                     '保存记录没有可核验的后端身份；不认证真实密码或科学有效性。')
    source = _answer_source(saved, summary)
    warning = ('限制：本周期最多三轮、每轮一把新私有密钥、每臂单次训练；科学预算与输出上限'
               '都是协作式检查，不是操作系统硬隔离。本页不重新计算、不补缺失值、'
               '不把未知或工程超限补成 0。')
    cards = _cards(summary)
    curve_note = ('曲线只连接已保存的每一轮训练集误差分数与开发集误差分数，'
                  '实线是训练集、虚线是开发集；不重新前向、不重算，也不补缺失点。')
    baseline_note = ('表中“TRAIN基线”是只用训练集标签的多数类常数，'
                     '不是公开参考 h，也不是把 h 当作模型的输入或标签。')
    detail = {'recipe': saved.get('recipe', {}), '实际计数': saved.get('counts', {}),
              'plan_identity': summary.get('plan_identity', {}),
              'output_bound': summary.get('output_bound', {}),
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
        '训练集：模型学习用的明文与答案。开发集：每一轮后用来挑“这轮更好吗”的'
        '明文与答案，内部字段名 dev。测试集：三个模型都冻结后才第一次看的'
        '明文与答案，内部字段名 test；它不参与升轮、阈值修改或补跑。',
        '甲（内部臂名 A，直接预测）：看原始64位明文直接猜答案。'
        '乙（内部臂名 B，原始输入学习纠错）：先学“答案与公开参考的区别”，再合并公开参考。'
        '丙（内部臂名 C，公开组合学习纠错）：先换成该轮公开组合，再学区别并合并。',
        '公开参考：用零主密钥（不是零轮密钥）按同样轮数算出的公开值，'
        '每一次都照常计入加密次数；它不进网络，只在输出端用于翻转恢复绝对答案。',
        '配对检验值（内部字段 p_one_sided）：只比较两条路线对同一条明文的判断，'
        '用精确条件分布算出的单侧尾概率；越小说明差别越难用“恰好抽到这种分歧”解释。'
        '尾概率按保存值以科学计数展示，非零值不会被四舍五入成 0；缺失显示未保存。',
        _label_note(source),
        '预定主比较：最终测试集上丙与甲的差大于 0 且单侧尾概率不超过 0.025/3。'
        '这是固定条件下的探索规则，不构成跨密钥确认。',
        '开发集升轮门槛：丙不低于55%、丙减甲为正、单侧尾概率不超过 0.025/3。'
        '55% 是资源门槛，这个尾概率也只是预定门槛而不是校准的显著性检验。',
        'pp：百分点，即猜对比例之差乘以100。',
    ]
    body_parts = [
        f'<h1>{_text(title)}：{_text(saved.get("run_id", "未知"))}</h1>'
        f'<p>执行状态：{_text(status)}；数据来源：{_text(backend)}</p>'
        f'<p>{_text(identity_note)}</p><p>{_text(warning)}</p>'
        f'<p>{_text(_limits_note(saved, summary))}</p>',
        '<h2>这个实验想回答什么</h2>'
        + ''.join(f'<p>{_text(paragraph)}</p>'
                  for paragraph in _question_parts(source, rounds, dimension))
        + f'<h2>方法图</h2>{_method_svg(source, rounds, dimension)}',
        f'<h2>猜对比例</h2>{cards}'
        f'<p>{_text(baseline_note)}</p>',
        '<h2>主比较：丙 与 甲（仅测试集，探索性）</h2>'
        f'<p>{_text(_comparison_text(summary, "C-A"))}</p>'
        f'<p>{_text(_test_rule_note(summary))}</p>',
        '<h2>开发集升轮门槛（与上面的测试主比较不同）</h2>'
        f'<p>{_text(_dev_gate_note(summary))}</p>',
        '<h2>次要比较（只作描述）：丙 与 乙</h2>'
        f'<p>{_text(_comparison_text(summary, "C-B"))}</p>'
        '<p>丙减乙只用于描述“公开组合输入”与“先学区别再合并”两种原因的相对表现，'
        '不另选结论，也不进入主比较或升轮规则。</p>',
        f'<h2>已保存的猜对比例图</h2>{_bars_svg(summary)}',
        f'<h2>全部学习曲线</h2><p>{_text(curve_note)}</p>{_curve_svg(summary)}',
        '<h2>结论边界</h2>'
        '<p>本页不声称跨密钥确认、不声称全部明文域或更高轮数也成立、不声称方法是新的。'
        '三轮各有独立私有密钥与面板，不能把跨轮差异全归轮数，也不把一次固定条件下的'
        '留出观察当作多密钥确认。若三条路线都接近一半，应讨论训练集能否拟合与'
        '是否有可泛化的公开规律，而不是直接否定所有公开表示。</p>',
        '<h2>附录</h2>',
        '<details><summary>技术词典（正文避免使用的术语在这里解释）</summary>'
        + ''.join(f'<p>{_text(line)}</p>' for line in glossary) + '</details>',
        '<details><summary>公开计划、输出上界与初始化</summary>'
        + ''.join(f'<p>{row}</p>' for row in _plan_rows(summary))
        + '<p>初始化是<b>预定初始化规则</b>，不是本页对本次运行的事实声明：每个线性层用公开种子'
        '与层名、输入维数派生独立随机流；规则要求甲/乙整份初始参数一致、三臂共有隐藏层/'
        '输出层初值一致。本页只依据已保存的完整合法 SHA 行判断本次是否一致：缺值或非64位'
        '小写十六进制一律记“证据不完整（未保存）”，不同则记“不同”。首层输入维数不同，'
        '因此参数总量不同，不能声称完全参数匹配。</p>'
        + ''.join(f'<p>{row}</p>' for row in _pairing_rows(summary)) + '</details>',
        '<details><summary>每条路线的参数、选模与资源</summary>'
        + ''.join(f'<p>{row}</p>' for row in _arm_rows(summary)) + '</details>',
        '<details><summary>四格表：逐条对照（丙对/丙错 × 对照对/对照错）</summary>'
        + ''.join(f'<p>{row}</p>' for row in _four_cell_rows(summary)) + '</details>',
        '<details><summary>比较原始数值与成本</summary>'
        + ''.join(f'<p>{row}</p>' for row in _comparison_rows(summary))
        + '<p>成本四单位：加密明文块次数、独立空标签抽样、优化更新次数、模型前向行数。</p>'
        '</details>',
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
    source_file = {'schema': 1, 'run_id': saved['run_id'], 'run_ref': saved['run_ref'],
                   'run_sha256': saved['run_sha256']}
    return {'report.md': (f'# {title}\n\n' + body + '\n').encode(),
            'index.html': html.encode(),
            'source.json': (json.dumps(source_file, ensure_ascii=False, sort_keys=True,
                                       indent=2) + '\n').encode()}
