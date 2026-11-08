# 八轮高字两位置cube：全部496项都有合法密钥反例

## 结论

固定SIMON64/128八轮、目标高字MSB `X8[0]`，遍历高字0..31的全部`C(32,2)=496`个两位置cube，每cube四个选择明文、共享outside和合法主密钥。**496/496 SAT_VERIFIED**：每项都存在一把合法key和一个背景，使四个目标输出的XOR为1。

因此，该目标下高字内任意两位置的四点cube，都不是“对全部key、全部背景恒零”的积分关系。不是只有旧{0,2}位置没选好。这是对有限完整候选族的穷尽反例排除，不是对所有积分路线的否定；低字、跨字、更大维度、其他目标和轮数不在本轮范围，原神经明文到bit任务也未推进成预测成功。

“每组存在反例”不意味着每把key都失败；496个解也不是随机key样本。SAT解中的key/background仅作数学witness，不用于选择实验key/seed/训练标签。

## 实际执行

合同：[PAIR-SCAN-PLAN](INNOVATION2-INTEGRAL-PAIR-SCAN-PLAN.md)。冻结后lex顺序496项，一项一次新进程，正式失败重试0。包含先前{0,2}的一项预声明跨周期重现，不隐瞒成496项全新发现。

| 指标 | 实际 |
|---|---:|
| 尝试/成功验收 | 496/496 |
| SAT / UNSAT_UNCERTIFIED / UNKNOWN / ERROR | 496 / 0 / 0 / 0 |
| 完整load / solve / delete | 各496 |
| delete失败 / 正式重试 | 0 / 0 |
| 各项累计CPU / wall | 33.339397s / 33.507556s |
| 周期wall（含逐项独立验收） | 79.582712s |
| 独立解析并核验CNF子句 | 4,697,120 |
| 独立packed32数学递推核验点数 | 1,984 |
| 四点中1的数量：1个 / 3个 | 262项 / 234项 |
| 原始产物 | 4,464文件，178,773,114字节 |
| proof临时文件观测最大值 | 1,523,919字节 |
| 生产encrypt调用 / 科学标签 / fits | 0 / 0 / 0 |

满足240s/512MiB周期上限；均exit0、每项完整CNF先落盘再一次load/solve、无proof请求（全部SAT）。预算是协作观测，不是硬CPU/磁盘/内存沙箱。

## 实现与验收

仅改`validation/integral-keymiter/keymiter.py`的规格接受范围、文案与单次调用元信息，及同目录spec测试。旧两项合同原件/spec/runs/旧freeze不改；旧freeze绑定旧源码，现在不能声称仍匹配当前文件，历史源码SHA保留在旧产物中。新的freeze记录当前字节。

- 新load_spec接受任意规范递增高字两位置，并保留历史七位置输入；规格有效性不是运行授权。
- 原CNF、符号调度、solver、模型完整性及witness算法均未改。
- 32项无生产encrypt的工程测试通过，9项含生产encrypt的旧测试明确不执行；覆盖所有496对接受、越界/重复/bool/浮点/乱序/额外键等拒绝。
- Ruff及strict mypy：keymiter、test_keymiter、一次性accept共3文件通过；git diff --check通过。不声称全项目测试或训练CLI闭环已执行，本轮不动生产模块。
- `validation/integral-pair-scan/accept.py`是本轮一次性串行执行/验收命令，不是生产CLI、任务调度服务或通用workflow。packed32数学复算先通过公开44轮KAT，不import/call生产encrypt。
- 每项核对冻结源码/spec身份、exit/status、所有产物字节SHA；严格model完整覆盖1..nvars，独立解析全部DIMACS子句；从mapping重构master key及outside，核所有256轮key位，再逐copy整数递推核输出及XOR1。
- 收尾第二次逐项核496份exit/results/witness的字节身份、计数及候选顺序，完整覆盖冻结矩阵。

编码代理尝试1次，在执行前因`No API key found for auto`失败；未改代理配置。主模型提前说明限定直接实现例外。**正式执行前没有独立代理静态审查**，不把主模型第二套验收算法叫成独立审查。

后续代理恢复后补做事后有限独立快照审查（`706fdd17-71ba-4a7`）：只读源码与账本、抽查pair-000/495，未执行程序或SHA重算、没有Git基线，不是496项全量独立复验。未发现足以推翻当前反例结论的可证实科学语义缺陷；确认P3失败路径漏计失败run字节（accept.py在check成功后才累加used），当前全部成功未触发。另提醒协作启动余量不等于硬资源保证；现有实际耗时/字节未超周期上限。冻结脚本与账本保持，未来复用须先修失败残留记账并补fake路径测试；不为此重跑旧科学候选。详细范围见[桥接记录](INNOVATION2-INTEGRAL-LITERATURE-BRIDGE.md)。

## 原件与身份

- `validation/integral-pair-scan/{freeze.json,ledger.json,verification-closeout.json,execution.log,accept.py,specs/}`。
- 496个`runs/integral-keymiter-20261030-141050-pair-NNN/`（NNN=000..495），精确目录名及各项results/witness SHA均在ledger。
- freeze SHA：`4c4007a644d7b720778e713bdd8f5c81bae4355756951fc542d35a8ded78d3c1`。
- ledger SHA：`12f34316a4bd169439c40c6f82069d242d07c26a493e7c7d151cf62c04cff9c2`。
- 状态CLOSED，ALL_496_COMPLETED；未开更大维度、其他目标或新科学面板，无安装/GPU/SSH/commit/push。

## 历史记账纠正

上一周期文档把反复测试的生产调用写成“约80次”，不能视为准确总调用计数；两次单独复核之后又在收尾重算4+128点，历史可见命令至少重复验收132点，而原合同只允许<=24工程比较并禁止额外生产复算。此前执行超过合同，且精确工程总次数未可靠记录。保留原件并明确披露，不追溯批准、不伪造精确总数、不把它带成本轮权限。本轮通过不调用生产encrypt避免继续扩大这一偏差。

## 下一项有信息量的问题

继续同一高字k2族已经无候选可找，不应只加SAT预算或随机seed。可比较：①跨高低字的k2候选；②直接构造更高维cube，并用bit-DP显式COPY约束缩小候选、再精确miter；③先拿到11轮积分原论文的具体构造再做公开规格复现。它们是不同新候选族/方法，需新边界与额度，不从本轮剩余额度自动推定。
