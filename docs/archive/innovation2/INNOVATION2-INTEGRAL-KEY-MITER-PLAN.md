# 合法主密钥约束下的固定cube判定：八轮{0,2}与{0..6}

状态：READY。继上一周期[精确miter结果](INNOVATION2-INTEGRAL-MITER-RESULTS.md)后继续推进；用户当前“继续推进任务”为本周期授权来源，前一周期的两次求解额度已用尽，不自动延续。

## 要回答的问题

上一周期在**独立轮常量**模型下给了两个SAT反例，于是不能说这两个八轮cube“对所有背景／常量恒零”。但真实SIMON的轮常量不是自由的：它们由一把128位主密钥经公开密钥调度决定。

本周期把同一个反例公式加上**合法密钥调度**，问：

> 固定cube的X8[0]输出XOR，是否对全部2^128把主密钥、全部outside明文都恒为0？

- **UNSAT**：则该cube在此公开模型下是真正的积分关系（对全部合法key），这比“对任意轮常量恒零”弱、但比任何单key实测强，且是合法key下才可能成立的新结论。
- **SAT**：则该cube对某把合法主密钥确实不恒零；此前的独立常量反例可能来自过强的常量自由度，也可来自真实失效。
- **UNKNOWN／超时**：什么都没证明，不得写成任何一种结论。

三种结果都必须如实分档；“有SAT反例”不等于“常用实验key会失败”，UNSAT也不等于破解。

## 方法与归属

- 复用上一周期已审查的编码骨架：每copy完整公开目标电路、AND 3条／XOR 4条Tseitin、active按MSB组合固定、末端parity断言1，全部copy共享outside明文。
- 唯一实质变化：符号常量不再自由，改由128位符号主密钥按公开密钥调度派生，且**全部copy共享同一把主密钥**。调度按 `src/cipher_experiments/ciphers/simon.py` 的公开公式，旋转用位序重排（无门），XOR用门，`z3` 与 `0xFFFFFFFC` 是公开常量。
- 8轮只需 `words[0..7]`，其中 `words[0..3]` 是主密钥切片，`words[4..7]` 各需4轮式派生；规模与主电路相比很小。
- 工程正确性关键：密钥调度与轮常量的接线必须与生产密码一致。因此除既有1..6轮公开cone全图等价外，另加**生产密码KAT级别对照**：固定公开主密钥与明文，令 rounds=1..4 与 rounds=8 的完整CNF被完全固定后求解，断言模型推出的目标bit等于 `cipher_experiments.ciphers.simon.encrypt` 的MSB bit0。调用次数有上限并披露；这是编码等价工程验证，不是科学标签实验。

职责：`validation/integral-keymiter/` 一次性临时只读分析工具（编码＋现成PySAT适配＋witness核验），不进入生产src、不改密码／模型／训练／CLI、不新增runner或registry。编码委派给 `implement-deep`（当前路由sol档），静态审查委派给 `code-review`，正式运行与验收由主模型执行。

## 冻结候选与预算

- 仅两项，顺序固定：①`{'rounds':8,'active':[0,2]}`；②`{'rounds':8,'active':[0,1,2,3,4,5,6]}`。两者都已在上一周期公开冻结，不按任何真实输出挑选。
- 每项单次启动：var cap 250000、clause cap 1000000、单文件64MiB、累计CPU与wall各60秒、solver conflict预算200000、`solve_limited(expect_interrupt=True)`＋wall剩余Timer、一次完整load／一次solve／一次delete。协作式限额，不是硬进程或内存限额。
- 正式重试0，正式solver尝试最多2次。第一项若UNKNOWN／ERROR／预算停止或整体异常，**不启动第二项**并关闭周期。
- 真实密码调用只允许KAT级工程对照（固定公开值，≤24次），不做cube面板、不生成标签、不做科学验证；若第一项出现UNSAT且需要实证交叉核对，必须另立合同，不在本周期自动进行。

## 验收与结论分档

- **SAT_VERIFIED**：模型必须覆盖1..nvars且无缺失／重复／冲突／0／非整数／越界；逐子句核全部CNF；再由独立32位word递推（用模型里的128位主密钥重算密钥调度）核全部copy target、与模型输出一致、XOR=1。保存主密钥witness与全部copy证据，但**该key不得用于挑选实验key、seed或训练数据**。
- **UNSAT_UNCERTIFIED**：仅当proof完整导出且保存成功；当前无独立proof checker，因此只写“求解器在该完整CNF上返回UNSAT且proof已保存”，**不宣称已认证的零和证明**，也不据此宣称攻破或新纪录。
- **UNKNOWN／ERROR／BUDGET_STOP**：保留原始answer、计数与部分产物，禁止晋级、禁止追加候选。
- 产物、收尾写入、proof临时文件超限、model非法等失败路径一律降级为非成功状态，不得以成功码收尾。
- 主模型独立核所有artifact字节SHA、完整map／model、必要时用第二套公开符号递推复算；代理自述的测试数／SHA不替代实测。

## 产物

`runs/integral-keymiter-<fresh-id>-r8-kN/`，独占fresh目录，保存spec原始字节、config、start（含源码／PySAT／Python身份）、circuit.cnf、mapping、witness或proof、progress、results、exit及全部字节SHA；缺exit视为unknown。
账本`validation/integral-keymiter/ledger.json`记录两次尝试上限、实际结果与停止原因，周期结束CLOSED。
