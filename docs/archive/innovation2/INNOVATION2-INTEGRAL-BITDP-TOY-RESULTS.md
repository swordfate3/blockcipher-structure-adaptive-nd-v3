# 显式COPY的全局division抽象：人工原型验收

状态：本轮人工工程范围COMPLETE；未启动SIMON正式分析。用户在文献/方案桥接阶段再次要求“继续”，按已说明边界委派implement-deep实现，只写validation/integral-bitdp-prototype三文件，无生产算法/数据/训练/CLI改动。

## 得到的新能力

原支持法逐节点合并投影mask，丢失共享子式约束。新人工原型把每个使用端口显式编译为COPY，并传播同一切面的全局指数向量前沿，不把同一来源的两条边当作独立来源。

`g=x XOR y; f=g AND g`是必须保留的未化简例子：

- x,y都活跃：全局前沿在XOR处排除双要求，输出判ZERO，与真实四点奇偶0一致。
- 仅x活跃：COPY保留单要求，经AND的10/01→1可达，与真实两点奇偶1一致；若错误混用“AND必须11”就会造假零。
- `x XOR x`仍允许抽象可达假阳性；本方法不追踪所有GF(2)抵消，不是精确函数判定器。

原型不依赖生产cipher、keymiter、runs、PySAT、模型或训练，只用stdlib及人工测试。接口严格限制n<=4、固定但未知参数q<=2、最多32节点、1..2输出；**不能直接输入64位SIMON图**。这是独立论证的division抽象toy prototype，**不是已核标准DP规范实现、十一轮再现或新密码结果**。

## 核心语义

输入多重集Z的矩`M_Z(u)=XOR_z product z_i^u_i`。前沿K的向上闭包覆盖所有可能非零矩。门表是最小前沿转移，不是数据值真值表或精确系数表：COPY0→00/1→10或01；XOR00→0、10/01→1、11无；AND00→0、10/01/11→1。DROP仅指数0可通过，其余线保持，支配删除保持向上闭包。

query用`exists k<=target`，不可替换成相等判断。REACHABLE不是实际反例；ZERO只来自完整抽象在给定输入假设下无覆盖，不是SAT-solver UNSAT认证。候选/转移截断返回UNKNOWN_RESOURCE，不附可用于零证书的最终前沿。

## 主模型实际验收

读取全部实现/测试/READY，核调用链和定义；执行一次无缓存/无外部插件的限定pytest，实际输出：

```
8 passed, 26 subtests passed in 0.16s
wall_seconds=0.395081179
child_cpu_seconds=0.394301
```

Ruff两文件、strict mypy两文件、git diff --check通过。命令为：

```
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts= -q -s -p no:cacheprovider --confcutdir=validation/integral-bitdp-prototype validation/integral-bitdp-prototype/test_bitdp.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --no-cache validation/integral-bitdp-prototype/bitdp.py validation/integral-bitdp-prototype/test_bitdp.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m mypy --strict --cache-dir=/dev/null validation/integral-bitdp-prototype/bitdp.py validation/integral-bitdp-prototype/test_bitdp.py
```

一轮主模型实测枚举：111次原图/编译求值对比；430个cube多重集/1576输出矩（488非零）；614个输入域子集/3442切面/27382切面矩（13634非零）；85个有序多重集/1020切面矩；15次平方自由多项式回代、16门表行、34闭包查询、2122支配约简查询。真实非零矩均被抽象覆盖。保留重复样本，不能set去重。

代理另记3次pytest累计wall1.196385s/CPU1.098240s（初次资源fixture预期错误，已修正且未删失败）；这些是代理记录，不替代上述主模型独立实测。加主模型本次后可见4次合成pytest累计wall约1.591466s/CPU1.492541s，低于预定90s合成额度；不把lint/typecheck耗时混入pytest时间。

## 独立静态审查

code-review只读三文件快照，不运行程序，无Git基线。未发现可证实P0–P3问题。核显式使用端口、重复入边/多输出共享、线消费与DROP、门表、全局支配约简、输出重排、资源未知、独立Boolean oracle/多项式回代和原图到编译图对照。

覆盖是14个固定人工fixture，不是全部合法图；没有密码级性能或标准DP原文一致性背书。内存异常等未统一转为UNKNOWN，但没有被转成ZERO；资源限额非硬隔离。

## 文献检索实际阻断

重新派发deep-research后6次请求耗尽，仅读到ePrint2015/090官方摘要和元数据。该次报告PDF请求的实际URL为`https://eprint.iacr.org/2015/090/090.pdf`，返回403；这是非规范路径，不能把它等同于本轮已核规范`/2015/090.pdf`。此前周期规范PDF也记录403，但两次事实不混写。搜索Exa429、DDG失败、S2 429、OpenAlex限流，没有取得正文。

摘要只支持SIMON64十一轮积分的报告轮数。输入集合/维度、输出位、数据复杂度、key量词、位序和轮边界继续未知，不能转换本地MSB(X,Y)公开规格。代理关于CRYPTO2016的未核线索不作为出版事实。

原两项异步任务在中间续轮后无法MessageAgent定位，且未收到交付；后续以重新同步派发实际返回为依据，不能把旧任务当已完成。新文献请求6次、无真实密码/solver。

## 范围与下一项

生产encrypt0、科学标签0、solver0、训练/GPU/SSH/安装/commit/push0。原型文件/测试/READY本地ignored，身份在validation/integral-bitdp-prototype/main-acceptance.json；不修改旧扫描冻结源码或重跑旧候选。

现在已有可测共享约束原型，但**还不能直接开展高轮密码声明**。下一项应先选：在原文可访问时对齐标准规范；或独立评审如何从有限toy到公开SIMON图（切面宽度/前沿爆炸、符号常量输入、无路径证据检查）的最小边界调整，再冻结人工压力测试和正式候选预算。不要简单放开n<=4接口并把无路径结果称作已认证密码证明。
