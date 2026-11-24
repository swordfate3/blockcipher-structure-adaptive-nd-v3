# 创新2 Cube：固定数据、最多50epoch与DEV早停

**状态：APPROVED。** 用户在“同seed重新初始化、复用原面板、50epoch/预热10/耐心8/min_delta1e-5、独立早停、3小时GPU1、一次发布运行/失败不重试、先验证退出捕获”的完整提案后回复“是”。本文件落实该简短设计，不增加科学候选或默许其它实验。

## 目标与选择

在已发布 `af784fcb1d8250add4f7013c09d6e3e84d85500f` 基础上，只扩训练上限/加入早停/只读复用数据/修复退出捕获；核心、模型、SIMON结构卡、编码、全128BCE训练及指定8位DEV/TEST指标均不变。

曾比较：①旧best继续训练（没有Adam状态，不是严格续训；best来自epoch2）；②同seed从头训练，复用同一数据，改变训练上限与停止规则。采用②。两臂MLP/Shared-M各一fit，初始化31，epoch顺序seed20261124不变，不加载旧权重或伪装从epoch6续训。

## 数据来源与边界

复用 `cube-sharedm-simon7-20261123-02` 原划分，源run.json SHA256=`bc2c87a5e2ab919021c7f4cdf556830f99b59380b858352ae72d30c93669f523`：

| 文件 | bytes | SHA256 |
|---|---:|---|
| dataset/train.npz |41943816|e01ca03e68a3feaaa98553cbb2d7662bdd943a31e304cc7abcec4ccaaaa1cf8a|
| dataset/dev.npz |2622216|116ebebf97f4a1df3685f53bc9aa02d98f8d8d58ab202d3e85477f0ba5b6f2a3|
| dataset/test.npz |2622216|f00dfe78821c28024b826eb8ad8444ebf1acd37e9525001061fe4b9c4f990ddc|

TRAIN1048576对/DEV65536对/TEST65536对；131072/8192/8192族，LSB0(0,21,42,60)，每族16变体/8互补对，7轮单key。只读校验来源/字节SHA/大小、NPZ仅3预定npy成员及uint64 C-order数组和形状/族标识、族完整性及划分互斥。不生成新标签、不访问秘密key/秘密熵、不运行密码。

使用原RunStore read_binary的逐文件size/SHA/路径安全契约。不能用read_saved扫描源run所有artifact来提前读TEST/模型。只把源run.json作为显式身份与artifact声明，不使用其旧TEST成绩参与训练。TRAIN/DEV预先读取并复用；两臂新best严格磁盘重载成功后才读取/解码源TEST文件。旧TEST已展示，本轮仍是适应性探索，不称独立确认或强密封。

新run可将读取的原NPZ字节写入自身dataset槽（原始bytes不再采样/序列化），登记原source run/SHA和各split来源；不改旧目录、不复制旧权重。模型/训练不根据文件路径猜数据身份。

## 训练与早停

- 各最多50epoch，batch256，FP32、Adam lr.001、betas(.9,.999)、eps1e-8、clip1、无权重衰减、固定lr；架构与原初始化/逐epoch数据顺序不变。
- 每epoch一次完整DEV指定8列BCE。两个串行fit独立早停，复用现core纯Python `EarlyStopping((arm,), Settings(10,8,1e-5), epoch_ceiling=50)`，不复制停止判定器。
- epoch1..10只更新参考最低；epoch11起，达到相对早停参考值至少1e-5改善才重置等待（微小改善可累计）。连续8轮未显著改善则停，最早epoch18。epoch50按上限结束。
- **模型选择和早停分离**：任何严格更低DEV均保存best，同分最早；min_delta不影响best选择。每臂保存epochs_completed/stop_reason/bestepoch。早停是正常完成；超时/设备/存储故障不是早停成功。
- 仍完整128bit BCE训练，仅8列DEV选模/TEST主ACC。两份新best均经size/SHA/严格模型加载后才读取TEST；两臂评分完整后统一发布，不输出单臂终分。基线仅TRAIN，族聚类区间沿用，不新增评估。

## 预算与权限

新数据加密0、秘密熵0、null0；本轮无需重复原18/2/32数值预检（已真实验证，且不应为复用数据重加密或免费微训练）。设备准入/已有运行时仍核验UUID/free/visible1/RSS。主控在训练前仅用空程序验证退出码0/7，禁止在开发代理或验收夹带真实NN/cipher。

最坏两臂各50轮：updates=409600，forward_pairs=111542272（50×TRAIN/DEV＋每臂最终一次TEST）。实际早停后按两臂**实际完整epoch数**核账：updates=(e_mlp+e_sharedm)×4096；forward_pairs=(e_mlp+e_sharedm)×1114112+131072。科学成功不得要求达到最大额度，不得把欠账当早停。失败保留已完成/尝试消耗。

总wall10800秒（3h），CPU43200秒（4线程），RSS8GiB、产物512MiB、单checkpoint16MiB沿用；最后模型重载、TEST和报告都在同一Meter/UTC期限内。最多2fit、一次运行、重试0，失败/身份错误/资源冲突立即停，不换卡/减batch/加seed/延长期限。使用物理GPU1UUID不变，不碰GPU0或其它服务。

本轮许可包含：本地实现/人工测试/独立审查、一次限定Git提交发布/远端exact checkout、无害退出码检查、一次GPU训练与取回保存结果。无安装/其它科学预检/新数据/自动重试/evaluate权限。科学程序不做SSH/Git。

## 模块归属与兼容

- `direct_block/fit.py`：加显式可选早停设置，旧默认关闭且原计量/选模语义不变；调用现core早停，不自建通用workflow。
- `cube_compare/reuse.py`（新）：严格只读源身份与compact NPZ加载，TEST懒读取，只此模块知道旧run引用；不加载模型、不执行密码。
- `config.py`：保留旧5epoch配方，新增具名50epoch-reuse配方及最大counts/来源SHA；固定配置描述不执行采样或文件操作。
- `training.py`：同一双臂装配接受明确面板provider/early-stop设置，实际epoch记账；不复制runner，不塞NPZ解析/停止数学。
- `cli.py`：按固定配方选择已有生成版或新只读版，准入/claim/错误/报告边界不降级；新版本明确不调用旧cipher或数值预检。不从主树import兄弟插件。
- `report.py`：保存值显示实际轮数/停止原因/来源/旧TEST探索限制，不重算指标。
- `scripts/start-cube-sharedm.ps1`：只增必要配置/来源参数和修退出码捕获，不改科学代码。先复现WindowsPS5.1 `$LASTEXITCODE`赋空/作用域问题，以可控空程序覆盖0/7，查清后最小修复，不能用stdout模拟原生码。

隔离源码从af784fcb提取，非主树覆盖；原main HEAD/index/其它改动和旧产物不动。新测试名字全局唯一；强制Ruff `--no-respect-gitignore`防止忽略workspace漏查。实现委派implement-deep，TDD、独立只读review、主控fresh及最小源码CLI闭环通过才发布。

## 交付与边界

保存完整实际训练/DEV曲线、best/epoch/早停原因、TEST和counts/nativeexit/来源SHA，取回后只核文件，不重新加载模型/加密/生成标签/forward/evaluate。失败不虚构结果；旧run的nativeexit UNKNOWN原样保留。本轮同TEST结果仍属探索，结论限于该key/seed/数据/50上限与本早停规则。
