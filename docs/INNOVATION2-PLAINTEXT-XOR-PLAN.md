# 两轮公开明文XOR输出路径：执行前冻结计划

用户在推荐的四项CPU对照后明确要求“试试推荐”“试试推进”。本次限于该有界方案，
不扩到GPU/SSH、多bit标签、真实轨迹、预训练或其他分支；不提交/推送。已有脏工作保留。

## 单一干预与信息合同

标准SIMON64/128减到2轮，固定未知key，MSB bit0（高半计算位）。
公开方程X2=X0 XOR F(X1) XOR K1，可提出g(P)=P[:,0]：
D0=C0 XOR g(P)，C0=D0 XOR g(P)。
新增`prediction_mode=plaintext-xor`返回`(1-2*g(P))*z(P)`，z仍来自完整原轮展开模型。
只改变输出参数化，**不去掉原模型内部low输入、不冻结cell、不改初始化**；
这不是无复制输入模型消融。绝对密文bit标签、BCE、Adam与DEV选模均保持不变。
不调用真实key/轮状态/额外密文位构造模型或残差，不生成额外训练标签。
不将连续隐状态解释为真实轮状态，不声称可验证密钥恢复或文献复现。

模式仅允许simon-unrolled、dataset=simon、full、r2、target_bit0..31。
默认direct保持旧行为；checkpoint字段/state tensor不新增，原raw配置SHA规则保持。
所有报告仍为原始绝对密文bit ACC/BCE，绝非残差准确率偷换。

## 四项、面板与预算

配置：`configs/innovation2-plaintext-xor/`。
顺序：`r2-direct-real`、`r2-direct-null`、`r2-xor-real`、`r2-xor-null`。
先完成全部工程检查、源码/配置冻结及check，再一次性运行四项；
不根据中途TEST决定改配置或是否追加槽。技术失败立即停止余项，不自动重试。

共同固定：CPU；simon-unrolled、width32、577可训练参数；full；
TRAIN4096/DEV1024/TEST8192；data_seed192801（新面板）、key_seed190502、init_seed17、null_seed192802；
batch256、50epochs、Adam lr.001、clip1；每项800更新、204800次呈现、120秒协作上限。
四项总3200更新、819200次呈现、480秒协作预算；不是硬进程截止。
所有臂使用相同面板、相同基础参数初态和shuffle序列，真实/null标签分别相同；
模型初态张量SHA必须四臂一致。与旧批次面板不同，不用旧TEST选配置或基线。
没有多key/init稳健性结论。

`configs/innovation2-plaintext-xor-smoke.json`为独立null 8/4/4、batch4、epoch1、2更新，
用于干净源码副本CLI check/train/evaluate工程闭环；不是第五项科学拟合。
单元测试中的小null/公开人工规则也只属于工程回归，不算新增科研seed。

## DEV判别与停止规则（查看本批指标前固定）

每臂仍由最低DEV BCE选best，平局保留较早epoch，strict SHA绑定加载后才请求TEST标签。
当前CLI自动落盘最终TEST，不能声称其访问强制封存；**本批决策首先只读取四臂保存的DEV字段**，
写入`dev-decision.json`后才展开TEST字段进行最终报告，不以TEST改实验或追加。

候选值得独立确认的工程推进门槛同时要求：

1. xor-real DEV ACC≥70%，且比direct-real至少高10个百分点；
2. xor-real DEV BCE同时优于该臂TRAIN-only常量与既有零主密钥公开参考常量基线；
3. 两个null臂DEV ACC均在[45%,55%]，并且xor-real DEV ACC比xor-null高至少10个百分点。

门槛只决定是否建议另起独立确认，不在本批自动执行。不是统计显著性或密码安全阈值。
若null超出范围则先标记异常并查工程证据，不先宣称有效，也不自动再跑。
若xor有提升但未达门槛则只报告有界探索信号；若仅TRAIN改善则不称泛化成功；
若均近随机则记录阴性、到此停止，不扩epoch/seed/轮数。
TEST ACC/BCE用于最终一次性描述，报告同面板基线和真实/null，单key/seed不作普遍结论。

## 工程验收和科学证据

- 单测验证所有高半bit的符号变换/BCE等价、0与1输入、577参数及初态匹配、边界拒绝。
- strict checkpoint/load/evaluate roundtrip；旧direct兼容；null禁止加密；TEST标签时序。
- Ruff/mypy/pytest；干净源码副本仅null CLI闭环；复用已有环境，不安装。
- 正式运行后只读保存JSON/进度/配置/源码SHA/文件字节SHA，核800更新/50epoch/exit0。
- 不加载正式checkpoint、不重生科学标签、不做本地前向重算；显式evaluate仅工程smoke。
- 冻结计划与配置/39份科学源码SHA写`validation/plaintext-xor/frozen-plan.json`。
  正式输出用全新`runs/innovation2-plaintext-xor-<timestamp>-<arm>/`，不覆盖旧结果。
