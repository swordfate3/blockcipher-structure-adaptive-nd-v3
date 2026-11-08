# 积分连续周期结果：六／七轮小cube实测通过，八轮碰到证明搜索墙

## 结论

按用户“继续推进，等推不动再停”及[有界合同](INNOVATION2-INTEGRAL-CYCLE-PLAN.md)，连续完成六轮与七轮新小cube的首次真实验证，再进入八轮公开证书搜索。

**六轮8点、七轮4点关系均通过16组真实加密交叉检查；八轮在固定2000万pair尝试预算内未找到证书，按停止门槛关闭，九／十轮未启动。**

八轮不是实测失败、不是已证不平衡，也不是证明不存在积分。卡住的是本次ANF投影支持判定器＋启发式＋固定资源。没有为了花完预算继续加key、换cube、扩大时间或训练。

## 全部已启动阶段

| 阶段 | 活跃明文位，目标高字MSB bit0 | 每cube点数 | 真实cube XOR0 | 随机对照XOR0 | 实际真实调用 | 状态 |
|---|---|---:|---:|---:|---:|---|
| 六轮验证 | {0,4,10} | 8 | **16/16** | 10/16 | 128 | exit0 |
| 七轮验证 | {0,2} | 4 | **16/16** | 6/16 | 64 | exit0 |
| 八轮公开搜索 | 固定三起点，k≤10 | 无真实调用 | 不适用 | 不适用 | 0 | exit3预算停止 |
| 九／十轮 | 未启动 | — | — | — | 0 | 前轮门槛未通过 |

本周期真实attempted/completed **192/192**，null192bit；fit／更新／DEV／TEST选择均0；GPU／SSH／安装／发布均无；正式重试0。
真实额度上限49344，未用49152不构成新许可：遇八轮墙后周期已关闭。

## 六／七轮实测细节

固定同一标签侧key seed190502；新背景seed分别194606／194607，null seed194706／194707。未知key没有进入公开证明器或任何模型；公开seed可重建key，不声称对所有读者保密。
每stage先生成16个不同背景、清活跃位，完整枚举所有活跃组合，组内唯一且组间无重叠；每点完整块encrypt一次，只保留目标bit。

| 阶段 | real目标bit总ones | null总ones | real奇偶序列（组0..15） | null奇偶序列 |
|---|---:|---:|---|---|
| 六轮 | 62/128 | 62/128 | 0000000000000000 | 1000000101011010 |
| 七轮 | 40/64 | 28/64 | 0000000000000000 | 1010101011110101 |

“积分零和”是目标bit数量的偶性，不要求每组恰一半1。比如七轮各组4bit可为0／2／4个1。
每stage事先固定全16组XOR0事件，理想随机函数下概率2^-16；计划至多五stage的保守union bound为5/65536，不依赖stage间独立。实际启动两个真实stage，没有将后验最显著结果当事先唯一事件。
单stage q=128／64与随机64位排列的switching差异上界分别约4.4062e-16／1.09288e-16；null实际使用独立伪随机bit对照，不是另跑完整随机排列。固定PRNG结果不是仍然随机，概率针对理想模型。

六轮这是新3维cube实测，不混用旧7维cube16/16；七轮这是首次4点cube实测，不称旧符号证书自动等于真实成功。两者仍为一把固定key和16组背景的有限检查，全key充分结论来自符号归纳及电路正确性前提，不由有限实验枚举证明。

## 八轮：具体卡在哪里

run `runs/integral-cycle-20261029-173002-search-r8/`，2026-10-29 17:30:02 UTC启动；exit3／`BUDGET_STOP`，原因`global_pair_attempts`。

- 196个unique query：192完成并为UNKNOWN，4个RESOURCE_LIMIT，0个PROVED_ZERO，0 ERROR，0缓存命中。
- 12个一至六轮公开源图／支持检查全部通过，两份传播pair总872，包含在本stage总预算中。
- pair尝试**正好20000000**；最后4查询中前三个各触及单query1000000上限，末个在758715次时触及全局限额。
- exit计时CPU14.7143504秒／墙钟14.283150278秒，均未到60秒；停止的是运算次数而非时间限额。
- 第一条轨迹从上一已验证cube{0,2}开始，逐步加位至{0,1,2,3,4,5,6,7,8}，在完整比较下一步10维候选期间停止；没有完成该步，不选择部分最优。
- 第二、三起点均NOT_STARTED；不能写成三条轨迹都失败或穷尽全部k≤10候选。
- `stage_choice=null`、`observed_best=null`，没有八轮真实验证，更没有跳过八轮去试九／十轮。

**观察到排序退化：192个完整输出支持都等于全部2^k个mask，support fraction全部为1。** 它们不仅verdict全UNKNOWN，启发式分数也完全打平，于是按冻结规则退到字典序，从{0,2}向连续小编号加位。
这解释了本次轨迹为何缺乏有效引导；不证明所有cube都如此，也不证明这些函数的真实ANF含全部项。过近似可能把不同路径／系数／相消信息丢掉。

因此，下一个有辨别价值的改进应先考虑**更有信息的证书／评分，或先覆盖多个起点的公平预算分配**，而不是原样追加更多秒／更多pair。标准division property、精确系数或SAT等需要单独论证和完整规范核读，不能把本次简单支持法叫作已实现标准DP，也不能直接从这里保证更高轮成功。

## 代码、边界及工程

生产src/tests/config/CLI未修改；临时`validation/integral-cycle/{validate,search}.py`只实现已冻结的验证／公开搜索两个职责，互不导入，不造训练器／通用workflow。
真实oracle始终`/workspace/src/cipher_experiments/ciphers/simon.py`，SHA256 `994418a2f4d6736500d20bd79216548c333e953f5c60c1d611f42e9ae2052de5`；Python3.12.15，`/workspace/.venv/bin/python`，显式PYTHONPATH当前src。

正式前发现并修复：
1. validate证书身份字段原为可选：改为必须匹配rounds／active／verdict。
2. null阶段中断可能丢partial bits：增加open-null保存／计数fixture。
3. search缓存可能绕过全局时钟、最终收口可能超时晋级：缓存前check、结束前check、非COMPLETE禁止stage choice。

均在任何相关正式执行前修复，无科学重跑。旧freeze保留；代理自写freeze-v2不是验收，主模型实际独立核源并写freeze-v3-main；search另freeze-r8。
主模型独立实测validate **44项fake测试**、search **58项fake／公开小fixture测试**，各自Ruff／两文件严格mypy通过，`git diff --check`通过；独立静态review确认上述P2关闭。未跑完整项目真实cipher测试／干净源码CLI训练，不以审计文件测试通过宣称整个后端重新验收。

最终validate SHA：`478b911f2dbcd565402ce0a307a25e2ff520e9ed3d5e46d16c9586d5711ce6d9`。
最终search SHA：`1cae4a2978ac58ddf78183ec284c9196380bb8264cbb3ed271bebce94a874689`。
Git HEAD仍e765eff105a9b1c610014ba7b4ce7fa6838b3ea9，已知无关dirty保留，审计源码未发布，不称该HEAD包含本周期工具。

## 原件与验收

| 阶段 | 原件目录 | 文件／字节 | results SHA256 |
|---|---|---|---|
| 六轮验证 | `runs/integral-cycle-20261029-172532-validate-r6/` | 5／23285 | 701965fb359c59c9a72ee7366d8a2890343a379e371fa0e51b21fdcebd460ab8 |
| 七轮验证 | `runs/integral-cycle-20261029-172602-validate-r7/` | 5／22955 | 7d9f5b1c06516079782922a30250abdaabfeddb1615d5392f6ae40d6c9c79291 |
| 八轮搜索 | `runs/integral-cycle-20261029-173002-search-r8/` | 6／1037840 | b7d38858a0dda8dfedd7b5633de1764e301f1b114cde7f6264acf530731d8ded |

主模型只读核对保存bit的ones／XOR、明文去重／MSB顺序／points SHA、全部文件字节SHA、冻结spec／证书／源码身份、calls、source与query总pair、完整支持mask／SHA、截断记录不含伪证书、选择为空与退出／轨迹未访问状态。没有再次encrypt、再生真实标签或运行evaluate。
八轮192条mask全空间核验是保存记录一致性，不是另实现重新传播全部候选。独立code-review另读聚合记录／调用链，确认六七轮状态、八轮饱和排序／后两起点未启动及合同停止相符，未发现具体协议违规；其未执行程序、rawbit重算或字节SHA，不将只读意见替代主模型验收。原件保留不改；验收记录`validation/integral-cycle/verification-closeout.json`，阶段账本`ledger.json`均CLOSED。

## 最终判断

积分方向已有**六轮8查询、七轮4查询**的符号证书与本周期真实交叉验证，比之前逐点预测近50%更有明确的结构依据。
这仍不是神经创新、随机明文零查询泛化、新文献纪录、全44轮破译或密钥恢复。
按照“推不动再停”，本轮在八轮当前方法／固定预算处停，不无限加预算，不把资源墙包装成安全证明。下一周期须先讨论如何避免当前评分饱和，再决定是否实现更强方法。
