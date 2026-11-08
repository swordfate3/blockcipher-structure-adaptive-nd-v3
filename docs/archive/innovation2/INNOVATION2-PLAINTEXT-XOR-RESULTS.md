# 两轮公开明文XOR路径：四项CPU结果

日期：2026-10-22。批次：`innovation2-plaintext-xor-20261022-104258`。
状态：四项完成，exit0；**预定DEV推进门槛未通过，批次关闭，无追加训练**。
执行前合同见[冻结计划](INNOVATION2-PLAINTEXT-XOR-PLAN.md)。本次用户明确要求试推荐方案，
已解除该四项CPU对照范围的暂停；不是文献报告后擅自恢复其他实验。

## 1. 改动及不变项

新增`prediction_mode=plaintext-xor`，仅允许simon-unrolled/simon/full、两轮、高半目标位。
计算为`absolute_logit=(1-2*plaintext_target_bit)*原轮展开selected_logit`。
没有改真实标签或训练损失；模型依然仅用最终绝对密文bit的BCE学习。
原轮内输入、共享cell、学习的条件与连续隐状态不变，没有删除内部low输入。
模式不增加参数或state tensor，原direct默认计算不受符号变换影响；严格checkpoint/raw配置SHA规则保留。
模型不读真实key、轮密钥、轨迹或其他密文标签，无旧权重初始化、无真实状态教师。

## 2. 数据、预算与证据

- 标准SIMON64/128减到2轮，MSB bit0，固定key_seed190502；不是全轮结果。
- full输入、width32、577参数；init17；四臂实际初始化state/trainable SHA完全一致。
- TRAIN4096/DEV1024/TEST8192唯一明文；新data192801、null192802。
  四臂各split的明文SHA一致；两种模式真实标签一致、null标签一致，真实/null标签不同。
- CPU，batch256，50epochs，每项800更新/204800次呈现；Adam.001/clip1。
- 每项120秒协作上限（非硬截止），四项实际训练计时和12.154542秒；
  合计3200更新、819200次呈现；不把单测/CLI工程控制算作额外科研seed。
- 每臂由DEV BCE最低epoch选best，strict加载best后才生成TEST标签。
  当前CLI自动保存最终TEST；本批先读取保存DEV并落盘`dev-decision.json`，再展开TEST报告。
  这不是访问强制封存或独立确认实验。

## 3. 完整结果（best checkpoint，绝对密文bit）

| 项目 | best epoch | TRAIN ACC | DEV ACC | TEST正确数/8192 | TEST ACC | TEST BCE | 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct真实 |9|49.8535%|50.1953%|4049|49.4263%|0.69325118|3.071813|
| direct-null |47|52.0264%|50.8789%|4030|49.1943%|0.69453403|3.017066|
| plaintext-xor真实 |50|55.5664%|53.4180%|4414|53.8818%|0.68130262|2.928706|
| plaintext-xor-null |32|51.3672%|51.1719%|4067|49.6460%|0.69426923|3.136957|

真实标签两模式共享的TRAIN-only基线：

| 基线 | DEV ACC | DEV BCE | TEST ACC | TEST BCE |
|---|---:|---:|---:|---:|
| TRAIN常量 |50.3906%|0.69311678|49.9390%|0.69319175|
| 零主密钥公开参考＋TRAIN残差常量 |50.8789%|0.69299567|50.0000%|0.69326170|

基线未改成新g(P)的拟合常量，保持原方案所述既有基线；不使用DEV/TEST拟合基线。

## 4. 预定DEV门槛：未通过

| 条件 | 实际 | 是否通过 |
|---|---|---|
| xor-real DEV ACC≥70% |53.4180%|否|
| 比direct-real DEV至少高10个百分点 |+3.2227个百分点|否|
| xor-real DEV BCE优于TRAIN常量 |.68417789 < .69311678|是|
| xor-real DEV BCE优于公开参考基线 |.68417789 < .69299567|是|
| 两个null DEV都在45%–55% |50.8789%、51.1719%|是|
| xor-real比xor-null DEV高至少10个百分点 |+2.2461个百分点|否|

门槛不是普遍统计成功标准，只决定是否建议推进独立确认。
本批未因TEST看起来提高而放宽70%或10个百分点要求，也没有把best落在末epoch作为追加预算的理由。

## 5. 能说什么、不能说什么

- 在本次同面板、同初始化、同预算的一对比较中，公开XOR输出路径比direct：
  DEV ACC高3.2227个百分点，TEST ACC高4.4556个百分点，且BCE降低。
  两种null最终TEST均约50%。这构成**单key/单seed下的小幅探索信号**，不是“完全没效果”。
- 但TRAIN也仅55.57%、DEV53.42%，远未把两轮函数学好；不是成功仿真、准确预测或轮数突破。
- 不能由此宣称跨key/init稳定、结构唯一原因已定位、局部相关理论已验证或标准密码安全下降。
  训练面板与shuffle同由data_seed控制，此处它们在四臂间匹配，不是单独研究各自效应。
- 该干预不增加秘密信息，只是改变可逆的输出参数化。成功或失败都不能证明神经隐状态等于真实轮状态。
- 根据冻结门槛，本批到此停止：没有三轮、额外seed/epoch、独立cell、BiLSTM、多bit监督或GPU批次。
  后续若要研究这一弱信号，需另行决定是做稳健性确认，还是先改变表示/优化假设，不自动执行。

## 6. 路径、checkpoint身份与验收范围

所有输出位于：

```text
runs/innovation2-plaintext-xor-20261022-104258-r2-direct-real/
runs/innovation2-plaintext-xor-20261022-104258-r2-direct-null/
runs/innovation2-plaintext-xor-20261022-104258-r2-xor-real/
runs/innovation2-plaintext-xor-20261022-104258-r2-xor-null/
```

best checkpoint SHA256：

```text
direct-real 1914463eebfff668f529099dc4ec1dbce0702168c274b7589874e02d0e7cc4eb
direct-null 586185eab9b0945c4e0ab2fd616c5f6ffdabdf6634355932524cbebed8a21e7e
xor-real    1470b5a65c822bc6c47fa09da37b6d411376f482d34c9275bbb87c209acf7ed2
xor-null    fadf00e15fd017f997d790544d59ebf57c6c16c1417ee366de52e70065add365
```

`validation/plaintext-xor/`保存冻结计划/配置/39份科学源码SHA、各CLI日志、
DEV决策和`verification.json`。核对48份运行文件字节SHA、原始配置/实际配置身份、
同面板/同初态、每epoch进度、best对应DEV最小BCE、选择时序标记、800更新与exit0。
**验收未加载正式checkpoint、未再生科学标签、未执行科学前向重算**。

实际工程检查：

```text
PYTHONPATH=src .venv/bin/python -m ruff check .   All checks passed
PYTHONPATH=src .venv/bin/python -m mypy         63 files, no issues
PYTHONPATH=src .venv/bin/python -m pytest       757 passed
```

干净源码副本：`validation/plaintext-xor-clean-20261022-104014/`，
仅plaintext-xor/null配置8/4/4、2更新，统一CLI check/train/evaluate均通过，
显式evaluate与其保存的TEST指标一致。不是科学第五臂或旧结果重放。
独立只读代码审查未发现可证实P1/P2；有限快照审查不是系统无缺陷证明。
已有环境复用，未安装、GPU、SSH、commit或push；main原有其他未提交工作保持。
