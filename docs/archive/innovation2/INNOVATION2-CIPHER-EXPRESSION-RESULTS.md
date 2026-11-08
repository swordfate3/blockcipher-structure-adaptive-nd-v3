# 六轮SIMON密文布尔表达式：五臂首轮结果

## 结论

已经按[固定方案](INNOVATION2-CIPHER-EXPRESSION-PLAN.md)在Windows A6000完成全部五项并取回完整产物。**本次没有观察到多bit表达式比单bit更易泛化**；三种组合表达式TEST准确率均约50%，所有模型TEST BCE均劣于其TRAIN常数概率基线。

这是单key、单初始化、固定四表达式的首轮探索；不是“所有密文表达式都不可预测”的证明，不扩大到其他表达式/模型/轮数。原任务仍为明文输入、单logit预测目标表达式值，不是平衡分类、积分cube或密钥恢复。

## 全部结果

TEST每项65536个未见明文；选择由各自32个DEV BCE决定，不按TEST重新选epoch。

| 标签 | 所选epoch | TEST正确数 | TEST ACC | TRAIN多数类在TEST ACC | TEST BCE | TRAIN常数概率在TEST BCE |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 1 | 33066 | 50.454712% | 49.519348% | 0.693299799 | 0.693178491 |
| C0 XOR C8 | 1 | 32671 | 49.851990% | 50.079346% | 0.693304567 | 0.693147330 |
| C0 XOR C8 XOR C16 | 1 | 32608 | 49.755859% | 50.056458% | 0.693424972 | 0.693147404 |
| C0 XOR (C8 AND C16) | 2 | 32764 | 49.993896% | 49.818420% | 0.693603059 | 0.693159587 |
| 独立随机bit null | 1 | 32451 | 49.516296% | 49.526978% | 0.693722696 | 0.693198833 |

单bit的50.455%比对应多数类准确率高，但BCE反而差；不把这一个有限面板、一次初始化的现象升格为可靠密码规律。没有按最高TEST追加模型/位组合或做跨key结论。三个组合标签也不支持相对单bit优势。

## 拟合与泛化

| 标签 | selected TRAIN BCE | selected DEV BCE | 第32轮在线TRAIN BCE | 第32轮DEV BCE |
|---|---:|---:|---:|---:|
| bit | 0.692943766 | 0.693249591 | 0.518707771 | 0.898817964 |
| xor2 | 0.692645994 | 0.693194611 | 0.519935614 | 0.888001660 |
| xor3 | 0.692872135 | 0.693322735 | 0.523855576 | 0.889797011 |
| xor-and | 0.692369182 | 0.693633610 | 0.526282297 | 0.880823018 |
| null | 0.692671638 | 0.692945860 | 0.521321853 | 0.900165156 |

五项都表现为继续降低在线训练损失、DEV恶化，null也一样，支持训练集拟合/过拟合解释。在线TRAIN BCE由一轮内不断变化的模型累计，**不是last.pt在固定完整TRAIN上的评估**；没有取回后加载last.pt再算。selected TRAIN属于早期DEV所选模型，不能与末轮混称。

## 训练身份与消耗

- 代码发布：`13e939bc8f305874a6f787012ded60a8fef3457f`，只提交批准的15文件，push后ls-remote一致。其他dirty work未提交。
- fresh远端：`G:\lxy\shared-m-cipher-nd-runs\cipher-expression-20261031-013340\repo`；exact Git检出、clean；46个tracked源码SHA及五份config文件SHA与本地发布树一致，每臂启动前再核。
- 物理GPU1 `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`映射逻辑cuda:0；每项启动前现场free>=24000MiB/util<=30%，无fallback；现有Python3.12.10/Torch2.14.0+cu130，无安装升级。
- 同MLP24961参数、相同初始参数SHA及TRAIN/DEV/TEST x SHA；seeds data195601/key190502/init17/null195602。数据与初始身份按远端保存SHA比对，没有本地再生数据。
- 每臂TRAIN65536/DEV8192/TEST65536，batch256、32epochs、8192更新、2097152呈现、32 DEV；五臂合计**40960更新、10485760呈现、160 DEV**，均exit0，正式retry0。
- 五项CLI保存elapsed合计295.313s（各约58–60s），不是含同步/检查/传输的全周期墙钟，也不是GPU独占计费量。单项1800s协作预算未超，不声称硬时限/断线存活。
- 由冻结代码和实际完成split规模可推出四真实臂合计557056次cipher调用；它们复用同一输入面板，独立明文仅139264。无单独逐调用计数器，不把这个推算称成独立测量。null139264随机标签、cipher0。
- 取回55文件、1146688字节至`runs/received/cipher-expression-20261031-013340-{bit,xor2,xor3,xor-and,null}/`。

## 发布与工程验收

提交前干净index副本：Ruff src/tests通过、strict mypy 100源码/测试文件通过；限定154 tests通过、1真实SIMON一致性测试未执行，2条既有Torch稀疏警告。主模型先前完成干净tracked源码副本五项null check/train闭环（10更新/40呈现、生产encrypt0）。没有声称全项目pytest通过。

独立代码审查发现synthetic/circuit标签元信息P3，委派修复并独立静态复核关闭；不影响本次SIMON标签数值。默认bit配置字典/SHA保持，checkpoint schema、DEV最小BCE选模、选中严格load后生成TEST时序不改。现有CLI自动输出TEST，不是访问强制封存，本轮固定矩阵探索。

## 原始错误与时间限制

首次启动在本地构造PowerShell时断言未替换`__OTHER_OUT__`，还未调用SSH，formal fits0。保留原始freeze/start及precli-failure-original；填空该旧模板门后生成formal-freeze-v2与start-v2，之后每臂仅一次实际CLI，无正式训练重试。不是隐去的第六fit。

独立保存记录审查发现：本地freeze-v2 UTC晚于远端首臂manifest UTC约8.25s。事后时钟观察显示远端UTC落后本地约19s（一次RTT0.715s），因此跨机时间不能直接排序；这个观测不证明训练期间偏差恒定，也不作为历史校时。启动源码顺序为先独占写freeze-v2，再逐项SSH命令，保存stage.utc是启动预留时间不是完成时间。保留两类事实，不凭原始时间戳宣称独立证明冻结先后。

## 结果验收范围

主模型核远端inventory与接收文件集合/每文件bytes/SHA；saved config SHA、manifest源码/commit clean、best checkpoint字节SHA、selection与32条DEV最低值、退出/更新/呈现、五项x和初始化身份一致。**没有checkpoint load、模型前向、标签再生、evaluate或本地指标重算**。

独立code-review只读指定JSON及五项results/progress/selection/manifest：未见数值/角色/选模矛盾，不支持“多bit更易学”；没有执行SHA或独立预测复验。null的experiment_kind仍沿用bit-baseline，其label_mode/source明确为随机，报告不算第五真实cipher臂。

账本与全精度结果：`validation/cipher-expression/{formal-freeze-v2.json,execution-completed.json,result-inventory.json,verification-closeout.json,clock-observation.json}`。CLOSED，5/5 fits耗尽，未追加表达式/seed/epoch。

## 下一步

现在已经回答了“简单换成这些密文表达式，准确率会不会改善”：本轮未见改善。若继续，先比较有代数消项依据的表达式设计与更适合布尔组合的模型，而不是随机增加更多bit组合或盲加epoch。新候选、训练、发布应另定有界合同，当前许可关闭。
