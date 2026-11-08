# 三轮SIMON固定半明文：部分目标出现明显泛化

## 全部结果与结论

按[固定10臂计划](INNOVATION2-PLAINTEXT-SUBSPACE-PLAN.md)完成训练、取回并核验。A=high32-zero（高32位固定0/低32位随机），B=low32-zero（低32位固定0/高32位随机）。密钥固定未知，完整64列明文输入同一个MLP，单logit预测密文表达式。

| 目标 | A TEST ACC | B TEST ACC | 先前三轮全64随机 ACC |
|---|---:|---:|---:|
| C0 | **100.0000%** | **99.8169%** | 50.3891% |
| C0 XOR C8 | **99.6384%** | 50.1053% | 50.0763% |
| C0 XOR C8 XOR C16 | 50.0443% | 50.2548% | 50.0061% |
| C0 XOR(C8 AND C16) | **87.0758%** | **66.4337%** | 50.3433% |
| null | 49.5117% | 50.1465% | 49.5163% |

每项TEST65536条。A单bit65536/65536正确，A-xor2为65299/65536；B单bit65416/65536。**有限TEST全对不等于整个2^32明文域上的数学证明。** 当前发现是固定key/固定零背景的受限子空间内，部分目标的同分布未见样本预测明显改善，不是完整随机明文或跨key突破。

## 基线与BCE

| arm | selected epoch | TEST BCE | TRAIN常数TEST ACC | TRAIN常数TEST BCE |
|---|---:|---:|---:|---:|
| A-bit | 32 | 0.0000042 | 49.8550% | 0.6931577 |
| A-xor2 | 32 | 0.0110388 | 50.6317% | 0.6931284 |
| A-xor3 | 1 | 0.6931973 | 50.1709% | 0.6931460 |
| A-xor-and | 32 | 0.2764772 | 50.2457% | 0.6931457 |
| A-null | 1 | 0.6938315 | 49.5270% | 0.6931988 |
| B-bit | 31 | 0.0174869 | 50.0687% | 0.6931471 |
| B-xor2 | 4 | 0.6936643 | 50.1312% | 0.6931450 |
| B-xor3 | 1 | 0.6931193 | 49.8962% | 0.6931474 |
| B-xor-and | 16 | 0.6237609 | 50.0000% | 0.6931500 |
| B-null | 2 | 0.6932184 | 49.5270% | 0.6931988 |

常数参数仅从TRAIN估计，TEST准确率约50%，因此A/B明显高分不是单纯多数类偏斜。B-xor3的微小BCE改善不作为有效发现；其ACC仍约50%，无独立重复支持。null是相同标签流、不同空间输入，不是两次跨key独立确认。

bit公开参考＋TRAIN残差常数TEST：A62.2437%/BCE.6628612，B62.7441%/BCE.6603270；它们未输入full/direct模型。这里神经网络明显超过这些既有参考，非bit不使用错配的单bit参考。

## TRAIN/DEV与机制边界

selected TRAIN/DEV ACC：A-bit100/100%，A-xor2 99.9252/99.6338%，A-xor-and90.4541/86.4746%；B-bit99.9268/99.7925%，B-xor-and70.4453/66.2354%。所选epoch为各32次DEV BCE最小点，不按TEST选；A-bit/xor2/xor-and在预定最后epoch32所选，不因上升趋势追加epoch。

A首轮X1=Y0 XOR K0、Y1=0，随机变量非线性交互被延后，给出结构性简化依据。观察到A-xor2显著优于B、A-xor-and也更好，符合输入位置影响学习难度的预期，但**没有机制消融或独立确认，不能说已证实唯一原因**。B单bit同样接近100%，说明并非只有A能学。

xor-and的部分高分可能来自较容易的子表达式/条件相关性，聚合ACC不能证明完整布尔组合或各密文bit都被恢复。xor3两侧仍约50%，显示当前限制没有让所有表达式都变简单。

## 工程与发布

仅3源码（config、无标签input sampler、training接线/来源）、1测试、20配置、1计划，共25文件；cipher/model/label构造/checkpoint/evaluate逻辑未改。新plaintext_space枚举full/high32-zero/low32-zero，默认full字典省略字段，旧输入分支不变；非full限制SIMON/MLP/full/direct。相同去重uint32流分别落入活跃半，完整pool后切DEV/TEST，space内部split无重复，未先采64位再截断。

独立审查发现历史兼容测试使用移动HEAD会导致提交后失败（P2），发布前委派修为固定d6c3c5d三组完整字典/SHA fixture，不依赖Git。主验收236tests通过；限定Ruff/strict mypy4文件通过；干净index源码Ruff/mypy src/tests102文件通过，新测试不含训练74通过/5未执行；十个null CLI check/train共20updates/80presentations、cipher0。代理早期CPU/null测试5fits8updates40presentations，主验收重复同组8updates40presentations，累计该开发验收最多36updates160presentations，与正式分开。没有全仓真实cipher测试或全项目pytest宣称。

发布commit `255ed9757b05ae620a087c7d64a1a1150a0c71a2`，push/ls-remote一致，无关dirty未带入。fresh远端`G:\lxy\shared-m-cipher-nd-runs\plaintext-subspace-20261031-084619\repo`，46源码/10配置SHA一致、clean，逐臂再核源码/配置、指定物理GPU1 UUID/free>=24000MiB/util<=30%，cuda:0，无fallback、无安装。

## 执行消耗与保存记录核验

10fits均exit0，正式retry0；每项8192更新/2097152呈现/32DEV，合计81920更新/20971520呈现/320DEV。CLI elapsed合计601.125s，不是全周期墙钟或GPU独占计费；协作每臂1800s预算未超，非硬限制。由代码/split推算8真实臂1114112次cipher调用、2null278528随机bit；每空间139264不同明文，两空间可能共享0点，不声称并集恰278528或独立运行时逐调用计数。

data195601/key190502/init17/null195602，所有初始参数SHA相同24961参数；每空间五项x SHA相同，两空间x SHA不同。模型/标签指标来自远端保存记录，不在本地复算。完整取回110文件2329299字节至`runs/received/plaintext-subspace-20261031-084619-{high32-zero,low32-zero}-{bit,xor2,xor3,xor-and,null}/`。

主模型核文件集合/bytes/SHA、manifest源码/commit、config SHA与输入来源、best.pt字节SHA、DEV32点最低选模、退出/更新/呈现和x/init声明身份。未load模型、前向、再生标签、重加密或evaluate；配对/固定位性质依据冻结采样实现及工程测试，不冒称接收后重建原明文实测。

独立结果审查只读freeze/closeout/execution及十项results/manifest/selection/progress，未见可证实数字/选模/预算矛盾，支持限定子空间内部分目标泛化；无独立模型/标签复验。selection的TEST=false属于生成TEST前快照，不与最终TEST指标冲突。CLI自动输出TEST，无技术封存，本轮单key/init探索不是独立确认。

## 后续与关闭

CLOSED，10/10耗尽，不自动扩大轮数/背景/seed。下一步优先固定一个成功目标（例如A-xor2），预定一个不同非零固定背景复验“同一受限采样模式训练是否仍可学”；跨背景重训与直接迁移是不同问题，不能混称。新背景、r4或新key均需单列合同。该结果不证明高轮密码被攻破。

精确记录：`validation/plaintext-subspace/{formal-freeze.json,execution-completed.json,result-inventory.json,verification-closeout.json,ledger.json}`。
