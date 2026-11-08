# 三轮两位XOR：非零背景仍有信号，但高分未保持

按[固定两臂合同](INNOVATION2-NONZERO-BACKGROUND-PLAN.md)完成非零背景重训并取回。高32位固定0xA5A5A5A5、低32位随机，同固定未知key，同MLP/full/direct、种子和预算。不是对零背景模型直接迁移测试。

## 全部结果

| 项目 | 两位XOR C0 XOR C8 | null |
|---|---:|---:|
| TEST ACC（65536行） | **58.836365%**（38559正确） | 49.551392% |
| TEST BCE | **0.653554310** | 0.693451450 |
| TRAIN常数在TEST ACC | 50.234985% | 49.526978% |
| TRAIN常数在TEST BCE | 0.693150074 | 0.693198833 |
| 所选epoch | 32 | 1 |
| selected TRAIN ACC/BCE | 67.752075% / 0.587183975 | 50.834656% / 0.692983644 |
| selected DEV ACC/BCE | 58.239746% / 0.658286864 | 50.659180% / 0.693220383 |

xor2较TRAIN-only常数TEST ACC高8.6014个百分点，BCE改善；null没有相应优势。说明本次非零背景内仍存在部分可学的预测信号，不是全局标签偏斜造成的虚高。但零背景99.638367%/BCE.011038780没有保持，准确率下降40.8020个百分点。单key/init/单非零背景，不能推广任意背景，更不是跨key/完整随机明文成功。

null虽然选epoch1，也完整训练32轮，不能只算一轮成本。xor2选预定最后epoch32，未追加训练。末在线TRAIN BCE .607010596不是selected模型完整TRAIN评估BCE .587183975，口径不同。

## 机制与下一步限制

高半固定任意B时，首轮F(B)都是常量；不能把性能下降说成“非零恢复首轮随机非线性”。背景可改变后续目标函数，同时full输入中的固定1通道也改变首层有效偏置/梯度/优化轨迹；本次没消除这种混杂。结果支持对固定背景的敏感性，不证明背景导致代数次数增加或唯一机制。

下一步若再推进，优先比较保持背景目标不变、仅把模型输入中固定半归零的无秘密表示控制（完整实际明文仍留标签侧），以区分输入表示/优化影响与密码目标差异。这个控制与新轮数/新key/更长训练均未自动授权。

## 实现与验证

发布commit4596264d409fd0b98676496aeff8c8f1c75d5bf7，仅本任务9文件：3源码+1测试+4配置+计划。新plaintext_fixed_value严格uint32/bool拒绝/full非零拒绝，默认0省略配置字段、旧身份保持；空间枚举历史名称high32-zero仍兼容，元数据明确实际非零常数。数据先去重uint32后拼固定半，活跃序列/分割不变。训练只接线/元数据，未改cipher/model/labels/checkpoint/evaluate算法。

主模型核实际diff；与旧对应formal raw配置逐字段仅name、plaintext_fixed_value不同。代码版本与零背景不同，身份透明，不能称完全同一代码；旧路径兼容由限定工程测试支持。独立静态实现审查未见可证实问题；主验收161 tests通过/9排除（避开已有CPU fit/fake labels）、Ruff/strict mypy4文件通过；干净index源码Ruff与mypy src/tests103文件通过、91新测试通过；两项CPU/null check/train共4updates16presentations、真实encrypt0。没有全项目pytest或实际工程真实cipher测试。

fresh远端G:\lxy\shared-m-cipher-nd-runs\nonzero-background-20261031-094902\repo，Git exact、clean、46源码/两配置SHA核验，逐臂再核；指定GPU1 UUID/live free>=24000MiB/util<=30%→cuda:0，无fallback/安装。

## 消耗、验收、关闭

2fits均exit0、正式retry0；每8192更新/2097152呈现/32 DEV，总16384更新/4194304呈现/64DEV；elapsed合计126.718s（不是含传输的全周期或GPU独占计费）。推算真实139264次cipher调用/null139264随机bit，各split65536/8192/65536，共享139264明文，无逐调用追踪计数器。每项1800s协作预算未超，非硬时限/断线存活保证。

取回22文件467103字节至runs/received/nonzero-background-20261031-094902-{xor2,null}/，主模型核集合/bytes/SHA、manifest源码/commit、config SHA/固定背景字段、best字节SHA、32 DEV最小选模、退出/计数、相同x/init身份；未加载权重、生成标签、模型前向或evaluate。输入固定半/配对性质依据冻结代码及工程测试，不称接收后独立重建样本复验。

独立结果静态审查核心指标/选模/预算无矛盾；未核字节或预测。执行stage.utc为调用前预留时间，不能当作退出确认时间；跨机时钟不强行排序。selection TEST=false是选模时点快照；CLI仍自动输出TEST，不是sealed独立确认。此前成功目标据已见TEST挑选，本轮为固定背景敏感性探索。

CLOSED，2/2正式fits已耗尽，不追加背景/表达式/epoch。原始记录validation/nonzero-background/{formal-freeze.json,execution-completed.json,result-inventory.json,verification-closeout.json,ledger.json}。
