# 三／四轮密文表达式：降轮未观察到明确泛化改善

## 结论与全部准确率

按[固定合同](INNOVATION2-CIPHER-EXPRESSION-LOW-ROUND-PLAN.md)完成r3/r4各五项。相同MLP、数据面板、初始化、每项8192更新；网络只看明文。所测多bit布尔表达式仍是单个二分类标签，不是预测表达式是否平衡。

| 标签 | r3 TEST ACC | r4 TEST ACC | 已完成r6 TEST ACC |
|---|---:|---:|---:|
| C0 | 50.3891% | 49.7894% | 50.4547% |
| C0 XOR C8 | 50.0763% | 49.8993% | 49.8520% |
| C0 XOR C8 XOR C16 | 50.0061% | 50.0534% | 49.7559% |
| C0 XOR(C8 AND C16) | 50.3433% | 50.0702% | 49.9939% |
| null | 49.5163% | 49.5163% | 49.5163% |

部分数值比r6上升，但均接近50%，**降至r3/r4后没有观察到明确泛化改善**，不应说所有指标完全没有上升。全部10项TEST BCE均劣于各自TRAIN常数概率基线。所测组合没有有说服力的优势；本次单bit直接MLP也没有显示可靠泛化，不意味着r3目标无可利用信号。

## BCE与选模

| arm | 所选epoch | TEST BCE | TRAIN常数在TEST BCE | 末在线TRAIN BCE | 末DEV BCE |
|---|---:|---:|---:|---:|---:|
| r3-bit | 3 | 0.6935501 | 0.6931453 | 0.5272 | 0.8623 |
| r3-xor2 | 1 | 0.6933071 | 0.6931474 | 0.5154 | 0.8997 |
| r3-xor3 | 1 | 0.6932338 | 0.6931500 | 0.5239 | 0.8757 |
| r3-xor-and | 2 | 0.6932114 | 0.6931347 | 0.5191 | 0.8916 |
| r3-null | 1 | 0.6937227 | 0.6931988 | 0.5213 | 0.9002 |
| r4-bit | 1 | 0.6932814 | 0.6932110 | 0.5230 | 0.8843 |
| r4-xor2 | 1 | 0.6933288 | 0.6931700 | 0.5242 | 0.9046 |
| r4-xor3 | 2 | 0.6936749 | 0.6931416 | 0.5222 | 0.8994 |
| r4-xor-and | 3 | 0.6935653 | 0.6931543 | 0.5198 | 0.8916 |
| r4-null | 1 | 0.6937227 | 0.6931988 | 0.5213 | 0.9002 |

表中近似值仅展示，精确保存数在verification-closeout.json。最小DEV BCE选择均与保存32点一致。在线TRAIN损失下降而DEV恶化，null也一样，支持过拟合；不将在线值误称last.pt固定完整TRAIN评估。

### 不应遗漏的公开结构基线

r3-bit保存的`public_reference_constant`在TEST为**62.6099%、BCE0.660999501**，明显好于本次直接MLP；r4为50.0916%、BCE0.693153495。它用公开明文结构参考和TRAIN残差常数，不读取真实key/状态，也没有作为full/direct模型输入。因此不能从MLP约50%说r3无结构信号，只能说该直接MLP在本设置下没有学到可靠泛化。

非bit表达式没有该单target公开参考，避免把错配参考强加到表达式。跨轮null保存的模型指标/训练轨迹与r6相同，是事前约定重复对照，不累计为三份独立确认；耗时、checkpoint字节身份、低轮公开reference附加指标可不同。

## 发布、执行、消耗

- 只新增20配置、1测试、1计划，发布commit `d6c3c5d3b89afe5dd494a948f6b313b43c1f6d39`；不改生产src，保留无关dirty工作。配置对r6只变rounds/name。
- 独立静态审查配置、测试、合同未发现可证实问题；主模型实测157tests通过（新54+既有表达式103，fake/null），Ruff/strict mypy限定通过；干净index副本Ruff与strict mypy src/tests共101文件通过，10个null CLI check/train每项2更新/8呈现，共20更新/80呈现，真实cipher0。
- fresh远端`G:\lxy\shared-m-cipher-nd-runs\cipher-expression-low-20261031-022738\repo`，exact Git SHA、clean、46源码及10配置文件SHA核对。每臂现场核指定GPU1 UUID与空闲>=24000MiB/util<=30%，无fallback，现有环境无安装。
- 10fits均exit0；每项8192更新/2097152呈现/32 DEV；总81920更新/20971520呈现/320 DEV，正式重试0，无启动失败。CLI保存elapsed合计630.687s，不是含传输等的全周期时间或独占GPU计费。
- 由冻结代码和split规模推算8真实臂1114112次cipher调用；共用139264个明文，非1114112个独立明文。2null278528随机标签，无额外密码调用，无独立逐调用计数器。
- 同key190502/init17/data195601/null195602，与r6面板及初始状态SHA一致；已见TEST再次用于固定探索矩阵，不是新独立确认。
- 取回110文件、2315277字节到`runs/received/cipher-expression-low-20261031-022738-r{3,4}-{bit,xor2,xor3,xor-and,null}/`。

## 验收与边界

主模型逐文件对远端inventory核集合/bytes/SHA，核保存manifest源码与commit clean、config SHA、best字节SHA、32条DEV最小选模、退出/更新/呈现及十项x/init与r6身份相同；只读记录，不本地load模型/前向/再生标签/调用evaluate。独立结果审查只读results/progress/selection和freeze/closeout，未发现数值/选模矛盾，未执行字节SHA或独立预测。

当前CLI在selected strict load后生成TEST但自动保存/输出，不是技术封存；单key/init、没有配对显著性或多种子，不宣称统计证明所有表达式等难或普遍无优势。历史r3曾有不同模型/训练配置的更好成绩，不能直接覆盖或当作本轮相同设置已复现。

本轮CLOSED，10/10 fits用尽，没有自动追加r1/r2/r5、更长训练或新组合。若继续，应先解释MLP在r3未学到公开结构基线已有信号，而非靠再降轮追分或随机堆表达式。

精确全量记录：`validation/cipher-expression-low/{formal-freeze.json,execution-completed.json,result-inventory.json,verification-closeout.json,ledger.json}`。
