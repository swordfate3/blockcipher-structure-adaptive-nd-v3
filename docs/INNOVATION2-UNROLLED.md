# 无真实轨迹监督的按轮展开神经预测

## 分支与任务

用户要求保存现有分支后回main推进。原`review-fixes-20261020`工作树干净，HEAD为
`5241c62d4adcca7542a191734e40913003ad58f9`，已额外保留本地引用
`saved/review-fixes-before-unrolled-20261021-173603`；未改动该分支、未合并。
本实现基于`main`的`111adab6fba0df0369e65c9a5a7312e1d8481fa6`。
其他分支中的STE/SAT/积分及结果没有迁入main，也没有丢弃；本实验不称其重跑。

任务保持：固定未知key的SIMON64/128，输入一个64位明文，输出预定MSB bit0的绝对logit。
**仅TRAIN最终单bit标签参与优化**；DEV同一目标选模；TEST严格加载选中checkpoint之后生成。
模型不读真实key、轮密钥、真实中间状态、其他密文位标签或历史checkpoint。
从样本学习到类密钥效应允许；不声称参数不可能编码密钥信息。

## 方法（新候选，不是成功结论）

`model_kind=simon-unrolled`，公开先验为Feistel复制路径与旋转1/8/2。
每轮32个位置使用同一个MLP `8→hidden_width→1`（Tanh）：

- 4个局部数值：`y[b], x[b+1], x[b+8], x[b+2]`（循环MSB索引），中心化到[-1,1]；
- 4个从随机初态学习的轮/位置条件，不填入实际轮密钥；
- 更新半块由MLP logit的sigmoid形成，下半块复制原上半块；
- 最终目标直接取对应更新logit；没有按阈值硬化隐藏状态、没有STE。

所有轮端到端反向传播，仅有最终目标位BCE。中间数值是连续隐表示，
**不是受监督或已验证正确的物理密码状态**；最终预测正确不证明轮间状态正确。
这不是“每轮得到真实答案后分别训练”，也不是复制完整布尔电路只搜未知常量。
共享cell与每轮位置条件提供更宽的可微参数化，但不保证解决组合学习困难。
无key schedule约束。不同轮数可有不同条件张量，单次训练不能冒称跨轮泛化。

允许1..6轮；r1低字目标只有复制、不含学习更新，因此配置拒绝。
主实验固定bit0，避免把Feistel低字浅一轮混作轮数提升。
现有MLP参数量与候选不匹配，必须报告，不能声称同参数公平比较。

## 冻结的本地CPU试验（首次正式拟合前制定）

所有项：TRAIN4096 / DEV1024 / TEST8192，full/direct，bit0，width32，
data192601/key190502/init17/null192602，batch256，50epochs，Adam .001/clip1，
**800更新/204800次训练呈现/项，120秒协作预算/项**。不保证硬截止。
不安装、不SSH、不GPU、不扩5轮、不追加seed/epochs、不自动重试。

1. r3 unrolled真实标签；r3 unrolled-null；r3 MLP真实标签，三项都报告。
2. 只有r3 unrolled的**DEV ACC≥70%且DEV BCE优于两种TRAIN-only基线**，
   才启动r4 unrolled、r4 unrolled-null、r4 MLP三项。
3. 使用DEV推进门槛，不用本批TEST选择下一阶段；TEST只作最终探索性报告。
4. 上限六项/4800更新/1228800次呈现/720秒协作预算。无后续自动任务。

配对真实/null相同明文和初始化，标签独立。MLP对照相同面板、最终标签与更新预算，
但输入全明文且参数化不同，结果只能比较该预算下行为。
这是一次有界可行性试验，不是稳健性、多key或安全轮数证明。
若正控失败，停止扩轮，保留阴性，不据此宣称逐轮网络不可行。

## 入口

```bash
PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli check --config configs/innovation2-unrolled-smoke.json
PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli train --config configs/innovation2-unrolled-smoke.json --output runs/unrolled-null-smoke-001
```

正式配置在`configs/innovation2-unrolled/`。配置存在不构成重跑授权。
加载仍采用原始配置SHA、exact tensor schema与weights_only；旧模型checkpoint格式不改。

## 来源

公开旋转和Feistel布局对应现有`ciphers/simon.py`及`models/prediction.py`的公开结构约定；
共享局部MLP＋轮位置条件是本地实验设计，不是某论文复现，不宣称新颖性已验证。
未提取backup科学源码、权重或求解赋值。见`provenance-unrolled.json`。
