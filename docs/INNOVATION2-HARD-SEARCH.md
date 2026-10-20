# 创新2：TRAIN-only硬布尔结构参数辨识

## 本轮固定方案（拟合前确定）

用户只需某个预定密文位在未见明文上有效，不需要全密文或完整密钥。
固定SIMON64/128的MSB编号bit0；不做TEST后挑位。
使用公开依赖图，将未知轮常量按排序坐标编码为整数，按升序检索TRAIN零误差组合。
允许学习与key相关的参数；不能从生成器获取真实key、轮密钥或真实中间状态。

本轮授权推进后选择本机CPU整数位并行搜索，**不使用GPU/SSH/安装/自动发布**。
统一入口`check/train/evaluate`，model_kind=`simon-hard-search`，新的run目录。
这不是Adam训练，也不将组合数/位并行行数虚称神经更新或样本presentation。

## 固定执行顺序与停止条件

配置位于`configs/innovation2-hard/`：

| 项 | TRAIN/DEV/TEST | 数据seed | 目标/搜索域 | 协作预算 |
|---|---|---:|---|---:|
| smoke | 1/4/4独立随机标签 | 191801 | r1 bit0，2组合 | 30秒 |
| r3 | 512/1024/16384 | 191801 | bit0，11坐标，2048组合 | 120秒 |
| r4 | 512/1024/16384 | 191801 | bit0，24坐标，16777216组合 | 120秒 |
| r4-null | 512/1024/16384独立随机标签 | 191801 | 同r4结构 | 120秒 |

key_seed190502、init17、null191402；init不决定硬搜索顺序。r3/r4同明文面板，按轮数生成不同标签。
这些seed是实验可复现设置，模型/拟合API不接受key_seed；真实密钥仅在标签生成路径。

1. 工程验证、干净源码smoke成功后执行一次r3。
2. r3需TRAIN/DEV/TEST准确率均100%，且TEST BCE优于两类TRAIN基线，才进入r4。
3. r4若同样通过，执行一次r4-null；独立随机标签预期不存在TRAIN全匹配组合。
4. 超时/失败/无拟合就停止下游，不重试、改目标位、追加样本/seed或扩大预算。
5. 本轮不试r5+：独立坐标域更大，且缺少完整key schedule约束；成功也不是自动扩批授权。

这是单key/seed机制探针，不是多key确认；其样本预算和旧神经方法不匹配，不能据此宣称架构公平胜出。
TEST用于固定候选评分和预先规定的轮数推进，不作为拟合条件或参数组合选择器。
逐级推进仍属探索性实验；若提出更强泛化结论需新的确认数据。

## 信息边界与算法

- `fitting/hard_simon.py`仅接收明文整数、TRAIN单bit标签、公开轮数/目标位及预算。
  不导入cipher/data/training，不采样或获取真实key。
- 公开AND/XOR图用Python整数位集进行并行：一个整数的各bit表示一个候选常量组合。
- 每块最多65536候选，对TRAIN行逐一求布尔输出，并去除不一致组合；若整块空则跳过后续行。
- 第一块有幸存者时，取最低整数编号参数；不再查看DEV以选择其他幸存参数。
- `fit.json`记录参数域、已处理块/覆盖候选数和位并行行求值数。覆盖候选数指被处理整块，
  不是逐候选跑完全部TRAIN的次数，也不是神经更新。
- CPU固定上限24个符号未知坐标，不尝试无限搜索；协作超时不是硬进程时限。
- 硬预测器保存±1符号状态，forward直接执行硬AND/XOR；加载验证配置SHA、tensor域和±1语义。
- `selection.json`及checkpoint严格加载完成后，才请求TEST标签。DEV只评分、不选参数。
- 找不到零误差模型时保存exhausted及计数，不生成TEST标签，不把无模型说成50%准确率。
- 完整遍历无解只说明这个固定模型族与给定TRAIN不一致，不是密码理论上的不可预测证明。

输出BCE将硬0/1概率裁剪到1e-6，正确时约1e-6、错误时约13.8；
这是固定评分约定，不是概率校准或真密钥可信度。
即使未见样本全对，也只能支持此目标位的预测；可能存在多个等价参数，不自动认定主密钥恢复。

## 入口

```bash
PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli check --config configs/innovation2-hard/r3.json
PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli train --config configs/innovation2-hard/r3.json --output runs/hard-r3-001
```

实际结果另行记录，不能将此准备方案当成完成证据。
