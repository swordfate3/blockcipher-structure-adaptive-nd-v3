# 审查后修订（分支 review-fixes-20261020）

本分支基于 `111adab`，只修改审查中确认的具体问题；主线 `main` 未改动，未 push。
已完成的四项修改、以及它们各自的实际验证，见下。未做的大项在末尾列出，避免把建议当成已完成。

## 1. 训练顺序种子与数据种子分离

新增可选字段 `shuffle_seed`。此前 batch 顺序固定为 `data_seed + 1`，导致
“换明文面板”和“换训练顺序”永远同时发生，STE 确认批失败后无法归因。

- `None`（缺省）保持历史行为 `data_seed + 1`，且 `to_dict()` 在缺省时不写该字段，
  **已记录的 configSHA 不变**，旧 checkpoint 仍可加载。
- 显式赋值后 `model_metadata.json` 记录 `order_seed` 与 `order_seed_source`，实验可复查。
- 回归测试：同面板不同顺序会产生不同结果；同顺序两次运行结果与 checkpoint 逐字节一致；
  只改顺序不改明文 SHA。

## 2. BCE 可比性

固定 ±2 logit 的模型正确样本 BCE 下限约 **0.126928**，与 0.6931 的常量概率基线不是同一尺度，
原先“BCE 优于基线”的门槛会误判。现在每次评价额外输出：

```text
bce_gain_vs_train_constant      = 常量BCE - 模型BCE     （正数=更好）
bce_gain_vs_public_reference    = 公开参考BCE - 模型BCE （存在参考时）
```

STE 的 `model_metadata.json` 同时记录 `score_floor_bce`。
原有 `bce`、`hard_constants`、两类基线字段未改名，历史结果仍可对照。

## 3. 参数可识别性审计（新增能力）

新增 `src/cipher_experiments/fitting/ambiguity.py` 与 `tools/audit_hard_assignment.py`：
读取已完成的硬拟合 run，把学到的常量与真实轮密钥逐位比较，并检验“同时翻转差异位是否仍保持函数”。

它是**事后审计**：不重训、不加载 checkpoint、不生成标签，也不进入训练/模型路径。
`ciphers/simon.py` 因此新增只读 `round_keys()`，`data/prediction.py` 新增 `key_for_seed()`（两者均用于审计）。

实测两个已保存的真实 run（`--probes 2048`，见 `validation/review-fixes/`）：

| run | 坐标 | 与真实轮密钥不同的坐标 | 差异位同时翻转 | 单独翻转均改变输出 | 与真密文不一致 |
|---|---:|---|---|---|---:|
| 硬 r3 | 11 | `(0,0) (2,0)` | 仍是同一函数 | 是 | 0/2048 |
| 硬 r4 | 24 | `(0,2) (1,4) (3,0)` | 仍是同一函数 | 是 | 0/2048 |

含义：**该目标位函数存在非平凡等价类，100% 预测成立但参数不可唯一识别。**
这次把“参数学习≠密钥恢复”从断言变成了可测结论，也解释了硬搜索为什么总选到与真值同数 0 的位
（升序枚举偏向 0）。另有坐标 `(0,2)` 在 r4 是结构性冗余叶，单独翻转不改变任何输出。

## 4. 未做的部分

- 配置/训练入口仍按 `model_kind` 分支（`config.py` 交叉校验、`training.py` + `hard_training.py` 双入口）；
  重构会牵动已冻结的实验身份，本分支不动。
- 未拆 `backups/`（28G）、未加 `evidence/` 入库、未加 CI 与 `uv.lock`。
- 未做跨 key 泛化实验；`shuffle_seed` 只是让以后能干净地拆开归因。

## 验证与证据

```text
PYTHONPATH=src .venv/bin/python -m ruff check .
PYTHONPATH=src .venv/bin/python -m mypy
PYTHONPATH=src .venv/bin/python -m pytest
```

全部通过（798 tests、mypy 72 files）。新增测试：
`tests/test_seed_separation.py`、`tests/test_ambiguity_audit.py`。
审计输出：`validation/review-fixes/audit-r3.json`、`audit-r4.json`。
本分支未执行新的密码拟合、GPU、SSH、安装或 commit/push。
