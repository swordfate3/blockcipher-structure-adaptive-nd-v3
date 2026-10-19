# 创新1统一CLI小训练验证（2026-10-19）

## 已落实的项目skills

- `.pi/skills/cipher-experiment-development/SKILL.md`：统一CLI、创新1/2边界、必要来源提取、功能训练验收、结果解释。
- `.pi/skills/cipher-git-result-workflow/SKILL.md`：本地commit/push→GitHub→Windows只fetch/checkout；
  Windows本机运行，runs经SSH/SCP取回；远端不提交Git、不通过Git回传结果。

skill-creator bundled validator两项实际通过。项目目录自动发现，无需复制全局skill或加载旧backupskills；
当前会话修改后需`/reload`更新发现上下文。本次已显式读取并按流程实施，未用skill授权扩大GPU/远端执行。

## 创新1能力

审阅提取旧纯算法及模型，修改导入为新包；绝不import备份运行时。
来源commit、原路径、SHA和修改分别在`provenance-models.json`与`provenance-data.json`，
总入口`provenance.json`。没有迁入旧训练器、checkpoint、优化器或审批系统。

统一入口：

```bash
uv run cipher-experiments check --config configs/innovation1-smoke.json
uv run cipher-experiments train --config configs/innovation1-smoke.json --output runs/innovation1-001
```

首轮开发源码验证使用：

```bash
PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli train --config configs/innovation1-smoke.json --output runs/innovation1-local-20261019
```

随后实际重建并安装新项目wheel（no-deps/no-index，不改第三方依赖），使用
`uv run --no-sync cipher-experiments train ... --output runs/innovation1-console-20261019`
再次完成同一4更新小配置，exit0/1.001944s。当前本机console入口已包含创新1。
这两次各4更新的工程调用分别记录，不称整个工作过程仅4更新。

## 实际训练结果（功能验证）

本地CPU，seed20261019/init17、Adam.001，1087509参数shared-M，一个joint state/public13/top2。
SIMECK64r16Δ40/PRESENT80r7ΔD00000/SIMON64r13Δ40/SKINNY64r8Δ2000。
每任务TRAIN/DEV/TEST各2groups，4pair/group、组内同key、负类独立明文真实加密。
1epoch/batch2：**每任务1更新，总4更新、8group训练呈现（32pair）**。

- 选模epoch1；normal exit0；1.4178933699949994s。
- 每任务TEST均50%（只有2groups），BCE分别1.564509/1.645229/1.714651/1.644266。
- 这些指标无质量推断价值，不能作为shared-M有效性/旧成绩复现。
- `best.pt`原始SHA：`68815c6d55f3ee7aa7085ccc546be22603a0a7dc3e8e68f9d8f624ca720bfb0f`。
- 产物：`runs/innovation1-local-20261019/`，在Git忽略范围。
- 实际执行发生在独立审查strictstate补丁前；原产物不改写。补丁后测试又验证正常/异常checkpoint路径。

## 验证与审查

全项目Ruff通过；strict mypy39files；pytest **39passed/2.02s**。
覆盖默认模型参数量/前向/梯度、四密码KAT、数据生成与真实小训练/选模/保存/显式评估。
负dtype和NaN checkpoint即使文件SHA正确也在模型前向之前拒绝。

独立readonly review081aa513-4075-4a2确认1项P2：strict=True不能拒绝dtype转换/NaN。
已增加逐tensor名称/shape/dtype/finite检查，用于selected重载和显式evaluate；补反例回归。
审查不执行程序，不能当作其复验了测试或训练。

没有SSH、新GPU占用、正式大数据拟合、Git提交/推送；用户此次“试训练”按有界CPU功能测试落实。
不是已完成远端正式实验，也不自动授权新GPU批次。
