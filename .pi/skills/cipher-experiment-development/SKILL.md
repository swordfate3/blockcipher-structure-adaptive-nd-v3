---
name: cipher-experiment-development
description: 开发、测试或解释本项目密码实验后端时使用，包含创新1 shared-M与创新2位预测。保持统一CLI和干净配置；不使用旧远程编排或继承历史实验授权。
---

# 本项目开发与实验

先读根目录[AGENTS](../../../AGENTS.md)、[README](../../../README.md)及[架构](../../../ARCHITECTURE.md)。

## 核心路径

- 用户入口只用`cipher-experiments check/train/evaluate`；源码开发可用`python -m cipher_experiments.cli`。
- `configs/smoke.json`为创新2合成工程检查；`configs/innovation1-smoke.json`为创新1四任务小训练检查。
  `check`只检查配置/设备，不启动拟合。`train`在调用CLI的机器运行；授权后可经SSH调用远端已Git同步的CLI。
  不通过SSH传任务包或科学源码，不恢复旧远程编排。
- 新配置独立完整、严格字段。输出必须是新的`runs/<run-id>/`；失败不可覆写旧结果或自动续训。
- 创新1：每decision恰4pair/组内同key，负类独立随机明文的真实加密；router只读public13。
  四任务SIMECK64r16Δ40/PRESENT80r7ΔD00000/SIMON64r13Δ40/SKINNY64r8Δ2000。
- 创新2：输入明文、输出单bit，不能把创新1分类准确率混作位预测。两分支来源/指标分开报告。
  支持full/support输入、direct/public-residual（SIMON1..4轮）与null对照；support是保守依赖集合。
  `configs/innovation2/`五份已完成一次GPU对照，不是smoke或重跑授权；合同见`docs/INNOVATION2-PREDICTION.md`，
  结果见`docs/INNOVATION2-FIRST-GPU-RESULTS.md`（r3有信号、r4近随机，仅单key/seed）。
- 模型与密码算法不依赖训练或CLI；禁止import backup/旧运行包。必要算法提取记录commit/原路径/SHA/修改。

## 开发闭环

1. 核对Git状态和现有入口，保留无关改动。
2. 最小实现，补无私有outputs依赖的合成/小tensor测试。
3. 执行`uv run ruff check .`、`uv run mypy`、`uv run pytest`；本机预备环境可加`--no-sync`。
4. 从CLI验证配置→真实小更新→DEV选择→bestcheckpoint→最终TEST/exit；工程结果不是科研质量。
5. 新机器/Windows兼容性必须现场验证，不能用本地测试数代替。skill维护本身不授权SSH/GPU/安装。

## 训练范围

先明确任务、设备、样本量、epochs/updates、次数及超时。用户要求“试训练”可准备有界功能配置，
但未明确GPU/正式预算时不要自行升级为长训练。这里的协作超时不是硬进程限额。
旧批次、窗口、备用checkpoint均不授予新训练许可。不能因50%结果追加seed/epoch。

## 保存结果

报告实际更新、分任务样本/ACC/BCE、选模epoch、退出、SHA及路径，区分失败/未知与成功。
创新2预建明文、严格加载DEV选中checkpoint后才生成TEST标签；创新1仍预建标签，均不称sealedconfirm。
新初始化无教师，不称旧M_reinit复现或SOTA。
取回远端结果只核文件SHA/源码配置身份/计数/退出；不自动本地重算指标、标签或模型前向。
显式evaluate是单独用户动作，不要加到默认取回验收链。
