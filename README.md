# Cipher Experiments

一个在**当前机器直接运行**的 Python 密码实验后端。Git负责代码版本，训练程序不调用SSH、Git同步或远程调度；本地和Windows训练机使用相同入口。

提供创新2合成启动检查与SIMON64/128单输出位MLP基线，以及创新1四任务shared-M小训练入口。
**不是旧实验复现，也不承诺任何准确率。**

## 1. 安装

需要 Python **3.12** 与 [uv](https://docs.astral.sh/uv/)。在仓库根目录：

```bash
uv sync --group dev
uv run cipher-experiments --help
```

CUDA训练需要与显卡驱动匹配的PyTorch。默认依赖安装不保证CUDA可用；先按PyTorch官方指引安装合适版本，再检查设备。程序不自动升级驱动、不安装CUDA、不在GPU缺失时偷偷退回CPU。

当前没有冻结锁文件：依赖版本范围写在`pyproject.toml`中；部署时应生成并审阅`uv.lock`，使用同一锁文件同步。不能把未实际执行的依赖安装当作已验证。

本次工作区已准备好独立`.venv`（复用已有第三方依赖，无下载升级）。**在当前机器**可用
`uv run --no-sync cipher-experiments ...`执行以下命令，避免重新解析依赖；新用户不要假设自己已有这个环境。

## 2. 首次运行：一分钟启动检查

```bash
uv run cipher-experiments check --config configs/smoke.json
uv run cipher-experiments train --config configs/smoke.json --output runs/first-smoke
```

这会进行**两个CPU更新**，验证训练、DEV选模、checkpoint、TEST评价和退出记录。标签是合成的明文首位，不是密码成绩。`runs/first-smoke`必须不存在；不会覆盖旧结果，也不自动续训。

看到`status: completed`后，查看：

```text
runs/first-smoke/
  config.json           # 实际配置
  manifest.json         # 源码SHA、Git身份（可得时）、环境
  data.json             # 各划分行数与张量SHA
  progress.jsonl        # 每epoch损失、DEV指标、真实更新数
  training.log          # 可读进度
  best.pt               # DEV BCE最低的checkpoint，平局保留最早
  last.pt               # 最后epoch checkpoint
  results.json          # best模型的TRAIN/DEV/TEST ACC、BCE与TRAIN常量基线
  exit.json             # completed/failed、更新计数、退出状态
  error.log             # 仅失败时保存异常栈
```

`max_seconds`是批次间协作式超时，**不是硬进程/GPU截止**。强杀进程或断电不保证产生exit记录；缺少exit必须视为未知，不能当成功。当前不含WindowsJob或后台调度器。

## 3. 运行SIMON基线

复制`configs/simon-baseline.json`，修改样本量、设备、轮数等，再在**实际训练机器本机终端**运行：

```bash
uv run cipher-experiments check --config configs/simon-baseline.json
uv run cipher-experiments train --config configs/simon-baseline.json --output runs/simon-r3-001
```

- 支持SIMON64/128，1—44轮；target_bit采用64位密文**MSB=0**编号。
- MLP输入64位明文；key仅用于数据生成，不输入模型。
- 各划分明文不重复；一把固定seed生成的密钥。没有跨key/init稳健性结论。
- 每epoch评价DEV并选模；只在训练及选模完成后评价TEST。
- TEST数据当前预先生成，不是封存confirm协议。不要据此宣称已复现旧实验验收或独立确认。
- 不读取旧`outputs/`、聊天记录、审批卡、历史权重或机器专属路径。

## 创新1：四任务shared-M训练

```bash
uv run cipher-experiments check --config configs/innovation1-smoke.json
uv run cipher-experiments train --config configs/innovation1-smoke.json --output runs/innovation1-001
```

当前机器已有环境可加`uv run --no-sync`；未重新安装的开发源码也可用
`PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli ...`（Windows使用安装后的CLI）。

- 一个共享模型，1,087,509参数，public13/top-2路由；路由不读密文、标签、key或专家预测。
- SIMECK64r16/Δ0x40、PRESENT80r7/Δ0xD00000、SIMON64r13/Δ0x40、SKINNY64r8/Δ0x2000。
- 每decision严格4pair、组内同key；负类为独立随机明文真实加密，不生成随机密文负类。
- `train_rows/dev_rows/test_rows`在此表示**每任务group数**，不是单明文数。
  小配置每任务2/2/2、batch2、1epoch；总4次更新，每任务1次，8个group训练呈现。
- `rounds/target_bit/key_seed/hidden_width`是位预测配置字段；创新1任务、模型宽度和组内key生成
  由固定四任务卡与`data_seed`决定，这些位预测字段不用于改变创新1模型/任务。
- 每epoch四任务平均DEV BCE选一个joint bestcheckpoint，最后逐任务报告TEST ACC/BCE。
  不加载历史权重，不称M_reinit或历史大实验复现；这份小配置只验证功能，不用于科研质量判断。
- 显式评估增加`--experiment innovation1`，同样需要实际checkpoint SHA：
  `cipher-experiments evaluate --experiment innovation1 --checkpoint <best.pt> --sha256 <SHA> --output runs/i1-eval`。

执行记录见[创新1小训练验证](docs/INNOVATION1-TRAINING.md)。

## 4. 显式checkpoint评估

从`results.json`读取`best_checkpoint_sha256`：

```bash
uv run cipher-experiments evaluate --checkpoint runs/first-smoke/best.pt --sha256 <实际SHA256> --output runs/first-evaluation
```

这是用户主动请求的本机评估：按checkpoint中配置重新生成TEST并执行前向。它**不是取回远程结果时的默认动作**。训练机已经保存`results.json`；本地取回时直接读取该文件，不必重算。

## 5. Git与训练机

1. **本地开发机**修改代码、测试、commit并push到GitHub。
2. **远端Windows训练机**只fetch并检出指定commit，确认工作区状态；不要在训练中pull修改源码。
3. 在训练机本机终端执行上述`train`命令，统一将结果写入仓库内的`runs/<run-id>/`。
   关闭终端可能终止训练；没有断线存活承诺。
4. **本地主动通过SSH/SCP取回**对应run目录，保存到本地`runs/received/<run-id>/`；
   读取远端已保存的`results.json`和`exit.json`，不自动重新训练、再生标签或重算指标。

**远端Windows不负责commit或push到GitHub，也不通过Git回传结果。**
代码同步和结果传输是两条独立通道：

```text
本地代码 → GitHub → Windows检出指定commit
Windows runs/<run-id>/ → SSH/SCP → 本地 runs/received/<run-id>/
```

`runs/`及备用产物目录`outputs/`、`results/`、`downloads/`均在`.gitignore`内；
备份、环境、权重和凭据也不纳入Git。CLI的`--output`可以指定其他路径，但发布流程统一使用`runs/`；
若另选仓库内目录，必须先补充忽略规则，不要执行无差别`git add .`。

仓库：https://github.com/swordfate3/blockcipher-structure-adaptive-nd-v3 。
没有自动push、Webhook、CI runner或常驻服务。

## 本项目skills

`.pi/skills/cipher-experiment-development/`规范开发、创新1/2合同与验证；
`.pi/skills/cipher-git-result-workflow/`规范本地发布、Windows只拉取/运行、SSH/SCP只取回结果。
项目skills按标准目录发现；修改后在Pi执行`/reload`，旧backup skills不再加载。
skill本身不授予训练/GPU/安装/发布权限，不包含旧审批卡或远端启动器。

## 开发与验收

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

模块职责见[ARCHITECTURE.md](ARCHITECTURE.md)，旧项目归档与实际验收范围见[重建记录](docs/REBUILD.md)。
