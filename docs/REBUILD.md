# 2026-10-19 重新建立实验后端

## 归档与新仓库

用户明确批准完整旧项目归档、新建独立Python实验后端。

- 旧项目：`backups/v3-before-rebuild-20261019/`。保存原`.git`、全部未提交/未跟踪工作、输出、环境、配置及历史失败，不重写旧实验记录。
- 更早的`backups/innovation-one-old-20261011/`保持原位置，不递归复制到新归档。
- `/workspace`是宿主挂载点，不能整体改名；采用同文件系统逐项rename，保留实际文件。
- `.pi-attachments`是独立嵌套挂载，保留原挂载，并复制其内容到归档；不试图移动挂载点。
- 归档`ARCHIVE-MANIFEST.json`记录旧HEAD `002181d71c9e774c4620d338fe00ba6a0039be4c`及原Git状态。
  检查归档后的状态：原项目各项保留，唯一新增是归档manifest；新staging`.rebuild/`不属于旧项目归档。
- 新根目录独立`git init -b main`，**没有commit、stage、remote配置或push**。不能说已经能远端git fetch；需要用户后续确定仓库地址并发布。

## 新项目范围

新项目名`cipher-experiments`，导入包`cipher_experiments`，命令`cipher-experiments`。
仅提取审阅后的SIMON标量算法，见`provenance.json`。没有导入旧模型、runner、SSH、审批卡、历史排除或私有输出。
MLP/训练/指标/CLI为新独立实现；旧5轮代数、神经成绩、失败记录都不是新项目结果。

## 实际完成的检查

- Ruff通过；strict mypy **17个文件通过**；pytest **15 passed**（新环境最终1.68s）。
- SIMON64/128 44轮KAT通过；小型数据位序、唯一性与确定性检查通过。
- wheel实际构建成功，包含`cipher-experiments` console entrypoint。
- wheel本体用`uv pip install --no-deps --no-index`安装验证；没有下载/升级第三方依赖。
- 不含backup/outputs的独立源码副本位于`validation/clean-source/`，实际调用安装后的console script，
  无PYTHONPATH旧runtime依赖，完成check/train/checkpoint load/evaluate。
- `shared_m_nd`和`cipher_output_prediction`在新环境中均不可import。
- 每次smoke是8/4/4合成样本、1epoch、batch4、**2次CPU更新**；测试/重复CLI验收也有各自小更新，不能称本次零更新。
  合成准确率没有科学意义，不作为新密码预测成绩。
- 独立静态审查9571a288-759d-47d未确认范围内P1/P2缺陷；未执行程序，不等于其复验了上述结果。
- `architect_preview`→用户确认→`architect_apply`生成13个基础文件，再按真实职责替换空层；`architect_review`已执行。

## 环境验证的边界

新`.venv`用持久Python3.12.15创建，复用已有第三方包文件（硬链接，不含旧项目editable或dist-info），
再安装新wheel。可直接执行：

```bash
uv run --no-sync cipher-experiments check --config configs/smoke.json
uv run --no-sync cipher-experiments train --config configs/smoke.json --output runs/my-new-smoke
```

`--no-sync`只适用于本机已经准备好的环境；普通新用户仍按README的`uv sync --group dev`安装。
实际第三方版本Torch2.14.1+cu130/NumPy2.5.3，Linux本机无CUDA现场验收。
全新联网依赖安装**未验证**。曾尝试`uv lock --offline`，缓存不足无法解析numpy候选，未生成锁文件，
不伪造可用lock或拷贝旧项目lock冒充新解析。后续发布前需在目标环境审阅并冻结依赖。

首次静态检查类型/行长问题已修复；新环境首次Ruff因遗漏二进制不能启动，补用现有Ruff二进制后通过。
原始失败日志与最终检查保存在`validation/`，不纳入Git。旧项目原包未因此改写。

## 未执行与限制

没有SSH、远端Git同步、GPU运行、正式SIMON拟合、新预测成绩、Git提交或推送。
没有硬进程限额/WindowsJob、resume、sealedconfirm、跨key/seed统计、WebAPI或自动调度。
`max_seconds`为协作预算，强杀没有保证exit记录。TEST预建但最后评分；显式evaluate会再生TEST，
这与仅传输并读取远端savedresults不同，不能自动把evaluate加入取回流程。

这些限制不阻止从干净代码副本启动小型训练，但不能据此宣称远端生产/科学协议全部验收。
