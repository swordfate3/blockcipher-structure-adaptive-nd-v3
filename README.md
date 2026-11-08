# Cipher Lab 工作区

当前开发项目：**[`cipher-lab/`](cipher-lab/README.md)**。

新后端已接通积分经验流程、位预测 MLP、创新1四任务 shared-M，以及直接读取保存模型和数据的独立评价。
命令、完整配置与限制以 [新项目 README](cipher-lab/README.md) 为准。

```bash
cd /workspace/cipher-lab
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /workspace/.venv/bin/python -m cipher_lab --help
```

这是当前已有环境的源码入口，不代表已安装或完成新机器部署。
工程合成测试通过不等于真实密码预测有效；真实密码/KAT、生产 RSS、GPU、Windows 尚未验收。

## 工作区保留内容

- `cipher-lab/`：当前源码、测试、配置及项目说明；运行后按需生成其自身 `data/`、`models/`、`result/`。
- `papers/`、`笔记本/`：共享论文和笔记。
- `docs/`：保留的设计、计划及历史文档档案。
- `.pi/`、`AGENTS.md`、`.learnings/`：项目规则与经验；新项目适用其自身 `AGENTS.md` 的入口和路径约定。
- `.venv/`：已有环境；本次未重新安装或解析依赖。
- `backups/`：既有约28GB历史备份，原位保留，本次没有整理或删除。
- `.git/`：现有版本历史与索引保留；本次未提交、发布或重建仓库。

## 旧项目归档

旧代码、配置和实验产物已在校验备份后从工作区移除；旧入口不再作为工作区入口使用。

- 统一备份根目录：`~/.local/share/backups/`
- 归档批次：`legacy-project-20261107T174214Z`
- 完整性索引：`workspace/legacy-project-20261107T174214Z.verification.json`
- 清理审计：`workspace/legacy-project-20261107T174214Z.cleanup.json`
- 备份清单：`bk list --category workspace`

归档包括当时的未提交文件和 Git 状态，不只是已提交代码。
清理仅涉及已核验的旧源码、测试、配置、运行/验证/结果目录及旧构建入口；共享资料不删除。
历史文档与来源登记仍保留原路径语义，原始旧来源需从归档查阅；不为了恢复旧链接覆盖当前项目。

Git 工作区中的旧文件删除尚未提交，这是归档清理的结果，不应使用 `git reset/clean` 来消除提示。
恢复时先按完整性索引定位所需备份，再明确恢复目的与目标位置；不要直接覆盖新项目。
