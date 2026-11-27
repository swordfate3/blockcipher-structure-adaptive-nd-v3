# 四密码结构卡实施计划

> 执行采用 writing-plans、executing-plans、test-driven-development，验收以固定合成测试为准。

**目标**　四种算法的小块取数表都由结构卡生成。

**架构**　`cipher-lab/plugins/cipher_lab_plugins/innovation1/four_single/structure_cards.py` 负责纯 Python 编译与哈希校验。配方 v2 在现有 interleave 运行时装配固定缓冲区，沿用原有 Top‑2 路由和 checkpoint 校验。

**设计**　[四密码结构卡](../specs/2026-11-27-four-cipher-structure-cards-design.md)

## 约束

工作目录位于 `feat/four-cipher-structure-cards`。v1 方法、配置字节和报告格式保留。四项使用原子集和训练设置。工程验证使用 guard、固定 CPU 构造和人工替身。

## 验收

- [x] 逐项比对四张 64×9 表。2,304 个索引与旧版一致，覆盖中心、零填充、SKINNY 跨列衔接、PRESENT 坐标及表哈希。
- [x] 核对 v2 配置、输出根互斥、结构卡与源闭包。干净源码副本在 `-S` 下通过两版 CLI `check`，v1 输出与基线逐字节一致。
- [x] 核对模型构造、Top‑2 取数与记录、固定缓冲区损坏拒绝，并用人工替身验证两版保存及报告流程。
- [x] 63 项定向测试、Ruff 与 17 文件严格 mypy 通过。主仓库 `main`、索引与工作区状态核对通过，`main.bundle` 校验通过。
- [ ] 项目独立审查。当前会话未发现配置中的 `code-review` 代理入口。
