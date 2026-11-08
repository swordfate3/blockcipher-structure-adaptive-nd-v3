# 当前文档导航

docs 根目录仅保留本导航、`superpowers/` 与 `archive/`。指南归档只是存放位置调整，不等于停用；下列使用与边界文档仍可阅读。

## 使用与边界

- [主使用说明](../README.md)：本机 CLI、安装及结果读取；[架构](../ARCHITECTURE.md)与[后端职责](archive/guides/BACKEND-BOUNDARIES.md)。
- [创新1训练](archive/guides/INNOVATION1-TRAINING.md)：四任务 shared-M 功能训练。
- [创新2位预测](archive/guides/INNOVATION2-PREDICTION.md)：单输出位任务合同；[研究状态记录](archive/guides/INNOVATION2-CURRENT-STATE.md)保留既有结论与限制，其中历史路径可查下方归档映射。
- [核积分](archive/guides/INNOVATION2-KERNEL-INTEGRAL.md)与[分裂消去](archive/guides/INNOVATION2-SPLIT-CANCEL.md)：相关方法及边界。
- [来源记录](archive/provenance/provenance.json)：其余 `provenance*.json` 同样位于 [archive/provenance/](archive/provenance/)。

## 当前方案与结果入口

- **2026-11-06 integral-scale：设计与计划，不是已执行科学结果。** [设计规格](superpowers/specs/2026-11-06-innovation2-integral-scale-design.md) · [实现计划](superpowers/plans/2026-11-06-innovation2-integral-scale.md) · [简版方案说明](../result/20261106-integral-scale-proposal/index.html)。该方案报告记录计划待审阅、实现/真实工程尚未开始，不能将历史数字当作本方案新结果。
- [已有创新2结果简版总览](../result/innovation2-index.html)与[文字摘要](../result/innovation2-summary.md)：只读取已保存结果，保留各分册来源和限制。
- [分裂消去简明说明](../result/20261106-split-cancel-explained/index.html)：历史保存结果的解释，不是新增实验。
- [Superpowers 规格与计划](superpowers/)保持原位；计划文档本身不构成实验完成或新增运行授权。

## 历史档案

历史文档已实际移至 [archive 索引](archive/README.md)；[manifest-final.json](archive/manifest-final.json)提供两轮共98份文件的总 old→new 映射、迁移前后大小与 SHA256，以及按原位置解析的相对链接扫描。[manifest.json](archive/manifest.json)仅为第一轮80份迁移的冻结快照。
历史正文、保留合同和冻结的 `result/`、`validation/`、`runs/` 产物不随归档重写；历史正文、旧报告及未维护入口中的旧路径**不自动跳转**，须按原位置解析后查总清单；来源 JSON 内部旧 record 路径未重写。归档不更新历史结论、不恢复历史权限。
