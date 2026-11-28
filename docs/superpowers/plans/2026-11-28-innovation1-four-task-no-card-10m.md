# 四任务撤卡实现计划

使用 executing-plans 和 test-driven-development。设计见 [撤卡对照](../specs/2026-11-28-innovation1-four-task-no-card-10m-design.md)。

- [ ] 先测试配方与全量池配置相同的字段、固定路线及通用缓冲区、基线相同的初始参数。
- [ ] 在 interleave 配方登记撤卡身份，用现有 NoCardM 计算常数路由并关闭专属小块取数。报告保存实际查表及有无卡口径。
- [ ] 核对旧配方与测试、干净源码 check、Ruff、strict mypy、独立审查和固定提交发布。
- [ ] 固定设备上启动一次，保存原生退出及完整结果。比较两套保存记录。
