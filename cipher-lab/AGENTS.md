# Cipher Lab

仅适用于本目录。新项目采用精简CCDS结构，不兼容旧项目；旧项目保持原状。
- 入口为已实现的cipher-lab check/run/evaluate；check支持三方法且零科学导入/输出，run显式backend：积分/位预测demo|simon，shared-M demo|ciphers。evaluate显式checkpoint/dataset路径与SHA，新run只前向、不复制源binary。不适用父旧项目的train、runs/、旧来源文件和旧报告绝对路径契约。
- 实验设置只在configs；原始记录data/<id>，权重models/<id>，报告result/<id>。来源提取记录只用docs/provenance.json。无runs/outputs/validation平行产物。
- 密码/数据/模型/训练/积分数学/评价/执行限制/保存各守职责，不import旧项目或历史产物，不造workflow/registry/巨大utils。秘密留在数据生成侧，不进入模型、符号层或公开产物。
- CLI只解析/claim/固定异常边界/呈现；training/application只装配provider与方法；evaluation/application独立组装保存文件评价，不importtraining/ciphers/数据生成/私有RNG。checkpoint无backend身份时evaluate必须UNVERIFIED/synthetic=null，不根据路径猜来源；报告只读保存指标，由原owner写。
- 每轮配方改参数，不复制执行器；新增真正算法才新增算法代码。每种永久文件须有消费者与保留规则，测试临时产物自行清理。
- 保留父规则的明确权限、代理分工、TDD、独立审查与fresh验证纪律。当前许可为已批准阶段2/3开发和无真实密码合成工程测试：CPU小tensor；每种完整合成闭环≤60s，位预测≤2次优化更新，shared-M四任务各1次共4次；不搜索、不多seed科研。真实密码/KAT/科学标签/正式拟合/GPU/生产RSS/SSH/安装/Git/备份切换/其他清理仍须另行授权。
- 仅服务本项目result/提供报告；不暴露data/models，不重新采样/计算指标来美化报告。依赖/Windows/密码实测未完成须直说。
