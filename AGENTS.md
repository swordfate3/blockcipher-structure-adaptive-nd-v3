# Cipher Experiments — development rules

- 新项目的目标是独立、可直接使用的Python实验后端；中文说明，命令必须对应已实现入口。
- 当前机器运行：`cipher-experiments check/train/evaluate`。不在训练中使用SSH/Git同步/远程包装。
- 相关任务先读`.pi/skills/cipher-experiment-development/SKILL.md`；Git发布/同步/取回再读`.pi/skills/cipher-git-result-workflow/SKILL.md`。旧backup skills不加载。
- 创新1用configs/innovation1-smoke.json：四任务shared-M/1087509参数，每任务2/2/2groups、batch2、epoch1总4更新，只是功能训练。不要混作创新2位预测或旧M_reinit科研复现。
- 保持明确依赖方向；模型不读key或其他cipher标签，密码不依赖模型/训练。不要通用workflow/registry/巨大utils。
- `configs/smoke.json`是两个CPU更新的合成工程检查，不是科学密码结果。正式密码拟合、GPU占用、安装、远程连接需用户明确授权；不继承备份中的历史批准/窗口。
- `backups/`包含完整旧项目、未提交工作及历史证据，只读参考；不import、不收集测试、不lint、不stage。新来源提取记录在docs/provenance.json。
- Git只管理代码/小配置/测试/文档；不存数据、runs、权重、缓存、凭据或环境。只有本地开发机commit/push；远端Windows只fetch/checkout和运行，不commit/push，不用Git回传产物。结果统一写runs/<run-id>/，本地SSH/SCP取回runs/received/<run-id>/。无明确请求不commit/push。
- 接收训练机保存结果不自动本地重算指标、再生标签或做模型前向；evaluate是用户主动调用的独立能力。
- 本机协作超时不是硬进程限额，缺exit记录是未知；不虚构训练完成、预测成功或跨key结论。
- 修改前调查现状，精确edit；保留无关工作。验证Ruff/mypy/pytest及干净源码副本CLI小闭环，报告实际执行范围；没有安装/GPU实测就如实说明。
