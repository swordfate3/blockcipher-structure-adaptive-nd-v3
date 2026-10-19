# Architecture

```text
cli → config + training / evaluation
training → data + models + evaluation + artifacts
data/simon → ciphers/simon
models → torch
ciphers → Python标准库
artifacts → Python标准库
```

- `cli.py`：唯一用户入口，check/train/evaluate，错误返回非零。
- `config.py`：冻结dataclass，严格JSON字段/类型/范围，不含审批或远程主机。
- `training.py`：组装数据/模型、Adam循环、DEV选模、保存、最后TEST，真实更新/样本计数。
- `evaluation.py`：绝对位ACC/BCE、TRAIN多数/概率基线、SHA绑定严格checkpoint加载；不导入training。
- `artifacts.py`：JSON/进度/字节SHA/源码身份；Git查询只用于可选身份记录，不同步/提交代码。
- `data/`：无重复明文划分；synthetic用于工程，simon是真实密码单bit标签。key不进入模型。
- `models/`：明文64位MLP，不依赖cipher/data/config/runner。
- `ciphers/`：纯标量SIMON算法与KAT，不依赖Torch或实验代码。

`innovation1.py`组装独立四任务分类训练，与位预测`training.py`平行；共享配置、保存工具和统一CLI，
不建立实验继承链。`models/shared_m`及`representations`不依赖数据/训练；
公共任务结构在`ciphers/tasks.py`，公开卡路由不读密文/key。算法来源分别在provenance子记录。

范围：一个同步本机实验入口。不包含SSH启动、通用工作流、插件注册、HTTP后端、数据库、自动调度或旧审批流程。
不迁移旧runner、旧环境配置和批次产物。算法提取记录在`docs/provenance.json`。

限制：协作超时不等于硬截止；全部split数据预建但TEST仅选模后评分；没有sealedconfirm、resume、跨key统计或GPU现场验收声明。扩展从实际需求出发，不预建多层空目录。
