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
- `two_round_diagnostics.py`：对选中local两轮模型只聚合TRAIN/DEV的公开特征分组、完整输入签名覆盖与错误；
  在TEST标签前保存，不参与选模/梯度，也不读取真实状态或key。角点hidden统计不是状态验证。
- `artifacts.py`：JSON/进度/字节SHA/源码身份；Git查询只用于可选身份记录，不同步/提交代码。
- `data/`：无重复明文划分；synthetic用于工程，simon是真实密码单bit标签。key不进入模型。
- `models/`：明文MLP、`logic.py`三层16真值表软门候选及公开依赖位/零主密钥参考；
  `prediction.py`提供MLP、logic16、人工任务circuit-aligned/random及simon-structure选择，不建registry；
  不依赖cipher/data/config/runner。
  logic固定接线持久化，加载后检查范围；train/eval均用软门，不做隐式硬化。
  `structured_simon.py`读取公开符号DAG，仅学习未知常量logit；不接收真实key/trace。
  evaluation对其额外报告hard_constants，不改变soft DEV BCE选模。
  `unrolled_simon.py`为共享局部神经轮模块及学习的轮/位置条件；direct支持1..8轮（6/8轮实测未超常量准确率），复制路径公开，
  隐状态连续且不受真实状态监督，仅最终目标bit参与BCE；不读key或密码模块。
  两轮高半位可用`plaintext-xor`将原selected logit乘公开明文位符号，不改轮内递推、标签或参数。
  `plaintext-xor-no-final-low`在此基础上仅将末轮center后的low输入通道置0；
  第一轮和其余输入不变，不增参数或标签，但有效函数类受限，不能视为真实状态学习。
- `models/simon_two_round.py`：两轮高半位的结构化四格正控与独立局部神经候选；
  公共特征或局部接线由明文计算，无真实key/轨迹或其他密文标签输入。
  `simon-r2-vector`为独立4通道局部变体（557参数），三个4→16→4与12→8→1，仍仅最终bit监督。
  正控的TRAIN四格覆盖/冲突元数据由训练层现有标签派生，不进入神经模型或改变选模。
- `data/prediction.py`：无标签唯一明文与按split请求的真实/null标签分离；训练严格加载选中模型后才请求TEST标签。
  基线仅用TRAIN拟合；显式evaluate独立重建数据。
  可选`train_pool_rows`固定训练池尾部的DEV/TEST边界，实际TRAIN取池前缀，unused行不生成标签；None保留旧划分。
  `dev_every_epochs`仅控制DEV选模间隔，末epoch必评估，未评估progress的dev为null，不改变优化更新。
- `ciphers/`：纯标量SIMON算法与KAT，不依赖Torch或实验代码。
- `data/circuit.py`：有限8-bit人工目标的无重复有效域划分/标签，不调用密码算法。
- `representations/simon_cone.py`：纯标准库公开目标位DAG，只有符号轮密钥占位，无key值/训练能力。

`innovation1.py`组装独立四任务分类训练，与位预测`training.py`平行；共享配置、保存工具和统一CLI，
不建立实验继承链。`models/shared_m`及`representations`不依赖数据/训练；
公共任务结构在`ciphers/tasks.py`，公开卡路由不读密文/key。算法来源分别在provenance子记录。

范围：一个同步本机实验入口。不包含SSH启动、通用工作流、插件注册、HTTP后端、数据库、自动调度或旧审批流程。
不迁移旧runner、旧环境配置和批次产物。算法提取记录在`docs/provenance.json`。

限制：协作超时不等于硬截止；创新2预建明文、选模后生成TEST标签，创新1仍预建split标签；
没有访问强制的sealedconfirm、resume、跨key统计或GPU现场验收声明。扩展从实际需求出发，不预建多层空目录。
