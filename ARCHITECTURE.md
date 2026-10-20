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
  可选`shuffle_seed`只控制batch顺序，缺省时保持`data_seed+1`历史行为且不写入`to_dict()`，
  因此已记录的configSHA不变。
- `training.py`：组装数据/模型、Adam循环、DEV选模、保存、最后TEST，真实更新/样本计数。
- `hard_training.py`：CLI平行硬拟合入口，TRAIN一致性选择、保存加载后TEST，不冒充Adam更新。
  `fitting/hard_simon.py`只依赖公开DAG/标准库，整数位并行消除不一致组合，不读取key/生成标签。
  `fitting/sat_simon.py`可选Glucose4，公开AND/XOR等价CNF、TRAIN共享未知坐标、单次中断式求解；
  无模型区分UNSAT/UNKNOWN，只有SAT且严格加载后TRAIN硬电路复核通过才生成TEST。
  `fitting/ambiguity.py`为训练后只读审计：比较学到的常量与真实轮密钥，检验差异位同时翻转是否保持函数。
  它不重训、不进模型/训练路径；`tools/audit_hard_assignment.py`是它的命令行入口。
- `evaluation.py`：绝对位ACC/BCE、TRAIN多数/概率基线、SHA绑定严格checkpoint加载；不导入training。
  额外输出`bce_gain_vs_train_constant`/`bce_gain_vs_public_reference`，
  避免把固定±2 logit模型的BCE直接与概率基线比大小。
- `artifacts.py`：JSON/进度/字节SHA/源码身份；Git查询只用于可选身份记录，不同步/提交代码。
- `data/`：无重复明文划分；synthetic用于工程，simon是真实密码单bit标签。key不进入模型。
- `models/`：明文MLP、`logic.py`三层16真值表软门候选及公开依赖位/零主密钥参考；
  `prediction.py`提供MLP、logic16、人工任务circuit-aligned/random及simon-structure选择，不建registry；
  不依赖cipher/data/config/runner。
  logic固定接线持久化，加载后检查范围；train/eval均用软门，不做隐式硬化。
  `structured_simon.py`读取公开符号DAG，仅学习未知常量logit；不接收真实key/trace。
  evaluation对其额外报告hard_constants，不改变soft DEV BCE选模。
  `ste_simon.py`继承公开结构，硬0/1前向、sigmoid-leaf STE反向、固定±2logit，
  与普通Adam训练对接，不调用fitting/SAT/枚举；属于梯度学习但替代梯度有偏。
  HardSimonPredictor是冻结±1常量的硬前向，checkpoint加载额外验证离散值域；
  hard-search以首个TRAIN一致组合选择，不使用DEV/TEST选参数，与soft分支的DEV选模不同。
- `data/prediction.py`：无标签唯一明文与按split请求的真实/null标签分离；训练严格加载选中模型后才请求TEST标签。
  基线仅用TRAIN拟合；显式evaluate独立重建数据。
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
