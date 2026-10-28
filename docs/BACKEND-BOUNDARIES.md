# 后端局部职责迁移：报告、checkpoint与训练分开

## 范围与不变量

用户要求先保证模块职权清晰，并将当前职责混杂的代码移回明确归属。本次仅做两项必要迁移，不重写后端，不新建通用workflow/registry或实验继承体系。

保持统一CLI `cipher-experiments check/train/evaluate`、配置与checkpoint格式、模型参数名称和初始化、采样和标签、Adam/BCE/clip1、shuffle、DEV选择及TEST标签时序不变。
创新1、模型、密码、数据生成、旧结果和其他分支均不改；没有新正式密码拟合、科学标签、GPU/SSH、安装或Git发布。

## 修改归属

| 模块 | 负责 | 不负责 |
|---|---|---|
| `training.py` | 数据/模型组装、预算、优化更新、DEV选模、调度保存与最终TEST | 模型专属报告字段、四格聚合实现、checkpoint格式实现 |
| `prediction_reports.py` | 根据模型、公开配置项和调用者给定TRAIN数据返回metadata与实验角色；四格覆盖/冲突聚合 | 文件IO、配置文件读取、标签生成、实际key、TEST、梯度/参数更新、选模 |
| `checkpoints.py` | 创新2 payload构造、CPU权重副本、临时文件替换保存、SHA/schema/raw-config/state/wiring严格加载 | 指标、数据生成、训练、报告、远程运行 |
| `evaluation.py` | 绝对bit ACC/BCE、TRAIN基线指标、结构模型hard报告 | checkpoint实现、训练；旧加载路径仅兼容导出 |
| `two_round_diagnostics.py` | 既有TRAIN/DEV只读诊断 | TEST与优化；本次不挪入报告以制造大模块 |

依赖：`training → reports / checkpoints / evaluation`；`reports → models / torch`；`checkpoints → config / models / artifacts`。
checkpoint不反向依赖evaluation/training。模型、密码和数据层不依赖新增报告或checkpoint模块。

## 具体迁移与兼容

- 将训练器中的模型说明、四格覆盖/冲突统计移到`prediction_reports.model_metadata`，纯返回字典或None；训练器仍在原位置决定写`model_metadata.json`。
- manifest的实验角色字符串归`prediction_reports.experiment_kind`，字段及取值不变；它不是模型调度器。
- 原`training._save_checkpoint`私有实现归`checkpoints.save_checkpoint`，原`evaluation.load_checkpoint`实现归`checkpoints.load_checkpoint`。
- `evaluation.load_checkpoint`与`evaluation.CHECKPOINT_SCHEMA`继续作为同一实现/常量的兼容导出，不增加新的加载分支。
- 选中两轮神经模型的诊断仍在strict selected load之后、TEST标签之前显式调用，未把诊断调度藏进报告helper。
- 不修改旧run/checkpoint。新run的source身份自然包含新模块并与旧source SHA不同，不声称源码或torch归档字节不变。

## 验证设计

1. 在修改前保存13组模型/模式、每组4更新的CPU合成行为记录；修改后相同输入、合成标签、seed再运行并比较。
   包含MLP direct/公开坐标/公开残差、logic、structured、unrolled三模式、table/local/vector、aligned/random circuit。
2. 比较best/last payload元数据及每个state tensor的dtype/shape/字节SHA、初始化、数据身份、选中epoch、所有指标、报告、诊断和progress（排除耗时）。
   没有用“退出0”或“文件数相同”替代行为比较；torch归档字节受保存路径影响，不要求归档SHA相等。
3. 新增持久测试：报告不改变输入、参数、梯度、模块模式、CPU/Python RNG；独立四格聚合期望；职责依赖检查。
4. checkpoint新增保存/加载、兼容导出、旧raw mapping SHA、weights_only/CPU、格式/类型/layout/有限值/接线拒绝测试。
5. 使用现有环境做Ruff/mypy/pytest及干净源码副本的8组CPU check/train/evaluate闭环；只用synthetic或null，不安装依赖。

证据保存在忽略的`validation/backend-boundaries/`，不是科研成绩，不包含旧正式checkpoint重放。
本次不是全仓库架构重写：CLI中的显式evaluate组装、数据/公共参考适配、config模型组合校验仍留原处；未为缩行强拆成转发层。

## 本地实际结果

- `training.py`从370行降至239行，`evaluation.py`从132行降至63行；新增报告159行与checkpoint91行。目的在职责归属，不以总行数减少作为成功标准。
- 13组重构前后合成对照完全一致，每阶段52更新/208呈现，0真实密码标签；不声称等价覆盖全部输入或CUDA。
- Ruff通过；mypy通过88文件；pytest **964 passed**。新增35项测试。两条PyTorch警告来自故意构造非法稀疏tensor的拒绝测试，不是正式checkpoint异常。
- 干净源码副本8组CPU check/train/evaluate均exit0，共16更新/64呈现；saved TEST指标与显式evaluate一致。仅synthetic/null工程数据，无安装或GPU实测。
- 与HEAD直接做AST比较：指标计算、严格加载完全一致，保存仅函数名称改变。模型、密码、数据、config、CLI、innovation1及原两轮诊断源码字节均未变。
- 独立只读审查未发现可证实新增缺陷；其无Git执行能力，旧源码对照来自已有干净副本，正式HEAD比较由主代理执行，二者证据范围不混称。
- 首次mypy发现新增测试通过模块隐式访问torch导致的类型问题；已改为本测试显式torch引用，最终全套复查通过。
- 未commit/push，未改旧run或重放正式checkpoint；此前未提交的parity、文档、papers等工作保留。

证据：`validation/backend-boundaries/{verification.json,before/summary.json,after/summary.json,ruff-final.log,mypy-final.log,pytest-final.log,clean-cli-final/verification.json}`。
首次干净副本与最终源码仅evaluation模块说明字符串不同，收尾字节校验发现后，在新路径用最终源码重复工程CLI核查；两份旧新工程记录均保留，无正式密码查询。
依赖AST测试是回归约束，不是阻止动态导入的安全沙箱；报告副作用未做CUDA RNG实测。
