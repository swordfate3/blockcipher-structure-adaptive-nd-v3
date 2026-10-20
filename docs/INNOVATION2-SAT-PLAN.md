# 更高轮次硬结构：单次TRAIN布尔约束拟合

## 预先确定的范围

用户要求推进轮数，沿硬布尔结构路线。继续固定目标MSB bit0，不看TEST后挑位。
5轮44个符号坐标直接枚举2^44不适用，6轮70个更大；新实现用公开DAG转CNF，
所有TRAIN样本共享同一组未知常量。真实key/roundkeys/trace不进入拟合器。
这是硬布尔参数学习，不是神经梯度，也不要求全密文/全key恢复。

公开结构准备检查尝试不含key值的明文单项式保守展开；4轮中间集合已出现9926项，超过8192工程上限，
未生成标签/拟合，未扩大上限。故本轮不继续单项式展开，改为保留紧凑AND/XOR图的SAT约束。
没有复制旧backup中的求解器或worker。历史r6 UNKNOWN不继承成当前成功，也不构成新批权限。

本机已有python-sat1.8.dev24/Glucose4，现场探测可创建后端；不下载或安装。
可选依赖声明`.[sat]`；没有该依赖时明确报错，不自动安装或切换后端。

## 运行合同（执行前冻结）

统一CLI `check/train/evaluate`，model_kind=`simon-hard-sat`，CPU，epoch1占位，0梯度更新。
config目录`configs/innovation2-sat/`：smoke / r5 / r6 / r5-null。

- 真实拟合固定TRAIN512、DEV1024、TEST16384，新data191901、key190502、init17、null191402。
- 每项预算120秒，从数据构建起计，协作检查；只一次Glucose4求解，不换seed/后端/重启。
- 先工程smoke和验证，再运行r5。r5需TRAIN/DEV/TEST均100%才进入r6。
- r6若超时UNKNOWN，无模型就没有预测指标，也不生成TEST标签；不自动重试。
- r5通过后允许独立r5-null对照一次，不论r6是否成功，最多三项真实/控制求解。
- r5失败/UNKNOWN/UNSAT或未通过精确门槛即不进入r6/null。
- SAT仅使用TRAIN选一个满足赋值，DEV评分不参与模型枚举，TEST在选中checkpoint严格加载和TRAIN独立硬电路复核后才生成。
- 不添加TEST约束，不用真实key校验候选；单key/单目标位机制探针，不称全key恢复。
- 没有匹配神经训练面板或多key确认，跨轮门槛含TEST观察所以属于探索性推进。

## 约束与超时语义

AND、XOR分别用等价CNF，样本内公开明文固定，样本间未知轮常量共享。
实际求解的变量/子句数和TRAIN行数写`fit.json`；不把变量/子句当神经参数更新。
计时器在预算到达时调用solver.interrupt，单次solve_limited返回None记UNKNOWN；
计时器退出时取消并join后再删除solver，避免使用已释放后端。
建模阶段也检查预算，耗尽不声称UNSAT。
Timer+native interrupt不是OS硬进程限额，不保证硬墙钟或断线存活。

拟合得到的赋值保存为同一HardSimonPredictor的±1状态；checkpoint严格校验tensor与离散语义。
生产模型只执行公开硬AND/XOR，没有SAT运行时依赖；显式evaluate仍用统一入口。
5轮以上独立轮效应没有强制完整密钥调度，所以找到的是兼容目标位的预测常量，
可能不对应合法主密钥，也不要求对应；真实函数在模型族内，而失败不证明不可预测。
SAT且TRAIN独立复核通过后仍须未见TEST评估。UNSAT仅说明给定模型/标签约束无满足赋值，
UNKNOWN既不是UNSAT也不是50%预测。

本批本机CPU、不SSH、不GPU、不安装、不commit/push；不继承旧预算，不自动扩到7轮。
实际结果另行记录。
