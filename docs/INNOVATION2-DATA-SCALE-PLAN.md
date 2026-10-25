# 远程6轮数据量对照合同

用户确认“是”，授权本次必要源码/小配置/测试/文档本地commit/push，远端Git同步，
现有A6000环境GPU训练四项并SSH/SCP取回结果；不安装升级。停止低轮科学拟合。

固定任务：SIMON64/128六轮、未知固定key、明文→绝对密文bit0；
现有simon-unrolled width32/full/direct，1089参数，BCE/Adam lr.001/clip1，无真实key/state教师。
四项顺序small-real、small-null、large-real、large-null，各一次，不重试/续训/按分数跳项。
small TRAIN4096/epochs512，large TRAIN65536/epochs32，batch256，共同各8192更新/2097152次呈现。
四项32768更新/8388608次呈现。单项max_seconds1800为协作上限，不是硬截止，不承诺SSH断连存活。
DEV8192/TEST65536完全相同，data193501/null193502/key190502/init17，cuda:0为选中物理GPU的映射。
选择空闲显存≥24000MiB且util≤30%的GPU，启动前每次重新检查，不占用不合格设备。

新增可选train_pool_rows65536：按pool+DEV+TEST生成全局唯一输入，smallTRAIN取large前4096，
unusedpool不生成标签。None保持旧输入划分。null TRAIN使用相同split流前缀，heldout同标签。
运行前只做无标签prefix/heldout/交集检查。各real实际生成TRAIN+DEV+TEST个标签：
small77824、large139264，合计217088次标量真实加密构造调用/139264不同明文；null无真实加密。
训练重复呈现不是新查询。无公开零key参考（现仅≤4轮），使用TRAIN-only常量基线。

同模型四项初态相同，样本/shuffle RNG seed相同但不同长度randperm不等于逐更新同样本顺序。
固定相同更新量并非固定同样本曝光：small每样本512遍，large32遍。
新增dev_every_epochs：small16/large1，因此都每256更新检查DEV一次，共32次选择；
末尾对best另计算一次DEV报告，不计选模次数。旧默认1保持旧行为，旧checkpoint按原raw配置SHA加载。

每臂最低DEV BCE选best、平局最早，strictload后才生成TEST标签。CLI自动保存TEST不称访问强制封存。
批次完成先只读取TRAIN/DEV写dev-decision，再看最终TEST。预定报告：large-small DEV ACC/BCE差值，
每臂TRAIN-DEV差、real-null差；探索信号每真实臂DEV≥55%、比其常量及null均高≥3pp且BCE优于常量、null45–55%。
TEST同条件单独报告，不选参续训。若大数据无改善，只说明等8192更新下无获益，不证明数据永远无关；
若改善，仍需独立key/init确认，不把单面板差值当普遍因果或显著性证明。

统一CLI check/train/evaluate；本地Ruff/mypy/pytest与干净副本null工程闭环后发布。
远端仅fetch/checkout明确commit到全新G:\lxy\shared-m-cipher-nd-runs\<id>\repo，无commit/push，
旧目录dirty即停、不reset/clean。现有Python依赖可用机器本地适配器，仅转交原CLI参数不承载科学算法。
先CUDA null小训练确认环境，再四项正式训练；技术失败停止剩余不重试。
产物runs/<id>/，取回runs/received/<id>/，核文件字节SHA/源码配置身份/计数退出，不本地重算指标/标签/模型。
工程smoke6轮null8/4/4，pool16，batch4/epoch2/DEV间隔2，共4更新，非科学臂。
发布只明确stage代码/小配置/测试/文档，排除papers/backups/runs/validation/outputs/环境/权重/凭据。
当前已积累未提交模型依赖和其测试/文档一并发布以形成自洽源码；不传科学任务包，不恢复旧远程编排。
