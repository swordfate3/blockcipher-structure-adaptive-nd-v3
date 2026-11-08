# 六轮MLP：数据量与训练计算的六项固定诊断方案

## 问题与权限

上一批公开坐标没有改善泛化；原MLP仅测过65536 TRAIN。旧4096→65536数据量实验针对1089参数轮展开，不能替代MLP数据量实验。
本批检验：四倍不同训练样本在固定更新下是否改善持出；同大数据再增加四倍更新是否改善。
这是单key/init下的探索诊断，不是跨key确认或不可能证明，不以低轮成绩为前置。
先按用户“推进”完成配置、本地工程与预算说明；随后用户再次“推进”，现确认本批六项发布/远程GPU/取回。
授权只限本批，无安装或追加实验，不继承已结束的旧批授权。冻结计划时尚无本批科学分数。

## 固定模型与信息边界

全部SIMON64/128六轮、预定绝对密文bit0，完整64bit原始明文→128 ReLU→128 ReLU→1，24961参数。
model_kind=mlp、prediction_mode=direct，Adam.001、clip1、batch256，key_seed190502/init17。
不改变科学源码，不加入坐标变换、其他输出bit、真实key/轮密钥/中间状态或教师。
参数从TRAIN学习key派生效应允许，实际秘密注入禁止。随机标签独立且split域分离、不调用真实加密。

## 六项固定预算

| 条件 | TRAIN | epoch | 更新 | TRAIN呈现 | DEV间隔epoch | DEV选模次数 |
|---|---:|---:|---:|---:|---:|---:|
| A 基准 | 65536 | 32 | 8192 | 2097152 | 4 | 8 |
| B 多数据/同更新 | 262144 | 8 | 8192 | 2097152 | 1 | 8 |
| C 多数据/同遍数 | 262144 | 32 | 32768 | 8388608 | 4 | 8 |

顺序a-real、a-null、b-real、b-null、c-real、c-null；每项一次，不按科学分数跳项。
合计**98304更新/25165824呈现**；是上批四项32768更新的三倍，不等于三倍实际耗时/FLOPs。
每项1800秒协作上限（非硬截止），远程A6000经实时eligible GPU→逻辑cuda:0；不安装/升级。
每项技术失败停止剩余、无科学重试或CPU fallback；缺exit仍是未知，不虚构完成。

A/B每1024更新一次DEV，C每4096更新一次DEV，各8次。
刻意统一选择机会数，而不是统一检查更新网格；因此短预算未选中的早期短暂信号可能漏过，这是预先声明的灵敏度限制。
不能直接用上一批32次DEV选模的best做本批因果比较。本批A是真正同期对照。
epoch在线TRAIN BCE按每轮汇总，A每轮256更新、B/C每轮1024更新；
比较训练过程时注意各epoch平均窗口长度不同，不把在线BCE当selected/last TRAIN accuracy。

## 数据配对与查询成本

新data_seed193801/null_seed193802，DEV8192/TEST65536、train_pool_rows262144。
所有配置生成同一335872明文池；A TRAIN为B/C TRAIN前65536，DEV从完整pool之后开始，与TRAIN严格不重合。
B/C输入、标签与前8epoch的shuffle路径相同；DEV调用频率不同，但现有评估不改变参数/RNG，
只将此前缀匹配作为预期，不在没有逐更新保存权重的情况下宣称独立验证逐步相同。
六项模型初始state与trainable SHA应全相同。相同seed配不同长度randperm不意味着A/B逐更新同顺序。
同split的null前缀相同；B/C完整真实/null标签相同；接收以保存SHA核对应身份，不再生旧标签。

本地仅无标签检查335872不同明文，核与事先列明面板（七个13312行旧面板192801..193401，
以及193501/193601/193701各139264行）零交集，不声称遍历所有历史数据。
真实标签加密构造次数：A=139264，B=C=335872，总811008次调用，涉及335872不同明文。
重复TRAIN呈现不算新oracle查询；null无加密，unused A TRAIN pool不生成标签。

## 选择、阈值与停止规则

每项最低DEV BCE，平局最早；严格加载选中checkpoint后才生成TEST标签。
CLI自动保存TEST，非访问强制封存；外层先只提取TRAIN/DEV写dev-decision，再展开TEST报告。
每条件DEV筛选：真实ACC≥55%，比本条件TRAIN-only常量与paired null各≥3pp，BCE优于常量，null45–55%。
TEST同条件仅作固定报告，不根据TEST选条件/续训；这是探索门槛，不是高精度成功。

- B优于A且达绝对信号门槛/区别于null：支持固定预算下数据量有帮助。
- C优于B且达门槛：支持同大数据增加计算有帮助，不单独证明数据量是瓶颈。
- A→C同时变数据与计算，不做单因素归因；未包含“小数据/长训练”格，不能完整估计交互效应或一般缩放规律。
- 仅在线TRAIN损失降低、DEV近随机，尤其null类似：未看到结构泛化，不能断言完全记忆或容量不足。
- 小幅相对提高但不达门槛不算突破。六项无信号则关闭，不追加百万数据、seed、key、epochs或调参。
- 若有信号，只建议另批新面板/init/key确认，不自动执行。

## 工程闭环与产物

正式配置`configs/innovation2-mlp-data-compute/{a,b,c}-{real,null}.json`，统一CLI check/train/evaluate，无新入口/registry。
CPU null smoke目录同名前缀-smoke，A=8TRAIN/4epochs/8更新、B=32TRAIN/1epoch/8更新、C=32TRAIN/4epochs/32更新，
pool32、DEV/TEST4、batch4、各1次DEV，data193811/null193812，60秒。仅工程、不是科学成绩。
测试配置预算、nested输入与null标签前缀、初始参数SHA、DEV选择点；Ruff/mypy/pytest与三条件干净源码null CLI闭环。
证据在忽略的`validation/mlp-data-compute/`；不读取正式历史checkpoint或重算历史TEST。
接收仅核远端保存JSON/大小/SHA/源码配置/计数/退出，不自动evaluate/生成科学标签或模型前向。
保留main、papers/、其他owner refs与旧run；不stage结果/环境/backups/validation。
