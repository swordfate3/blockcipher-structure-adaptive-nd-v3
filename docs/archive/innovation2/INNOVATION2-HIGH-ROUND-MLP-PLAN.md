# 六轮直接MLP与轮展开：固定四项方案

用户在数据量对照完成后要求“推进”，最初仅准备、验证新的高轮比较配置；
在解释四项方案并明确本批发布/远程预算后，用户再次“推进”，确认本批发布、远程GPU四项运行与取回。
授权仅此四项，不继承上批许可，也不自动扩展。计划冻结时尚无本批科学分数。
不回低轮，不重放旧正式checkpoint/TEST，不新增真实状态/其他密文bit监督。

## 研究问题与限制

上一批现有轮展开候选在TRAIN4096与65536下都近随机。
下一步比较它与不经过逐轮隐状态传递的直接明文MLP，检验替代端到端表示能否在相同数据/更新量下学到信号。
不改源码模型，不加registry或新CLI；调用现有`mlp`与`simon-unrolled`。

- unrolled：width32，共享8→32→1 Tanh轮模块、连续sigmoid隐状态、公开Feistel/旋转与轮/位置学习条件，1089参数。
- MLP：全64bit明文→128 ReLU→128 ReLU→1，**24961参数**，无中间轮表示。
- 两者均输入明文、输出SIMON64/128六轮固定未知key下的预定绝对密文bit0，不读实际key/轨迹/其他标签。
- MLP参数更多、激活与深度/拓扑/先验/初始化均不同，**不是等参数或纯“取消压缩”的单因素对照**。
  即使MLP更好，也只能支持该具体替代候选在本预算更易学，不能唯一归因于sigmoid/隐状态压缩。
- 同更新与batch不代表同FLOPs或GPU时间；需要分别报告时间。一次单key/init也不能建立总体结论。
- 两篇AN/Fix只提供机制控制与查询/计算分别计账的实验思路，不是该MLP六轮成功的论文依据。

## 拟定四项与预算

顺序unrolled-real、unrolled-null、mlp-real、mlp-null，各一次；不根据某项分数跳过其他项。
共同CPU工程预备后，拟用远程A6000 physical eligible GPU→cuda:0：

| 设置 | 固定值 |
|---|---|
| 任务 | prediction/simon，6轮、target_bit0、full/direct |
| TRAIN / DEV / TEST | 65536 / 8192 / 65536 |
| train_pool_rows | 65536 |
| data / null seed | 193601 / 193602，新面板 |
| key / init seed | 190502 / 17 |
| batch / epoch / DEV间隔 | 256 / 32 / 1 |
| 优化 | Adam lr.001、clip1、最终绝对bit BCE |
| 每项更新 / 呈现 | 8192 / 2097152 |
| 总计 | 四项32768更新 / 8388608呈现 |
| 每项时间参数 | 1800秒协作max_seconds，非硬截止 |

四项共享同一新明文面板与shuffle、real/null分别配对。相同seed并不意味着跨架构同初态；
只要求同架构real/null初始化SHA相同。秘密标签构造每真实臂139264次标量加密，
合计278528次调用，139264不同明文；null不调用真实加密。
公开参考当前仅≤4轮，本批只保留TRAIN-only常量基线，不额外延伸基线实现。
新面板预检只计算明文身份，全局唯一且与上一数据规模批large面板及此前六高/浅轮相关面板零交集。
仅检验预列面板，不宣称与所有历史实验都无重合。数据seed也改变shuffle。

## 选模、判断与停止

各项32次DEV选择，每256更新一次，最小DEV BCE，平局最早；strict selected load后才请求TEST标签。
最后另评价选中模型DEV/TEST。外层先只读取TRAIN/DEV写dev-decision，再展开TEST。
CLI自动保存TEST，不称访问强制封存，不用TEST挑架构或追加训练。

主要描述两个真实臂的TRAIN拟合、DEV ACC/BCE差值与real-null差，不能用相对更好替代绝对信号。
每模型探索信号条件：DEV ACC≥55%，比本模型TRAIN-only常量及paired null均高≥3pp，
DEV BCE低于常量，null DEV45–55%；TEST同条件单独报告。不是99%可靠预测或完整44轮攻击标准。
若TRAIN高而DEV/TEST近随机，标为过拟合/泛化失败观察；若TRAIN也随机，不能单凭阴性定位优化或表达能力。
若MLP/null都记住TRAIN，也不能把TRAIN高分当真实密码泛化。
四项技术失败停余项，不重试；科学失败完整报告固定四项后停止，不加seed/epoch/数据或自动升轮。

## 工程状态与待确认事项

配置目录`configs/innovation2-high-round-mlp/`，每项是已实现CLI可读取的完整配置，不包含运行授权字段。
smoke目录`configs/innovation2-high-round-mlp-smoke/`各模型null8/4/4、pool16、batch4、epoch2、DEV间隔2、
data193611/null193612、CPU/60秒，共各4更新。仅工程，不生成正式密码标签。
先本地Ruff/mypy/pytest、干净源码副本两模型check/train/evaluate、无标签面板检查及独立只读配置审查。

本次没有修改科学模型/训练源码，也不重新安装环境。用户现已确认：
**允许发布这些新小配置/测试/文档，并按上述四项预算远程GPU运行和取回。**
远端新目录Git检出明确commit、核clean/源码配置SHA、
实时GPU阈值及必要CUDA工程检查，仍用既有统一CLI与环境，不上传科学源码包、不安装、不远端commit/push。
取回只验JSON/字节SHA/计数/退出/选模，不自动本地再生标签、重算指标或模型前向。
