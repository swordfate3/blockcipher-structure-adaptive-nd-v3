# 六轮公开可逆坐标MLP：固定四项方案

用户在六轮MLP与轮展开均近随机后说“好，推进”，先完成新机制推导、实现与本地工程验证。
解释四项预算并要求本批发布/远程运行确认后，用户再次“推进”，确认下述四项的发布、GPU运行与取回。
授权限于本批，不继承旧批、也不自动扩大；计划冻结时尚无本批科学分数。

## 1. 新假设：先消去一个已知的非线性输入映射

SIMON的公开轮函数为
`F(X)=(ROL1(X) AND ROL8(X)) XOR ROL2(X)`，采用32bit字、64bit明文MSB顺序`(X,Y)`。
定义无key的公开坐标变换：

```text
phi(X,Y) = (U,V) = (Y XOR F(X), X)
phi_inverse(U,V) = (V, U XOR F(V))
```

直接代入可见互逆，保存全部64bit信息。对均匀明文，坐标仍均匀；
它不是增加标签、挑选偏置样本或泄露秘密，也不改变待预测的最终绝对密文bit。
真实第一轮状态为`(U XOR K0,V)`，其中K0从未输入模型或用于变换。
**U不是实际X1**，也没有从样本外获知K0。新坐标下仍要学习未知第一轮密钥平移及后续真实轮的复杂效应，
不是已经攻破/移除一个带key轮，更不是保证六轮会变容易的定理。

假设：把公开第一轮非线性从MLP必须发现的表示中显式提取，可能改善固定预算的端到端拟合。
竞争解释：未知xor平移和后续非线性仍可能足够困难，或者这种坐标改变了优化几何却没有帮助。
均匀分布保持不意味着两种目标函数对有限ReLU MLP同样易学。

## 2. 与旧方案的差别及边界

- 不是公开零key**输出残差**：不计算六轮参考输出、不对logit作sign恢复；标签仍绝对bit。
- 不是展开隐藏轮状态：模型只在输入处作一次确定公开可逆变换，之后普通MLP，不传递真实轨迹。
- 不是对U作局部目标监督：没有U标签、true F教师或其他cipher输出bit标签。
- 不搜索或枚举K0、不恢复完整key；MLP权重可以从TRAIN学习key派生效应，禁止的是实际秘密注入。
- 同64→128 ReLU→128 ReLU→1、**24961参数**、层名/state key和init RNG消耗均保持。
  新模式`prediction_mode="public-coordinate"`仅允许prediction/mlp/simon/full、rounds2..44。
  接口支持这些轮数不等于已验证其科学学习能力，本提案只跑六轮bit0。
- 参数量/初态匹配比上一跨架构比较更可控，但输入非线性导致有效函数类与优化几何不同，
  不是相同函数类下的纯计算加速，也不做单一原因的机制证明。
- 独立实现于`models/prediction.py`，只依赖Torch与已有BitMLP；生产模型不读cipher/data/key。
  用独立标量公开位运算及合成往返测试核位序，无作者代码执行或新颖性声明。

## 3. 固定四项科学预算

顺序direct-real、direct-null、coordinate-real、coordinate-null，各一次，不看某项分数跳项。
全部使用MLP128、SIMON64/128六轮、目标绝对密文bit0，full输入，固定未知key190502/init17。

| 设置 | 提案 |
|---|---|
| TRAIN/DEV/TEST | 65536 / 8192 / 65536，train_pool_rows65536 |
| data/null seed | 193701 / 193702，新面板 |
| batch/epochs/DEV间隔 | 256 / 32 / 1 |
| 优化 | Adam lr.001、clip1、最终bit BCE |
| 每项更新/呈现 | 8192 / 2097152 |
| 合计 | 32768更新 / 8388608呈现 |
| 设备 | 远程A6000 eligible physical GPU→cuda:0 |
| 时间参数 | 每项1800秒协作上限，不是硬截止 |

四项网络参数初始SHA应完全相同，明文/shuffle相同，真实/null分别标签配对。
秘密标签构造每真实臂139264次标量真实加密，合计278528调用/139264不同明文；null无真实加密。
坐标预处理只是公开函数计算，不是秘密oracle查询；等更新不意味着等FLOPs/运行时间，需分别报告。
只检查明文身份，全局唯一且与前两批large及七个已列高/浅轮面板零交集，不再生旧标签，不声称所有历史面板无重合。

每项32次DEV选择，最低BCE、平局最早。strict selected load后才生成TEST标签。
外层先只提取TRAIN/DEV写dev-decision再展开最终TEST；CLI自动保存TEST，不称访问强制封存。
主要对照为coordinate−direct的DEV ACC/BCE和TRAIN拟合，不能用微小相对提高替代绝对信号。
每模型探索条件：DEV≥55%，比本模型TRAIN-only常量和paired null均≥3pp、BCE优于常量、null45–55%；TEST同条件单列。
这不是99%高精度标准或完整44轮攻击。若坐标模型有信号，只建议另授权新面板/init/key确认。
若无信号，报告本机制在预算下失败，不自动增加坐标轮数或扫网络宽度。
技术失败停止剩余、无重试；科学失败完成固定四项后关闭。不得按TEST选配置/延长训练。

## 4. 本轮实际工程范围

正式配置目录`configs/innovation2-public-coordinate/`为本批固定四项，配置本身不是已训练成绩。
smoke目录同名前缀-smoke，direct/coordinate两模型均null8/4/4、pool16、batch4、epoch2、DEV间隔2、
data193711/null193712、CPU/60秒，各4更新，仅工程，不生成正式真实key标签。

验证公式/逆变换、dtype/shape/模式边界、网络参数与RNG、manual forward、strict损坏state拒绝/旧checkpoint、
null禁加密及TEST先load时序；Ruff/mypy/pytest、干净源码副本两模型check/train/evaluate、独立只读审查。
工程准备阶段未安装、GPU、SSH、commit/push，未重放旧正式checkpoint/TEST。保留main与papers/等无关工作。
用户现已确认本批发布/远程GPU四项运行和取回；先Git同步/核clean与源码配置SHA、实时GPU门槛、CUDA null预检，
再执行远端已同步CLI。无科学源码/任务包SSH上传，无环境安装升级；接收只验保存JSON/字节/计数/退出，不自动重算科学标签/指标。
