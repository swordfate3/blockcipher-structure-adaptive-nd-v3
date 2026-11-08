# 三轮SIMON：两种32位激活明文空间

用户在明确方法推荐后授权“试试”。仅执行以下固定10项，沿用本地限定发布→Windows exact Git同步→指定GPU CLI→完整取回，不安装、不上传科学源码/任务包，不扩大候选或正式重试。

## 假设与两种方案

明文打包高32位X0、低32位Y0。A=`high32-zero`：X0固定0，Y0随机；B=`low32-zero`：Y0固定0，X0随机。均有32个激活位；同data_seed使用相同去重uint32序列分别放入活跃半，非先64位采样再截断。模型仍输入完整64位。

SIMON X(t+1)=Y(t) XOR F(X(t)) XOR K(t)，Y(t+1)=X(t)。A的首轮X1=Y0 XOR K0，延后随机变量非线性交互，可能比B易学，但次数上界不能保证神经网络成功。与两半全随机比较分离维数/位置因素，不能从相同维数推断复杂度相同。仅此两半空间，不枚举任意mask。

各空间五臂：C0；C0 XOR C8；C0 XOR C8 XOR C16；C0 XOR(C8 AND C16)；独立null。先A五臂再B五臂。固定未知key不变，非密钥激活/恢复，无内部状态/密文teacher，模型不读key。

## 模块边界与兼容

config新增严格plaintext_space枚举默认full；默认序列化省略字段、旧config/checkpoint SHA不变；非full限定prediction/simon/mlp/full/direct。data/prediction.make_inputs负责无标签32位采样/去重/配对/划分，labels不改。training只传参和记录SIMON真实输入空间来源，不向synthetic/circuit错误添加均匀明文来源。模型、密码、checkpoint schema、DEV选模及strict load后TEST标签时序不改，evaluate沿已有input_panels路径。新tests及两目录各10配置；只发布本轮3源码+1测试+20配置+本计划，共25文件。

独立审查发现旧身份测试以移动HEAD作历史基线会在提交后失败，发布前改为固定d6c3c5d的3组字典/SHA fixture、无Git运行时依赖；有限代表不冒充全部历史config比对。无关dirty不发布。

## 冻结预算与设备

- 正式max10fits/retry0，三轮固定、MLP64→128→128→1共24961参数，full/direct，Adam .001/clip1；不因TEST改候选或增加epoch。
- 每臂TRAIN65536/DEV8192/TEST65536，pool65536/batch256/32epochs，每epoch DEV；8192更新、2097152呈现、32 DEV。总81920更新/20971520呈现/320 DEV。
- data195601/key190502/init17/null195602，各臂相同；两空间x不同但活跃uint32一一对应。数据新采样规则不等于新独立key/确认。
- 八真实臂每139264个标签、合计1114112次encrypt，按实现/split推算，不是独立逐调用计数；两空间各139264不同明文，P=0若被采到可能跨空间重合，不能保证两空间并集恰278528；每空间内与各split严格去重。两null共278528随机bits/cipher0，null共享同一流，不是独立确认。
- GPU物理1 UUID GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5→cuda:0，逐臂free>=24000MiB/util<=30%预检，无fallback不强占。现有环境无安装。
- 每臂max_seconds1800，总18000s协作训练预算；连接/构建/同步/取回另计，非硬时限/断线保证。任一身份错、资源不合格、非零/缺exit、预算停即停止新启动，无正式重试。
- 本地工程：实现代理5个CPU/null fits共8updates/40presentations；主验收仅一次重复这组测试最多8updates/40presentations，加干净CLI十null共20updates/80presentations。全部工程生产encrypt0，不运行全仓真实cipher测试。

## 观测与解释

主观察相对TRAIN-only常数概率的TEST BCE改善，附ACC、TRAIN/DEV、最后在线TRAIN损失、null；按DEV最小BCE选epoch，固定tie不改。单bit公开结构参考只报告、不输入模型，组合不套错配参考。固定子空间上的标签**不保证平衡**，高ACC而不胜多数类不是成功。

A/B有优势仅说明当前固定背景子空间可学，不泛化为任意明文或跨key；未试其他背景。若均无信号停止，不追seed/位组；有信号亦先报告再另定背景/新key复验。此前TEST已见且CLI仍输出TEST，无技术封存，整个10项为探索而非独立确认。

取回完整runs至新runs/received目录，核字节SHA/commit/source/config/x身份/退出/计数/DEV选择，不本地load checkpoint、模型前向、标签再生或evaluate。完成或触停即CLOSED。
