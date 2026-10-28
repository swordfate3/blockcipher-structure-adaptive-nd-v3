# 六轮单bit：连续邻接与公开旋转邻接的残差CNN

状态：本地实现/工程验证阶段；**正式数据标签、科学拟合、GPU、SSH、安装、commit/push均未授权**。
本次用户明确要求换不同网络、尝试卷积识别结构关系；不自动承接已关闭批次许可。

## 1. 方法问题，不是继续小MLP调参

问题：保持固定未知key、六轮SIMON64/128、完整明文→绝对MSB bit0不变，多通道残差CNN能否学到有用关系？
公开旋转邻接是否比普通连续邻接更有帮助？不把单次阴性看作所有CNN不可能。

| 候选 | 隐表示/计算 | 优势假设 | 风险/区分方式 | 本轮决定 |
|---|---|---|---|---|
| 连续邻接残差CNN | 32位置×64通道，空间四tap(0,1,2,3)，6独立残差块 | 保留多维特征、卷积共享学习局部交互，残差提供信息/梯度通路 | 未直接匹配SIMON跨距，有限样本难发现深层Boolean关系 | 实现，对照 |
| 公开旋转邻接残差CNN | 同参数布局/通道/块，tap仅换成(0,1,2,8) | 更早组合公开轮函数使用的相对位置 | 改善也可能来自更快传播/优化，而非真实结构恢复 | 实现，主候选 |
| 全局attention | token位置+内容依赖全局交互 | 自适应远距关系 | 同时改变容量/计算/归一化/位置机制，不能干净隔离邻接因素 | 本轮不实现/不运行，未来另论证 |

这不是复现某篇高轮位预测论文，也没有新颖性结论。只有最终目标bit BCE监督；不注入真实key/轮key/实际状态/其他密文bit教师/旧解权重。
相对旧unrolled同时变了容量、隐藏维数、残差/归一化、非线性和跨层共享；跨历史改善不能归因单一机制。

## 2. 明确结构与参数

输入[N,64] float32二进制，按MSB reshape[N,2,32]，high与low为通道，center到±1。
1×1 Conv把2通道升到64，加零初始化的可学习[1,64,32]位置参数。
六个独立残差块，每块两次：逐位置跨通道LayerNorm→GELU→四tap空间卷积；block输出identity+branch。
末逐位置LayerNorm/GELU→channel-major flatten(64×32)→一个raw logit。
没有中间sigmoid，没有实际轮状态约束，没有池化丢弃位置，没有BN跨样本依赖。
**六块是网络深度，不对应六真实密码轮；两种CNN都不是Feistel仿真。**

卷积实现：gather按位置i在外、tap在内，索引(i+offset)%32得到[N,C,128]，Conv1d kernel4/stride4输出32位置。
MSB数值ROL_r输出i读取i+r；local等价右侧circular pad3、kernel4/stride1的普通互相关。
两者使用同一算子实现，训练参数同尺寸、同seed逐张量一致，持久化邻接buffer不同，所以完整state SHA本来就不同。
加载checkpoint后严格核对每层完整规范邻接，不能仅检查范围和shape。

设通道C，参数公式为：
- stem3C、位置32C；
- 每block两conv各4C²+C，两LayerNorm各2C，共8C²+6C；
- 末LN2C、读出32C+1；
- 合计 **48C²+105C+1**；C64为**203329参数**（已实现实测交叉核对），工程C8为3913。
这是旧unrolled1089参数约187倍，不再把小模型作为主要能力覆盖。

卷积理论MAC近似每样本6×2×32×4C²，C64约629万MAC，加stem4096/readout2048；不是实测FLOPs或耗时。
batch256时单特征图256×64×32×4=2MiB，单gather8MiB，12层约96MiB gather，不含其余激活/梯度/库workspace。
参数FP32约0.78MiB，Adam/梯度另计；没有GPU现场显存/速度声明。

位置参数与flatten允许固定key/固定目标打破平移等变约束，不是key输入，也不保证解决key效应。
即使没有位置参数，flatten也能区分位置；不能称位置参数是预测绝对bit的必要条件。
12次空间卷积后两邻接都覆盖整个32环，但传播路径/早期非线性交互不同。
flatten从一开始可汇总全输入；感受野不等于输出能否看到全输入。
若rotation获益，之后需非密码长跳/传播速度匹配对照才可更强归因，本批不自动追加。

## 3. 模块归属与不变行为

- `models/simon_conv.py`：卷积、残差前向、规范接线验证；只依赖Torch/模型输入校验，不读数据、cipher或配置文件。
- `models/prediction.py`：两显式选择`simon-cnn`/`simon-rotation-cnn`，不加registry。
- `config.py`：两模型限定prediction/simon/full/direct/rounds6/width8..256，bit0..63沿用统一编号；正式本批固定width64/bit0。
- `checkpoints.py`：新增返回类型与接线严格检查；既有schema、SHA绑定、weights_only和旧模型兼容不变。
- `prediction_reports.py`：只读说明邻接/维度/读出及非真实状态定位，不访问TEST或修改模型。
- 测试：独立卷积oracle、方向/wrap、梯度、参数配对、checkpoint篡改拒绝、null CLI时序。
- 不改密码、数据、训练循环、BCE/Adam/clip1、DEV选模、TEST生成时序、CLI和创新1。

## 4. 有界正式提案（未运行）

四项固定矩阵：local-real/null、rotation-real/null，`configs/innovation2-conv-structure/`。
SIMON6/full/direct/bit0，width64，key190502仅数据侧/init17，data194201/null194202。
TRAIN65536/DEV8192/TEST65536，pool65536，batch256，Adam.001，32epochs，DEV每4epochs。
每项**8192更新/2097152呈现/8个DEV点**；四项**32768更新/8388608呈现**，真实16个DEV候选。
每个真实项139264加密调用，两个合计**278528调用/139264不同明文**；两个null不调用密码。
同panel/shuffle/训练初态，完整state因接线buffer不同；等steps不等于相同初始函数、优化进度或实测墙钟。
每项3600秒协作预算，总14400秒上限计划；不预测实际耗时，不宣称硬GPU截止。
没有正式重试、第二key/init、追加lr/宽度或低轮科研fit；确认预算本批为0，需另定新面板验证。
正式执行仍需用户一次明确允许代码发布、Windows同步/预检、指定A6000 GPU、四项拟合和取回。
现场不可用/脏工作区/身份不符/缺exit时停止；不覆盖目录、不fallback，不恢复远程任务包。

## 5. 预定观察与决策

各项最低DEV BCE选checkpoint，平局最早；跨两真实候选取DEV BCE最低，平局local先。披露所有8×2次真实DEV机会。
先保存TRAIN/DEV选择决定，再展示TEST；当前CLI自动保存/输出TEST，不称技术封存。TEST不用于下一候选设计或扩大预算。
所有真实/null的selected TRAIN/DEV/TEST与末在线TRAIN/DEV分别报告，不把online BCE当last重评估。
推进到另批独立确认的探索门槛：DEV ACC≥55%、比TRAIN常量和配对null各高≥3pp、BCE优于常量，null45%..55%。
这是有限预算下筛选值得确认的实际效应的工程门槛，不是数学阈值或多重比较校正显著性；小于门槛的BCE差异只记探索观察。
TEST如不支持DEV，不作成功宣称；若有信号也必须新面板/init/key按研究问题另批准，不因TEST好分自动启动。
若TRAIN下降而DEV不变：拟合但未泛化；若TRAIN也不下降：未观察有效拟合，尚不能区分优化/容量/数据机制。
若两者相似：该预算未识别邻接优势，不否定其他CNN；若rotation更好：只支持本邻接归纳偏置，非状态/轮结构恢复证明。

## 6. 工程门槛与证据来源

本地无科学标签工程：两模型独立oracle/正反向、MSB脉冲方向、same trainable init/参数数、合法输入与篡改接线拒绝。
小null check/train/evaluate：各8/4/4、width8、2epoch/batch4→4更新/16呈现；只验证CLI/strictload/TEST时序。
人工16角点Boolean规则正控只作局部工程，不调用密码或实际key：固定预算300step/Adam.01，不等于六轮训练证据。
工程未通过不启动正式矩阵；失败先诊断，不根据其效果无限调预算。
两模型人工正控均固定300更新/4800呈现达到16/16，BCE local .70135260→.00001161，rotation .70302081→.00000706；
合计600更新/9600呈现、0密码调用。相同训练/评估16角点，无独立泛化含义；不称一轮密码学习。
两个干净源码CPU null CLI各4更新/16呈现，check/train/evaluate保存指标一致，合计8更新/32呈现。
无标签139264明文与14个明确列出旧面板不重叠，不声称核对所有历史数据。
验证日志置`validation/conv-structure/`；正式权限未给，当前没有本批科学结果。

架构参考已读：
- [Gohr公开train_nets.py](https://github.com/agohr/deep_speck/blob/master/train_nets.py)：reshape成位置/通道、1×1升维、两Conv残差块、flatten/Dense读出。
  其代码为SPECK差分区分、两密文输入、MSE/BN/ReLU/循环lr与更大数据；不与本任务协议相同，作者源码未执行。
  本设计独立实现，改为明文2通道、环形tap、LayerNorm/GELU和绝对单bit BCE；不声称论文复现。
- [He等Identity Mappings in Deep Residual Networks](https://arxiv.org/html/1603.05027)：已读取网络结构/恒等路径与预激活讨论；不是密码可学习性保证。
- PyTorch Conv1d在线页面提取失败，算子语义以现有Torch的独立前向/梯度oracle验证，不更改网络配置或装依赖绕过。
