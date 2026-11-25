# SIMON32/64：四个独立网络各预测两位的8位小试

**状态：APPROVED。** 用户对上述完整提案回复“是”：SIMON32/64第1–3轮，固定高8位，四个网络各2位/每轮，TRAIN262144/DEV32768/TEST65536，12fit、50上限早停、GPU1总3小时、一次限定发布运行/失败不重试。批准不是已实现、发布、GPU启动或科学结果。

## 已确认目标与推荐固定口径

单条32位明文P，预测对应减轮绝对密文C_r的8个预先固定位置。无差分输入/差分标签、无明文对、无密钥/教师/其他真实密文bit输入。每个网络都读取完整32位P，不是只读它负责的两位。拟保持已讨论的第1/2/3轮，三轮使用同一把新合成未知64位key和相同明文划分。

输出位置建议固定为**MSB0列0..7（密文高8位）**，属于SIMON输出高16位x半。第1轮的低16位y半只是复制输入x；选高8位避免只靠复制撑高分数。四个独立MLP按顺序负责(0,1)/(2,3)/(4,5)/(6,7)，最后拼接为8位；不在看过DEV/TEST后挑容易位，也不宣称拼出了完整32位密文。

网络选用上一轮推荐的普通前馈MLP，而非继续末态LSTM：32→128→128→2，两隐藏层ReLU，**每个20994参数、四个83976参数**。所有输入位能直接进入首层，检验输出任务拆分后的可学性，绕开末态记忆路径；这是候选机制，不保证成功。只运行四个分块网络，不额外加整32位或8位大头对照。每个round四fit，三round合计**12fit**。模型不得共享梯度/可变参数或把其预测当其它模型教师。

主准确率=四网络合计正确比特/(TEST条数×8)。另保存每网两列和8列逐位正确数、标签0/1、TRAIN冻结的8位多数串（tie0）。不要求两位或8位全部正确才得分；整8位全对可作明确附诊断，非主指标。无训练额外模型或配对统计比较。

## 本轮固定试水预算

- TRAIN262144条、DEV32768条、TEST65536条。32位域内无放回采样，跨split全局去重，不复用旧64位密文截断，不用旧cube差分族采样；四网及三round使用同一明文和原划分。
- 新key只在数据provider闭包持有，全部round和split共用，不写日志/产物或传模型。真实低轮标签每P每round生成一次，四网复用同一标签，不各自重复加密。
- 主加密=3×(262144+32768+65536)=**1081344blocks**；新增SIMON32实现先按原规范公共完整32轮KAT核对标量/批接口各1block，额外2blocks，总上限**1081346blocks**，无额外数值模型预检。KAT对应原始来源/字序/公式需独立核实，不从SIMON64输出截断。
- 各网max50epochs、batch256、Adam lr.001/betas(.9,.999)/eps1e-8/clip1/weight_decay0/FP32；各2位DEV BCE每epoch一次最早最低选best，warmup10/patience8/min_delta1e-5独立早停，最早18epoch。seed31+j（j=0..3）初始化，跨round保留对应j的相同初始化seed；同一epoch行顺序seed20261124。
- 最大updates=**614400**；最大forward_rows=**177733632**（按每模型样本行计，12fit各50×TRAIN/DEV加最终TEST）。实际成功updates=1024Σepochs，forwards=294912Σepochs+786432。科学attempt/completed分别核对，不要求用满cap。
- 12份本轮best全部持久化size/SHA/strictload通过之后才生成或读取TEST低轮标签。全部模型/round评分完整后一次性提交结果；中途失败保留实际消耗但不部分终分。完整终分冻结后可选report失败仍保留完整记录/原生非零，不改core事务语义。
- 仅GPU1 UUID GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5，单logicalcuda0/free≥16GiB，总wall10800秒/CPU43200秒，RSS8GiB、产物512MiB、ckpt16MiB；一次限定发布同步/前台SSH运行/取回，任一失败停，retry0，不自动换卡/加seed/新模型/第4轮。

这是从262k训练样本起步的探索配方，不把量级当充分性证明；若无信号，不自动扩大样本或预算。新32位域比旧64位小，跨split不去重会混入见过的P，须强制检查；仍单key/seed小轮数，不代表实际完整32轮可预测或可攻击。

## 模块边界和前置门槛

现有core仅SIMON64/128，SPECK32/64虽有实现但不是SIMON32；需要独立纯SIMON32密码模块，明确16位字/64位key/32轮/z0/key word order/高x低y，与密码数据/模型/训练无反向依赖。引用规范表和公开KAT先只读核实，开发用人工替身，真实KAT只在批准批次中。

方法逻辑放`direct_block/split8/`明确子owner，显式插件入口`cipher_lab_plugins.direct_block.split8.cli`，复用Meter/RunStore/父fit/cube_compare严格checkpoint/runtime及resource采样；不再把新算法堆进cube_compare的巨型mode分派、不新建通用registry/workflow。允许共用TorchEngine的显式输入width增加32且保留旧64/128默认，输出两位以实际logits/labels shape严格校验，不复制Adam/loss逻辑。新纯密码模块`src/cipher_lab/ciphers/simon32.py`不修改原simon64模块，provenance登记原规范/交叉来源。provider负责去重采样/秘密标签；模型只参数forward；训练只四网×三round装配/选模/重载/TEST时序；统计/报告只显式输入和保存值。允许原PS启动器增加本fixedconfig的plugin/无source/10800路径，不第二调度器。实现由implement-deep先TDD，代理无真实cipher/entropy/NN数值/Adam/GPU/RSS/SSH/Git/安装权限，不派子代理。

用户已确认预算，执行：writing-plans→隔离基线允许清单→委派实现→独立review与主控fresh人工/静态/干净CLI验证→一次Git发布/远端exactcheckout→公共2blockKAT和唯一训练批次→取回仅保存值核验。保留main/index/无关改动和旧产物，不加载旧权重或报告后重新前向/evaluate。缺真实退出证据保持UNKNOWN。

## 规范来源核对状态

已读作者仓库README、作者NIST2015正文与ASIC论文：32位分组/64位key/16位字、完整32轮、Feistel轮函数旋转1/8/2。原始eprint2013/404 PDF当前403，未直接读其向量表；公开转换副本cryptography.academy（明确自动转换非作者审定）已核对Table3.1/伪码/附录：SIMON32/64用z0=`11111010001001010110000111001101111101000100101011000011100110`，m4调度tmp=ROR3(k[i-1]) XOR k[i-3]，tmp XOR ROR1(tmp)，k[i]=k[i-4] XOR tmp XOR 0xfffc XOR z0[(i-4)%62]。KAT高→低key=`1918 1110 0908 0100`、P=`6565 6877`、C=`c69b e9bb`；接口low-word-first=(0100,0908,1110,1918)，block高x低y。独立公开实现inmcm的SimonCipher源码中32/64配置和反向z0存储可交叉核对，仍待只读research进一步KAT来源核对，不能假称原PDF可访问或已执行KAT。来源：https://github.com/nsacyber/simon-speck ，https://csrc.nist.gov/csrc/media/events/lightweight-cryptography-workshop-2015/documents/papers/session1-shors-paper.pdf ，https://eprint.iacr.org/2013/404 ，https://cryptography.academy/papers/the-simon-and-speck-families-of-lightweight-block-ciphers-2013 ，https://raw.githubusercontent.com/inmcm/Simon_Speck_Ciphers/master/Python/simonspeckciphers/simon/simon.py 。
