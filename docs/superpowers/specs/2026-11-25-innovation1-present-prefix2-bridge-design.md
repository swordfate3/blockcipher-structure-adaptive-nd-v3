# PRESENT-80：理想二轮状态辅助区分2–7轮

状态：APPROVED。用户依次确认前向二轮完整状态语义、13fit三路对照，以及本稿32768/8192/32768组、max30/batch128/独立早停、GPU1总4小时、一次Git发布/SSH启动/取回、零重试的具体设计与执行权限。实现与正式批次尚未开始；下文“提案/待确认”保留设计演变语境，以本批准声明冻结的数值/边界为准。

## 1. 用户目标与已确认事项

用户希望先假设明文能够准确预测完整两轮输出，不训练该预测器，以真实二轮输出S2代替；以PRESENT为例，对总2、3、4、5、6、7轮使用创新1的Shared-M做区分，观察准确率。

已确认采用带对照13fit：联合(S2,Cr)六个、仅Cr六个、仅S2一个。分类单元为同key四对样本组成的一组，真假各半，主准确率=正确组数/TEST组数，随机基线50%。这不是上一批绝对位预测任务，也不把SIMON32高8位100%说成PRESENT完整二轮预测器已经成立。

## 2. 方案选择

- 简单6fit联合输入：成本低，但S2自身可能一直泄露正负类信号，不能说明最终密文是否被利用。用户未选。
- 13fit三路对照：本轮选择。按总轮数同时列Cr-only、S2-only、joint，观察联合增益是否超出任一单路。
- 训练实际两轮预测器或在密文末尾回退两轮：不同问题，未选、不做。本轮是真实S2作为理想上界，不是假装能获得未知秘密的现实能力。

## 3. 关键科学边界

1. 数据侧用完整PRESENT-80算法生成S2与Cr；模型仅得到声明的输入，不得到P、key、roundkeys、真实标签教师、内部调度或其它状态。
2. S2定义为当前API `present.encrypt_many(..., rounds=2)` 的完整二轮缩减密码输出，含第三个轮密钥的末白化。Cr同样含第r+1个轮密钥的末白化。不是含糊的未白化trace。r2时S2=Cr，联合输入是同状态重复，须明示；不得double-whiten继续计算。
3. 正类每对初始明文关系P'=P XOR 0xD00000；负类每对P与P'独立均匀采样，二者都经过真实密码运算。两类统一计算S2/Cr，不能一类真实一类随机状态，不按标签改变前端流程，不筛选预测正确样本。
4. 每组四对同key；提案本轮全周期固定一把新合成80位key，正负/三split/六轮完全相同，吻合单key预测器假设。不采用每组新key的更强oracle跨key条件。这是主控建议、待确认。
   理想状态oracle是明确更强的输入条件：相邻的白化输出S2与C3之间存在可还原部分轮密钥的公开关系，因此不能以“未给key字段”声称模型在信息论上没有秘密相关信息。本轮有意允许声明的理想状态输入来测上界，不实现密钥推导/恢复，不将该输入视为现实攻击者已具备的能力；模型可以学习输入的关系，不额外注入实际轮密钥。用户确认设计时需看到此限制。
5. 三路共享同一批P/P'/labels/group和split，所有r共享同批初始样本与key；每组四对为唯一统计单位，不能把8个加密块、4个pair或64个bit当作独立样本。S2-only只训练一次并跨r显示同分，不能假装六次独立结果。
6. 看到joint在7轮高分只能说理想S2辅助分类高，不能自动称7轮密文本身可区分或加两轮成功。必须同时看joint相对Cr-only及S2-only；即使有增益也只是有限面板探索，不证明现实攻击。
7. 本轮只考查一个cipher的Shared-M适配骨干，不是创新1原四任务迁移共享效果复现，也不训练额外专用专家D。

## 4. 数据、模型与训练提案

### 数据与持久化

- TRAIN32768组、DEV8192组、TEST32768组；每组4pair/8块，各split标签严格平衡，正负类统一公开采样/洗牌流程，group identity显式区分split。
- 每个P/P'只走一次PRESENT前缀轨迹，物化r2..7六个缩减轮输出；S2就是r2那一份，不额外计算或保存重复状态。跨split不复用同组；检查并拒绝跨split完整原始pair的意外重复，而不是按模型/标签筛选样本。碰撞重采样只能在加密前发生且记录，不生成加密失败后的科学重试。
- TRAIN为固定面板按epoch重放，不称在线新样本；每个模型相同epoch使用同一组排列，独立参数/Adam。DEV按组独立固定；TEST仅在13份best全磁盘严格恢复后生成/评分，标签访问约定不称安全封存。
- 新private key仅数据闭包采样一次，用完丢弃，不保存key/roundkeys或可恢复其内容的日志。公开采样seed20261125、排列seed20261126、初始化seed31（13fit同构同初值，均fresh参数）。不得复用旧权重、NPZ或key。
- 每split只在内存保留一份6时刻uint64密文、标签和无秘密组identity；**不落盘原始P/P'、S2、Cr或跨轮状态面板**，以免产物直接包含可还原轮密钥的相邻状态。保存seed/配方/计数/面板hash及TEST每组labels与13模型对应logits，以支持结果溯源和已保存配对统计；没有私钥也不能据此声称可复现该密文面板。取回不重算指标。权重可能学习到固定key相关效应，这不等同保存key字段，也不承诺无法从权重推断任何秘密。

### 模型

复用创新1Shared-M的local/word/nibble公开邻域、public13 PRESENT结构卡、top2路由、GN8/96通道、残差kernel3/5/7/9/11、SharedReadout四pair聚合。PRESENT结构卡固定，所以给定参数下各样本路由一致，不称按样本自适应。

只改变输入适配：原每pair的3路[C,C',C XOR C']扩成6路[S2,S2',delta2,Cr,Cr',deltar]，邻域9点的首层Conv输入27→54，后部不改。仅S2屏蔽后三路为0，Cr-only屏蔽前三路为0，joint六路均有；13fit同结构/总参数/初值，报告实际参数计数，也注明单路有效连接数较少，不宣称严格同有效容量。S2-only不接受r或最终Cr内容。

不改旧InputRepresentation/GeometryStem的固定3通道API。子owner用同一邻域索引对两组3路分别调用现有gather再拼接，并在本owner构造扩输入stem，复用现有Residual/SharedReadout/PublicRouter，不复制公共网络或改旧实验前向。

### 优化和选模

- batch128组，max30epoch；每网独立DEV BCE严格最低、同分最早选best。
- core EarlyStopping预热10epoch、patience8、min_delta1e-5，独立于严格best；最早18epoch结束，到30为EPOCH_CEILING。
- FP32 Adam lr0.001/betas(.9,.999)/eps1e-8/decay0/clip1，禁AMP/TF32，无额外正则/teacher/预训练；同一Meter覆盖全部13fit。
- 不额外训练MLP/专用网络，不额外seed/key/轮数，不因50%或100%自动增减数据/改差分/改输入。
- 执行顺序S2-only一次，再按r2..7各Cr-only/joint；每fit独立早停。13best全部严格磁盘校验/身份恢复完成后才进行任何TEST评分，完整结果统一终分。任一fit/restore/TEST/save失败无部分终分冒充完成。

## 5. 指标与解释

主表：总轮数、Cr-only正确组/总组及ACC/BCE、共享S2-only正确组/总组及ACC/BCE、joint正确组/总组及ACC/BCE、joint−Cr、joint−S2百分点差。

另存逐组配对正确性四格表（joint对/错×对照对/错）及实际epoch/DEV选中epoch/stopreason；不引入额外bootstrap/统计优化框架。先报告描述性差异，不把一个key/一个seed称普适规律。若后续需要显著性或独立确认，另定相应面板/多重比较，不事后挑轮次冒充预注册单一结论。

若joint≈S2-only但Cr-only≈50%，结论仅是相对S2-only未观察到额外准确率收益，不能说区分轮数扩展，**也不能据此因果判定训练好的joint忽略Cr或主要使用S2**。三路是独立训练的网络，同准确率可以判对不同组；本轮不增加针对同一个joint模型的输入遮挡/置乱前向，因此不做模型依赖归因。r2重复输入的增益可能来自表示冗余/优化，也不能称桥接。若joint同时优于单路，仅为理想辅助信息下的探索信号，不等于实际预测器能维持该效果。

## 6. 公共算法验证与精确计量提案

现有present.encrypt_many是标量Python参考实现，round1..31、每轮保留pLayer并含末白化；拟新增纯numpy有界批前缀实现，依赖只在cipher层，不能依赖采样/模型/训练或暴露key。公式由规范/独立源码静态核，开发时禁止真实加密执行或benchmark。

规范来源：Bogdanov等作者论文《PRESENT: An Ultra-Lightweight Block Cipher》CHES2007，图1/§3/附录I，https://crypto.orange-labs.fr/papers/ches2007-450.pdf ，公开全零31轮KAT5579C1387B228445。SageMath独立源码 https://raw.githubusercontent.com/sagemath/sage/develop/src/sage/crypto/block_cipher/present.py 可交叉公式，但默认doFinalRound=False，比较必须明确True，不能漏末pLayer。作者规定逻辑位串，不规定Python low-word-first三元组，该wordorder是本项目契约。资料核对只读，不是实际KAT已过。

唯一正式批次内先做14block公开门禁：
- 公开80位零key、P0、完整31轮KAT：原标量1块、新批接口1块，共2。
- 固定同一公开非零80位key=0x00010203040506070809、P=0x0123456789ABCDEF，r2..7各原标量1块与批接口1块做独立路径等值校验，共12。接口low-word-first=(0x06070809,0x02030405,0x0001)，只读按整数位移检查该拆装，不执行加密。r2..7内部实现一致性不冒称6条官方KAT，非零用来增加wordorder/roundboundary覆盖。任一失败停止，不采新私钥/训练；修后重跑须新批准。

科学面板共32768+8192+32768=73728组，每组8块、每块输出6个round结果，共 **3538944个计费block-output**，再加门禁14，real_blocks上限 **3538958**、null0。必须明确一条前缀轨迹虽然复用轮运算，但每个已生成round输出保守收费1block；另记录独立轨迹589824、roundstateoutputs3538944，不能把两者混报。失败批计量先reserve后调用、验证响应再complete，不退款/漏记多输出。

设13fit实际epoch和为E（最多390）：
- updates=256E，最多99840。
- forward group rows=40960E+32768×13，最多16400384。
- TRAIN呈现=32768E，最多12779520组。
- 每epoch固定256更新、64DEV批，TEST每fit256批；四pair不重复计成四个group forward row。

## 7. 资源、发布与停止——均待用户一并批准

主控提案：物理GPU1（既有A6000 UUID），总wall4小时/CPU16小时、RSS8GiB、产物512MiB、单checkpoint16MiB。一次限定发布/远端Git exact checkout/正式批次/结果取回，无安装，无额外真实数值预热/吞吐测试/微训练，正式retry0。预算是协作Meter/UTC期限而非OS硬隔离。

SSH远端写入仍只G:\lxy\本轮新目录，使用现有唯一前台启动脚本最小分派，不恢复旧编排/新调度器。保持SSH至结束，分别立即记录原生exit和本地wait实际结果；缺任何退出证据保持UNKNOWN，不由后续stdout/PID缺失补0。旧holder消失问题未定位，不宣称现有记录机制已修复；若运行期间控制连接未知，停止新启动/不自动重试，取回已保存证据。

首FAILED/STOPPED/UNKNOWN即停止整批，不换卡、不缩batch、不延时、不追补fit。3个输入类型不做适应性轮数晋级，六轮固定全测但任失败止；时间不够如实标未执行，不把上限当成功保证。关闭后剩余额度失效。

## 8. 模块职责与实现边界

- 新纯cipher批前缀模块：PRESENT密钥调度、轮公式、选择物化缩减轮输出；输入受限uint64/显式80位key/目标轮数，输出同shape有限数组，无RNG、模型、文件或资源操作；不改旧标量行为。
- 新innovation1/prefix_bridge子owner：config数据契约；data唯一样本/秘密闭包与显式计量；model六通道表示及组件组装；runtime数值适配和checkpoint；training仅装配公共fit、13best门禁、按保存时序调度；metrics显式组ACC/BCE/配对计数；report只展示保存值；CLI解析/claim/错误呈现。
- 复用已有通用direct_block.fit（BatchPanel不限制输入维数、N×1标签可用），其公共单fit只增加一个可选完整batch-update回调，默认旧Engine路径完全保持。回调路径由fit仅预留/完成一次optimizer update，交给现有innovation1.TorchRuntime.update完成forward/BCE/backward/clip/step/同步，其中训练forward由原runtime计一次，禁止双计。新owner的Runtime仅inference_only初始化、登记当前一slot/调用add_adam、适配输入与public卡/域校验/predictFP32，禁止构造默认五slot或复制Adam实现；不能把包办更新的TorchRuntime.update硬塞原分段Engine.update而造成二次forward或漏计。
- 严格checkpoint复用innovation1.checkpoint.encode及single_train.weights.decode_state，用本owner显式构造新method/round/mask/S2boundary/recipe/source域身份、CPU模板及全部fixedgeometrybuffer；不复用旧五slot identity。numerical shape由本owner适配，禁止改旧cube宽度白名单/伪装plain32。共享fit接口改动单列TDD旧路径回归，不暗改其它方法语义。
- Source identity必须覆盖实际导入的core＋innovation1生产＋复用direct_block生产闭包，新owner不能漏入原显式清单；tests/cache不当生产文件。不用扩大到兄弟未用实验替代真实闭包调查。
- Config放configs，原始记录和分类输出data/<id>（本轮无原始状态面板），权重models/<id>，原报告result/<id>；汇总只在/workspace/result新目录，用保存值做方法图/曲线/结果图。不造runs/registry/workflow或新的远程取回框架。
- 基于已发布cb5e78c独立archive开发，主main/index/旧产物/无关脏工作不动；独立allowlist发布，不合入main冒充同环境。

## 9. 验证与交付

开发仅人工cipher oracle/private source/panel/runtime/clock/资源/FFI替身，模型可构造/参数shape核对，禁止真实密码、私密熵、NN数值forward/backward/Adam/GPU/SSH/安装/发布，直至对应设计和权限获批。TDD先RED后GREEN，implement-deep实现、独立code-review审查，主控fresh Ruff(--no-respect-gitignore --no-cache)/mypy/新owner显式tests/相关旧行为与默认fullsafe分别报告。

必须人工覆盖r2重复输入、13输入mask/label不泄露、共享组与不同模型无参数别名、S2-only无Cr/round依赖、whitening边界、14block门禁失败不取私钥、sharedstates只生成一次、首checkpoint写失败计量持久化、最后restore失败无TEST、TEST后臂失败无部分终分、实际epoch计数、超时/存储故障零重试与source身份闭包。测试guard封底层密码、数值alias和缓存句柄，不称通用沙箱。

主控取回实际data/models/result/必要日志核原字节SHA、源码配置身份、实际fit/epoch/消费/退出；不加载权重或解码科学NPZ重评。交付并列科学终态/nativeexit/SSHexit；区分设计、静态验证、KAT、有限TEST探索与独立科学确认。

## 10. 设计只读复核记录

- architect bd32fc9b：确认6通道应在stem gather/首Conv适配，原fit本身允许多维组输入，数值runtime整批更新与fit分阶段接口不能硬接；已写入可选batch-update复用方案，旧默认路径不变。指出原数据按round派seed/每组换key不适合本轮，明确新provider固定onekey/splits。
- deep-research af7b8885：作者论文/Sage核对公式、末pLayer/白化、KAT与wordorder边界；只读，无真实执行。14block提案不增加调用，r2..7改用固定非零公开key/P以增加拆装覆盖。
- builder-check a60f9fe0两个实质问题均已定案：原始相邻状态不落盘并明示oracle秘密关系；三路比较只报告观测增益，不做已训练模型因果依赖归因。此二项不增加fit或forward。用户随后批准本稿数值预算/输入条件与一次执行权限。
