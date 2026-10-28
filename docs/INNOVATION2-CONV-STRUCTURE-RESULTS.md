# 六轮残差CNN：能拟合TRAIN，但未泛化；旋转邻接未见优势

## 1. 结论

用户“是”明确确认[四项固定提案](INNOVATION2-CONV-STRUCTURE-PLAN.md)的代码发布、Windows同步、A6000训练和取回后，四项全部完成，无正式重试。
**普通连续环邻接CNN TEST49.7543%，公开旋转邻接CNN49.6429%，null均约49.7%，没有可靠预测信号。**
DEV预定推进门槛全失败，按真实DEV BCE选local，两真实TEST BCE也都差于TRAIN-only常量。
本批关闭，不自动加宽、加数据、延长、换lr/init/key/attention或低轮追分。

与此前小型轮展开不同，本次多通道CNN的末epoch在线TRAIN BCE显著降到.0613/.0410，但DEV升到3.2141/3.7910，随机标签同样可强拟合。
这支持**训练拟合与泛化严重分离**的诊断，而非“网络完全没学动”。没有last模型重评估，不能据此声称末TRAIN准确率100%或逐样本完全记忆。
结果只覆盖这两个架构、一个key/init/panel及本预算，不证明卷积/结构先验永远无效。

## 2. 实际模型与同期对照

完整明文high/low为32位置上的2通道→64通道，零初始化位置参数，6独立残差块（每块两次逐位置channel LayerNorm/GELU/四tap卷积），末LN/GELU/flatten→单raw logit。
每项**203329参数**，无中间sigmoid，不输入真实key/状态/其他bit教师。
local offsets(0,1,2,3)，rotation offsets(0,1,2,8)。六块是网络深度，不是六真实轮状态仿真。
同参数布局/可训练初态、数据与shuffle、优化/steps；邻接持久buffer不同，完整state SHA按两种结构分组。
同初态不等同初始函数；邻接改变传播速度、交互路径和优化，不因rotation若胜就证明恢复密码结构。

固定SIMON64/128六轮/绝对MSB bit0/full/direct，key190502只在数据侧/init17/data194201/null194202。
TRAIN65536/DEV8192/TEST65536、pool65536，batch256、Adam.001/clip1、32epochs/DEV4。
每项8个DEV选择点，最低BCE选best、平局最早；跨两真实候选按DEV BCE，平局local先，共16个真实候选选择机会。

| 项 | best epoch/更新 | selected TRAIN ACC | selected TRAIN BCE | DEV ACC | DEV BCE | TEST正确/65536 | TEST ACC | TEST BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local real | 4/1024 | 51.8951% | .69241618 | 50.5249% | .69326784 | 32607 | **49.7543%** | .69342484 |
| local null | 4/1024 | 51.4526% | .69253322 | 49.7314% | .69347474 | 32574 | 49.7040% | .69355860 |
| rotation real | 4/1024 | 52.1088% | .69224200 | 49.9146% | .69335785 | 32534 | **49.6429%** | .69352752 |
| rotation null | 4/1024 | 51.2711% | .69263060 | 49.7559% | .69334370 | 32579 | 49.7116% | .69349534 |

两真实项共同TRAIN-only常量：bit1、平滑概率.5010223232677724；TRAIN50.1022%/BCE.69314509，DEV49.8901%/.69315376，TEST50.0565%/.69314696。
两null共同常量：bit1/概率.5014648213985993；DEV50.0610%/.69314790，TEST49.9283%/.69315567。
local相对rotation的DEV准确率高.6104pp、TEST高.1114pp，但二者都近基线，不能宣称普通卷积显著优于结构卷积。
没有做配对逐样本检验，也没有将结果反转/挑选TEST最高项后追加实验。

预定DEV探索门槛（ACC≥55%、比常量及paired null各高≥3pp、BCE好于常量、null45..55%）两个真实项均失败，尤其BCE也未胜常量。
门槛是推进规则，不是统计显著性定理；本批不是多key/多init独立确认。

## 3. 训练轨迹：不能用selected指标描述末模型

| 项 | 末epoch在线TRAIN BCE | 末DEV ACC | 末DEV BCE |
|---|---:|---:|---:|
| local real | .06133201 | 49.9512% | 3.21412960 |
| local null | .06595320 | 49.1333% | 3.03107468 |
| rotation real | .04097317 | 49.8657% | 3.79099992 |
| rotation null | .04025920 | 50.3418% | 3.51299484 |

四项best都在首次DEV点epoch4；这不确定epoch1..3是否更好（本批没在这些点选模）。
后期TRAIN loss很低且DEV BCE很高、ACC仍约50%，表明模型对未见样本产生了大量高置信错误；null表现相似。
末epoch在线TRAIN不是同一固定last参数的完整重评估，不由它推算最终TRAIN ACC，也不加载last另算。
与小unrolled的TRAIN BCE≈.693相比，当前网络显著更能拟合此训练集，但相对历史同时改了容量/表示/优化路径/面板，不能单独归因为卷积。

较合理的有限结论：**单纯换成这两个更有容量的CNN还不足以得到六轮预测泛化；本次没有识别出旋转邻接的收益。**
未排除：数据相对目标组合复杂度不足、训练协议、邻接/读出选择、缺少显式Boolean交互、多通道表示不合适、单seed偶然性等。
不能据“TRAIN能拟合”宣布优化和架构已充分解决，也不能据“DEV没信号”宣布所有结构学习不可行。

## 4. 发布、执行、资源和取回

运行commit：**`5fbf5871ac0087d9a6c9cd03b3439d47b6334435`**，GitHub refs核对成功。
只发布19个CNN必要源码/配置/测试/计划/provenance文件；本地未提交研究skill、parity、搁置MLP、papers等未夹带，不删除。
实际发布干净源码副本**1002tests/mypy91/Ruff通过**；本地完整工作区1064tests/mypy96包含未发布内容，分别报告。
Windows exact detached checkout、前后clean、44源码模块/4正式配置SHA相符；两CNN CUDA null预检各4更新/16呈现通过，没有安装/升级依赖。

批次：`conv-structure-20261028-170706`；本地UTC启动日志17:08:30到17:18:12（含SSH/CLI开销），不是严格同步远端时钟的GPU计时。
依次local-real/null、rotation-real/null，每项**8192更新/2097152呈现**，四项**32768更新/8388608呈现**。
共128progress、32DEV点（真实16），四exit0，全部预算完成，无重跑、续训或追加。
每项elapsed135.844/134.203/130.391/134.531秒，合计**534.969秒**约8.92分钟；这是CLI保存训练elapsed总和，不含所有启动/传输开销，也非精确GPU忙碌时长。
预定每项3600秒、总14400秒为协作预算，不是硬GPU截止。

两真实项各139264标签构造，总**278528标量加密调用/139264不同明文**；由冻结配置与既有调用链推导，不是独立逐调用探针计数。
null无真实密码调用。没有本地再生正式标签或模型前向。

主机DESKTOP-BBLPACJ，物理GPU1 UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`→逻辑cuda0。
每项启动前现场快照free48429MiB/util0%，满足≥24000MiB/≤30%；快照不是全程GPU监控。
所有远端生成物在G:\lxy，不上传科学源码任务包、不创建远程编排/服务、远端不commit/push。

远端原件：`G:\lxy\shared-m-cipher-nd-runs\conv-structure-20261028-170706\repo\runs\<id>-<local|rotation>-<real|null>`。
本地：`runs/received/conv-structure-20261028-170706-{local,rotation}-{real,null}/`。
每项12文件，共**48文件**，对远端清单的字节数/SHA逐一匹配；同时验manifest/config/源码/init/data/progress/selection/exit身份和聚合计数。
这不是独立重现实验，不加载best/last、重算科学指标、再生标签或做前向。

## 5. TEST访问、证据与下一步

外层先保存`dev-decision.json`（local，两个DEV门槛失败），后进行TEST数值验收/展示。
但CLI早已生成、保存并输出TEST；外层DEV阶段也读取完整results JSON/文件字节，只排除了TEST数值参与选择与展示。
**不是技术封存TEST，不声称决定前完全没有读取TEST；没有TEST-directed选择或后续实验。**
候选与预算在运行前固定，所有真实/null完整报告；任何新方案需要新的有界研究合同。

证据：`validation/conv-structure/{frozen-plan,remote-preflight,remote-inventory,remote-postrun,dev-decision,verification}.json`及四份SSH日志。
合成工程正控和1064测试只减少实现风险，不替代本次科学结果；以前的16角点拟合不写作六轮密码正控成功。

本批关闭。下一步优先比较“为何拟合与泛化分离”的可区分假设，是否需不同交互机制或采样/正则化设计，先有方法依据再分配预算。
这只是建议，不自动启动更大网络、attention、数据扩展或独立确认。
