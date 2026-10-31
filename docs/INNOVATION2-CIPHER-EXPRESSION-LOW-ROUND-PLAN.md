# 三／四轮密文表达式：固定降轮对照

状态：用户在六轮五臂结果后要求“你做低点轮次看看有没有变化”。本轮仅沿相同CLI/Git/Windows GPU流程改变rounds/name，固定r3、r4各五臂，不改模型或密码实现、不扩大表达式/seed/训练预算。限定发布本轮20配置+1测试+本计划，其他dirty不发布。

## 问题与比较

六轮组合标签约50%、后期过拟合。降轮检验这些特定表达式能否在浅轮更易学习，区分“仅六轮难度”与“该MLP对这些标签普遍学不动”。不是积分/是否平衡预测。此前6轮TEST已见，所以这是后续探索、不是独立确认。

每轮顺序bit(C0)、xor2(C0 XOR C8)、xor3(C0 XOR C8 XOR C16)、xor-and(C0 XOR(C8 AND C16))、null；先r3后r4。模型始终只看64bit明文、单logit，不给key/密文/内部状态。MSB编号0..63，四真实表达式完整明文域严格平衡，有限样本常数基线另报。

## 职责与保持

仅新增`configs/innovation2-cipher-expression-low-round/`与`-smoke/`各十份、`tests/test_cipher_expression_low_round_configs.py`、本计划。不改生产src。每份与六轮对应配置raw/resolved字段仅rounds和name不同。label层沿用已发布严格枚举，bit低轮保留旧公开参考基线，非bit禁不匹配参考；public_reference是公开明文变换，不影响full/direct模型。r3/r4 null也是bit声明随机标签，参考只作已有报告、不输入模型。

## 资源与权限

- 正式10fits，r3/r4各5，失败/缺exit/身份错/预算停即不启动后续；正式retry0，不按结果增臂。
- 同MLP24961参数、TRAIN65536 DEV8192 TEST65536、pool65536；batch256/epochs32/lr.001/clip1、每epoch DEV BCE最小选模；每臂8192更新、2097152呈现、32 DEV，总81920更新/20971520呈现/320 DEV。
- 同data195601/key190502/init17/null195602；与六轮已见面板相同，按保存x SHA核，不称全新TEST。跨轮key调度前缀相同，不称独立key。各轮null同标签/面板/初始化，是预声明重复对照，不是独立统计确认。
- 8真实臂每139264 cipher calls，合计1114112（由代码/规模推算，未装独立逐调用计数器），只有同一139264个明文；2null共278528随机bit，cipher0。
- 物理GPU1 UUID GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5→cuda:0，free>=24000MiB/util<=30%，每项启动前查，无fallback、不等待强占。
- 每项max1800s协作训练预算，总10*1800=18000s；连接/同步/取回另计。非硬进程/断线/GPU沙箱。已有Python/Torch不安装、不升级。
- 本地工程仅fake/CPU null，10份smoke最多20更新/80呈现，生产encrypt0。只发布本轮22文件，Windows exact Git checkout，不SCP科学源码或任务包，所有远端生成文件G:\lxy边界、无reparse祖先。

## 结果口径

报告每项selected TRAIN/DEV/TEST ACC/BCE及TRAIN常数、r6同标签参考，区分末在线TRAIN与固定checkpoint指标，不把null复跑当新证据。TEST仍在selected strict load后生成并由CLI自动保存/输出，无封存隔离；只按DEV选epoch，不据TEST换表达式。单key/init，不支持广泛表达式难度排序。

取回全run只核字节SHA/源码config/data/退出/计数/选模，不本地再生标签/模型前向/evaluate。全部完成或触停止条件即CLOSED，不自动加r1/r2/r5或新位组合。
