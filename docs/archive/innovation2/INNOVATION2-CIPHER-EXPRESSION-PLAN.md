# 六轮SIMON密文布尔表达式预测：首轮五臂

状态：本地实现与限定验收完成；用户在四真实表达式+随机对照方案后答“试试看”，随后在明确列出15文件发布/同步/五臂训练范围的确认问题后答“推进”。本轮授权限定15文件本地commit/push、exact Windows同步、指定GPU五臂、结果取回。尚未正式训练；不发布其他dirty工作。

## 科学问题与边界

输入为一个64位明文，MLP单logit预测固定六轮密文的一个布尔表达式。不是预测表达式是否平衡，不是选择明文cube积分，不给模型密文/key/内部状态/其他标签。固定未知key由标签侧局部生成，模型只从TRAIN学习参数。

预先固定（MSB=0，LSB=63）：

| arm | 标签 |
|---|---|
| bit | C0 |
| xor2 | C0 XOR C8 |
| xor3 | C0 XOR C8 XOR C16 |
| xor-and | C0 XOR (C8 AND C16) |
| null | 与明文独立的均匀随机bit |

相同MLP是首轮标签可学习性对照，不预先认定XOR/非线性一定更易。四真实表达式在完整均匀明文域上均严格平衡（固定key加密是置换），有限面板不必精确1:1，仍报告TRAIN常数基线和BCE。不得把有限偏斜称成密码单bit在全域天然不平衡。

## 模块职权

config描述并校验固定枚举/位表，非bit仅prediction/simon/mlp/full/direct；data从每明文已有的一次完整加密计算标签；training只透传/呈现，不实现表达式或改变训练时序。模型/cipher/checkpoints/evaluation/CLI不改。bit默认序列化省略新增字段保持旧配置SHA；新表达式绑进完整config和checkpoint身份。

不引入表达式语言、registry、额外训练入口或远程编排。不扩大旧结构模型到其target cone之外。null不调用cipher，合成任务元信息按实际输入标签描述。

## 冻结矩阵与资源

- 正式五配置：`configs/innovation2-cipher-expression/{bit,xor2,xor3,xor-and,null}.json`；顺序同表。
- SIMON64/128 r6；MLP64→128→128→1，24961参数；full/direct；Adam lr.001、BCE、clip1；同初始化和相同shuffle/输入面板。
- TRAIN65536、DEV8192、TEST65536，pool65536；batch256、32epochs，每epoch DEV一次。每臂8192更新/2097152呈现，合计40960更新/10485760呈现/160 DEV。
- seeds：data195601（本轮新面板）、key190502（历史同key，不能称跨key确认）、init17、null195602。最终按保存x SHA核相同面板，不按概率猜。
- 正式真实cipher calls：每真实臂65536+8192+65536=139264，四臂合计557056；不同表达式重复用同一批139264明文，不宣称557056个独立明文。没有cache复用；null139264随机bit、cipher0。
- 拟用指定GPU1 UUID GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5映射cuda:0；启动前现场空闲>=24000MiB、util<=30%，无fallback。单臂max_seconds1800协作预算，总5*1800=9000s训练预算；连接/同步/传输另计。无安装。
- 正式重试0，最多5fits；缺exit/非零/预算失败/源码身份错/设备不符时不启动下一臂。尚未定义硬进程限额或断线存活保障，不伪称有。

## 验收和解释

DEV BCE各臂独立选checkpoint，严格load之后才生成TEST。CLI仍自动保存/输出TEST，不是技术封存；所有表达式事前固定，本周期不按TEST追加位组合/key/init/epoch。全五臂都报告，不只报最好。

主指标：TEST accuracy/BCE相对TRAIN多数类/常数概率基线、null，结合selected TRAIN/DEV及末在线TRAIN BCE判断拟合/泛化。单key/init首轮探索，不作普适可学或跨key结论；若挑最好臂后再证实需新面板单独冻结。没有预设某准确率为论文成功。

工程测试使用fake ciphertext、synthetic、null，生产encrypt0。正式数据只在批准训练机运行CLI生成，取回只核字节/身份/记录，不load模型、重算标签或evaluate。

## 发布/远端/接收

只发布本任务三源码、新测试、十配置、本计划及必要架构说明，逐文件stage；保留所有无关dirty，不reset/clean/stash，不把runs/validation/环境/旧文档带入提交。仅本地commit/push，Windows fetch/checkout exact SHA；不SCP科学源码或任务包。fresh目录G:\lxy\shared-m-cipher-nd-runs\<fresh-id>\repo，路径规范化且祖先无ReparsePoint；已有Python/Torch环境不升级。

结果写runs/<run-id>/，取回全量到本地fresh runs/received/，核manifest/code/config/data SHA/updates/presentations/exit。无自动evaluate/指标重算。
