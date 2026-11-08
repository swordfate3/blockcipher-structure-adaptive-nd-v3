# 公开XOR弱信号：新面板复查计划

用户在四项结果与独立复核后要求“继续”。本批是新授权下的一次有界样本稳健性复查，
不是恢复已关闭批次或因剩余预算自动追加。保留main与全部已有未提交工作，不改科学源码。

## 问题

前批xor真实TEST53.88%、direct49.43%，但DEV53.42%未达70%推进门槛。
本批只检查弱信号在一个新面板上方向是否保留，不以失败为由扩大数据/epoch/模型。
不加载旧checkpoint或对旧样本前向；每个模型独立从原init17初始化。

## 固定配置与唯一变化组

四臂为direct真实、direct-null、plaintext-xor真实、plaintext-xor-null。
标准SIMON64/128减到2轮、预定bit0、full、simon-unrolled width32、577参数。
TRAIN4096/DEV1024/TEST8192、batch256、50epochs、Adam.001、clip1，
CPU、每项800更新/204800次呈现、120秒协作上限。合计3200更新/819200次呈现/480秒，非硬截止。
key_seed190502、init_seed17及全部模型/优化语义保持；新data_seed192901、null_seed192902。
配置目录`configs/innovation2-plaintext-xor-panel/`；输出使用全新runs目录，不覆盖旧目录。

**data_seed同时决定明文抽样和shuffle顺序**，因此与前批相比变化的是“新面板＋新shuffle”，
不是严格只改数据且控制shuffle的因果分离。null_seed也随新面板更换随机标签。
四臂之间面板/shuffle/init相同，真实和null标签各自配对，不把此复查说成跨key/init确认。
全局输入唯一性沿用make_inputs；在运行前只生成无标签明文，核对新面板各split与上一批
所有明文无交集（旧明文从其公开配方重建，不重新生成任何标签）。

## 判别与停止

保持前批推进门槛，不降格为“超过50%就成功”：

1. xor-real DEV ACC≥70%，比direct-real至少高10个百分点；
2. xor-real DEV BCE同时优于TRAIN常量和既有公开参考常量；
3. 两个null DEV均在[45%,55%]且xor-real比xor-null至少高10个百分点。

单独描述弱信号方向是否重复：xor-real是否在DEV和TEST都高于direct且BCE更低，
是否优于TRAIN-only基线、是否与null分离；不把方向一致当70%门槛通过或正式统计证明。
保留正向、消失、反向结果，不根据中途分数挑seed、换阈值或追加第五臂。
四项固定顺序一次运行；技术错误即停止余项，不重试。四项完结无论门槛如何均关闭，
不自动跑三轮/其他key/init、增加epoch、GPU、SSH、多bit、真实状态教师或新架构。

每臂最低DEV BCE选best，strict加载后生成最终TEST。
当前CLI自动落盘TEST而非访问强制封存；先只提取DEV写decision，然后展开TEST最终报告，
不使用TEST选择任何新配置。所有指标仍是绝对最终单密文位ACC/BCE，不报告替代目标成绩。

## 验收

复用已有环境，科学源码39份SHA必须与前批冻结值完全一致。
配置/device check后冻结计划、配置文件/解析SHA及源身份；以现有统一CLI check/train执行。
Ruff/mypy/pytest验证现有实现与新增小配置可解析；本批无源改动，沿用相同源码此前
干净副本null CLI检查证据，不重新执行正式checkpoint评价。
结束只读保存JSON/进度/字节SHA，核同面板/同初态、计数/exit/最低DEV BCE。
不加载正式.pt、不再生科学标签、不重算前向。无安装/GPU/SSH/commit/push。
