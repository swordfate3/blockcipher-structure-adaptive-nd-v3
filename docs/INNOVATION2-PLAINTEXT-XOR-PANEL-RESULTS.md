# 公开XOR弱信号的新面板复查结果

日期2026-10-22；批次`innovation2-plaintext-xor-panel-20261022-120247`。
用户在前批关闭后要求“继续”，本批为另一次有界CPU复查，不恢复前批。
[执行前计划](INNOVATION2-PLAINTEXT-XOR-PANEL-PLAN.md)；前批[结果](INNOVATION2-PLAINTEXT-XOR-RESULTS.md)。
**四臂完成、门槛再次未通过，批次关闭。没有追加训练。**

## 1. 不变项与变化项

无科学源码修改，39份源码SHA与前批一致。原simon-unrolled、full、width32、577参数，
标准SIMON64/128减到2轮，最终bit0单标签；不喂key/轨迹/其他密文标签。
plaintext-xor只在原输出logit上乘公开明文位符号；绝对标签BCE不变。
每臂CPU、TRAIN4096/DEV1024/TEST8192、batch256、50epochs、800更新、204800次呈现，
Adam.001/clip1，120秒协作上限而非硬截止。key190502、init17保持，无旧权重初始化。

新data192901同时改变明文面板和shuffle，null192902改变随机标签；不是纯数据因果分离。
运行前只构建无标签明文，检查13312条唯一输入与前批所有split零交集，未再生旧标签。
四臂各split明文SHA一致，真实/null标签各自配对，初始化SHA四臂及前批一致。
这是一个新面板复查，不是跨key/init复现，两个面板仍共用同一key及init。

## 2. 保存的best结果（绝对最终密文bit）

每项最低DEV BCE选best，严格加载后生成TEST标签。CLI自动保存TEST；先落DEV决策，再展开TEST。

| 项目 | best epoch | TRAIN ACC | DEV ACC | TEST正确数/8192 | TEST ACC | TEST BCE | 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct真实 |23|50.9521%|51.6602%|4046|49.3896%|0.69371679|3.382928|
| direct-null |7|51.7090%|50.9766%|4076|49.7559%|0.69342870|3.691264|
| plaintext-xor真实 |50|52.4414%|51.5625%|4218|51.4893%|0.69200057|4.574264|
| plaintext-xor-null |2|49.1943%|51.4648%|4077|49.7681%|0.69344740|3.726061|

真实标签的TRAIN-only基线：

| 基线 | DEV ACC | DEV BCE | TEST ACC | TEST BCE |
|---|---:|---:|---:|---:|
| 常量 |49.0234%|0.69316673|50.1587%|0.69314456|
| 公开零主密钥参考＋TRAIN残差常量 |50.0000%|0.69329316|50.3418%|0.69317635|

真实XOR的DEV BCE为0.69176349，direct为0.69292037。

## 3. 判别结果

原70%/10个百分点门槛保持，不因前批弱信号或本批TEST而下调：

- xor-real DEV51.5625% < 70%，失败；
- 相对direct DEV准确率差为**−0.0977个百分点**，未达+10；
- DEV BCE同时优于两种TRAIN-only基线，通过；
- 两null DEV50.9766%/51.4648%均在45%–55%，通过；
- xor-real比xor-null DEV仅+0.0977个百分点，未达+10。

因此DEV推进门槛再次未通过。
另列弱信号方向：TEST准确率仍高于direct（+2.0996个百分点）、DEV/TEST BCE仍更低，
但DEV准确率优势没有保留，故预先列出的“DEV和TEST准确率与BCE均优于direct”方向条件也未全部重复。
这不是完全反向，也不是稳健成功：前批XOR TEST53.88%，本批51.49%，小幅效果变弱。
不把best仍在epoch50用作加训练理由，不按TEST挑下一个seed。

## 4. 有限结论与停止

给定架构/信息条件/预算下，公开XOR的BCE改善在两个面板方向一致，
但准确率增益小且DEV不一致，不能称为可靠的两轮预测器或解释唯一优化瓶颈。
不将单bit均匀分布相关推导推广成通用学习不可能证明，也不声称隐状态等于真实轮状态。
未进行独立key/init稳健性、配对统计检验或标准全轮攻击；不从这两批挑最优值作确认成绩。
两次试验仅改变面板/打乱条件，仍可能受固定key/init、有限样本和优化轨迹影响。

本批合计3200更新、819200次呈现，四exit0，训练计时和15.374517秒。
止于当前两轮，不自动添加epoch/seed、其他key、第三轮、GPU、多bit监督或新架构。
后续如继续，应先形成新的机制假设和可区分的最小对照，而不是反复换面板追分；本报告不执行它。

## 5. 证据与工程范围

输出为`runs/innovation2-plaintext-xor-panel-20261022-120247-<arm>/`，四个arm：
`r2-direct-real`、`r2-direct-null`、`r2-xor-real`、`r2-xor-null`。

best checkpoint SHA256：

```text
direct-real 8f30b7b5be95c7c736f0e787e7c73a4e4270fee23f76f354aff94c5c8eb5861d
direct-null 22867d9c29a854bdb09618afdee90b6e5ff1414288127edd2b7d34bf0f89c7e9
xor-real    436a2d11e8489bf13747c191fa3a52f850538a2609bc4924e94b1a7f429167d0
xor-null    45fe1f05cf17d35f8ef8c8d8232b559a456e35800c1d569ad40eb8d2b0071666
```

`validation/plaintext-xor-panel/`保存preflight、冻结计划、check/CLI日志、dev-decision、verification。
核验48份运行文件字节SHA、源码/计划/配置身份、同初态/同面板/标签配对、50epoch计数与
最低DEV BCE选模及选择时TEST标签尚未生成标记。未加载正式checkpoint、未再生科学标签或前向。
新配置回归锁定两批种子和预算；Ruff通过、mypy63文件通过、pytest758通过、diff检查通过。
本批科学源码与已有干净副本CLI闭环`validation/plaintext-xor-clean-20261022-104014/`完全一致，
沿用其null check/train/evaluate证据，未新跑工程smoke或正式evaluate。
已有环境复用，无安装/GPU/SSH/commit/push，保留main及其他分支引用和全部已有工作。
