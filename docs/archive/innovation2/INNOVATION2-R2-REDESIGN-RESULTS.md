# 两轮重新设计首批CPU结果：正控100%，神经候选96.68%，仍未达目标

## 1. 结论与停止

用户要求“推进下实验”，按[运行前合同](INNOVATION2-R2-REDESIGN-PLAN.md)完成固定四项，
批次 **`innovation2-r2-redesign-20261024-013329`**。输入明文，预测固定未知key下的
**SIMON64/128两轮、绝对密文bit0**；不是差分分类、完整44轮、整块恢复或完整密钥恢复。

- **A：四格结构化正控通过本批所有TRAIN/DEV与TEST门槛**。4个从零训练的logit达到TRAIN/DEV/TEST100%，
  TEST BCE0.02966991；只说明很强的公开结构约化加最终bit标签能完成该两轮任务，不算神经组合学习成功。
- **B：332参数独立局部神经候选明显学到信号，但未通过门槛**。
  TRAIN96.7773%、DEV97.0703%、TEST96.6797%（7920/8192），TEST BCE0.07668525。
  未达TRAIN≥99.9%、DEV/TEST≥99%、BCE<0.05。不是“因为接近所以算过”，不追加epoch。
- 两个null在DEV/TEST均处于45–55%，支持本批优势不是普遍拟合独立随机标签；不能因此证明没有任何其他实现风险。
- 四项各800更新、204800次TRAIN呈现，均exit0；**本批关闭，不自动加seed、key、轮数或预算**。

## 2. 协议、来源与预算

实现位于`src/cipher_experiments/models/simon_two_round.py`，
[推导与边界](INNOVATION2-TWO-ROUND-REDESIGN.md)、[来源](provenance-two-round.json)。

A从明文计算公开`U=Y XOR F(X)`，并不获取真实第一轮状态：
`A=U[j+1]、B=U[j+8]、T=X[j] XOR U[j+2]`，输出`(1-2T)*table_logits[2A+B]`。
四个未知logit从零开始，未赋真实常量。B仅用公开局部接线、三个独立`4→16→1 Tanh`模块，
sigmoid/center后由独立`3→8→1 Tanh`组合器输出，再乘公开明文`X[j]`符号。
B没有实际轮函数教师、真实key/状态/其他密文bit或旧权重；连续隐表示不被认定为真实密码状态。

共同设置：

| 项目 | 固定值 |
|---|---|
| TRAIN / DEV / TEST | 4096 / 1024 / 8192，明文全局唯一 |
| data / null seed | 193101 / 193102 |
| key / init seed | 190502 / 17 |
| device / batch / epoch | CPU / 256 / 50 |
| 优化 | Adam，lr0.01，clip1，最终绝对bit BCE |
| 预算 | 每项800更新、204800次TRAIN呈现；四项3200/819200 |
| 时间参数 | 每项120秒协作式max_seconds，不是硬截止；实际训练elapsed合计6.338026秒 |
| 选模 | DEV BCE最小，平局最早；严格加载选中checkpoint后才生成TEST标签 |

顺序固定为table-real、table-null、local-real、local-null，各一次，无失败重试或按分数跳项。
四项都用同一明文面板，两个真实臂标签相同、两个null标签相同；real/null不同。
同模型real/null初始权重SHA相同；**A与B的参数数及初始化机制不同，不宣称跨模型同初始化**。
面板与此前data192801/192901/193001面板均无跨划分重合；data seed同时影响明文与shuffle。

秘密标签构造预算与优化分开：每个真实臂各13312次标量真实加密构造调用、合计26624次，
对应13312个不同明文。两个null不调用真实加密。公开零key参考计算另列，不属于秘密oracle查询。

CLI已自动保存最终TEST，**并非访问强制封存**。外层先只提取TRAIN/DEV并写
`validation/r2-redesign/dev-decision.json`，之后才展开TEST作固定报告；不按TEST选择配置或继续训练。

## 3. 全部结果

指标均来自DEV选中的checkpoint，非最后epoch任意挑分。

| 项目 | 参数 | best epoch | TRAIN ACC | DEV ACC | TEST正确数/8192 | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| table-real | 4 | 50 | 100.0000% | 100.0000% | 8192 | **100.0000%** | 0.02966991 | 1.500917 |
| table-null | 4 | 1 | 51.3184% | 48.7305% | 4191 | 51.1597% | 0.69299153 | 1.325222 |
| local-real | 332 | 49 | 96.7773% | 97.0703% | 7920 | **96.6797%** | 0.07668525 | 1.812966 |
| local-null | 332 | 1 | 51.1230% | 48.2422% | 4019 | 49.0601% | 0.69386711 | 1.698920 |

真实臂best模型的TRAIN/DEV BCE：

| 模型 | TRAIN BCE | DEV BCE |
|---|---:|---:|
| table | 0.02966902 | 0.02966988 |
| local | 0.07118756 | 0.06803673 |

真实标签基线由TRAIN确定常量/概率，再固定评价其他划分，不读DEV/TEST答案来拟合：

| 划分 | 常量ACC / BCE | 公开零key参考常量ACC / BCE |
|---|---|---|
| TRAIN | 51.2695% / 0.69282480 | 50.8301% / 0.69300937 |
| DEV | 49.8047% / 0.69356865 | 49.7070% / 0.69338220 |
| TEST | 49.0479% / 0.69395297 | 49.7070% / 0.69338220 |

A/B的TRAIN、DEV、TEST BCE都优于这两个基线。A满足全门槛；B的准确率和BCE绝对目标均失败，
即使比基线好、null正常，也不改变合取门槛判定。

## 4. TRAIN四格正控检查

只读取训练时由已有TRAIN输入/标签生成的元数据；验收不重新生成标签或加载正式checkpoint。
另有拟合前独立公开Boolean占格检查，未涉及标签或模型前向。

| 格索引2A+B | TRAIN行数 | real残差1数 | null残差1数 |
|---|---:|---:|---:|
| 0 | 1024 | 1024 | 509 |
| 1 | 1058 | 1058 | 525 |
| 2 | 988 | 0 | 508 |
| 3 | 1026 | 1026 | 546 |

四格全部覆盖；真实臂各格内部无冲突，null四格均有冲突。该统计只写元数据，
不用于初始化、损失或选模，不向神经候选提供额外监督。
学到的真值效应与固定key有关，**不等于恢复完整128bit密钥**。

## 5. 能说什么、不能说什么

**观察：** 当前独立局部候选已从此前同任务候选的68.76%跃升到96.68%；仍有272个TEST错误，
不能说两轮已经学透。TRAIN也只有96.78%，不是只有TEST掉分的典型情况；其原因尚未定位。

**局限：** 与旧批跨面板比较，同时改变拓扑、共享、条件、参数量及学习率（旧0.001，本批0.01），
不是单因素控制，不能唯一归因于“不共享”。A更直接地编码公开两轮公式，不与B构成同函数类、
同参数公平竞赛。A成功不证明B优化一定容易；原width32的数学表达性存在见证也不等于Adam可学习。

**未验证：** 新面板/新init/新key独立确认、其他target位实际拟合、三轮及更高轮、内部潜变量与真实状态一致性、
更大预算能否补齐余下错误。不得把单key/init的两轮结果外推到高轮或完整密码安全性。

**建议而非已执行：** 若后续另行授权，先在固定预算下判断剩余错误属于稳定结构性缺口还是优化不充分，
并做新的面板/init确认；不应直接拿本批TEST挑改动，或自动升三轮。本批不实施这些建议。

## 6. 工程与保存证据

- `PYTHONPATH=src .venv/bin/python -m ruff check .`：通过。
- `PYTHONPATH=src .venv/bin/python -m mypy`：68个源码文件通过。
- `PYTHONPATH=src .venv/bin/python -m pytest`：**782 passed**。
- 干净源码副本两种null模型各`check → train → evaluate`、各2更新，保存TEST与显式evaluate一致：
  `validation/r2-redesign-clean-20261024-012705/`。仅工程检查，不是额外科学臂。
- 独立实现审查发现测试直接赋值`training.load_checkpoint`导致测试结束后不恢复；已改`monkeypatch.setattr`，
  全套检查再次通过，生产源码未因此修改。前一轮工程日志保存在`before-test-cleanup-fix/`。
- 执行门禁审查发现原封存脚本只凭日志存在判通过；正式拟合前修为实际退出码与源码/测试/配置/日志SHA绑定，
  只读复核与失败退出负例验证通过。不是说实际Ruff/mypy/pytest曾失败。
- 正式四项48个保存文件的字节SHA、40个科学源码身份、配置、200条epoch进度、更新/呈现、
  最低DEV BCE选择、初始化配对、面板与标签SHA配对及exit均已核对。
- 正式验收**没有**重放checkpoint、重新生成科学标签或额外执行科学模型前向。
- 未安装依赖、GPU、SSH、Git提交/push；main及他人分支/保存引用保持不变。

证据目录：`validation/r2-redesign/`，含`frozen-plan.json`、`dev-decision.json`、`verification.json`、
`coverage-verification.json`、`engineering.json`、`checks.json`、`clean-cli.json`及`panel-preflight.json`。
结果原件：`runs/innovation2-r2-redesign-20261024-013329-{table-real,table-null,local-real,local-null}/`。

best checkpoint SHA256：

```text
table-real ad3ea29c7c5fb6198aa0568b8d42985c52a1a40a43ad9e2989e01d2d35e3c01a
table-null efa28f249195e1f4c705e33f1d9323b9f0ea3904541bd32abaf7f3be02a9f8e2
local-real beaa1fb27e030ac654faeffd7ce27cda53a861a33ab3b2857c255fb6cd5bf3ae
local-null a77d3a7534037203308871a04caa4ad11bf44c085875a742df9c35a849f9a190
```
