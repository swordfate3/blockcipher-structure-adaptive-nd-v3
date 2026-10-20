# GPU规模STE：4轮59.58%（对照50.10%），5轮无信号

## 执行方式

分支`review-fixes-20261020`，提交`a9b37c5`（含`shuffle_seed`、`bce_gain_*`等审查修订）。
Windows机器DESKTOP-BBLPACJ，代码经GitHub fetch后detached checkout，无SSH传源码/任务包；
物理GPU1 `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`映射`cuda:0`。
四次启动前现场门槛均满足：free≥24000MiB、util≤30%（首次48406MiB/0%）。
用既有机器依赖适配器（SHA`e893a295…`），未安装、未改驱动、未提交远端代码。

本轮范围是用户在4轮确认失败后要求“拿这个分支去A6000训练”的放大重跑：
把样本量与更新数放大16倍，并补上5轮真实/null配对。

| 项 | 值 |
|---|---|
| TRAIN / DEV / TEST | 262144 / 16384 / 65536 |
| batch / epochs / updates / presentations | 1024 / 20 / 5120 / 5242880 |
| data_seed / shuffle_seed / init_seed | 192201 / 192202 / 17 |
| key_seed（仅标签生成） | 190502 |
| 目标位 / 模型 / 优化器 | MSB bit0 / simon-ste / Adam .01, clip1 |
| 协作预算 | 每项1800秒（未触及；实测108–185秒） |

四项依次为r4、r4-null、r5、r5-null，全部exit 0，合计20480次梯度更新、20,971,520次presentation，
保存elapsed相加567.67秒（非GPU利用率或墙钟性能基准）。

## 结果

| 项目 | r4 | r4-null | r5 | r5-null |
|---|---:|---:|---:|---:|
| 最佳DEV epoch | 9 | 8 | 18 | 15 |
| TRAIN ACC | 59.4406% | 49.9622% | 49.9142% | 50.1873% |
| DEV ACC | 59.2834% | 50.6104% | 51.0742% | 50.5676% |
| **TEST ACC** | **59.5764%** | **50.0977%** | **49.8428%** | **49.7421%** |
| TEST正确数 | 39044/65536 | 32832/65536 | 32665/65536 | 32599/65536 |
| TEST BCE | 0.935400 | 1.124975 | 1.130071 | 1.132086 |
| TEST常量基线ACC | 49.8138% | 49.7162% | 50.1587% | 49.7162% |
| `bce_gain_vs_train_constant` | **−0.2422** | −0.4318 | −0.4369 | −0.4389 |

同面板、同初始化（r4与r4-null的`initialization.json`逐字节相同）、明文SHA一致而标签SHA不同，
因此 r4 的 **+9.48个百分点**是当前最干净的真实label vs 随机label对照。

## 如何解读

1. **4轮有真实但有限的信号。** 12.8万个TRAIN标签、5120次更新后，未见TEST比随机标签高约9.5pp，
   且在TRAIN/DEV/TEST三划分上一致（约59.3%–59.6%），不是单点波动。
   按审查新增的审计口径，这意味着STE在该预算下确实学到了部分未知常量。
2. **放大没有变好：早先小面板74.90%未能升级为更强效果。** 本轮训练量是那批的16倍，
   TEST反而低约15pp。结合上一轮四轮确认（新面板约50%），
   目前证据支持“STE优化轨迹对面板/顺序高度敏感”，而不支持“数据越多越准”。
   74.90%应视为有利抽样，不宜作为该方法的能力估计。
3. **5轮仍无信号。** 同一面板、同一预算下 r5 49.84%（null 49.74%），比4轮掉近10pp。
   这是“当前STE参数化+预算在4→5轮之间失效”的负结果，
   不等于5轮不可预测（历史硬枚举/SAT/旧代数各有其独立证据），也不等于所有逻辑网络都不行。
4. **BCE全部为负增益是预期而非矛盾。** 模型输出固定±2 logit，正确样本BCE下限约0.1269，
   与0.6931的常量概率基线不同尺度；审查新增的`bce_gain_*`把这件事显式化。
   本轮结论只应依据**准确率**与label/null对照，不能因为BCE差就否定59.58%，反之也不能用BCE好来宣称成功。
5. **单seed、单key、单目标位。** 未做跨key或跨seed矩阵；不称稳定复现，也不称密码被“破解”。
   4轮24个未知常量对应主密钥字的前缀，参数可识别性另见审查审计：预测正确不等于唯一恢复参数。

## 证据与核验

```text
runs/received/innovation2-gpu-scale-20261020-114438/
  artifact-inventory.json   verification.json
  innovation2-gpu-scale-20261020-114438-{r4,r4-null,r5,r5-null}/
validation/gpu-scale/       # sync/check/each-run 日志、batch.txt
```

48份原始产物取回，逐份字节长度与SHA均匹配；核对原始配置SHA、Git身份（commit一致且dirty=false）、
逐源文件SHA（7个文件仅因Windows CRLF与LF差异，按换行归一化匹配）、
初始化身份、配对明文/标签身份、20行progress、DEV选模与checkpoint文件SHA、exit与计数。
远端原件保留在`G:\lxy\shared-m-cipher-nd-runs\git-validate-20261019-060000\repo\runs\`。
取回后未加载checkpoint、未再生成标签、未重跑模型；`evaluate`仍是独立用户动作。

| arm | configSHA | best.pt SHA256 |
|---|---|---|
| r4 | `286a85c9…1b4db3d` | `ce0b30067a11b46f3ee829e0c08831314eea18c10cc70f0a19d91535bc90881e` |
| r4-null | `55683836…06487e1` | `7311581b9faa6021ce7788bc89f5088e89beb0053b4f80a4e66d6ff06ac382be` |
| r5 | `1df761bd…2bade74` | `e654ffa4aef0bbde8e551b8e1822c8b6927d9c3a05960dae12eece1170518461` |
| r5-null | `2bd2f86c…ff3cecc` | `ab2940005e0ff46c0bf30346672788f5d71ec80fa37282c84665a8d52e038edf` |

## 下一步（未执行）

- 用固定面板、只改`shuffle_seed`与`init_seed`做3–5次，估计4轮59.6%的波动范围；
  现在只有1个seed，区间未知。
- 5轮若要再试，应先做梯度/符号翻转诊断，而不是继续放大样本或改成硬枚举。
- 差分bit任务仍按用户要求搁置。
