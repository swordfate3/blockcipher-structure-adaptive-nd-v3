# 无真实轨迹的神经轮展开：首次CPU试验未通过3轮门槛

## 执行与信息边界

基于main `111adab6fba0df0369e65c9a5a7312e1d8481fa6`加本地未提交实现，
不是另一分支STE/GPU/SAT的重跑。原分支保存在
`review-fixes-20261020`及`saved/review-fixes-before-unrolled-20261021-173603`，均指向5241c62。
未合并、未提交或push、未SSH/GPU/安装。

方案在拟合前冻结，见[无轨迹轮展开](INNOVATION2-UNROLLED.md)。
只输入明文，TRAIN最终绝对bit0提供BCE；没有真实key、轮密钥、真实轮状态、其他密文位
或求解器赋值输入。随机初始化共享局部MLP及轮位置条件，隐藏状态连续可微，无STE。
未知key只在数据标签生成层使用；允许从TRAIN学习未知key效应。

CPU，TRAIN4096/DEV1024/TEST8192，batch256，50epochs，800更新/204800次训练呈现每项，
width32，Adam .001/clip1，data192601/key190502/init17/null192602。
每项120秒协作预算，实际均未触及。

## 实测结果

| 项 | 参数量 | 最佳epoch | TRAIN ACC | DEV ACC | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| r3 unrolled真实 | 705 | 38 | 52.8076% | 51.8555% | **50.3052%（4121/8192）** | .69408073 | 4.46896 |
| r3 unrolled-null | 705 | 1 | 48.1934% | 52.1484% | **49.1333%（4025/8192）** | .69482723 | 4.08661 |
| r3 MLP真实 | 3169 | 6 | 56.3965% | 50.9766% | **49.8413%（4083/8192）** | .69466294 | 1.34590 |

真实标签两项同一个TRAIN常量基线TEST50.6958%/BCE.69312787；
TRAIN拟合的公开零key参考残差常量基线TEST62.6709%/BCE.66075689。
候选和MLP都未超过基线，不能称预测成功。两模型参数量不匹配。

三项均exit0、实际800更新；总2400更新/614400次训练呈现，elapsed相加9.90147秒。
这是保存elapsed之和，不是密码安全复杂度或受控性能基准。

## 门槛与结论

提前规定：r3 unrolled **DEV ACC≥70%且DEV BCE优于两种TRAIN-only基线**，才启动r4三项。
实际DEV51.8555%、BCE.69187497，公开基线DEV61.8164%/BCE.66539774，门槛失败。
因此**r4 unrolled、r4-null、r4 MLP全部未启动**；配置存在不授予重跑权限。
无追加epoch/seed/5轮，批次关闭。

本设置没有有效预测证据。TRAIN本身仅52.81%，不是高TRAIN/低TEST的典型记忆过拟合；
与共享局部参数化、连续隐藏表示、单末端监督及有限预算有关的优化/表示困难仍需区分。
梯度测试证明链路可反传，不证明梯度足以学会密码。
本批没有独立r1/r2函数学习正控，不能声称已验证局部轮模块能学会完整单轮规则，
也不能把失败归结为某一个机制或宣布所有逐轮方法不可行。
同期full MLP也失败，不是历史support MLP约82%的同预算复现。

若后续获准继续，优先补无秘密信息的单轮人工布尔规则控制并诊断输入/早期梯度，
而不是直接GPU放大。模型内部逐轮推演与用真实状态逐轮监督是不同任务，后者仍未实施。

## 产物与检查

```text
runs/innovation2-unrolled-20261021-180627-r3-unrolled/
runs/innovation2-unrolled-20261021-180627-r3-unrolled-null/
runs/innovation2-unrolled-20261021-180627-r3-mlp/
validation/unrolled/frozen-plan.json
validation/unrolled/r3-gate.json
validation/unrolled/verification.json
validation/unrolled-clean-20261021-175955/
```

核验35份原始文件字节/SHA、39份科学源码SHA、配置、初始化、同面板与真实/null标签身份、
50行progress、DEV最优epoch、严格加载前TEST声明、实际计数与exit；
核验阶段只读JSON和字节，没有加载正式checkpoint、再生标签或重算前向。

真实/null初始化记录逐字节等价，所有项明文SHA一致；MLP与unrolled真实标签SHA相同。
源码身份冻结前后相同；后补测试只加固输入依赖梯度及null禁止加密，没有改模型或训练。

| 项 | best.pt SHA256 |
|---|---|
| r3 unrolled | `e6330dd14e9e3e8756ca25480f969aa26d7eeddaf8510de37a9a9ae0d1f9c394` |
| r3 unrolled-null | `b9311991898ac9b3471f5ffbf6f1d6f0eae264090245ab5f5b1ff4b07fa0b37c` |
| r3 MLP | `4ce582fe8ef0e12587110cc50b0ee1bab7ef2b6a8fe24a69e03d1ba829e7990d` |

已验证Ruff/mypy/pytest、干净源码副本null标签check/train/evaluate两更新闭环。
独立只读审查未发现已确认P1/P2；补了其建议的两个边界回归。
没有新安装环境、CUDA或Windows现场测试。
