# 无轨迹轮展开浅轮定位：一轮学会，两轮失败

本批基于main及既有未提交轮展开实现；原review-fixes分支及保存引用均未改动。
用户“继续推进”后按[执行前计划](INNOVATION2-UNROLLED-LOCAL-PLAN.md)进行本地CPU诊断，
未GPU/SSH/安装/commit/push。没有改变模型/训练源码或读取上一批正式checkpoint。

## 公开人工规则正控

同一`UnrolledSimonPredictor(width32,rounds1,target0)`，随机init17，
全部16种(y,a,b,c)输入角点，标签 `y XOR (a AND b) XOR c XOR z`。
每个z独立随机初始化，Adam .01/clip1，1000个全表更新，各16000次呈现：

| 人工极性z | 初始BCE | 最终BCE | 训练角点ACC | 首步cell梯度绝对和 | 首步condition梯度绝对和 |
|---|---:|---:|---:|---:|---:|
| 0 | .706711 | .000078 | 16/16 | .940378 | .0269063 |
| 1 | .707195 | .000092 | 16/16 | .922474 | .0303637 |

无实际密码/key/真实轨迹接口；人工极性只通过标签体现，没有填入模型参数。
全部角点用于训练，所以**仅证明局部规则拟合，不是持出泛化/密码成绩**。
该正控作为确定性工程回归测试可被测试套件重复执行，不是新增科研seed。

## 正式浅轮最终位预测

固定未知key的SIMON64/128，MSB bit0，只给明文和最终目标位标签。
两轮项也没有额外一轮标签或轨迹；每项独立随机初始化，没有迁移人工控制或一轮权重。
TRAIN4096/DEV1024/TEST8192，batch256，50epochs/800更新/204800次呈现每项，
width32，Adam .001/clip1，data192701/key190502/init17/null192702，CPU。
120秒协作预算/项，均未触及。

| 项 | 参数量 | DEV最优epoch | TRAIN ACC | DEV ACC | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1真实 | 449 | 50 | 100% | 100% | **100%（8192/8192）** | .03288889 | 1.97026 |
| r1-null | 449 | 21 | 52.0020% | 49.0234% | **49.6582%（4068/8192）** | .69360253 | 1.95121 |
| r2真实 | 577 | 41 | 51.2939% | 51.6602% | **48.9380%（4009/8192）** | .69444168 | 3.19661 |
| r2-null | 577 | 41 | 51.1719% | 50.8789% | **50.0610%（4101/8192）** | .69353218 | 2.98906 |

所有项exit0；总3200次更新/819200次呈现，保存elapsed合计10.10714秒。
人工全表正控使用不同步长和预算，不与真实任务当单因素实验比较。

### 基线与门槛

r1真实的TRAIN常量基线TEST50.1465%/BCE.69328887；
公开零key参考残差常量基线TEST100%/BCE.00012206。
所以一轮神经成功是学习正控，**并未优于公开结构基线**，不构成攻击优势。
未知一轮key效应仅是一个输出极性，不能据此声称学会整块轮函数。

提前规定的一轮推进门槛为**DEV ACC≥99%、BCE<.05**，实际100%/.03275885，通过，
因此执行了两轮真实与null；未用TEST决定扩轮。
两轮真实的TRAIN常量基线TEST48.7915%/BCE.69326302，公开参考基线49.3286%/BCE.69366637；
两轮候选没有有效优势。本批按计划止于两轮，无自动r3/4/5，无额外seed/epochs。

## 本批新增了什么认识

1. 同一模块确实能用梯度学会公开局部规则，也能从真实一轮最终标签学出目标映射。
   所以先前只靠“有非零梯度”的功能证据已补成实际局部学习证据。
2. 本设置从单轮变成两轮、仍只给末端一位监督，就没有学起来；TRAIN也只有51.29%，
   不是明显的高训练拟合/低TEST泛化现象。
3. 这只能定位到组合学习的困难，**没有隔离出唯一原因**：连续标量隐藏状态、共享cell、
   随机轮位置条件、最终单bit稀疏监督及训练预算都可能起作用。
4. 正式模型没有接受真实轮状态、key、轮密钥或其他密文位；不是通过真实状态对恢复key。
   从合法TRAIN标签学到未知效应允许，不声称参数绝不包含秘密信息。
5. 一轮目标成功不证明全部32个更新位置学对；无真实状态审计，不能声称内部轨迹正确。

后续若另行推进，较有针对性的干预是对比共享cell与逐轮独立cell，或诊断早期梯度/隐藏表示，
先固定两轮、同面板和预算；不应因为两轮失败直接GPU放大或加入秘密轨迹监督。
这是下一步建议，不是本批已执行内容。

## 验证与产物

- Ruff通过，mypy 62文件通过，pytest **715 passed**；干净源码副本null CLI两更新check/train/evaluate通过。
- 单元测试控制存于`tests/test_unrolled_simon_learning_control.py`；正式配置为
  `configs/innovation2-unrolled-local/r{1,2}-{real,null}.json`。
- 核验48份原始产物字节/SHA、39份源码身份、冻结配置、初始化、明文/标签SHA、
  50行进度、DEV选模、严格加载前TEST声明、计数与exit。核验只读保存的JSON/字节，
  没有再加载正式模型、生成标签或执行前向。
- 与上一批科学源码逐文件SHA相同；未从其他分支合并任何实验实现。

```text
runs/innovation2-unrolled-local-20261022-001531-r1-real/
runs/innovation2-unrolled-local-20261022-001531-r1-null/
runs/innovation2-unrolled-local-20261022-001531-r2-real/
runs/innovation2-unrolled-local-20261022-001531-r2-null/
validation/unrolled-local/frozen-plan.json
validation/unrolled-local/artificial-control.log
validation/unrolled-local/r1-gate.json
validation/unrolled-local/verification.json
validation/unrolled-local-clean-20261022-001046/
```

| 项 | best.pt SHA256 |
|---|---|
| r1真实 | `55982374fd9e9443c34cc62dbc9881274b0270aa8e94e24e69a16cd0f1426c4b` |
| r1-null | `d6632e9cf136e69f5255063548eb6de0c65f3da160373fea8f67163785d39e94` |
| r2真实 | `f4df7aa751ace776b1f92feb1c854a1b47982a5ba4d24c5bda6c075ffee85060` |
| r2-null | `5423d914335d362dcbf23dae7672b1bf1f266ab34c464a15672ed1b7ba2d3a20` |
