# 创新2首批GPU对照：2026-10-19

## 结论

**五项均完成。r3直接预测有持出信号；r4直接/公开残差预测均接近随机，未观察到突破。**
本批仅一个固定key、一个初始化、一个目标bit，不构成跨key或方法普遍有效结论。

本次按用户澄清的方式：代码/配置先经Git同步，再通过SSH执行远端已有CLI启动指令；
没有SSH任务包、科学worker传输、WMI、Job模板、调度服务或历史checkpoint。
机器环境适配器仅引入既有Python依赖并转交CLI参数。五项顺序调用，不重试、不追加。

## 身份与范围

- 代码：`d29e1d9e7031e1d26f67a233951c5e39d618bc75`，Windows检出同一提交，训练后Git仍干净。
- 主机：`DESKTOP-BBLPACJ`；物理GPU1 RTX A6000，UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`。
- 每项开始前重新查询GPU，门槛空闲≥24000MiB、利用率≤30%；五次起始均报告48448MiB/0%。
- `CUDA_VISIBLE_DEVICES=1`，程序设备`cuda:0`，未CPU回退。
- Python3.12.10，Torch2.14.0+cu130，复用现有依赖，未安装/升级。
- 每项TRAIN65536 / DEV8192 / TEST65536，目标MSB bit0、support、hidden114、batch256、50epochs。
- data_seed191401 / init_seed17 / key_seed190502；null标签独立域分隔，null_seed191402。
- 每项12800更新、3276800训练呈现，五项共64000更新、16384000呈现。
- 每项协作时间上限1800秒，**不是硬截止**；实际均低于上限、exit0。
- 五个训练记录elapsed总和523.405秒（约8分43秒），不是精确GPU利用时间；不含所有进程间开销。
- r3参数15619，r4参数17101。真实key仅存在标签生成层，不输入模型。

## 保存的最终TEST结果

下面直接摘录训练机的best模型`results.json`，没有本地重算标签/前向/指标。
每项均完成50epochs；选模epoch为DEV BCE最小值，不是最后epoch。

| 实验 | 选模epoch | TEST ACC | TEST BCE | TRAIN常量ACC | 公开参考残差常量ACC | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|
| r3 direct | 49 | 81.9412% | 0.389746 | 50.2411% | 62.4023% | 90.187 |
| r4 direct | 1 | 50.1770% | 0.693389 | 49.8367% | 50.0061% | 87.437 |
| r4 public-residual | 1 | 50.0214% | 0.693208 | 49.8367% | 50.0061% | 128.844 |
| r4 direct null | 2 | 50.0153% | 0.693226 | 49.7162% | 50.0320% | 87.859 |
| r4 public-residual null | 1 | 50.0275% | 0.693347 | 49.7162% | 50.0320% | 129.078 |

r3在本key/明文划分上优于原常量和公开参考常量，说明这次低轮数管线能学到持出信号；
但本批没有r3专属null或多key重复，不称独立普遍确认。
r4真实标签与null均约50%；没有证据说明公开残差重参数化在该设置下带来泛化改进。
r4 direct最后epoch DEV BCE0.861235，残差0.868525，均明显差于其epoch1的约0.693；
这表明继续优化训练集没有改善DEV，因此使用早期best，而不是挑最后模型或追加epoch。
不把近随机结果称为不可能学习的证明，也不将当前神经结果与旧代数100%成绩混用。

## 文件与核验

远端原始run保留在：

```text
G:\lxy\shared-m-cipher-nd-runs\git-validate-20261019-060000\repo\runs\
  innovation2-first-d29e1d9-20261019-r3-direct-positive\
  innovation2-first-d29e1d9-20261019-r4-direct\
  innovation2-first-d29e1d9-20261019-r4-residual\
  innovation2-first-d29e1d9-20261019-r4-direct-null\
  innovation2-first-d29e1d9-20261019-r4-residual-null\
```

完整指定run已取回本地同名子目录：`runs/received/innovation2-first-d29e1d9-20261019/`。
五项每项11文件，共55文件，均与远端生成清单核对字节数和SHA256。
核对config/manifest/results配置SHA、selected checkpoint字节SHA、epoch/更新/退出/时间记录。
两个r4真实标签机制的data.json完全一致，两个null机制亦一致；配对初始化记录完全一致。
source身份比较允许既有文件纯LF→CRLF转换，不能宣称所有源码字节完全相同。
TEST标签时序由实现与`selection.json`声明支持，不是访问强制封存的独立审计证明。

只读JSON和文件字节；**本地没有加载远端checkpoint、生成标签、执行模型前向或重算指标**。
本地核验：`runs/received/innovation2-first-d29e1d9-20261019/verification.json`。
SSH日志和准备证据：`validation/innovation2-publish-sync/`。产物均被Git忽略，不从Windowscommit/push。

本批到此结束；未追加seed、epoch、模型、其他位或轮数，下一项拟合需要新的具体范围。
