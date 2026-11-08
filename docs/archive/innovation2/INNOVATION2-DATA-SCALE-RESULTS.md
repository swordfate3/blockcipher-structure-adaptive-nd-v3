# 远程6轮数据量对照：增加16倍TRAIN未使当前模型学起来

## 结论

用户确认发布与远程GPU四项预算后，按[运行前合同](INNOVATION2-DATA-SCALE-PLAN.md)完成：
6轮SIMON64/128绝对密文bit0，TRAIN4096/65536各真实/null，每项8192次更新。
**真实TEST分别50.2060%/50.1068%，未观察到数据增加带来的有效预测优势。**
两个null为50.0000%/50.1862%。四项exit0，均完整8192更新，本批停止，无追加训练。

这支持的仅是：**在本候选、固定8192更新、单key/init/面板下，将TRAIN增加16倍没有解决学习失败。**
不证明数据永远无关；大数据每样本曝光更少，其他优化/架构或更大计算尚未测试。
TRAIN本身也近常量水平，不是已经拟合TRAIN却单纯泛化失败。不能称6轮安全证明、高轮不可能或完整44轮攻击。

## 全部结果

均为最低DEV BCE选中的checkpoint，平局最早。模型simon-unrolled/full/direct/width32，1089参数，
仍是公开Feistel/旋转与共享神经局部轮模块，仅最终单bit BCE，无真实key/状态或其他密文bit教师。

| 项目 | TRAIN行数 | best epoch | best更新 | TRAIN ACC | DEV ACC | TEST正确/65536 | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small-real | 4096 | 16 | 256 | 50.4395% | 50.1465% | 32903 | **50.2060%** | .69314406 | 145.000 |
| small-null | 4096 | 272 | 4352 | 51.2939% | 50.2930% | 32768 | 50.0000% | .69358757 | 144.516 |
| large-real | 65536 | 10 | 2560 | 50.2258% | 50.4028% | 32838 | **50.1068%** | .69315900 | 146.734 |
| large-null | 65536 | 26 | 6656 | 50.3510% | 50.3540% | 32890 | 50.1862% | .69315312 | 145.875 |

真实TRAIN/DEV BCE：small **.69310526/.69314859**，large **.69312220/.69311426**。
大−小DEV ACC为 **+0.25635个百分点**，DEV BCE **−.00003433**；
大−小TEST ACC **−0.09918个百分点**，TEST BCE **+.00001494**。
这些微小变化没有满足预定≥55%、优于常量及null均≥3pp的探索信号条件，不能包装成数据规模获益。
两组DEV/TEST BCE虽略优于自己的TRAIN-only常量，亦不足以抵消准确率条件失败。

| 真实臂常量基线 | TRAIN ACC / BCE | DEV ACC / BCE | TEST ACC / BCE |
|---|---|---|---|
| small | 50.4395% / .69310856 | 50.1465% / .69316004 | 50.2060% / .69314959 |
| large | 50.2075% / .69313857 | 49.8535% / .69316795 | 49.7940% / .69317289 |

small真实各split正确数与常量相同，不据汇总声称逐条预测完全相同。
large TEST比其常量高0.31281pp，但比paired null低，且未达门槛；非显著性/跨key结论。

## 公平性和实际预算

共同：6轮bit0，固定key190502/init17，data193501/null193502，batch256、Adam lr.001、clip1。
相同模型初态SHA、相同DEV8192/TEST65536和标签；smallTRAIN为largeTRAIN前4096，
train_pool_rows65536固定持出边界。预检139264条全局唯一明文，unused pool不生成标签。

| 项目 | small | large |
|---|---:|---:|
| TRAIN | 4096 | 65536 |
| epoch | 512 | 32 |
| 每epoch更新 | 16 | 256 |
| 每项总更新 | 8192 | 8192 |
| 每项TRAIN呈现 | 2097152 | 2097152 |
| DEV间隔epoch | 16 | 1 |
| 选模DEV检查点 | 每256更新，共32次 | 每256更新，共32次 |

末尾还对选中模型计算一次DEV报告，不参与选模。等更新/等batch呈现不等于同逐更新样本顺序、同样本曝光或端到端FLOPs；
small每样本512遍，large32遍。相同shuffle seed但不同长度randperm，不宣称逐步相同样本顺序。
每项1800秒max_seconds为协作上限，非硬截止。本次result elapsed合计 **582.125秒**，不是开发/SSH/传输总耗时。

秘密标签构造small-real77824、large-real139264，合计217088次真实标量加密构造调用，
涉及139264个不同明文；两个null无真实加密。无新增中间/其他bit标签，没有公开零key参考（仅≤4轮现有能力）。
重复epoch不计新的oracle查询。

## 发布、远端执行和取回

发布过程全部在本地，远端不commit/push：
- `e5372240bd3977dd058c1b6f36f5a2581b7f72fd`：发布已积累自洽模型依赖、配置/测试/文档与本批pool/DEV间隔能力；
- `2ed5a791b74be104114923614ab4e8438ba1fc2b`：修正首次stage检查发现的来源JSON末尾空格；
- **`f151d0c12360fbc3050049d1456d2a2e0cfcf20e`**：新增独立CUDA null smoke配置，为正式运行版本。

首次stage的空白检查指出一处trailing whitespace，但shell后续仍提交；随后明确修正并普通commit/push，
没有rewrite/force。未提交papers、数据、runs、权重、validation、backups或凭据。

远端目录：
`G:\lxy\shared-m-cipher-nd-runs\data-scale-20261025-021431\repo`。
旧源仓库只读检查clean，新目录Git复制并fetch明确commit。SSH到GitHub的fetch失败，
随后对公开仓库作一次HTTPS fetch成功，未修改凭据；不是通过SSH传科学源码/任务包。
全部实际科学代码和配置来自Git，SSH只运行已有CLI命令。

物理GPU1：`GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`，映射logical cuda:0。
每项启动前核空闲≥24000MiB/util≤30%，选中GPU实际CUDA null4更新预检成功后才正式训练。
使用已有Python3.12.10及Torch环境，无安装/升级。机器本地适配器只设置既有dependency路径、DLL/cache目录并转交CLI参数，
不承载科学算法；所有新远端产物位于G:\lxy内。

顺序small-real、small-null、large-real、large-null，全部一次完成；
SSH调用在本次连接中完成，不据此承诺断线存活或恢复能力。正式训练UTC约02:17:36–02:28:08。

DEV BCE选模→strictload→selection.json后才生成TEST。CLI自动保存TEST，不是访问强制封存。
取回后先仅提取TRAIN/DEV保存`dev-decision.json`，再展开TEST作固定报告，无TEST选参或继续训练。
本地接收后**未**加载正式checkpoint、重建科学标签或执行科学模型前向。

## 验证与证据

- 本地Ruff、mypy **76个源码文件**、pytest **849 passed**及干净源码null CLI check/train/evaluate通过：
  `validation/data-scale-clean-20261025-021135/`。后来新增CUDA smoke仅改小配置，无科学源码改变。
- 远端CUDA smoke6轮null8/4/4、4更新实际通过；它是工程，不是科学成绩。
- 远端clean明确commit、41个源码和四正式配置字节SHA与本地一致。
- 核四run共48文件的远端原件SHA和接收字节SHA一致，
  1088条epoch进度、每臂32次DEV选择位置、8192更新/2097152呈现、选模、配对和退出。
- 总计 **32768更新/8388608次TRAIN呈现**；源/配置/manifest/commit绑定一致。
- 独立只读pool/间隔与合同审查未见确认缺陷；不是远端训练的独立复现。

接收目录：`runs/received/data-scale-20261025-021431-{small-real,small-null,large-real,large-null}/`。
远端原件保留。证据：`validation/data-scale/{frozen-plan,remote-preflight,remote-inventory,panel-preflight,dev-decision,verification,test-report}.json`。

best SHA256：

```text
small-real c3479bcd5a9381109086b90a3ff5202fe051b5fb9cc90e9c9e1bb7332ec8011e
small-null 24e3ed2fe18fe16d23b965fa3019b383edabdf89e30acca65f193e26d588ea91
large-real f70b4d0221d2e16e81f94f9404a9232678a1c0957c5baf3da7ac5c6e60b4a678
large-null e894e8f56badacae882c7f2d80878d7d32223b022249e7e9d4cf67bf24d2fb4c
```

## 后续

本批停止，不自动加数据/epoch/seed/GPU预算，也不退回低轮追分。
下一步若另授权，应讨论高轮候选的学习机制/优化与可检验替代路线，
而不是把本结果归结为单纯缺数据。更大预算的联合数据×计算研究仍未做，不能由本批阴性排除。
