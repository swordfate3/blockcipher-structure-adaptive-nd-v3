# 六轮直接MLP与轮展开：两者均未显示泛化优势

## 1. 结论

用户在方案及本批发布/远程GPU预算解释后回复“推进”，按[固定四项合同](INNOVATION2-HIGH-ROUND-MLP-PLAN.md)
完成A6000运行与取回。批次 **`high-round-mlp-20261026-091102`**，正式运行版本
**`3a7ba03eefd5349cd6135ac5a6d06a1bc727ada2`**，四项均exit0、各8192更新，无重试、调参、续训或升轮。

- 六轮轮展开TEST **49.7238%**；直接MLP **49.7620%**，两个null **50.1343% / 50.1495%**。
- 两模型DEV与TEST均未达到预定55%、优于常量及paired null各≥3个百分点的探索信号条件。
- **MLP后期能降低训练过程BCE，但没有改善DEV**；null也出现类似现象。
  不支持当前直接MLP在该预算下提供真实密码泛化优势，也不能把问题唯一归因于轮展开的sigmoid压缩。
- 这是单key/init/面板、两种候选的有限阴性结果，不是六轮不可预测证明或完整SIMON64/128的安全证明（完整44轮）。

本批关闭，不因两者约50%自动加epoch、seed、数据或GPU，不回低轮追分。

## 2. 全部选中模型结果

最低DEV BCE选best、平局最早；strict selected load后才生成TEST标签。以下为选中模型而非最后epoch：

| 项目 | 参数 | best epoch / 更新 | TRAIN ACC | DEV ACC | TEST正确/65536 | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unrolled-real | 1089 | 4 / 1024 | 50.1770% | 50.9521% | 32587 | **49.7238%** | .69336382 | 142.015 |
| unrolled-null | 1089 | 2 / 512 | 49.8566% | 50.2930% | 32856 | 50.1343% | .69315275 | 139.453 |
| mlp-real | 24961 | 1 / 256 | 50.9064% | 50.7202% | 32612 | **49.7620%** | .69356489 | 55.047 |
| mlp-null | 24961 | 1 / 256 | 50.0595% | 50.1953% | 32866 | 50.1495% | .69356606 | 54.922 |

真实臂选中模型TRAIN/DEV BCE：
- unrolled：**.69321916 / .69297351**；
- MLP：**.69278733 / .69323435**。

MLP−unrolled DEV ACC为 **−0.23193个百分点**，DEV BCE **+.00026084**。
TEST ACC差值 **+0.03815个百分点**，不构成可靠优势；未作显著性或逐样本预测配对检验。

相同真实标签、TRAIN拟合的常量基线：

| 划分 | 常量ACC | 常量BCE |
|---|---:|---:|
| TRAIN | 50.1770% | .69314091 |
| DEV | 50.9521% | .69308603 |
| TEST | 49.7238% | .69317300 |

unrolled真实各split正确数与常量相同，但未保存逐条预测，不能据此声称所有预测逐条一致。
MLP TEST比常量只高25/65536（.03815pp），BCE反而更差；两个真实臂TEST BCE都差于常量。
两个null的DEV/TEST均45–55%，仅是负对照正常，不补偿真实信号条件失败。
六轮没有扩展零key公开参考，仍只报告TRAIN-only常量基线。

## 3. 训练过程与DEV恶化

| 项目 | 第32epoch在线TRAIN BCE | 第32epoch DEV ACC | 第32epoch DEV BCE |
|---|---:|---:|---:|
| unrolled-real | .69317473 | 49.1821% | .69336312 |
| unrolled-null | .69321204 | 49.5728% | .69317673 |
| mlp-real | **.51799918** | 50.6958% | **.87031289** |
| mlp-null | **.52063109** | 50.6348% | **.86954753** |

这里TRAIN BCE是epoch内各更新时刻的在线损失平均，不是最后checkpoint统一前向的TRAIN指标，
不把它换算成未记录的最终TRAIN准确率，也不称MLP完整记住了所有训练标签。
MLP真实/null都降低训练过程损失而DEV恶化，符合训练拟合不能转化为泛化的观察。
选中MLP仍是epoch1，因此“best TRAIN约50%”不能误解为整个训练过程中损失从未下降。
没有为检查最终模型额外加载last.pt、重新生成标签或执行前向。

## 4. 模型、信息与预算

- unrolled：64bit明文，公开Feistel复制/旋转，共享8→32→1 Tanh局部轮模块、连续sigmoid隐表示及学习条件，1089参数。
- MLP：全64bit明文→128 ReLU→128 ReLU→1，24961参数，不传递逐轮隐表示。
- 两者都只监督同一固定未知key下的最终绝对密文bit0，不注入实际key/轮key、中间状态或其他密文bit。
  参数量、激活、深度、先验和初始化不同，不是等参数或纯取消压缩的单因素实验。

共同配置：SIMON64/128六轮，full/direct，data193601/null193602/key190502/init17，
TRAIN65536/DEV8192/TEST65536、train_pool_rows65536、batch256、epochs32、dev_every_epochs1、Adam lr.001/clip1。
四项固定顺序unrolled-real、unrolled-null、mlp-real、mlp-null，各一次。
每项8192更新/2097152次TRAIN呈现，合计 **32768更新/8388608次呈现**；
每256更新一次DEV选择，共32次，末尾对选中模型再报告DEV不参与选模。
每项1800秒协作上限非硬截止；四项result elapsed合计 **391.437秒**，不是SSH/开发/传输端到端耗时。
正式SSH调用UTC约2026-10-26 09:14:32–09:21:58。

同一新面板和shuffle，real/null分别标签配对；同架构real/null初态SHA相同，不宣称跨架构同初态。
139264个不同明文全局唯一，与预列的八个以前面板无交集（含上一65536数据规模面板）；不称所有历史无重合。
每真实臂139264次真实标量加密标签构造，合计 **278528调用/139264不同明文**；null无真实加密。
重复训练呈现不计新秘密oracle查询。AN/Fix仅提供机制控制和查询/计算分开计账的方法学，不是六轮MLP有效性的依据。

CLI自动保存TEST，不是访问强制封存。完整取回后先核字节并只提取TRAIN/DEV写dev-decision，
之后才展开TEST作固定报告；不按TEST选择模型/延长训练。

## 5. 发布、远程和接收

仅本地commit/push，发布11个明确的小配置/测试/文档；没有修改科学模型或训练源码。
GitHub main ref核对成功，Windows新目录由Git复制/fetch/checkout明确提交，训练前后clean。

远端目录：
`G:\lxy\shared-m-cipher-nd-runs\high-round-mlp-20261026-091102\repo`。
物理GPU1 RTX A6000 UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`→logical cuda:0。
每次CUDA smoke及正式臂前实时核空闲≥24000MiB、利用率≤30%；无CPU回退、无安装/升级。

机器适配器从已存在的上批适配器复制，只替换repo路径，设置G:\lxy下cache/TEMP和既有依赖，
然后转交不变CLI参数；不含科学算法。新科学代码/配置均来自Git，没有SSH任务包或科学worker上传。
两模型各作一次CUDA null4更新/16呈现预检，smoke从已同步CPU null配置只修改device，所有产物均在新work目录内。
没有远端commit/push、环境安装或常驻服务；普通SSH不承诺断线存活。

本地接收路径：
`runs/received/high-round-mlp-20261026-091102-{unrolled-real,unrolled-null,mlp-real,mlp-null}/`。
远端原件保留。接收后只核保存JSON与字节SHA，没有加载正式checkpoint、重新生成科学标签或本地模型前向。

## 6. 工程和验收

- 本地Ruff通过、mypy **77个源码文件**通过、pytest **850 passed**；本轮发布前再次执行通过。
- 准备阶段两模型干净源码副本null的check/train/evaluate各4更新，保存TEST与evaluate一致；源/测试/配置/日志SHA已绑定于engineering.json。
- 远端CUDA null两模型各4更新现场通过；不是科学预测成绩。
- 本地与Windows clean明确提交、41个科学源码和四配置字节SHA一致。
- 验收 **46文件**的远端SHA与接收字节SHA一致，核128条epoch进度、每臂32次DEV位置、8192更新、
  初始化和面板/标签配对、最低DEV BCE选模、配置/manifest/commit绑定和正常退出。
- MLP原本不写model_metadata.json，因此每臂11文件；unrolled每臂12文件，共46，不能套用旧全unrolled48文件假设。
  初次验收脚本沿用旧12文件断言而停止；检查目录后仅修正模型对应文件数，未重跑训练/评价。
- 独立只读发布前合同/config与必要调用链审查`76b20c90-3a13-496`，指定范围无确认缺陷；不等于独立复现实验。

证据：`validation/high-round-mlp/{remote-location,remote-preflight,frozen-plan,remote-inventory,remote-postrun,dev-decision,verification,test-report}.json`。
本批不自动继承为下一次GPU或训练许可。若继续，先提出能区别其他机制的新假设与预算，不再仅扩大这个MLP或unrolled训练量。

best SHA256：

```text
unrolled-real 880eee271f4ba5d6a25d586d29eabc339f1c2219a73d5987e18a881c418e678c
unrolled-null b4ac32ea157e5597b9b89021fc4250711bea74c5e435c4bebabf6d7818d1dca5
mlp-real ed2894a070054669ce8fd926c3f77aaa862a8be3dc2d63a708e41136d6ea2e04
mlp-null 3840f81b0890947f3bc200a1da074c5a93670c7d843adff9dd67231fe45162dc
```
