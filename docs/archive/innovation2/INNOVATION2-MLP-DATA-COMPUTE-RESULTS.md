# 六轮MLP数据量／计算量六项结果

## 回答数据量问题

本次四倍TRAIN、以及在大数据上再四倍更新，都没有带来可靠的六轮泛化信号。
真实TEST：A65536行/8192更新 **50.1160%**，B262144行/8192更新 **50.2106%**，
C262144行/32768更新 **50.2106%**。三个随机标签对照50.1282%/49.9969%/50.0488%。
所有条件DEV/TEST均未达到预定55%、比常量及paired null均高3pp等探索门槛。

B比A的DEV提高1.07422pp、TEST仅提高.09460pp；B/C真实结果相同，均选中大数据第4epoch。
不能把这解释为有效突破，也不能断言数据量永远无关；这只是当前MLP/优化器/单key/init/预算的阴性结果。
现有证据不支持把失败简单归结为“65536样本太少，堆到262144就行”，也不支持自动继续上百万样本。
不是六轮不可学习、完整44轮安全或任何模型样本复杂度下界证明。

## 合同与实际运行

用户先确认准备、后在六项预算说明后“推进”，授权本批发布/远程GPU运行/取回。
[冻结前合同](INNOVATION2-MLP-DATA-COMPUTE-PLAN.md)。运行commit **`9335105b74aea724bd17096a6e8be1afacd9cac0`**，
GitHub发布核ref；远端只fetch/checkout clean exactcommit，不commit/push。科学源码与上批一致，未安装升级。
运行ID `mlp-data-compute-20261026-120656`，六CLI UTC约12:08:57–12:20:27（含启动/构造等，非训练elapsed总和）。

共同：SIMON64/128六轮、绝对密文bit0，full/direct MLP64→128 ReLU→128 ReLU→1、24961参数，
key_seed190502/init17/data193801/null193802，Adam.001/clip1/batch256，不读取真实key/状态/其他bit教师。
六项初始state/trainable SHA全部一致；A TRAIN为B/C前65536，B/C全TRAIN相同，DEV8192/TEST65536共同。
335872不同明文，与预列10旧面板零交集，不声称核遍全部历史；unused A池不生成标签。
B/C前8epoch保存的在线TRAIN BCE、更新/呈现计数及共同DEV点数值一致，不据此声称独立核验每次更新的权重。
各split的配对真实/null标签SHA检查通过；A TRAIN标签前缀性由实现/工程小样本支持，未本地再生科学标签。

| 条件 | TRAIN | epoch | DEV间隔epoch | 每项更新 | 每项呈现 | DEV选择机会 |
|---|---:|---:|---:|---:|---:|---:|
| A | 65536 | 32 | 4 | 8192 | 2097152 | 8 |
| B | 262144 | 8 | 1 | 8192 | 2097152 | 8 |
| C | 262144 | 32 | 4 | 32768 | 8388608 | 8 |

依次a-real/null、b-real/null、c-real/null全部一次完成、exit0，**98304更新/25165824呈现**。
A/B同每1024更新选模，C每4096更新选模：机会数相同但网格不同，未提供完整2×2因子格（缺小数据长训练），
不能完整估计交互或所有缩放规律；A→C同时变数据与计算，不做单因素归因。
真实标签构造A139264、B/C各335872，共811008加密调用/335872不同明文；null无真实加密。
重复epoch呈现不算新oracle查询。同更新不等于等FLOPs/耗时；每项1800秒为协作参数，不是硬截止。

## 最低DEV BCE选中模型

| 项 | best epoch／更新 | TRAIN ACC | DEV ACC | DEV BCE | TEST correct/65536 | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-real | 4／1024 | 54.6936% | 49.8413% | .69476651 | 32844 | 50.1160% | .69461804 | 52.765 |
| A-null | 4／1024 | 53.3768% | 50.8911% | .69431286 | 32852 | 50.1282% | .69464525 | 50.829 |
| B-real | 4／4096 | 52.1973% | 50.9155% | .69294288 | 32906 | 50.2106% | .69332200 | 57.578 |
| B-null | 3／3072 | 51.1555% | 50.2686% | .69321372 | 32766 | 49.9969% | .69337885 | 55.172 |
| C-real | 4／4096 | 52.1973% | 50.9155% | .69294288 | 32906 | 50.2106% | .69332200 | 197.234 |
| C-null | 4／4096 | 51.4332% | 50.2930% | .69360516 | 32800 | 50.0488% | .69350910 | 190.625 |

真实selected TRAIN BCE：A.68774811、B/C.69219239。B−A DEV BCE−.00182362、TEST BCE−.00129604，
但B/C真实TEST BCE仍差于各自TRAIN-only常量（A.69314483、B/C.69314535）。
真实TEST常量准确率共同50.1251%；B/C比常量仅+.08545pp，远未达3pp。
B/C真实selected各split指标相同，不据此证明逐条预测一致或checkpoint文件相同；checkpoint保存config不同，字节SHA不同。

各项DEV最低BCE、平局最早；strict load选中checkpoint后才生成TEST标签。
外层先核字节并只取TRAIN/DEV写`dev-decision.json`，再展开TEST；CLI已保存TEST，不称强制封存。
A在DEV仅null范围通过；B/C在DEV另有BCE优于常量，但ACC及3pp条件失败。TEST三条件均仅null范围通过。
没有以小幅波动放松门槛或依TEST选择新配置/继续拟合。

## 训练过程：更多数据减轻了损失恶化，但未出现预测信号

以下是每项末轮**在线TRAIN BCE**及末轮DEV，不是last checkpoint重评估TRAIN准确率：

| 项 | 末epoch | 在线TRAIN BCE | DEV ACC | DEV BCE |
|---|---:|---:|---:|---:|
| A-real | 32 | .51246277 | 49.4629% | .89671205 |
| A-null | 32 | .52611248 | 50.0977% | .88409656 |
| B-real | 8 | .68850242 | 49.7681% | .69645572 |
| B-null | 8 | .68873421 | 49.2554% | .69879702 |
| C-real | 32 | .63218678 | 49.6826% | .76204356 |
| C-null | 32 | .63362854 | 50.2197% | .75417718 |

固定更新A→B，训练过程损失下降更少、DEV BCE恶化较轻，符合不同样本重复减少的现象，不能把“少过拟合”说成“学会结构”。
B→C继续降低TRAIN过程损失但DEV恶化，真实/null类似；在线BCE也不支持完整记忆或未记录的最终TRAIN准确率推算。
每epoch平均窗口A256更新、B/C1024更新，不能将其当同窗口逐步曲线。

## 工程、取回与停止

本地875tests/mypy80/Ruff/diff通过，三条件干净源码CPU null check/train/evaluate通过（A/B8更新、C32更新）。
远端CUDA工程预检辅助命令首次PowerShell解析失败（f-string花括号未转义）；解析发生在执行前，没有CLI/科学更新。
仅修正忽略的辅助脚本并保留失败日志，三条件CUDA null预检随后通过，未修改Git科学源码或重试正式fit。
主机DESKTOP-BBLPACJ、G:\lxy边界、fresh repo/clean exactcommit、41源码/6配置SHA匹配；原有adapter只转CLI与环境路径。
物理GPU1 UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`→逻辑cuda0，各项前核free≥24000MiB/util≤30%，无fallback。
未SSH上传科学源码/任务包、安装环境、远端commit或运行中pull；普通SSH不承诺断线存活。

六run各11文件，共**66文件**SCP完整取回、核远端字节SHA/大小、保存manifest/config/source/commit/init/panel/labels/selection/count/exit；
A/C各32progress、B各8progress，合计144记录/48DEV选择点；elapsed合计604.203秒，不是端到端耗时。
接收没有加载正式checkpoint、再生科学标签、evaluate或模型前向。静态审查不是独立复现。

最佳checkpoint SHA256：

```text
a-real 11c111235a1cc93f2b27dff0e79029fb13f409b4db19dd831729622b8fce0371
a-null 1ed736f66eb10ede50ec440825945ff63974310053a5c1c87da599d8ee5ea96e
b-real d58ca36dad06429bc075824c6dc592b8cd8dfc0b685d5e9ae42ca3f024cbd2be
b-null 1135054854e0493b6fc29863d428be8e4faa64b51341ce5b144e37a26dafdc6c
c-real a041ba22b4543072e9d24edc0a6b209160ca1a4b3ad26ef13fc611e333270776
c-null e2ed08b88822eb86b96f158c54744e637e4f883f23d472d822d7ac5bad9e5646
```

证据：`validation/mlp-data-compute/{frozen-plan,remote-inventory,remote-postrun,dev-decision,verification,test-report}.json`。
接收：`runs/received/mlp-data-compute-20261026-120656-{a,b,c}-{real,null}/`；远端原件保留于
`G:\lxy\shared-m-cipher-nd-runs\mlp-data-compute-20261026-120656\repo\runs\`，repo保持运行commit且clean。
本批关闭，无追加拟合/seed/key/数据/epochs。下一步须新假设和单独授权，不自动扩规模或回低轮。
