# 公开SIMON结构未知效应学习：首次GPU试训练

## 状态

用户在结构模型与r1→r3→r4有界探针方案后明确要求“推进”。
本轮**实际运行r1和r3，r3失败后停止，r4未启动**；批次关闭，不追加seed/epoch/重试。
公开结构及从TRAIN学习未知密钥效应是允许的先验，不将参数辨识本身视为泄露。

科学代码本地提交并push：`111adab6fba0df0369e65c9a5a7312e1d8481fa6`，GitHub ref已核对。
Windows通过既有GitHub SSH fetch明确commit，干净detached checkout；没有SSH传科学源码/任务包。
普通SSH调用已同步CLI，使用既有机器依赖适配器，没有安装/服务/远程编排。

## 固定合同

- 模型`simon-structure`，SIMON64/128目标bit0、full/direct，无额外head。
- TRAIN4096 / DEV1024 / TEST8192，data191701 / key190502 / init17。
- batch256，50epochs，每项800更新、204800presentations，Adam.01、clip1。
- DEV软绝对BCE选模；硬化`key_logits>0`仅报告，不选模。
- TEST标签在选中checkpoint严格加载后生成，不称访问隔离的sealed confirm。
- 每项1800秒协作上限；不是硬进程/GPU时限；失败不重试。
- r1门槛soft/hard TEST≥99%且soft BCE<.05；r3门槛soft TEST≥70%且BCE优于两类TRAIN基线。

设备：DESKTOP-BBLPACJ，物理GPU1 RTX A6000，UUID
`GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`，映射逻辑`cuda:0`。
每次启动前现场门槛：free≥24000MiB且util≤30%，两次均通过；首次GPU1 free48401MiB/util0%。
Python3.12.10、Torch2.14.0+cu130、CUDA13.0，复用已有依赖。

## 实际结果

| 项目 | r1正控 | r3结构模型 |
|---|---:|---:|
| 可训练参数 | 1 | 11 |
| 最佳epoch | 50 | 50 |
| TRAIN软ACC | 100% | 49.3652% |
| TRAIN软BCE | 0.00563239 | 0.69318103 |
| DEV软ACC | 100% | 49.9023% |
| DEV软BCE | 0.00563239 | 0.69318715 |
| TEST软ACC | **100%** | **50.5127%** |
| TEST软BCE | 0.00563239 | 0.69316650 |
| TEST硬化ACC | **100%** | **49.8291%** |
| TEST硬化BCE | 0.0000009833 | 6.92798296 |
| TEST原始TRAIN常量ACC/BCE | 49.7192% / 0.69316387 | 51.0620% / 0.69311715 |
| TEST公开参考＋TRAIN残差常量ACC/BCE | 100% / 0.00012206 | 62.3291% / 0.66255161 |
| 保存elapsed_seconds | 9.281 | 15.750 |
| updates / presentations | 800 / 204800 | 800 / 204800 |
| exit_code | 0 | 0 |

r1通过最小未知效应正控，但公开常量基线本就能100%，不是方法优势。
r3在TRAIN和未见TEST均无有效信号，不是“训练准确率高、TEST失效”的典型过拟合。
其DEV BCE从epoch1的0.705757降到0.693187，只是改善到近随机，不能称学习了密码规律。
r3硬化DEV正确数每epoch均510/1024，支持硬路径没有出现可见的有效改进；
没有加载权重检查参数轨迹，因此不进一步断言具体哪些参数跨越阈值或趋向0.5。

合计1600更新、409600presentations，保存elapsed相加25.031秒，非精确GPU利用时间或端到端时间。
两个run使用同一明文panel，密文标签按轮数不同。只有一个key/seed，未跑新面板MLP/null。
历史MLP r3约81.94%采用不同数据与样本预算，不能当作本轮严格匹配架构对照。

## 结论与下一方向

**当前“逐个未知位sigmoid＋软AND/XOR＋随机有符号初态”的训练参数化没有通过r3正控。**
这不是公开结构不可用、学习密钥效应违规、r3不可预测或不存在正确电路的证据。
硬布尔模型可表示真实目标，但“包含正确函数”不保证本优化过程能找到它。

先前任意标签梯度小检查已经提示软化会趋近平坦；本轮BCE趋近随机与此相容，
但尚未隔离局部极值、符号对称性、重汇聚松弛和样本预算的具体贡献，不宣称已定位唯一原因。
下一步应优先考虑TRAIN-only离散参数搜索或更保留布尔结构的目标，
不直接扩样本/加epoch/换seed，也不把当前失败改写成“更多训练就会好”。
r3只有11个坐标（2048硬组合），可作为未来穷举TRAIN一致性的廉价诊断候选，
但本轮**没有执行**该搜索、没有用TEST选组合、没有从生成器取真实参数。

## 可追溯证据

远端代码：
`G:\lxy\shared-m-cipher-nd-runs\git-validate-20261019-060000\repo`

远端run与本地接收同名：

```text
innovation2-structured-111adab-20261020-r1
innovation2-structured-111adab-20261020-r3
```

本地：`runs/received/innovation2-structured-111adab-20261020/`。
24份原始产物已取回，远端字节长度与SHA逐份验证；`artifact-inventory.json`、`verification.json`
记录配置、Git/source、初始化、DEV选择、checkpoint文件SHA、50行progress、计数和exit核验。
38个源码文件匹配指定commit，其中8个仅LF/CRLF差异（以verification实际列表为准）。
取回后没有本地加载checkpoint、生成标签、模型前向或重算指标。远端原件保留。
远端只读确认`runs/innovation2-structured-111adab-20261020-r4`不存在。

| 产物 | r1 | r3 |
|---|---|---|
| configSHA | `b829f5edddd3f66253dbf3181d1fdc9e1b1870bce5475d969f582f6dc2371be7` | `445bf34bb33a06b0b2da26676ebb70ed6a3486f18bc968f7ef4cb21d00a1def0` |
| bestSHA | `5f32867735a7361d795b5bac7dccd8fdeda6c3a3a8026dd0793ae6e67a882283` | `f08b5834031e4cf7eb050d7b62e2ca29a020f3acec1ac5a8bdcf756b7bc237c1` |

本地执行证据：`validation/structured-gpu/`。
初次同步命令已在远端完成，但本地读取PowerShell中文stderr时UTF-8解码失败；
随后只读确认目标commit、clean状态及adapter SHA，没有重复训练。
发布前681tests/Ruff/mypy57files通过；本轮未改训练实现，仅新增结果记录。
