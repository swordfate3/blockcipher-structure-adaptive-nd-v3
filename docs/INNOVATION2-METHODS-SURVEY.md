# 预测/攻击方法谱系与适配判断（检索记录）

检索途径：IACR ePrint 站内搜索（`eprint.iacr.org/search`）。通用搜索 API 本次不可用（限流/无密钥），
故只使用 ePrint 可核实的标题与摘要，未引用未读原文的具体数值。

## 一、我们已试过的

| 方法 | 结果 |
|---|---|
| 逐 key 精确布尔搜索（TRAIN 一致性） | r4：32 条样本辨识成功；r5：2⁴³ 不可行 |
| 朴素 Tseitin CNF + Glucose4 | r5 不可用；r4 可 0.2s 解出；预算不被守住 |
| 软图 / STE 梯度（硬前向） | 绝对位无样本效率优势；r5 ≈50% |
| 差分位 + MLP | 不及 key 边缘基线，且新面板不复现 |
| 代数单调式展开 | 次数上界 r4=16、r5=32、r6=64，线性化不可行 |

## 二、文献里的方法家族

### 1. 差分与差分-神经（differential-neural）

- 起点：Gohr, CRYPTO 2019（ePrint 2019/037）：Speck32/64 的 9 轮神经区分器，
  11 轮密钥恢复剩余安全度降到约 38 位。
- 后续：Bao 等（ePrint 2021/452 等）用中性比特把密钥恢复推到 Speck 13/22 轮、
  **SIMON32/64 16/32 轮**。
- 综述：Gerault 等《Survey: Six Years of Neural Differential Cryptanalysis》给出分类法。
- 输入表示：Gholizadeh & Atani《Rich Input Representations in Neural Differential
  Cryptanalysis: A Taxonomy and Survey》——单密文对、差分、多对输入的信息量差异很大。
- 数据复杂度比较：同作者《Fixed-Budget Allocation in Neural Differential Distinguishers》
  指出"固定样本数比较"会误导，应按固定密文预算比较。
- **冷水**：Peyrin 等《On the Mismatch between Neural-Discovered Differential-Linear
  Features and Long-Round Distinguisher Construction》指出：现有差分-神经工作
  **尚未显示出相对最强经典分析明确的轮数优势**；且差分-神经学到的特征可用
  经典差分-线性掩码解释。Gohr/Leander/Neumann《An Assessment of Differential-Neural
  Distinguishers》也在系统校验各类声称。

> 对我们是关键提醒：这条线"再训练久一点"不会有质变，必须有更好的特征和更大的数据预算。

### 2. 线性与神经-线性（linear / neural-linear）

- 经典：Chen & Wang《Improved Linear Hull Attack on Round-Reduced SIMON with Dynamic
  Key-guessing》——**目前 SIMON64/128 的最强公开结果（31/44 轮）就来自线性 hull**。
- 神经版：Zhou 等《Neural-Linear Attack Based on Distribution Data and Its Application on DES》
  把深度学习接到线性攻击上（DES 3–5 轮单比特密钥恢复）。
- 对我们的价值：SIMON 的"最好经典成绩"在线性侧，而不是差分侧。若要和大数对话，
  线性是更现实的主线；但同样需要掩码搜索与密钥猜测框架。

### 3. 积分 / division property（可证明型）

- Wang 等《Cryptanalysis of Reduced-round SIMON32 and SIMON48》：积分、零相关线性、
  不可能差分，给出当时最好的积分区分器。
- Hadipour 等《Improved Search for Integral, Impossible-Differential and Zero-Correlation
  Attacks...》（EUROCRYPT 2023）：**约束规划（CP）自动化搜索**，覆盖 Ascon/SKINNY/
  PRESENT 等，并强调"不仅找区分器，还找完整攻击"。
- Khalesi & Ahmadian：用 division property **可证明地**给出最小数据复杂度的积分区分器。
- Zhang 等《Neural-Inspired Advances in Integral Cryptanalysis》《Improving Neural-Inspired
  Integral Distinguishers via a Linear-Algebraic Approach》：用神经网络的判别器当"导航"
  来加速积分搜索。

> 对我们的价值最高：结论是**平衡位/积分性质**这种可验证事实，而不是 54% 这种统计数字；
> 搜索在 CPU 上可跑，且能与我们已有的 cone 工具互相校验。

### 4. 不可能差分 / 零相关线性

- 与积分同属"结构性质"，同样依赖 MILP/CP 自动搜索（见上条 Hadipour 等）。
- 好处：给出的是"某位必然是 0/1/平衡"的断言，**可被穷举小实例直接验证**。

### 5. SAT / SMT / CP / MILP 求解

- 我们的教训：直接对全轮电路建 CNF 求密钥，r5 就崩；且 Glucose4 不响应中断。
- 文献的正常用法是**用求解器搜索特征/路径**（差分、线性、积分），而不是直接解密钥。
- 相关：《What to Guess in Key-Recovery Attacks?》（Beyne 等）讨论"该猜哪些密钥位"，
  这正是我们 r5 需要的算法层问题。

### 6. 代数（Gröbner / XL / MQ）

- 我已实测：目标位作为密钥多项式，次数上界 r4=16、r5=32、r6=64。
  线性化需要 ~C(44,≤32) 个单项式，**量级上不可行**。
- 结论：对我们这条任务线，代数方向可以划掉（除非只用它做低轮小规模验证）。

### 7. 学习侧的方法学

- 《A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input
  Differences》：**自动找 Δ**，正好补我们"手挑 Δ"的短板。
- 《More Insight on Deep Learning-aided Cryptanalysis》《On the Explanation and Enhancement
  of Neural-inspired Differential Cryptanalysis》：解释神经判别器到底用了什么特征。
- 《Rethinking Learning-based Symmetric Cryptanalysis: a Theoretical Perspective》：
  CoTo 模型 + 合取奇偶形式（CPF），给这类"预测"任务一个理论刻画。
- 相关密钥、多对聚合：《Related-Key Multi-Pair Neural Distinguishers》等。

### 8. 侧信道 / 故障（不同威胁模型）

- 《DLFA: Deep Learning based Fault Analysis》《DL-SITM》等：需要功耗轨迹或故障注入，
  与我们"只有明文-密文"的设定不符，**不适用**。

## 三、适配我们条件的排序建议

约束：单机（CPU + 一张 A6000）、无大规模数据、需要可复现、结论必须可验证。

| 优先级 | 方向 | 理由 | 成本 |
|---|---|---|---|
| **1** | 积分 / division property 自动搜索（先做小轮验证） | 结论可证明、CPU 可跑、能与现有 cone 互证；不依赖训练 | 低 |
| **2** | 线性 hull 掩码搜索（MILP/CP）+ 小轮密钥猜测 | SIMON 最强经典结果在线性侧，路径清晰 | 中 |
| **3** | AutoND：自动找 Δ + 多对输入 + 固定密文预算比较 | 唯一能让神经方法"够得着文献"的做法 | 中高 |
| 4 | 换 XOR 原生求解器重试 r5 | 只解决可行性，不一定提升能力 | 低（需安装） |
| — | 代数 MQ / 朴素 SAT 解密钥 | 已由我们自己证明不可行 | 放弃 |

## 四、期望管理（来自文献的冷水）

- 差分-神经**尚未证明**相对最强经典分析有轮数优势（Peyrin 等）。
- 我们要的那类"单比特可预测"结论，在文献里通常只是完整攻击的中间产物，
  单独报告价值有限。
- 因此更现实的目标不是"打败 31 轮"，而是：**在积分/线性这类可证明结构上，
  用自动化搜索给出别人没报过的小轮结论，并且可复现。**

## 未核对项

- 上述论文的具体数据/时间复杂度只按摘要转述，**未逐篇读原文核对表格**。
- ePrint 搜索本次未覆盖期刊/会议非 ePrint 的成果（如部分 FSE/ToSC 论文）。
