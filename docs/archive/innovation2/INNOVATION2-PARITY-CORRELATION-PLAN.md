# 六轮固定64奇偶特征：TRAIN-only固定相关性核查

## 当前权限与目的

用户在[无标签候选核查](INNOVATION2-LINEAR-FEASIBILITY-RESULTS.md)后问如何做，并说“推进看看”。
先按约定完成纯统计实现、合成测试、固定字典和预算。用户随后明确“确认”，授权一次本地CPU正式核查：
65536条真实TRAIN标签及同明文随机标签，零模型更新、无DEV/TEST、GPU/SSH/安装或commit/push。
授权仅限本次，不把已关闭批次解释成新权限。目标仍是六轮最终密文bit0，不回低轮追分。

本次问题只有：**已有64个、完全不看标签生成的奇偶特征里，是否有可测且跨TRAIN两半保持的相关？**
这不是高精度预测、模型攻击或神经轮组合学习；即使相关存在也不自动获得55%准确率。

## 固定字典与一次数据预算

- 字典：上次六轮bit0/beam64/局部变量cap12保留的64个去重非零明文mask；顺序与内容冻结，禁止新搜索。
- 固定已有编译文件的字节SHA及canonical mask-list SHA；默认不重编、不按相关性改beam、mask、输出位或key。
- 现有`parity_statistics.py`只接收numeric明文、binary标签和mask，不接受key/key_seed、cipher evaluator或模型。
- 未来真实标签：SIMON64/128六轮、bit0、key_seed190502、仅数据层使用，不返回/记录key值或真实状态。
- 新data_seed193901/null_seed193902；65536个不同64bit明文，前32768为TRAIN-A、后32768为TRAIN-B。
- 只构造一份真实TRAIN标签（65536次真实加密）和一份相同明文上的split=train随机标签（无真实加密）。
- 不构造DEV/TEST、不读取其标签/旧正式checkpoint、不访问旧科学预测或执行evaluate。
- 一次CPU统计，零模型更新、零GPU；max_seconds提案120秒协作预算，非硬截止，技术失败保留记录不重做科学标签。
- 明文仅无标签核预列历史面板交集；不能声称覆盖所有旧历史。正式分析仍是单key、单面板、一次null。

## 位序、估计量与方向冻结

mask为两个32bit整数(high,low)，内部mask bit i表示MSB坐标i，不能直接当numeric word bit i。
转换后numeric_mask=(reverse32(high)<<32)|reverse32(low)，每行
`chi=(-1)^parity(plaintext & numeric_mask)`、`z=(-1)^y`。
每半每特征保存精确整数`sum(z*chi)`、N、rho=sum/N、`sum(chi)`及`sum(z)`、原始未翻转正确数。
rho是符号乘积的均值，不是减均值/标准差后的Pearson系数；在不平衡样本上也不能混写为centered covariance。
附带两种均值帮助观察样本偏置，但它们只是描述量，不额外声称被rho置信区间覆盖。

方向**只由A**决定：sumA≥0选+（0固定+），否则−；正方向预测parity bit、负方向预测其补。
将A锁定方向应用B，B正确数=(N+directionA*sumB)/2。不能按B再翻转后把`(1+abs(rhoB))/2`称独立准确率。
64个特征全部报告；不以TRAIN-B挑最好mask后声称它有独立未选择验证集。
若需要一项描述性首选，只按A的abs(sum)最大、平局原mask顺序最早，报告它在B的锁定符号结果；
B仍是同一TRAIN内复查，不是最终TEST确认，也没有额外key独立验证。

## 预先固定的同时误差带

总体rho定义为固定未知key、固定mask、均匀明文下E[z chi]。
对理想独立均匀样本，Hoeffding对[-1,1]量给
`Pr(|rho_hat-rho|>eps)<=2 exp(-N eps^2/2)`。
一起报告64mask×2半×real/null = **256估计**，family_alpha=.05，
`eps=sqrt(2 ln(2*256/.05)/32768)`约.02374；区间裁剪[-1,1]。
API默认只覆盖单次调用两半2M，正式联合real/null必须显式family_tests=256，不偷用128或仅数事后挑出的mask。
估计之间不要求独立，union bound仍成立；独立采样假设涉及每一估计内的样本。
项目生成器实际是伪随机唯一明文、无放回；以上是理想抽样参考，而非证明PRNG具有概率覆盖或精确p值。
明文总体极大、采样比例很低，也不能将这种直觉写成抽样假设已经被审计。

每个mask的稳定相关标记要求：A/B的同时带均排除0且同号；仅同号但落在带内不算检测。
一次null是异常对照，不是经验零分布；真实减一个null的差值不能直接称显著。
如果null也有此强标记，停止推进、报告异常，不能直接剔除null或换seed重做。
总体单特征最佳符号准确率=(1+abs(rho))/2，55%要求|rho|≥.1；上述带宽对应约1.187pp的单特征准确率不确定量级。
带宽.02374也意味着无法可靠排除1/1024量级的弱真实相关；未检测不是rho=0证明，亦非多特征组合无效证明。
未知路径hull仍可能增强/抵消，不能把编译路径1/1024直接拿来当rho或当统计检验的真值。

## 事先停止与解释

- 检测到稳定相关，且A所定方向在B保留、null无异常：仅形成TRAIN内证据；另设计新DEV/TEST的小型正则化head提案。
- 稳定但微弱：记录结构基线证据，不直接启动GPU，不降低历史55%/3pp门槛。
- 无检测或符号不稳／与null类似：关闭此固定字典核查，不自动扩beam、数据、seed、key、特征或epochs。
- 核查未执行、失败或缺exit：报告未执行/失败/未知，不能填50%或“无信号”。

## 当前工程范围

新纯标准库`src/cipher_experiments/parity_statistics.py`，不可变报告、样本/类型/唯一性/掩码/家族数校验与MSB转换。
`tests/test_parity_statistics.py`只用合成数据，含独立字符串位序oracle、正负/零相关、A符号锁定与反转、误差带和坏输入。
不在生产训练系统加新入口、registry或自动科学分析；正式核查不是train/evaluate，不伪装成模型拟合。
准备阶段忽略validation存字典冻结、无标签面板预检、工程检查与干净源码CPU null CLI回归（不是本核查的科学成绩）。
正式核查写新runs目录，保存聚合统计、身份与退出，不输出逐行标签或实际key；无训练CLI替代或新入口。
正式执行前应另冻结数据源码/字典/计划身份；正式执行后保存聚合整数/SHA/退出，不输出实际key或逐样本秘密标签。
保留未提交编译器工作、papers/和其他owner refs；没有明确发布请求不commit/push。
