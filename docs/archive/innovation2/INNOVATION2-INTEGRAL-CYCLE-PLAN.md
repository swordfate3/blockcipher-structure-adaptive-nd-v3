# 积分连续推进周期：先验证六／七轮，再有界推进至十轮

状态：READY；用户“继续推进，等推不动再停”后，本周期按下面固定方法族连续执行，不每个局部步骤重新询问；不解释成无限资源授权。

## 范围与方法选择

已有[公开搜索](INNOVATION2-INTEGRAL-FRONTIER-RESULTS.md)得到六轮{0,4,10}／七轮{0,2}证书，但两个小cube尚未真实验证。
先验证两者，通过后逐轮八→九→十轮同公式搜索、验证；任一科学反例、无证书或资源墙立即停。最多十轮，即使通过也在周期边界关闭。

本轮仍是选择明文积分／结构分析，不等价于原随机明文单bit神经预测。目标始终SIMON64/128高字MSB bit0。未知key不进入公开证明器；无实际状态／额外输出bit老师。
继续已审查的ANF投影支持过近似，不引入新科学方法。标准division-property/SAT实现和神经排序暂不选：前者尚无完整核读实现与批准安装，后者旧外推／排序未显示优势。成功不宣称文献创新或key恢复。

## 模块职责

- 生产cipher/model/config/data/training/evaluation/CLI保持不变，当前`simon_cone`生产六轮限制不扩。
- ignored的`validation/integral-cycle/search.py`仅负责公开递推、支持判定、固定启发式和其证据；不得接收真实key／密文／标签。
- 同目录`validate.py`只负责已冻结证书的选择明文构造、标签侧完整块oracle调用、目标bit提取与XOR统计；不训练／不搜索／不把真实答案回馈选择另一cube。
- 现有`ciphers/simon.py::encrypt`是唯一真实oracle，不复制密码算法。真实key仅标签侧局部派生，不保存真实key／轮key／完整密文／状态。
- 两脚本是本地一次性研究工具，不是通用runner、registry、新生产入口或远程编排。现有check/train/evaluate没有积分选项，不虚构参数。
- GLM实现search/test_search/SEARCH；DS实现validate/test_validate/VALIDATE，文件互不重叠；主模型写协议、冻结、运行、账本与验收。不再派子代理。

## 固定验证协议

每个已选cube验证16个新背景，每背景枚举全部2^k点，背景先清活跃位并拒绝重复保证组间不相交；输入位和输出位均按64位MSB编号。

- key seed190502，标签侧`random.Random(seed).getrandbits(128)`，沿用既有key身份，仅新背景；公开seed可重建key，不称对读者保密。
- 背景seed=`194600+rounds`；null seed=`194700+rounds`，局部独立RNG。
- null为每个不同明文独立均匀目标bit的可复现伪随机实例，代表随机函数单bit零假设，不直接模拟随机排列。
- 六轮S={0,4,10}：16×8=128次；七轮S={0,2}：16×4=64次；先六后七。
- 八／九／十轮每轮仅验证公开搜索选中的唯一cube，k≤10，每轮最多16×1024=16384次。候选只依赖公开结构，不看真实／null结果择优。
- 正式最大总真实调用192+3×16384=49344＜50000；null同上。每point oracle一次，禁止重复科学复验；attempted调用先扣账，失败不返还。
- 每个stage CPU＋monotonic墙钟协作60秒，调用前检查，非硬进程限制。
- 首个真实cube XOR非零立即停止整个周期，保存partial，不换key／cube重试。异常／未知退出同样停；缺exit不当零消耗。
- 保存每组完整目标bit字符串／ones／XOR／背景／points SHA，不保存真实key或其他ciphertext bit。结果验收不重新加密。

预定每个stage事件为16组全部XOR0，理想随机函数下概率2^-16；至多5个stage的保守family union bound≤5×2^-16（不声称stage间独立）。候选自适应只看公开结构／前轮是否通过，不从本轮标签挑选。随机排列与随机函数的概率加性差异可用每stage q(q−1)/(2×2^64)或全周期总q对应界说明；不同轮不是同一个oracle，不混用跨轮独立假设。

## 八至十轮公开搜索族

依次仅在前轮真实验证通过后启动；每轮独立冻结spec，包含轮数和上一轮已验证cube。

- 同一公开SIMON递推展开至X_r[0]，输入候选仅高字0..31且在目标cone内，非空k≤10。
- 三个固定起点：上一轮已验证cube、0..6、{0,4,8,12,16,20,24}；过滤cone外位置，不能利用缺失依赖制造成功。
- 每起点先baseline；有证书则逐位删除，成功即重启扫描。未证则逐步加位，每步完整比较可行加位候选，优先PROVED_ZERO，再输出支持比例len(Q)/2^k小者，再字典序。k10仍未证则该轨迹停止。
- 截断候选RESOURCE_LIMIT不当UNKNOWN，不以部分支持作低分或证书；全部候选截断即停该轨迹。完整输出全集mask缺失才可PROVED_ZERO。
- 三轨迹结束，从所有完整证书选k最小、再字典序最小的唯一cube；不声称全局最优。最后单删全部完成UNKNOWN才称该判定器下局部极小；截断／未访问存在则列pending。
- 每stage最多512个unique query，cache命中另记；每query pair尝试≤1000000，每stage总≤20000000（含一致性检查、失败尝试、重复OR）；node支持≤2048。k≤10时最多1024mask，实际主要预算防线是pair与时间。
- 每stage CPU＋wall各60秒协作，query/cache/node/pair内检查；最多3stage／1536query／60000000pair／180秒搜索协作。全局限额立即partial并关闭，不追加search。
- 正式搜索每轮最多一次，验证每轮最多一次，所有正式重试0；工程人工fixture修正分列，不能用于掩盖科学重试。
- 当前生产cone一至六轮全图／支持一致性核查（12个固定小案例）在每stage预算内记账；七至十轮静态核递推同公式，不靠旧代码动态导入或改生产接口。

## 停止含义与权限

1. 真实出现反例：停止，调查证明／模型／位序；不能继续更高轮或自动修完重跑。
2. 某轮有界搜索无证书或全局资源耗尽：停止，报告未证与截断／未访问，而非密码不平衡／该轮不可能。
3. 真实调用预算剩余不足最坏stage成本：不启动。
4. 十轮验证通过也停：方法在此周期未卡死，但授权边界已到；不称推不动，另报剩余问题。
5. 安装、GPU、SSH、发布、checkpoint replay、训练／evaluate均未授权；本周期只允许已列本地公开计算与最多49344真实加密。

## 冻结、验证与归档

先用fake oracle／人工Boolean小图检验位序、source等价、証书绑定与全集缺失、预算／attempt/completed、截断／异常和独占输出，主模型独立运行Ruff、两组mypy／pytest与diff-check。
每阶段在真实／公开调用前冻结脚本、spec、证明来源／source SHA及输出目录。脚本或spec身份不符／输出已存在即停。
各stage新输出`runs/integral-cycle-<UTC>-<stage>/`，保存config/start/progress/results/exit及文件SHA；raw产物不Git，源码不经SSH传送。
主模型只读核保存目标bit、XOR／counts、公开明文去重／points SHA、证书mask／source及byte SHA，不再生科学标签。独立审查须说明覆盖是静态／保存记录，不能冒称重新加密。
阶段状态与实际消耗记录于`validation/integral-cycle/ledger.json`，周期结束另写结果／当前状态，保留无关dirty和历史证据。无commit/push。
