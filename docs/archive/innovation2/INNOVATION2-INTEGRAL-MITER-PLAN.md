# 固定cube精确Boolean反例判定：八轮两候选

状态READY；继用户“是推进”与分散搜索无证书后，在同一推进请求内针对证明器本身换判定信息。不恢复旧周期额度，不无限加预算，不改变密码／训练／CLI。

## 方法选择和边界

旧支持法丢弃系数／共享路径抵消，65个分散完整候选仍全支持。下一项不是换seed。
比较：
1. **完整Boolean SAT miter：选择**。复用本机PySAT现有Glucose4，不实现solver，只常规AND/XOR CNF门模板和结果验收。固定cube的2^k份电路共享outside明文／符号轮常量，输出XOR=1为反例。
2. 显式outside-coefficient ANF／BDD：先不做，k≤10不限制每个系数复杂度，当前没有BDD依赖。
3. 标准bit-DP：具备显式COPY需求约束，可强于当前独立父支持合并，不贬为同一算法；完整规范实现另论。本文不叫标准DP。

主模型负责方法／预算／验收；DS编码代理仅新建ignored临时miter.py/test_miter.py/READY，职责publicDAG→完整CNF→现成solver→数学witness核验。不改密码／生产cone上限／模型／训练，也不开发复杂求解算法或自研proof checker。已核没有verified sol代理，不冒称DS就是sol。

## 数学问题

固定r8 highX8[0]，两项依序：S={0,2}；S={0,1,2,3,4,5,6}。均为既有公开起点，绝不按真实密文选cube。

令u为64-k个outside明文bit，c为各轮每位**独立符号轮常量**，每cube内固定跨全部副本共享；x为active变量。

I_S(u,c)=XOR_{a∈{0,1}^k} f8(a,u,c)。构造Phi=[所有副本公开递推] AND [I_S=1]，存在SAT解就给该符号模型的反例；UNSAT才可能支持对全部u,c恒零。

常量不是0，不读取实际key／roundkey／状态／其他cipher标签；递推包含完整公开Feistel接线。SAT的c可能不满足SIMON主密钥调度，因此只反驳独立轮常量下恒零，**不能声称存在合法key反例**。公开witness只作数学检查，不用于选择实际key／seed，不作为模型输入或教师。

## 预定预算和执行

- 两候选，各单次启动，默认正式重试0。编码cap250000vars／1000000clauses；CNF／proof／model类文件各64MiB上限。
- 各项编码／求解／核验总体协作CPU与wall各60秒；solver conflict cap200000；Glucose4 solve_limited配wall剩余Timer interrupt、before/after双时钟核对；不是硬CPU／内存进程限。创建空solver后一次完整append_formula，使部分加载异常也能finally清理；只有append完整成功才计loaded，partial永不solve。
- 全部副本、共享叶、末端parity与单位约束完整生成，写CNF+mapping后才完整加载solver。任何partial／不完整加载不得得UNSAT结论。构造／加载／求解／取model或proof异常均保存partial状态与实际计数；所有路径取消并结束Timer，之后释放solver，不能让晚到interrupt触碰已释放对象。
- 60秒计时累计覆盖构造、序列化、完整加载、求解、proof／model导出与核验；spec读取也≤64MiB，所有逐条写入以文件累计字节检查64MiB，不因append重置，结束再核文件尺寸。solver内部proof临时文件另以协作监测／取回时fstat检查，采样之间可能短时超限，不能冒称hard disk quota。超限停止科学处理，只保留有界状态；协作检查不是硬时限。
- 模型若SAT：要求所有CNF变量有值且每子句满足，不缺值补0；独立32位word公开递推重算2^k个符号输出，与miter副本output模型逐项比较并核XOR1；保存核验witness。
- False在正常处理时只记UNSAT_UNCERTIFIED；保存原始proof若可用，但**当前无独立proof checker，不晋级零和证书**。proof读取／导出异常或超限也必须顶层UNKNOWN／ERROR并阻止第二候选，不能掩盖为正常待认证UNSAT；保留原始False。None／超限／异常记UNKNOWN／ERROR；坏模型记拒绝／未验证，不能说SAT反例已成立。
- 第一项验证SAT或正常UNSAT_UNCERTIFIED才允许第二项；第一项UNKNOWN／异常／预算停止即停，不自动追加其它cube／key／时间／方法。第二项任何结果后关闭。
- 真实encrypt0／null0／训练0／GPU0／SSH0／安装0／发布0，无任何结果触发真实密码验证或更高轮。

## 产物和验收

fresh runs/integral-miter-<UTC>-r8-kN/，exclusive目录；spec只rounds8和上述精确active之一。保留CNF、变量映射、spec原字节、source／PySAT／Python／solver身份、进度、完整计数、SAT模型／UNSAT待认证proof或unknown原因、exit与byteSHA。缺exit未知，不能把工具timeout冒作已知零调用。
冻结前主模型独立fake／人工小图／固定公开1..6**全部可达节点、边、坐标与根**等价测试、Ruff／mypy／diffcheck及只读review；这些工程检查不执行r8正式求解或真实cipher。≤6全节点比较是工程验证，r8同实现互验也不冒充独立正确性证明。
数学重算仅公开符号Boolean函数，不调用现有encrypt；模型核验不是新真实密码标签实验。
保存结果核SHA、CNF counts／映射／副本完整性、solverattempts和witness是否真正核验；原件不改、不自动重跑。
