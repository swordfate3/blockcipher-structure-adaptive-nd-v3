# 积分分散轮转周期：避免饱和评分与起点饿死

状态：READY。用户在上轮八轮资源墙报告后明确“是推进”，本轮继续同一可靠判定器的有界搜索策略改进，不把上次预算墙当研究不可推进。
前一周期已关闭，许可不继承；本轮范围来自当前继续推进请求与本次明确的限额，不无限扩展。

## 方法与归属

依据[上轮结果](INNOVATION2-INTEGRAL-CYCLE-RESULTS.md)：192个完整支持分数全1，greedy退化lex；第一条轨迹耗光20Mpair，后两起点未开始。

| 方法 | 本轮决定 |
|---|---|
| 原样追加pair／时间 | 不选：继续购买饱和分数下的连续小位扩展，没有区分信息 |
| 固定种子分散候选＋轮转预算 | **选择**：隔离当前已知调度／饱和评分问题，保留已审查证明语义 |
| 标准DP／SAT／精确系数方法 | 另需规范原文与实现核查；当前未实现，不伪称本轮就是更强证明器 |

目标仍SIMON64/128输出高字MSB bit0，只限输入高字0..31且位于公开目标cone内，非空k≤10。
ANF投影支持：活跃叶单mask，其它明文／符号常量用保守{0}，XOR并集，AND所有pair OR；全集mask缺失才能已证零和，出现只UNKNOWN。RESOURCE_LIMIT不能当UNKNOWN或证书。分散候选不改变此判定。

生产密码／模型／数据／训练／checkpoint／CLI完全不改，当前生产cone≤6轮不扩大。
GLM仅在ignored `validation/integral-diversity/`实现search_diverse.py／test_diverse.py／READY；职责为公开递推、固定候选族、轮转与证据，不接收真实key／密文／label。
真实验证复用已审查`validation/integral-cycle/validate.py`，主模型构造新冻结spec／完整证书；不修改旧源码／产物／合同，保留其他dirty工作。不造通用registry／训练入口／远程编排。

## 冻结的候选与调度

每stage8条候选流，轮转每条每turn最多1个query request：
- lane0 baseline：上一轮已验证cube（八轮用七轮{0,2}）；
- lane1：0..6；lane2：{0,4,8,12,16,20,24}；
- lane3..7：各独立RNG从允许位抽7位作为baseline。
每lane RNG seed=`194800+100*rounds+lane_id`，不看真实输出。

baseline以后未找到proof时，不再用全1的support分数排序：维度按固定循环7,8,9,10,5,6，各自RNG直接sample允许位并canonical排序。记录所有来源、seed、坐标和维度；重复候选可缓存但request照计，不重新搜索到“看起来好”为止。
某lane找到proof后，后续turn做单点删除，成功即重启扫描。最终单点删除全部完成UNKNOWN才称该判定器下局部极小，截断／未访问列pending；一位cube不接受空集合删除。保存已证cube即使删减未完成，不虚构最小性。
每lane最多32requests（含cache），实际pair最多2400000；单query最多1000000pair。某lane额度耗尽不阻止其它lane获得轮转机会。
正常8lane均DONE／REQUEST_CAP／LANE_BUDGET_DONE后，从所有完整PROVED_ZERO记录按k最小再lex选唯一候选；允许选中尚未完成删减的证书但不称最小。全局异常／限额下只留observed best，禁止晋级。

## 阶段与预算

八轮→九轮→十轮，前轮唯一选中cube的真实验证通过才启动下一轮。
正常收口却无完整证书、全局资源限制／异常、真实非零即停；query／lane局部截断只按各流规则结束或继续，不阻断其它流，也不自动取消其它完整证书的晋级。十轮通过也在边界关闭。本轮最多3search／3validation，正式重试0，未找到不能写不可能。

- 每search unique≤256，8lane×32requests（缓存另计unique/attempt），k≤10。
- 每lane pair≤2.4M，query≤1M，stage total≤20M，三个stage≤60M；重复OR／失败均累计。前置12个当前cone1..6全图／支持等价检查pair计全局不计lane。
- 每search CPU＋wall各60秒，query/cache/node/pair内及结束前检查；不是硬进程限制。support cap2048在k≤10最多1024mask下不是主要防线。
- 工程验证只fake／人工小电路／固定公开小fixture，无真实密码。
- 真实验证每stage16个互不相交背景×2^k点，最多16384次；总≤49152。key seed190502只标签侧派生，不保存真实key／额外ciphertext bit／状态。
- 验证使用已审工具既有固定seed schema：背景**194600+rounds**、null **194700+rounds**。八九十轮在旧周期均未真实验证，因此仍是新面板；不改旧工具、不虚构新seed，也不回放旧六七轮面板。
- 每validation CPU/wall各60秒，调用前attempt++，有效返回后completed++，首个真实cube XOR1停止整个周期，无改cube／key重试。null同样最多49152bit。
- 理想随机函数预定每stage全16组零和概率2^-16，至多3stage union bound≤3/65536；调用既有validate产物保留其旧5stage保守上界，不改旧脚本冒称3。随机排列差异另以q(q−1)/(2×2^64)界说明。概率非固定PRNG文件随机性，非结果错误概率。
- fit0／梯度更新0／DEV或TEST选择0／GPU0／SSH0／安装0／发布0。无已授权新复杂求解器。

## 冻结与验收

每stage前核前轮验证、剩余额度、源码／合同／spec／证书SHA与freshoutput。公开搜索新spec只含rounds／previous_cube，start记录原始字节与身份。
新搜索产物`runs/integral-diversity-<UTC>-search-rN/`；验证沿用工具要求`runs/integral-cycle-<UTC>-diversity-validate-rN/`，不称旧周期重跑。
工程主模型独立Ruff／mypy／pytest限定新文件、diffcheck，静态review预算／缓存／截断／lane公平性。真实保存验收只读rawbit／points／SHA／count，不重新encrypt或evaluate。
发生代码／基础设施错误后停，不自动科学重跑；工程人工修复不掩盖已消耗调用。
正常结束或触墙即写结果和ledger CLOSED，报告八条流实际覆盖／截断、证明／实测区别，不把本轮未证扩成所有积分或神经方法失败。
