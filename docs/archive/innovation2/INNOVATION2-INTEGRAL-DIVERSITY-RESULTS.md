# 分散轮转搜索结果：覆盖问题已解决，八轮支持饱和仍在

本轮按用户“是推进”，执行[冻结合同](INNOVATION2-INTEGRAL-DIVERSITY-PLAN.md)中的八轮搜索，不重开上一周期，不新增训练／GPU／SSH／安装／发布。

## 实际结果

- 原件 `runs/integral-diversity-20261029-183700-search-r8/`（路径与`validation/integral-diversity/r8-search-location.txt`一致，下述汇总由原件读取）。
- COMPLETE／exit0，所有8lane正常达到局部预算终态，不是全局异常。`stage_choice=null`，`observed_best=null`。
- 总pair **19200872**＝8lane×2400000＋source872，含重复OR／失败；80requests／80unique／0cache，65完整UNKNOWN／15RESOURCE_LIMIT／0ERROR／0PROVED。
- source12/12完成，是固定1–6轮图／支持对照，不是八轮独立完整验证。
- CPU **16.825908828s**，wall **15.819176637s**。每lane实际pair均2400000，请求数依次9,10,10,10,10,10,11,10。
- 首轮请求严格lane0..7，各起点都访问，不再有前一lane饿死其它流。
- 全部65个完整支持仍恰好包含全部2^k个mask：k2×1、k5×8、k6×8、k7×23、k8×15、k9×9、k10×1。完成不等于证明，截断也不当负例。
- 没有证书，因此无八轮真实验证，无九／十轮启动；新周期真实调用0／null0／fit0／正式重试0。前周期六七轮192次不重复计入此周期。

## 解释

轮转确实修复了调度覆盖；固定随机分散替代了无信息lexgrow，但在已访问的完整候选上仍无区分力。不能因此说调度修复无效、所有八轮cube无积分、或标准division property失败；也不能靠再添随机seed掩盖判定器支持膨胀。

下一步转到证明方法本身的只读论证：考虑保留相消系数、共享DAG代数化简、或对固定cube的符号SAT miter。不是直接安装新求解器／扩大真实样本，也不冒充已经实现更强DP。现有pysat可导入，z3／dd／pyeda不可导入；尚未因此启动任何solver实验。文献搜索仍遇Exa429，不能从未读全文宣布标准DP与当前方法等价或更强的具体结论。

## 边界与工程验收

生产src／tests／config／CLI未改。GLM只新建三个ignored文件，main独立最终**64项fake／公开小测试**、Ruff、两文件strict mypy与diffcheck通过。静态review最终PASS；中间交付63→64项身份变化在正式前以实测64和新testSHA解决，不是科学重试。

搜索SHA `3601fce23ba29465b1473de4326992426faf60b46d648137afff12aa4026c229`；testSHA `74c9ce095ebb923106f8e1361aa587110410369f4715ca2001bf06292455d2d4`。冻结`validation/integral-diversity/freeze-r8-search.json`。

main实际只读核全部artifact字节SHA、完整mask／maskSHA、pair加和、8lane首轮顺序／终态、无proof与选择为空；没有再传播全部候选、重算密码或再生标签。ledger已CLOSED；此结果是有界策略消融，不是更强密码学证明。
